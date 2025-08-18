# -*- coding: utf-8 -*-
import math
from utils import _float, _int

def _mix_psnr_from_fb(cfg, fb: dict) -> float:
    """返回观测到的 PSNR（按 cfg.psnr_mode 使用 Y 或 YUV 融合）"""
    py = _float(fb.get("psnr_y", 0.0))
    if cfg.psnr_mode == "y":
        return py
    pu = _float(fb.get("psnr_u", 0.0))
    pv = _float(fb.get("psnr_v", 0.0))
    if py > 0 and pu > 0 and pv > 0:
        return (6*py + pu + pv) / 8.0
    return py

def _target_psnr(cfg, rq_meta: dict) -> float:
    """决定目标得分（dB）：优先 RL 侧 target；否则回落到 rq 的 score_*；再否则用 cfg.psnr_target_db。"""
    if getattr(cfg, "use_rl_targets", False):
        t = _float(getattr(cfg, "target_score_avg", 0.0)) or _float(getattr(cfg, "target_score_min", 0.0))
        if t > 0:
            return t
    t = _float(rq_meta.get("score_avg", 0.0)) or _float(rq_meta.get("score_min", 0.0)) or float(getattr(cfg, "psnr_target_db", 40.0))
    return t

def _target_bpf(cfg, rq_meta: dict) -> float:
    """决定目标每帧比特：优先 RL 侧 target；否则回落到 mg 的剩余预算均摊；最后不低于 min_bpf。"""
    if getattr(cfg, "use_rl_targets", False):
        bpf = float(getattr(cfg, "target_bpf", 0.0))
        if bpf > 0:
            return bpf
        br_kbps = float(getattr(cfg, "target_bitrate_kbps", 0.0))
        fps = float(getattr(cfg, "target_fps", 0.0))
        if br_kbps > 0 and fps > 0:
            return (br_kbps * 1000.0) / fps
    mg_bits_rem = _float(rq_meta.get("mg_bits_rem", 0.0))
    frames_left_mg = _int(rq_meta.get("frames_left_mg", 0))
    if mg_bits_rem > 0.0 and frames_left_mg >= 1:
        return max(getattr(cfg, "min_bpf", 0.0), mg_bits_rem / max(1, frames_left_mg))
    return float(getattr(cfg, "min_bpf", 0.0))

def compute_reward(cfg, fb: dict, rq_meta: dict, prev_psnr_cached: float = 0.0,
                   mg_info: dict | None = None,
                   global_info: dict | None = None) -> float:
    """
    奖励组成（按优先级）：
      1) score_term（最高优先）：鼓励 PSNR ≥ target_score（低于目标强烈拉负）
      2) bit_term  （第二优先）：惩罚当帧 bits 偏离 target_bpf 的程度（平方）
      3) smooth_pen：平滑项，惩罚 ΔPSNR^2
      4) mg_pen：mini-GOP 结束后对“平均比特>目标2x”和“平均PSNR<目标”的强惩罚
      5) global_pen：全局平均 bpf 明显偏离目标的强惩罚（逐帧累计）
    最终 r 按 cfg.reward_scale 缩放。
    """
    psnr = _mix_psnr_from_fb(cfg, fb)
    bits = _float(fb.get("bits", 0.0))

    tgt_psnr = _target_psnr(cfg, rq_meta)
    tgt_bpf  = _target_bpf(cfg, rq_meta)

    # 1) 目标得分项（最高优先）
    score_term = getattr(cfg, "w_score_main", 1.0) * ((psnr - tgt_psnr) / max(1e-6, getattr(cfg, "psnr_norm", 45.0)))

    # 2) 目标比特项（第二优先）——偏差平方惩罚
    dev = (bits - tgt_bpf) / max(1.0, tgt_bpf)
    bit_term = - getattr(cfg, "w_bit_main", 0.8) * (dev ** 2)

    # 3) 质量稳定（ΔPSNR^2）
    if prev_psnr_cached > 0.0:
        d_psnr = psnr - prev_psnr_cached
    else:
        d_psnr = 0.0
    smooth_pen = getattr(cfg, "w_smooth", 0.30) * (d_psnr / max(1e-6, getattr(cfg, "smooth_ref_db", 5.0))) ** 2

    # 4) mini-GOP 强惩罚（在 mini-GOP 最后一帧触发）
    mg_pen = 0.0
    if mg_info and mg_info.get("done", False):
        mg_avg_bits = float(mg_info.get("avg_bits", 0.0))
        mg_avg_psnr = float(mg_info.get("avg_psnr", 0.0))
        ratio = mg_avg_bits / max(1.0, tgt_bpf)
        thr = getattr(cfg, "overshoot_factor_mg", 2.0)
        if ratio > thr:
            mg_pen += getattr(cfg, "w_mg_overshoot_hard", 2.0) * (ratio - thr) ** 2
        # 平均 PSNR 低于目标
        if (mg_avg_psnr > 0.0) and (tgt_psnr > 0.0) and (mg_avg_psnr < tgt_psnr):
            mg_pen += getattr(cfg, "w_mg_score_below_hard", 2.0) * (tgt_psnr - mg_avg_psnr) ** 2

    # 5) 全局最终偏差强惩罚（逐帧累计）
    global_pen = 0.0
    if global_info and getattr(cfg, "use_rl_targets", False):
        g_frames = int(global_info.get("frames", 0))
        g_avg_bpf = float(global_info.get("avg_bits", 0.0))
        if g_frames > 0 and tgt_bpf > 0:
            g_dev = abs(g_avg_bpf / tgt_bpf - 1.0)
            tol = float(getattr(cfg, "global_bits_dev_tol", 0.15))
            if g_dev > tol:
                global_pen = float(getattr(cfg, "w_global_bits_dev", 0.5)) * (g_dev - tol) ** 2

    r = score_term + bit_term - smooth_pen - mg_pen - global_pen

    # GOP 风险项（若仍想联动 gop_credit，可保留）
    gop_credit = _float(rq_meta.get("gop_credit", 0.0))
    gop_bits_rem = _float(rq_meta.get("gop_bits_rem", 0.0))
    if getattr(cfg, "use_gop_credit", True) and (gop_credit < 0.0) and (gop_bits_rem > 0.0):
        r -= getattr(cfg, "w_gop_risk", 0.2) * (min(1.5, (-gop_credit / gop_bits_rem)) ** 2)

    return float(r * float(getattr(cfg, "reward_scale", 1.0)))
