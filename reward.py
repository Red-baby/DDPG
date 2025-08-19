# -*- coding: utf-8 -*-
import math
from utils import _float, _int

def _mix_psnr_from_fb(cfg, fb: dict) -> float:
    """返回 PSNR（根据 cfg.psnr_mode 使用 Y 或 YUV 融合）"""
    py = _float(fb.get("psnr_y", 0.0))
    if cfg.psnr_mode == "y":
        return py
    pu = _float(fb.get("psnr_u", 0.0))
    pv = _float(fb.get("psnr_v", 0.0))
    if py > 0 and pu > 0 and pv > 0:
        return (6*py + pu + pv) / 8.0  # 6:1:1 加权
    return py

def compute_reward(cfg, fb: dict, rq_meta: dict, prev_psnr_cached: float = 0.0) -> float:
    """
    目标层次：PSNR > 稳定 > 比特。
    - 帧级：优先使用 rq 的 bits_plan_frame 与本帧实际 bits 做偏差（分梯度）；
            当 norm_pf ≥ over_hard_ratio_frame（如2×）时，叠加“硬惩罚”；
            PSNR 达标 → 欠码不罚且奖励省码；未达标 → 欠/超都较严格；
            若 ΔPSNR 很小但超码明显 → 低效超码惩罚。
    - mini-GOP 级：按“到当前为止的累计超出比例”每帧分梯度惩罚；
                   若累计已 ≥ mg_over_hard_ratio（如2×），再叠加硬惩罚；
                   受 PSNR 门控（达标宽松、未达标严格）。
    """
    # —— 观测 —— #
    psnr = _mix_psnr_from_fb(cfg, fb)
    bits = _float(fb.get("bits", 0.0))

    # —— rq / meta（动作前视图） —— #
    plan_pf        = _float(rq_meta.get("bits_plan_frame", 0.0))  # 帧级预估码率（关键）
    mg_bits_tgt    = _float(rq_meta.get("mg_bits_tgt", 0.0))
    mg_bits_rem    = _float(rq_meta.get("mg_bits_rem", 0.0))
    frames_left_mg = _int(rq_meta.get("frames_left_mg", 0))
    gop_credit     = _float(rq_meta.get("gop_credit", 0.0))
    frames_left_gop= _int(rq_meta.get("frames_left_gop", 0))
    gop_bits_rem   = _float(rq_meta.get("gop_bits_rem", 0.0))

    # —— 目标 PSNR —— #
    psnr_target = float(getattr(cfg, "psnr_target_db", 40.0))
    psnr_met = (psnr >= psnr_target)

    # —— 回退 BPF（若缺 bits_plan_frame）+ GOP credit 微调 —— #
    if mg_bits_rem > 0.0 and frames_left_mg >= 1:
        bpf_mg = mg_bits_rem / max(1, frames_left_mg)
    else:
        bpf_mg = max(cfg.min_bpf, 500.0)
    credit_share = 0.0
    if getattr(cfg, "use_gop_credit", True) and frames_left_gop >= 1 and abs(gop_credit) > 0.0:
        credit_share = cfg.alpha_credit_share * (gop_credit / max(1, frames_left_gop))
    bpf_fallback = max(cfg.min_bpf, bpf_mg + credit_share)

    # ========== 1) 帧级分梯度偏差（优先 plan_pf） ========== #
    denom_pf = plan_pf if plan_pf > 0 else bpf_fallback
    norm_pf = bits / max(1.0, denom_pf)  # 实际/预估
    over_pf = max(0.0, norm_pf - 1.0)
    under_pf = max(0.0, 1.0 - norm_pf)

    gate_lo, gate_hi = float(cfg.bit_gate_lo), float(cfg.bit_gate_hi)

    # 达标：欠码不罚 + 省码奖励；超码宽松；未达标：欠/超都较严格
    if psnr_met:
        frame_over_pen = gate_lo * cfg.w_over * (over_pf ** 2)
        frame_under_pen = 0.0
        bit_saving_bonus = float(getattr(cfg, "w_save_bonus", 0.0)) * under_pf
    else:
        frame_over_pen = gate_hi * cfg.w_over * (over_pf ** 2)
        frame_under_pen = gate_hi * cfg.w_under * (under_pf ** 2)
        bit_saving_bonus = 0.0

    # 帧级硬惩罚：若该帧 ≥ 指定倍数阈值（默认2×）
    over_hard_ratio = float(getattr(cfg, "over_hard_ratio_frame", 2.0))
    if norm_pf >= over_hard_ratio:
        # 额外叠加一个强惩罚项（与 (norm_pf - over_hard_ratio)^2 成正比）
        frame_over_hard = float(getattr(cfg, "w_over_hard_frame", 3.0)) * ((norm_pf - over_hard_ratio) ** 2)
        # 同样受 PSNR 门控
        frame_over_pen += (gate_lo if psnr_met else gate_hi) * frame_over_hard

    # 低效超码惩罚：ΔPSNR 很小却超码明显
    if prev_psnr_cached > 0.0:
        d_psnr = psnr - float(prev_psnr_cached)
    else:
        d_psnr = 0.0
    if over_pf > 0.0 and d_psnr < float(getattr(cfg, "eff_gain_eps", 0.10)):
        ineff = (1.0 - d_psnr / max(float(getattr(cfg, "eff_gain_eps", 0.10)), 1e-6))  # 0~1
        frame_ineff_pen = float(getattr(cfg, "w_ineff_over", 1.0)) * over_pf * max(0.0, ineff)
    else:
        frame_ineff_pen = 0.0

    # ========== 2) mini-GOP 级累计分梯度惩罚（每帧都计算） ========== #
    mg_pen = 0.0
    if mg_bits_tgt > 0.0:
        used_before = max(0.0, mg_bits_tgt - mg_bits_rem)  # 到上一帧为止已用
        used_after  = used_before + bits                   # 加上本帧后的累计
        over_mg = max(0.0, used_after / mg_bits_tgt - 1.0) # 累计超出比例（分梯度）

        # 常规累计惩罚（分梯度，每帧生效）
        mg_pen = float(getattr(cfg, "w_mg_over", 0.8)) * (over_mg ** 2)

        # 累计严重超出（≥ mg_over_hard_ratio）再叠加强惩罚
        mg_hard_thr = float(getattr(cfg, "mg_over_hard_ratio", 2.0)) - 1.0
        if over_mg >= max(0.0, mg_hard_thr):
            mg_pen += float(getattr(cfg, "w_mg_over_hard", 3.5)) * ((over_mg - mg_hard_thr) ** 2)

        # 仍受 PSNR 门控（达标更宽松）
        mg_pen *= (gate_lo if psnr_met else gate_hi)

    # ========== 3) 质量稳定（ΔPSNR^2） ========== #
    prev_psnr = float(prev_psnr_cached) if prev_psnr_cached > 0.0 else psnr
    dps = (psnr - prev_psnr)
    smooth_pen = cfg.w_smooth * (dps / max(1e-6, cfg.smooth_ref_db)) ** 2

    # ========== 4) GOP 风险项（保留原逻辑，可整体关闭） ========== #
    if getattr(cfg, "use_gop_credit", True) and (gop_credit < 0.0) and (gop_bits_rem > 0.0):
        gop_risk_pen = cfg.w_gop_risk * (min(1.5, (-gop_credit / gop_bits_rem)) ** 2)
    else:
        gop_risk_pen = 0.0

    # ========== 5) PSNR 主收益 ========== #
    q_gain = cfg.w_psnr * (psnr / max(1e-6, cfg.psnr_norm))

    # 汇总（PSNR正向；其余为惩罚；省码给奖励）
    r = q_gain - smooth_pen - frame_over_pen - frame_under_pen - frame_ineff_pen - mg_pen - gop_risk_pen + bit_saving_bonus
    return float(r) * 0.1  # 数值缩放，保持 Critic 稳定

