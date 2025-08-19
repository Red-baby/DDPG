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

def compute_reward(cfg, fb: dict, rq_meta: dict, prev_psnr_cached: float = 0.0,mg_ctx: dict | None = None, ) -> float:
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
        frame_over_pen = gate_lo * cfg.w_over * ((over_pf+0.6) ** 2)
        frame_under_pen = 0.0
        bit_saving_bonus = float(getattr(cfg, "w_save_bonus", 0.0)) * under_pf
    else:
        frame_over_pen = 4 * gate_lo * cfg.w_over * ((over_pf+0.6) ** 2)
        frame_under_pen = 4 * gate_hi * cfg.w_under * under_pf
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

    # ========== 2) mini-GOP 级累计分梯度惩罚（每帧都计算；越早超出惩罚越大） ==========
    mg_pen = 0.0
    if mg_bits_tgt > 0.0:
        used_before = max(0.0, mg_bits_tgt - mg_bits_rem)  # 到上一帧为止已用
        used_after = used_before + bits  # 加上本帧后的累计
        over_mg = max(0.0, used_after / mg_bits_tgt - 1.0)  # 累计超出比例（分梯度）

        if over_mg > 0.0:
            # —— 早超权重：剩余帧 / 总帧；越早（比例越接近1）→ 惩罚越大 —— #
            # 需要在 io_runner 侧把 mg_ctx={"frames_so_far": st["frames"], ...} 传进来（见下方备注）。
            if mg_ctx is not None:
                frames_so_far = int(mg_ctx.get("frames_so_far", 0))
                mg_total_frames = max(1, frames_so_far + int(frames_left_mg))  # 到当前帧为止的总帧估计
            else:
                mg_total_frames = max(1, int(frames_left_mg) + 1)  # 兜底：假设当前帧是第1帧
            early_ratio = float(frames_left_mg) / float(mg_total_frames)  # 0~1，越早越接近1
            early_boost = 1.0 + float(cfg.mg_early_amp) * (early_ratio ** float(cfg.mg_early_exp))

            # —— 常规累计惩罚（分梯度）+ 严重超出附加惩罚 —— #
            mg_pen = float(getattr(cfg, "w_mg_over", 0.8)) * (over_mg ** 2)

            mg_hard_thr = float(getattr(cfg, "mg_over_hard_ratio", 2.0)) - 1.0
            if over_mg >= max(0.0, mg_hard_thr):
                mg_pen += float(getattr(cfg, "w_mg_over_hard", 3.5)) * ((over_mg - mg_hard_thr) ** 2)

            # —— 质量门控 + 早超放大 —— #
            mg_pen *= (gate_lo if psnr_met else gate_hi) * early_boost
        else:
            mg_pen = 0.0

    # ========== 3) 质量稳定（ΔPSNR^2；未达标时弱化或关闭） ========== #
    prev_psnr = float(prev_psnr_cached) if prev_psnr_cached > 0.0 else psnr
    dps = (psnr - prev_psnr)

    # 欠标幅度（单位 dB）
    psnr_target = float(getattr(cfg, "psnr_target_db", 40.0))  # 你上面已算过 psnr_target/psnr_met，也可以直接复用
    deficit = max(0.0, psnr_target - psnr)

    # 欠标时的平滑权重缩放：w_smooth * scale * extra
    # - w_smooth_under_scale：统一缩小（0=完全关掉；0.25=保留25%）
    # - smooth_under_boost：随欠标再衰减（0=不随欠标变化）
    if deficit > 0.0:
        scale = float(getattr(cfg, "w_smooth_under_scale", 0.25))
        extra = 1.0 / (1.0 + float(getattr(cfg, "smooth_under_boost", 0.5)) * (deficit / max(1e-6, cfg.smooth_ref_db)))
        w_smooth_eff = cfg.w_smooth * scale * extra
    else:
        w_smooth_eff = cfg.w_smooth

    smooth_pen = w_smooth_eff * (dps / max(1e-6, cfg.smooth_ref_db)) ** 2

    # ========== 4) GOP 风险项（保留原逻辑，可整体关闭） ========== #
    if getattr(cfg, "use_gop_credit", True) and (gop_credit < 0.0) and (gop_bits_rem > 0.0):
        gop_risk_pen = cfg.w_gop_risk * (min(1.5, (-gop_credit / gop_bits_rem)) ** 2)
    else:
        gop_risk_pen = 0.0

    # ========== 5) PSNR 主收益（分段：最低线/目标/达标以上） ==========
    # 两条阈值优先读 rq_meta 的 score_min / score_avg，否则回退到 cfg
    psnr_min = float(getattr(cfg, "psnr_min_db", 37.0))
    psnr_tar = float(getattr(cfg, "psnr_target_db", 40.0))
    # 归一参考（用于“达标以上”的线性上升终点），沿用 cfg.psnr_norm
    psnr_norm = float(getattr(cfg, "psnr_norm", max(psnr_tar + 5.0, 45.0)))

    # 形状参数
    q_between_neg = float(getattr(cfg, "q_between_neg", 0.10))  # psnr==min 时的负幅度
    q_under_min_k = float(getattr(cfg, "q_under_min_scale", 1.00))  # 低于最低线的负斜率
    q_above_k = float(getattr(cfg, "q_above_scale", 1.00))  # 高于目标的正斜率
    q_cap = float(getattr(cfg, "q_gain_cap", 1.00))  # 夹紧

    # 保证阈值有序
    if psnr_tar <= psnr_min:
        psnr_tar = psnr_min + 1e-3

    span_min_tar = max(1e-6, psnr_tar - psnr_min)
    span_tar_norm = max(1e-6, psnr_norm - psnr_tar)

    # 分段核心值 q_core（未乘 w_psnr）
    if psnr < psnr_min:
        # 低于最低线：从 psnr=psnr_min 的 -q_between_neg 开始，越低越负
        q_core = -q_between_neg - q_under_min_k * ((psnr_min - psnr) / span_min_tar)
    elif psnr < psnr_tar:
        # 介于 min 与 target：略微为负，psnr 越接近 target 越靠近 0-
        q_core = -q_between_neg * ((psnr_tar - psnr) / span_min_tar)
    else:
        # 达标以上：从 0 起，线性增长；到 psnr_norm 大约增长到 +q_above_k
        q_core = q_above_k * ((psnr - psnr_tar) / span_tar_norm)

    # 夹紧以避免过大（可选）
    q_core = max(-q_cap, min(q_cap, q_core))

    # 乘以主权重
    q_gain = cfg.w_psnr * q_core

    # 汇总（PSNR正向；其余为惩罚；省码给奖励）
    r = q_gain - smooth_pen - frame_over_pen - frame_under_pen - frame_ineff_pen - mg_pen - gop_risk_pen + bit_saving_bonus
    return float(r) * 0.1  # 数值缩放，保持 Critic 稳定

