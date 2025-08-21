# -*- coding: utf-8 -*-
import math
from utils import _float, _int

def _huber_abs(x: float, delta: float) -> float:
    ax = abs(float(x)); d = float(max(1e-9, delta))
    return (0.5 * (ax*ax) / d) if ax <= d else (ax - 0.5 * d)

def _soft_tol_penalty(rel_err: float, tol: float, delta: float) -> float:
    """对 |相对误差| 先减容忍 tol，再做 Huber；结果≥0。"""
    over = max(0.0, abs(rel_err) - float(tol))
    return _huber_abs(over, delta)

def compute_reward(cfg, fb: dict, rq_meta: dict,
                   prev_psnr_cached: float = 0.0,
                   mg_ctx: dict | None = None) -> float:
    """
    r = s_q * w_q * q_term + s_b * ( w_bf * b_frame + w_bmg * b_mg )

    - q_term：仅基于 PSNR 的单调奖励，使用 tanh 压到 (-1,1)
        q_term = tanh( (psnr - q_mid_db) / q_span_db )

    - b_frame：帧级码率偏差的“容忍+Huber”惩罚（≤0）
        使用预测帧比特或 mini-GOP 的剩余均摊作为参考：
          ref_pf = bits_pred_frame  (若提供且>0)
                or mg_bits_rem/frames_left_mg
                or bits_plan_frame
                or cfg.min_bpf
        err_pf = (bits - ref_pf) / ref_pf
        b_frame = - SoftTolHuber(|err_pf|, tol=pf_tol, delta=pf_huber_delta)

      ——“单帧有偏差是正常的”体现在 tol（如 0.20=±20%）里，tol 内不罚，超出再罚。
         你也可以把 w_bf 设小一些，让单帧偏差更多地“交给 agent 学”。

    - b_mg：mini-GOP 级的强惩罚（仅在 mini-GOP 末帧给；≤0）
        在 frames_left_mg == 1 时，计算：
          used_after = (mg_bits_tgt - mg_bits_rem) + bits
          err_mg = (used_after - mg_bits_tgt) / mg_bits_tgt
          b_mg = - SoftTolHuber(|err_mg|, tol=mg_tol, delta=mg_huber_delta)

    - s_q, s_b：可选“自适应平衡系数”，用 EMA 对齐两路项的平均绝对幅度，
      让“失真奖励”和“码率惩罚”处在同一量级，避免谁压制谁。
      关闭它也行（见 config）。

    返回值会被 clip 到 [-clip, +clip]，再乘 reward_scale。
    """
    # ---------- 观测 ----------
    psnr = float(_float(fb.get("psnr_y", 0.0)))  # 也可用你之前的 6:1:1 融合
    bits = float(_float(fb.get("bits", 0.0)))

    mg_bits_tgt   = _float(rq_meta.get("mg_bits_tgt", 0.0))
    mg_bits_rem   = _float(rq_meta.get("mg_bits_rem", 0.0))
    frames_left_mg= _int(rq_meta.get("frames_left_mg", 0))
    plan_pf       = _float(rq_meta.get("bits_plan_frame", 0.0))
    pred_pf       = _float(rq_meta.get("bits_pred_frame", 0.0))  # 你提供的“单帧预测比特”

    # ---------- 配置 ----------
    q_mid   = float(getattr(cfg, "q_mid_db", 38.0))   # PSNR 中位
    q_span  = float(getattr(cfg, "q_span_db",  2.0))  # 每 2dB 变化 ~ tanh 的 1 个尺度
    w_q     = float(getattr(cfg, "w_q",    1.0))

    pf_tol  = float(getattr(cfg, "pf_tol", 0.20))     # 单帧容忍 ±20%
    pf_hub  = float(getattr(cfg, "pf_huber_delta", 0.20))
    w_bf    = float(getattr(cfg, "w_bf",   0.6))

    mg_tol  = float(getattr(cfg, "mg_tol", 0.05))     # mini-GOP 容忍 ±5%
    mg_hub  = float(getattr(cfg, "mg_huber_delta", 0.05))
    w_bmg   = float(getattr(cfg, "w_bmg",  1.0))

    # 自适应平衡（让两路项同量级）
    use_balance = bool(getattr(cfg, "reward_balance_auto", True))
    bal_ema_mom = float(getattr(cfg, "reward_balance_momentum", 0.95))
    target_mag  = float(getattr(cfg, "reward_balance_target_mag", 0.8))  # 目标平均幅度

    # 其他
    min_bpf     = float(getattr(cfg, "min_bpf", 500.0))
    clip_mag    = float(getattr(cfg, "reward_clip", 1.5))
    scale       = float(getattr(cfg, "reward_scale", 1.0))

    # ---------- 1) 失真项：PSNR 单调奖励 ----------
    q_term = math.tanh((psnr - q_mid) / max(1e-6, q_span))  # 约 (-0.96, +0.96)

    # ---------- 2) 帧级码率项（容忍 + Huber） ----------
    if pred_pf > 0.0:
        ref_pf = pred_pf
    elif mg_bits_rem > 0.0 and frames_left_mg >= 1:
        ref_pf = mg_bits_rem / max(1, frames_left_mg)
    elif plan_pf > 0.0:
        ref_pf = plan_pf
    else:
        ref_pf = min_bpf

    err_pf = (bits - ref_pf) / max(1.0, ref_pf)
    b_frame_mag = _soft_tol_penalty(err_pf, tol=pf_tol, delta=pf_hub)
    b_frame = - b_frame_mag

    # ---------- 3) mini-GOP 级码率项（仅末帧） ----------
    b_mg = 0.0
    if frames_left_mg == 1 and mg_bits_tgt > 0.0:
        used_before = max(0.0, mg_bits_tgt - mg_bits_rem)
        used_after  = used_before + bits
        err_mg = (used_after - mg_bits_tgt) / mg_bits_tgt
        b_mg_mag = _soft_tol_penalty(err_mg, tol=mg_tol, delta=mg_hub)
        b_mg = - b_mg_mag
    else:
        b_mg_mag = 0.0

    # ---------- 4) 自适应平衡（可关） ----------
    s_q = 1.0; s_b = 1.0
    if use_balance:
        if mg_ctx is not None:
            ema_q = float(mg_ctx.get("ema_abs_q", 0.0))
            ema_b = float(mg_ctx.get("ema_abs_b", 0.0))
        else:
            ema_q = 0.0; ema_b = 0.0

        cur_abs_q = abs(w_q * q_term)
        cur_abs_b = abs(w_bf * b_frame + w_bmg * b_mg)

        ema_q = bal_ema_mom * ema_q + (1.0 - bal_ema_mom) * cur_abs_q
        ema_b = bal_ema_mom * ema_b + (1.0 - bal_ema_mom) * cur_abs_b

        # 目标把两路项都拉到 ~target_mag 的平均幅度
        s_q = (target_mag / max(1e-6, ema_q)) if ema_q > 0 else 1.0
        s_b = (target_mag / max(1e-6, ema_b)) if ema_b > 0 else 1.0
        # 可再夹一下，避免剧烈缩放
        s_q = float(max(0.5, min(2.0, s_q)))
        s_b = float(max(0.5, min(2.0, s_b)))

        if mg_ctx is not None:
            mg_ctx["ema_abs_q"] = ema_q
            mg_ctx["ema_abs_b"] = ema_b

    # ---------- 5) 汇总 ----------
    r = s_q * (w_q * q_term) + s_b * (w_bf * b_frame + w_bmg * b_mg)
    r = max(-clip_mag, min(clip_mag, r))
    return float(r * scale)


def compute_reward_dual(cfg, fb: dict, rq_meta: dict) -> tuple[float, float]:
    """Return (rD, rR) for the dual-critic setting.
    - rD: per-frame quality reward (PSNR-monotonic, normalized to ~(-1,1)).
    - rR: GOP/mini-GOP terminal rate reward (0 at non-terminal steps; negative absolute deviation at terminal).
    This mirrors Sec. 2.3 (r_D immediate; r_R only at terminal) in the dual-critic paper.
    """
    # Observations
    psnr_y = _float(fb.get("psnr_y", 0.0))
    pu = _float(fb.get("psnr_u", 0.0)); pv = _float(fb.get("psnr_v", 0.0))
    psnr_mode = str(getattr(cfg, "psnr_mode", "y")).lower()
    if psnr_mode == "yuv" and psnr_y > 0 and pu > 0 and pv > 0:
        psnr = (6*psnr_y + pu + pv) / 8.0
    else:
        psnr = psnr_y

    # Distortion reward: monotonic in PSNR, tanh-shaped (scaled by q_span around q_mid)
    q_mid = float(getattr(cfg, "q_mid_db", 38.0))
    q_span = float(getattr(cfg, "q_span_db", 2.0))
    rD = float(math.tanh((psnr - q_mid) / max(1e-6, q_span)))

    # Rate reward: only at terminal of mini-GOP/GOP
    mg_bits_tgt = _float(rq_meta.get("mg_bits_tgt", 0.0))
    fac = float(getattr(cfg, "rR_target_factor", 1.0))
    mg_bits_tgt_eff = mg_bits_tgt * fac
    mg_bits_rem = _float(rq_meta.get("mg_bits_rem", 0.0))
    bits = _float(fb.get("bits", 0.0))
    # frames_left_mg can be 0 at terminal (some encoders emit 0 or 1; we accept both conventions)
    flm_val = fb.get("frames_left_mg", rq_meta.get("frames_left_mg", None))
    terminal = (flm_val is not None) and (int(flm_val) == 0)

    # —— mini-GOP 末帧的码率误差项（dual: rR）
    if terminal and mg_bits_tgt > 0.0:
        # 1) 优先使用 RL 端传入的“上一帧为止的真实累计”
        hint = rq_meta.get("mg_used_before", None)
        if hint is not None:
            used_before = float(hint)
        else:
            # 2) 回退用 (tgt - rem)，注意不要 clamp；允许 rem 为负表示已超
            used_before = (mg_bits_tgt - mg_bits_rem)

        used_after = used_before + bits  # 整个 mini-GOP 实际总用比特
        # 若你启用了 rR_target_factor（放宽/收紧目标），这里先得到有效目标
        tgt_eff = mg_bits_tgt * float(getattr(cfg, "rR_target_factor", 1.0))
        rR = - abs(used_after - tgt_eff) / max(1.0, tgt_eff)
    else:
        rR = 0.0

    return float(rD), float(rR)
