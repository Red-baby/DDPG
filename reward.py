# -*- coding: utf-8 -*-
"""
Reward function with Nash-style log-utility, intra-miniGOP & inter-miniGOP quality smoothing,
and scene-cut aware handling. Designed to work both in training (with rewards) and deployment
(no reward) by ensuring *the policy* sees cross-miniGOP context via runner-fed mg_ctx/state
instead of reward-internal memory only.

Public API (kept stable):
    compute_reward(cfg, fb, rq_meta, prev_psnr_cached=0.0, mg_ctx=None) -> float
    compute_reward_dual(...)  # alias

Expected keys:
- fb: {"psnr_y" or "psnr": float, "bits": int}
- rq_meta: {
    "mg_bits_tgt", "mg_bits_rem", "frames_left_mg", "mg_size",
    "bits_pred_frame" or "bits_plan_frame",
    "scene_cut"/"is_scene_cut" (0/1),
    "scene_mg"/"is_sc_mg" (0/1)  # optional
}
- mg_ctx (dict) is maintained by the runner for BOTH train & deploy:
    {
      "lambda": float, "lambda_init"/"lambda_lo"/"lambda_hi",
      "global_psnr_ema": float, "mg_frame_idx": int, "has_sc_in_mg": bool
    }
"""
import math
from utils import _float, _int


# -------------------- utilities --------------------
def _huber_abs(x: float, delta: float) -> float:
    ax = abs(float(x)); d = float(max(1e-9, delta))
    return (0.5*(ax*ax)/d) if ax <= d else (ax - 0.5*d)

def _soft_tol_penalty(rel_err: float, tol: float, delta: float) -> float:
    over = max(0.0, abs(rel_err) - float(tol))
    return _huber_abs(over, delta)

def _psnr_to_utility(psnr_db: float) -> float:
    # stable and monotone
    return 10.0 ** (float(psnr_db) / 20.0)

def _get_bool(d: dict, *keys, default=False):
    for k in keys:
        if k in d:
            try:
                return bool(int(_int(d[k])))
            except Exception:
                return bool(d[k])
    return default

# -------------------- lambda update --------------------
def _nash_update_lambda(mg_ctx: dict, eta: float, err_mg: float,
                        lo: float, hi: float) -> None:
    lam = float(mg_ctx.get("lambda", mg_ctx.get("lambda_init", 1e-3)))
    lam = lam * math.exp(float(eta) * float(err_mg))   # multiplicative, keeps >0
    lam = float(max(lo, min(hi, lam)))
    mg_ctx["lambda"] = lam


# -------------------- main reward --------------------
def compute_reward(cfg, fb: dict, rq_meta: dict,
                   prev_psnr_cached: float = 0.0,
                   mg_ctx: dict | None = None) -> float:
    """Nash-like reward with quality smoothing and scene-cut-aware P boost."""
    psnr = float(_float(fb.get("psnr_y", fb.get("psnr", 0.0))))
    bits = float(_float(fb.get("bits", 0.0)))

    mg_bits_tgt   = _float(rq_meta.get("mg_bits_tgt", 0.0))
    mg_bits_rem   = _float(rq_meta.get("mg_bits_rem", 0.0))
    frames_left   = max(1, _int(rq_meta.get("frames_left_mg", 1)))
    mg_size       = max(1, _int(rq_meta.get("mg_size", frames_left)))
    pred_pf       = _float(rq_meta.get("bits_pred_frame", rq_meta.get("bits_plan_frame", 0.0)))

    # scene flags
    is_sc_frame   = _get_bool(rq_meta, "scene_cut", "is_scene_cut", default=False)
    is_sc_mg_meta = _get_bool(rq_meta, "scene_mg", "is_sc_mg", default=False)

    # ----- per-frame normalization scale (for bit term) -----
    if mg_bits_tgt > 0.0 and mg_size > 0:
        pf_norm = mg_bits_tgt / float(mg_size)
    elif pred_pf > 0.0:
        pf_norm = pred_pf
    else:
        pf_norm = float(getattr(cfg, "min_bpf", 500.0))

    # ----- Nash bargaining style log-utility (no per-layer bias) -----
    # disagreement point tracked via mg_ctx or default target
    psnr_min_db = float(getattr(cfg, "psnr_min_db", 38.0))
    if mg_ctx is not None:
        ud_db = float(mg_ctx.get("ud_db", psnr_min_db))
        beta_ud = float(getattr(cfg, "ud_ema_beta", 0.9))
        ud_db = beta_ud * ud_db + (1.0 - beta_ud) * psnr
        mg_ctx["ud_db"] = ud_db
    else:
        ud_db = psnr_min_db

    U  = _psnr_to_utility(psnr)
    Ud = _psnr_to_utility(ud_db)
    eps = float(getattr(cfg, "nash_eps", 1e-6))
    barg = math.log(max(U - Ud, eps))

    # ----- quality smoothing (intra- and inter-miniGOP) -----
    # a) intra-mg EMA deviation
    if mg_ctx is not None:
        local_ema = float(mg_ctx.get("psnr_ema_mg", psnr))
        beta_local = float(getattr(cfg, "smooth_ema_beta", 0.90))
        local_ema = beta_local * local_ema + (1.0 - beta_local) * psnr
        mg_ctx["psnr_ema_mg"] = local_ema
    else:
        local_ema = psnr

    delta_smooth = float(getattr(cfg, "smooth_huber_delta", 0.50))
    w_smooth     = float(getattr(cfg, "w_smooth", 0.35))
    smooth_pen = - w_smooth * _huber_abs(psnr - local_ema, delta_smooth)

    # b) adjacent-frame gradient with SC amplification
    if mg_ctx is not None:
        prev_psnr = float(mg_ctx.get("prev_psnr", psnr))
        mg_ctx["prev_psnr"] = psnr
        # track has_sc_in_mg and mg_frame_idx (runner should set these; fallback here)
        start_of_mg = (frames_left == mg_size)
        if start_of_mg:
            mg_ctx.setdefault("has_sc_in_mg", False)
            mg_ctx["mg_frame_idx"] = 1
        else:
            mg_ctx["mg_frame_idx"] = int(mg_ctx.get("mg_frame_idx", 1)) + 1
        if is_sc_frame:
            mg_ctx["has_sc_in_mg"] = True
        has_sc_in_mg = bool(mg_ctx.get("has_sc_in_mg", False) or is_sc_mg_meta)
    else:
        prev_psnr = psnr
        has_sc_in_mg = bool(is_sc_mg_meta or is_sc_frame)

    delta_grad = float(getattr(cfg, "grad_huber_delta", 0.70))
    w_grad     = float(getattr(cfg, "w_grad", 0.20))
    sc_grad_amp= float(getattr(cfg, "sc_grad_amp", 0.80))
    grad_weight= (1.0 + sc_grad_amp) if has_sc_in_mg else 1.0
    grad_pen   = - grad_weight * w_grad * _huber_abs(psnr - prev_psnr, delta_grad)

    # c) inter-mg alignment using *runner-fed* global_psnr_ema (policy sees same info)
    inter_enable= bool(getattr(cfg, "inter_smooth_enable", True))
    inter_pen = 0.0
    if inter_enable:
        # these are fed by runner at both train/infer time
        g_ema      = float((mg_ctx or {}).get("global_psnr_ema", psnr))
        mg_idx     = int((mg_ctx or {}).get("mg_frame_idx", 1))
        delta_inter= float(getattr(cfg, "inter_smooth_huber_delta", 0.80))
        w_inter    = float(getattr(cfg, "w_inter", 0.15))
        inter_K    = int(getattr(cfg, "inter_smooth_first_k", 3))
        inter_gate_sc = float(getattr(cfg, "inter_gate_sc", 0.0))  # 0 disables in SC mg
        if mg_idx <= max(1, inter_K):
            gate = (inter_gate_sc if has_sc_in_mg else 1.0)
            inter_pen = - gate * w_inter * _huber_abs(psnr - g_ema, delta_inter)

    # ----- bit-term with PSNR target gating -----
    if mg_ctx is not None and "lambda" not in mg_ctx:
        mg_ctx["lambda"] = float(getattr(cfg, "lambda_init", 1e-3))
        mg_ctx["lambda_lo"] = float(getattr(cfg, "lambda_lo", 1e-6))
        mg_ctx["lambda_hi"] = float(getattr(cfg, "lambda_hi", 1e+2))
    lam = float((mg_ctx or {}).get("lambda", getattr(cfg, "lambda_init", 1e-3)))

    if mg_ctx is not None:
        avg_psnr_ema = float(mg_ctx.get("avg_psnr_ema", psnr))
        ema_b = float(getattr(cfg, "avg_psnr_ema_beta", 0.98))
        avg_psnr_ema = ema_b * avg_psnr_ema + (1.0 - ema_b) * psnr
        mg_ctx["avg_psnr_ema"] = avg_psnr_ema
        psnr_target_db = float(getattr(cfg, "psnr_target_db", 40.5))
        gate = float(getattr(cfg, "bit_gate_hi", 1.0)) if avg_psnr_ema >= psnr_target_db \
            else float(getattr(cfg, "bit_gate_lo", 0.25))
    else:
        gate = 1.0

    bits_norm = bits / max(1.0, pf_norm)
    bit_term  = - gate * lam * bits_norm

    # ----- scene-cut mg: boost P (last frame) -----
    sc_p_quality_boost = float(getattr(cfg, "sc_p_quality_boost", 0.80))
    sc_p_bit_gate      = float(getattr(cfg, "sc_p_bit_gate", 0.60))
    is_last_frame_of_mg= (frames_left == 1)
    if has_sc_in_mg and is_last_frame_of_mg:
        barg     = (1.0 + sc_p_quality_boost) * barg
        bit_term = sc_p_bit_gate * bit_term

    # ----- update lambda at mg end -----
    if mg_ctx is not None and is_last_frame_of_mg and mg_bits_tgt > 0.0:
        used_before = max(0.0, mg_bits_tgt - mg_bits_rem)
        used_after  = used_before + bits
        err_mg = (used_after - mg_bits_tgt) / max(1.0, mg_bits_tgt)
        amp = float(getattr(cfg, "mg_early_amp", 1.0))
        exp = float(getattr(cfg, "mg_early_exp", 0.9))
        prog = float(max(0.0, min(1.0, (mg_bits_tgt - mg_bits_rem) / max(1.0, mg_bits_tgt))))
        early = (1.0 + amp * (prog ** exp))
        eta = float(getattr(cfg, "lambda_eta", 0.5))
        _nash_update_lambda(mg_ctx, eta * early * err_mg,
                            getattr(cfg, "lambda_lo", 1e-6),
                            getattr(cfg, "lambda_hi", 1e+2))

    # ----- magnitude balancing (stabilize training) -----
    use_balance = bool(getattr(cfg, "reward_balance_auto", True))
    bal_ema_mom = float(getattr(cfg, "reward_balance_momentum", 0.95))
    target_mag  = float(getattr(cfg, "reward_balance_target_mag", 0.8))
    s_q = 1.0; s_b = 1.0
    if use_balance and mg_ctx is not None:
        ema_q = float(mg_ctx.get("ema_abs_q", 0.0))
        ema_b = float(mg_ctx.get("ema_abs_b", 0.0))
        cur_abs_q = abs(barg + smooth_pen + grad_pen + inter_pen)
        cur_abs_b = abs(bit_term)
        ema_q = bal_ema_mom * ema_q + (1.0 - bal_ema_mom) * cur_abs_q
        ema_b = bal_ema_mom * ema_b + (1.0 - bal_ema_mom) * cur_abs_b
        s_q = (target_mag / max(1e-6, ema_q)) if ema_q > 0 else 1.0
        s_b = (target_mag / max(1e-6, ema_b)) if ema_b > 0 else 1.0
        s_q = float(max(0.5, min(2.0, s_q)))
        s_b = float(max(0.5, min(2.0, s_b)))
        mg_ctx["ema_abs_q"] = float(ema_q)
        mg_ctx["ema_abs_b"] = float(ema_b)

    # ----- aggregate -----
    r = s_q * (barg + smooth_pen + grad_pen + inter_pen) + s_b * (bit_term)

    # final mg deviation penalty (soft) to tighten budget
    if is_last_frame_of_mg and mg_bits_tgt > 0.0:
        used_before = max(0.0, mg_bits_tgt - mg_bits_rem)
        used_after  = used_before + bits
        err_mg = (used_after - mg_bits_tgt) / mg_bits_tgt
        mg_tol  = float(getattr(cfg, "mg_tol", 0.05))
        mg_hub  = float(getattr(cfg, "mg_huber_delta", 0.05))
        r += - _soft_tol_penalty(err_mg, tol=mg_tol, delta=mg_hub)

    clip_mag = float(getattr(cfg, "reward_clip", 1.5))
    scale    = float(getattr(cfg, "reward_scale", 1.0))
    r = float(max(-clip_mag, min(clip_mag, r))) * scale
    return r


def compute_reward_dual(cfg, fb: dict, rq_meta: dict,
                        prev_psnr_cached: float = 0.0,
                        mg_ctx: dict | None = None):
    return compute_reward(cfg, fb, rq_meta, prev_psnr_cached, mg_ctx)
