# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Dict
from utils import _float, _int

_INTERNAL: Dict[int, Dict[str, float]] = {}

def _ctx_for_doc(rq_meta: dict) -> dict:
    doc_id = int(_int(rq_meta.get("doc", -1)))
    if doc_id not in _INTERNAL:
        _INTERNAL[doc_id] = {
            "global_psnr_ema": 0.0, "psnr_ema_mg": 0.0,
            "prev_psnr": 0.0, "ud_db": 38.0,
            "ema_abs_q": 0.0, "ema_abs_b": 0.0,
            "mg_frame_idx": 0, "has_sc_in_mg": False,
            "lambda_b": 0.0, "lambda_q": 0.0,
            "ref_bits_total": 0.0, "ref_psnr_avg": 0.0,
        }
    return _INTERNAL[doc_id]

def _huber_abs(x: float, d: float) -> float:
    ax = abs(float(x)); d = float(max(1e-9, d))
    return (0.5*(ax*ax)/d) if ax <= d else (ax - 0.5*d)

def _psnr_to_utility(psnr_db: float) -> float:
    return 10.0 ** (float(psnr_db) / 20.0)

def compute_reward(cfg, fb: dict, rq_meta: dict,
                   prev_psnr_cached: float = 0.0,
                   mg_ctx: dict | None = None) -> float:
    """
    逐帧：质量+平滑项（无预算/预计量）；末段：与 2-pass 的 Lagrange 代价（支持末K帧分摊）。
    - 首帧不做“相邻帧梯度”惩罚（避免跨 miniGOP 惩罚）
    - 跨 miniGOP 对齐：仅段首前K帧贴近全局EMA，可在场景切换时弱化/关闭
    - 基础项先 clip/scale，再叠加末段代价（末段可不clip或另设较大clip）
    """
    # ---------- inputs ----------
    psnr = float(_float(fb.get("psnr_y", fb.get("psnr", 0.0))))
    bits = float(_float(fb.get("bits", 0.0)))
    frames_left = int(_int(rq_meta.get("frames_left_mg", 1)))
    mg_size = int(_int(rq_meta.get("mg_size", getattr(cfg, "mg_size", 16))))
    start_of_mg = (frames_left == mg_size)
    is_last_frame = (frames_left == 1)

    # 帧/段级 SC 标志
    is_sc_frame   = int(_int(rq_meta.get("scene_cut", rq_meta.get("is_scene_cut", 0)))) != 0
    is_sc_mg_meta = bool(rq_meta.get("scene_mg", rq_meta.get("is_sc_mg", False)))

    # ---------- context ----------
    if mg_ctx is None:
        mg_ctx = _ctx_for_doc(rq_meta)
    mg_ctx["mg_size"] = mg_size

    # ---------- Quality term: Nash log-utility 或线性差 ----------
    psnr_min_db = float(getattr(cfg, "psnr_min_db", 38.0))
    ud_db = float(mg_ctx.get("ud_db", psnr_min_db))
    beta_ud = float(getattr(cfg, "ud_ema_beta", 0.90))
    ud_db = beta_ud * ud_db + (1.0 - beta_ud) * psnr
    mg_ctx["ud_db"] = ud_db

    if bool(getattr(cfg, "use_nash", True)):
        U  = _psnr_to_utility(psnr)
        Ud = _psnr_to_utility(ud_db)
        barg = float(getattr(cfg, "nash_scale", 1.0)) * math.log(max(U - Ud, float(getattr(cfg, "nash_eps", 1e-6))))
    else:
        barg = float(getattr(cfg, "linq_scale", 1.0)) * (psnr - ud_db)

    # ---------- smoothing: intra-EMA ----------
    local_ema = float(mg_ctx.get("psnr_ema_mg", psnr))
    beta_local = float(getattr(cfg, "smooth_ema_beta", 0.90))
    delta_local = float(getattr(cfg, "smooth_huber_delta", 0.50))
    local_ema = beta_local * local_ema + (1.0 - beta_local) * psnr
    mg_ctx["psnr_ema_mg"] = local_ema
    w_smooth = float(getattr(cfg, "w_smooth", 0.35))
    smooth_pen = - w_smooth * _huber_abs(psnr - local_ema, delta_local)

    # ---------- smoothing: adjacent gradient（首帧不惩罚、不跨mg） ----------
    if start_of_mg:
        prev_psnr = psnr
        mg_ctx["mg_frame_idx"] = 1
        mg_ctx["has_sc_in_mg"] = False
    else:
        prev_psnr = float(mg_ctx.get("prev_psnr", prev_psnr_cached))
        mg_ctx["mg_frame_idx"] = int(_int(mg_ctx.get("mg_frame_idx", 1))) + 1

    if is_sc_frame:
        mg_ctx["has_sc_in_mg"] = True
    has_sc_in_mg = bool(mg_ctx.get("has_sc_in_mg", False) or is_sc_mg_meta)

    mg_ctx["prev_psnr"] = psnr

    delta_g = float(getattr(cfg, "grad_huber_delta", 0.70))
    w_grad  = float(getattr(cfg, "w_grad", 0.20))
    sc_grad_amp = float(getattr(cfg, "sc_grad_amp", 0.80))
    grad_pen = - (1.0 + (sc_grad_amp if has_sc_in_mg else 0.0)) \
               * w_grad * _huber_abs(psnr - prev_psnr, delta_g)

    # ---------- inter-MG alignment（段首前K帧贴全局EMA，SC时可弱化/关闭） ----------
    inter_pen = 0.0
    if bool(getattr(cfg, "inter_smooth_enable", True)):
        g_prev = float(mg_ctx.get("global_psnr_ema", 0.0))
        if g_prev <= 0.0:
            g_prev = psnr
        mg_idx = int(_int(mg_ctx.get("mg_frame_idx", 1)))
        delta_inter = float(getattr(cfg, "inter_smooth_huber_delta", 0.80))
        w_inter = float(getattr(cfg, "w_inter", 0.15))
        if has_sc_in_mg or is_sc_frame:
            w_inter *= float(getattr(cfg, "inter_sc_scale", 0.0))  # 默认0=关闭
        inter_K = int(getattr(cfg, "inter_smooth_first_k", 3))
        if mg_idx <= max(1, inter_K):
            inter_pen = - w_inter * _huber_abs(psnr - g_prev, delta_inter)
        beta_g = float(getattr(cfg, "inter_global_ema_beta", 0.98))
        mg_ctx["global_psnr_ema"] = beta_g * g_prev + (1.0 - beta_g) * psnr

    # ---------- base reward（仅基础项做小范围clip/scale） ----------
    r_base = barg + smooth_pen + grad_pen + inter_pen
    base_clip  = float(getattr(cfg, "reward_clip_base", 1.5))
    base_scale = float(getattr(cfg, "reward_scale_base", 1.0))
    r = max(-base_clip, min(base_clip, r_base)) * base_scale

    # ---------- 2-pass Lagrangian penalty（末K帧分摊，可放大、可不clip） ----------
    K = int(getattr(cfg, "end_span_k", 3))  # 末K帧分摊；K<=0则等同仅末帧
    if K <= 0:
        K = 1
    if frames_left <= K:
        ref_bits_total = float(_float(mg_ctx.get("ref_bits_total", rq_meta.get("ref_bits_total", 0.0))))
        ref_psnr_avg   = float(_float(mg_ctx.get("ref_psnr_avg",   rq_meta.get("ref_psnr_avg",   0.0))))
        # λ来源：mg_ctx -> rq_meta -> cfg 初值
        lambda_b = float(_float(mg_ctx.get("lambda_b",
                  rq_meta.get("lambda_b", getattr(cfg, "lag_b_init", 0.0)))))
        lambda_q = float(_float(mg_ctx.get("lambda_q",
                  rq_meta.get("lambda_q", getattr(cfg, "lag_q_init", 0.0)))))

        if ref_bits_total > 0.0:
            used_before = float(_float(rq_meta.get("mg_used_before", 0.0)))
            used_after  = used_before + bits
            rho = used_after / max(1.0, ref_bits_total)

            tol_b = float(getattr(cfg, "ref_bits_tol", 0.10))
            c_b = max(0.0, abs(rho - 1.0) - tol_b)

            frames_so_far = int(_int(mg_ctx.get("frames_so_far", rq_meta.get("frames_so_far", 0))))
            psnr_sum_prev = float(_float(rq_meta.get("mg_avg_psnr_so_far", -1.0)))
            if psnr_sum_prev >= 0.0 and frames_so_far > 0:
                avg_rl = (psnr_sum_prev * frames_so_far + psnr) / (frames_so_far + 1)
            else:
                avg_rl = psnr

            # 质量项门控：硬门 or 软门（指数衰减）
            if bool(getattr(cfg, "ref_gate_soften", False)):
                tol_q = float(getattr(cfg, "ref_bits_tol_q", tol_b))
                gate = math.exp(- max(0.0, abs(rho - 1.0) - tol_q) / max(1e-6, tol_q))
                c_q = gate * max(0.0, ref_psnr_avg - avg_rl)
            else:
                c_q = 0.0 if abs(rho - 1.0) > tol_b else max(0.0, ref_psnr_avg - avg_rl)

            # 线性分摊权 w ∈ (1..K)/sum(1..K)
            denom = K * (K + 1) / 2.0
            w_k = (K - frames_left + 1) / denom
            end_penalty_scale = float(getattr(cfg, "end_penalty_scale", 1.0))
            penalty = end_penalty_scale * w_k * (lambda_b * c_b + lambda_q * c_q)

            if bool(getattr(cfg, "end_penalty_no_clip", True)):
                r = r - penalty
            else:
                end_clip = float(getattr(cfg, "reward_clip_end", 4.0))
                r = max(-end_clip, min(end_clip, r - penalty))

    # ---------- final guard clip（宽边界，防数值爆） ----------
    final_clip = float(getattr(cfg, "reward_clip_final", 8.0))
    r = max(-final_clip, min(final_clip, r))
    return float(r)


# 兼容旧名
def compute_reward_dual(*args, **kwargs):
    return compute_reward(*args, **kwargs)
