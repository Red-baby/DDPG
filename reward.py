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
    层次化目标（优先级：PSNR > 稳定 > 比特）：
      r = w_psnr*(psnr/psnr_norm)  -  smooth_pen  -  bit_gate * bit_dev_pen  -  gop_risk
    其中 bit_gate 由 PSNR 是否达标控制：psnr>=target → 放大比特要求；否则弱化比特要求。
    """
    psnr = _mix_psnr_from_fb(cfg, fb)
    bits = _float(fb.get("bits", 0.0))

    mg_bits_rem     = _float(rq_meta.get("mg_bits_rem", 0.0))
    frames_left_mg  = _int(rq_meta.get("frames_left_mg", 0))
    gop_credit      = _float(rq_meta.get("gop_credit", 0.0))
    frames_left_gop = _int(rq_meta.get("frames_left_gop", 0))
    gop_bits_rem    = _float(rq_meta.get("gop_bits_rem", 0.0))

    # 目标 PSNR：优先读 rq 传来的 score / 阈值，否则用 cfg.psnr_target_db
    psnr_target = _float(rq_meta.get("score_avg", 0.0)) or _float(rq_meta.get("score_min", 0.0)) or float(cfg.psnr_target_db)

    # mini-GOP 动态目标 BPF
    if mg_bits_rem > 0.0 and frames_left_mg >= 1:
        bpf_mg = mg_bits_rem / max(1, frames_left_mg)
    else:
        bpf_mg = max(cfg.min_bpf, 500.0)  # 兜底

    # GOP 信用软调整（可整体关闭）
    credit_share = 0.0
    if cfg.use_gop_credit and frames_left_gop >= 1 and abs(gop_credit) > 0.0:
        credit_share = cfg.alpha_credit_share * (gop_credit / max(1, frames_left_gop))

    bpf = max(cfg.min_bpf, bpf_mg + credit_share)

    # 码率偏差罚（基础权重）
    norm = bits / max(bpf, 1.0)
    if norm >= 1.0:
        bit_dev_pen = cfg.w_over * (norm - 1.0) ** 2
    else:
        bit_dev_pen = cfg.w_under * (1.0 - norm) ** 2

    # PSNR 门控后的比特权重倍率
    bit_gate = (cfg.bit_gate_hi if psnr >= psnr_target else cfg.bit_gate_lo)

    # 平滑（ΔPSNR^2），prev_psnr=0 时不罚
    if prev_psnr_cached > 0.0:
        prev_psnr = prev_psnr_cached
    else:
        prev_psnr = psnr
    d_psnr = (psnr - prev_psnr)
    smooth_pen = cfg.w_smooth * (d_psnr / max(1e-6, cfg.smooth_ref_db)) ** 2

    # GOP 风险（可随 use_gop_credit 一并关掉）
    if cfg.use_gop_credit and (gop_credit < 0.0) and (gop_bits_rem > 0.0):
        gop_risk_pen = cfg.w_gop_risk * (min(1.5, (-gop_credit / gop_bits_rem)) ** 2)
    else:
        gop_risk_pen = 0.0

    q_gain = cfg.w_psnr * (psnr / max(1e-6, cfg.psnr_norm))
    r = q_gain - smooth_pen - bit_gate * bit_dev_pen - gop_risk_pen
    return float(r)*0.1
