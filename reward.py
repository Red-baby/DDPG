# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict

def compute_reward_mg(cfg, fb: dict, ref: Dict[str, float]) -> float:
    """
    MiniGOP reward:
      - Keep bit_avg within [low*ref_bit_avg, high*ref_bit_avg]
      - Keep vmaf_avg >= ref_vmaf_avg
    Positive reward for higher VMAF, penalties for bit out-of-band or quality drop.
    """
    bit_avg   = float(fb.get("bit_avg", 0.0) or 0.0)
    vmaf_avg  = float(fb.get("vmaf_avg", 0.0) or 0.0)
    ref_bavg  = float(ref.get("bit_avg", 0.0) or 0.0)
    ref_vavg  = float(ref.get("vmaf_avg", 0.0) or 0.0)

    low  = float(getattr(cfg, "rate_band_low",  0.90))
    high = float(getattr(cfg, "rate_band_high", 1.05))

    # bits ratio
    if ref_bavg <= 0.0:
        rho = 1.0
    else:
        rho = bit_avg / max(1e-6, ref_bavg)

    # penalties for band violation
    k_b = float(getattr(cfg, "mg_bits_penalty_gain", 2.0))
    pen_b = 0.0
    if rho < low:
        pen_b = (low - rho) * k_b
    elif rho > high:
        pen_b = (rho - high) * k_b

    # quality reward (symmetric)
    kq_pos = float(getattr(cfg, "mg_vmaf_gain_pos", 0.20))
    kq_neg = float(getattr(cfg, "mg_vmaf_gain_neg", 0.30))
    dv = vmaf_avg - ref_vavg
    if dv >= 0:
        rew_q = dv * kq_pos
    else:
        rew_q = dv * kq_neg  # negative

    r = rew_q - pen_b

    # optional terminal scaling
    if int(fb.get("gop_end", fb.get("gopend", 0)) or 0) == 1:
        r *= float(getattr(cfg, "end_penalty_scale", 1.0))

    # clip + global scale
    r = max(-float(getattr(cfg, "reward_clip", 3.0)),
            min(float(getattr(cfg, "reward_clip", 3.0)), r))
    return r * float(getattr(cfg, "reward_scale", 1.0))
