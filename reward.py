# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict

def compute_reward_mg(cfg, fb: dict, ref: Dict[str, float]) -> float:
    """
    MiniGOP reward (hard cap on bitrate):
      - Hard constraint: bit_avg must NOT exceed high * ref_bit_avg. If violated,
        return a strong negative reward, with NO quality bonus allowed.
      - If under the hard cap:
          * Soft penalty when bit_avg < low * ref_bit_avg
          * Quality bonus: vmaf_avg - ref_vmaf_avg (as before)
    """
    bit_avg   = float(fb.get("bit_avg", 0.0) or 0.0)
    vmaf_avg  = float(fb.get("vmaf_avg", 0.0) or 0.0)
    ref_bavg  = float(ref.get("bit_avg", 0.0) or 0.0)
    ref_vavg  = float(ref.get("vmaf_avg", 0.0) or 0.0)

    low  = float(getattr(cfg, "rate_band_low",  0.90))   # 下限仍为软约束
    high = float(getattr(cfg, "rate_band_high", 1.05))   # 上限改为硬约束

    # 通用系数
    end_scale   = float(getattr(cfg, "end_penalty_scale", 1.0))
    scale       = float(getattr(cfg, "reward_scale", 1.0))
    clip_val    = float(getattr(cfg, "reward_clip", 3.0))

    # 硬惩罚配置（新）：超上限时直接用，不允许被质量奖励抵消
    hard_pen    = float(getattr(cfg, "overbit_hard_penalty", 10.0))   # 建议 > reward_clip
    bypass_clip = bool(getattr(cfg, "overbit_bypass_clip", True))     # True: 硬惩罚不受 clip 限制

    # ---------- 硬约束：bit_avg > high * ref_bavg ----------
    if ref_bavg > 0.0:
        upper = high * ref_bavg
        if bit_avg > upper:
            # 超上限：强负反馈，且强度随超出比例增加；不计算任何正向质量奖励
            over_ratio = bit_avg / max(1e-9, upper) - 1.0   # >0 表示超了多少
            r = - hard_pen * (1.0 + over_ratio)             # 至少 -hard_pen，更超越更重罚
            if int(fb.get("gop_end", fb.get("gopend", 0)) or 0) == 1:
                r *= end_scale
            # 是否绕过 clip（默认绕过，确保“硬约束”不被弱化）
            return (r * scale) if bypass_clip else (max(-clip_val, min(clip_val, r)) * scale)

    # ---------- 未触发硬约束：按原逻辑（低比特软罚 + 质量奖励） ----------
    # 比特比
    rho = 1.0 if ref_bavg <= 0.0 else (bit_avg / max(1e-6, ref_bavg))

    # 低于下限的软惩罚（维持你之前的需求：>=0.9×ref）
    k_b = float(getattr(cfg, "mg_bits_penalty_gain", 2.0))
    pen_b = (low - rho) * k_b if rho < low else 0.0

    # 质量项：高于参考给正向奖励，低于给惩罚
    kq_pos = float(getattr(cfg, "mg_vmaf_gain_pos", 0.20))
    kq_neg = float(getattr(cfg, "mg_vmaf_gain_neg", 0.30))
    dv = vmaf_avg - ref_vavg
    rew_q = dv * (kq_pos if dv >= 0 else kq_neg)

    r = rew_q - pen_b

    # GOP 结束可选缩放
    if int(fb.get("gop_end", fb.get("gopend", 0)) or 0) == 1:
        r *= end_scale

    # 常规裁剪与缩放
    r = max(-clip_val, min(clip_val, r))
    return r * scale
