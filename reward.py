# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict

def compute_reward_mg(cfg, fb: dict, ref: Dict[str, float]) -> float:
    """
    MiniGOP reward with DUAL hard constraints:
      - HARD constraint 1: vmaf_avg MUST be >= ref_vmaf_avg (quality requirement)
      - HARD constraint 2: bit_avg MUST be in [0.9, 1.05] * ref_bit_avg (bitrate requirement)
      - Only when BOTH constraints are satisfied, give quality-based rewards
      - Violation of either constraint results in strong penalty
    """
    bit_avg   = float(fb.get("bit_avg", 0.0) or 0.0)
    vmaf_avg  = float(fb.get("vmaf_avg", 0.0) or 0.0)
    ref_bavg  = float(ref.get("bit_avg", 0.0) or 0.0)
    ref_vavg  = float(ref.get("vmaf_avg", 0.0) or 0.0)

    low  = float(getattr(cfg, "rate_band_low",  0.90))   # 硬下限
    high = float(getattr(cfg, "rate_band_high", 1.05))   # 硬上限

    # 通用系数
    end_scale   = float(getattr(cfg, "end_penalty_scale", 1.0))
    scale       = float(getattr(cfg, "reward_scale", 1.0))
    clip_val    = float(getattr(cfg, "reward_clip", 3.0))

    # 硬约束惩罚配置
    hard_pen_bitrate = float(getattr(cfg, "hard_penalty_bitrate", 15.0))  # 比特率违约惩罚
    hard_pen_quality = float(getattr(cfg, "hard_penalty_quality", 20.0))  # 质量违约惩罚
    bypass_clip = bool(getattr(cfg, "overbit_bypass_clip", True))

    # 计算比特率范围
    upper = high * ref_bavg if ref_bavg > 0.0 else float('inf')
    lower = low * ref_bavg if ref_bavg > 0.0 else 0.0

    # ---------- 硬约束1：质量必须优于参考 ----------
    if vmaf_avg < ref_vavg:
        quality_violation = ref_vavg - vmaf_avg  # 质量差距
        r = -hard_pen_quality * (1.0 + quality_violation / max(1.0, ref_vavg))
        if int(fb.get("gop_end", fb.get("gopend", 0)) or 0) == 1:
            r *= end_scale
        return (r * scale) if bypass_clip else (max(-clip_val, min(clip_val, r)) * scale)

    # ---------- 硬约束2：比特率必须在目标范围内 ----------
    if ref_bavg > 0.0 and (bit_avg < lower or bit_avg > upper):
        if bit_avg > upper:
            # 超上限
            over_ratio = bit_avg / max(1e-9, upper) - 1.0
            r = -hard_pen_bitrate * (1.0 + over_ratio)
        else:
            # 低于下限
            under_ratio = (lower - bit_avg) / max(1e-9, lower)
            r = -hard_pen_bitrate * (1.0 + under_ratio)
        
        if int(fb.get("gop_end", fb.get("gopend", 0)) or 0) == 1:
            r *= end_scale
        return (r * scale) if bypass_clip else (max(-clip_val, min(clip_val, r)) * scale)

    # ---------- 两个硬约束都满足：给予质量奖励 ----------
    # 质量奖励：鼓励进一步提升质量
    kq_pos = float(getattr(cfg, "mg_vmaf_gain_pos", 0.30))   # 质量提升奖励
    dv = vmaf_avg - ref_vavg  # 此时 dv >= 0（已通过硬约束）
    rew_q = dv * kq_pos

    # 比特率优化奖励：在合法范围内，越接近下限越好（节省比特率）
    bit_efficiency_gain = float(getattr(cfg, "bit_efficiency_gain", 0.10))
    if ref_bavg > 0.0:
        # 比特率效率：越接近下限奖励越高
        bit_ratio = bit_avg / ref_bavg
        efficiency_bonus = (high - bit_ratio) / (high - low) * bit_efficiency_gain
        r = rew_q + efficiency_bonus
    else:
        r = rew_q

    # GOP 结束时的缩放
    if int(fb.get("gop_end", fb.get("gopend", 0)) or 0) == 1:
        r *= end_scale

    # 裁剪与缩放
    r = max(-clip_val, min(clip_val, r))
    return r * scale