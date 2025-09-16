# -*- coding: utf-8 -*-
"""
state.py — miniGOP 向量状态（一次性输出当前 miniGOP 所有帧 QP 的方案）

功能：
- build_mg_state(rq, cfg)：
  把 mg_*.rq.json 中的 frames[*] 提取为 16×7 特征矩阵（不足复制末帧补齐），
  再展平为一维张量，返回 (state_tensor, meta, poc_list)：
    state_tensor: torch.FloatTensor, shape=[16*7]
    meta: {
        "mg_id": int, "mg_size": int, "mg_bits_tgt": float,
        "gop_end_hint": 0/1, "base_q_list": List[int]
    }
    poc_list: List[int], 仅前 mg_size 个用于 2-pass 参考统计
- 可选：build_frame_state(...) 保留一个精简逐帧状态接口，避免旧代码报错。

特征顺序（每帧 7 维）：
[ temporal_id,
  base_q,
  log1p(bits_plan),
  log1p(i_rdcost_accum),
  log1p(i_intra_cost_accum),
  log1p(i_inter_cost_accum),
  cor_coef ]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
import torch

from utils import _float, _int

# ===== miniGOP 特征定义 =====
MG_FEATS: tuple[str, ...] = (
    "temporal_id",
    "base_q",
    "bits_plan",
    "i_rdcost_accum",
    "i_intra_cost_accum",
    "i_inter_cost_accum",
    "cor_coef",
)
MG_FEAT_DIM: int = len(MG_FEATS)   # = 7


def _safe_float(v, dv: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(dv)


def _row_from_frame(frame: Dict[str, Any], qp_mid: int) -> tuple[list[float], int, int]:
    """
    从单帧 RQ 记录抽取一行特征（长度7），并返回 (row, base_q_int, poc)
    """
    tid = _safe_float(frame.get("temporal_id", 1.0), 1.0)
    bq  = _safe_float(frame.get("base_q", qp_mid), qp_mid)

    bp  = _safe_float(frame.get("bits_plan", 0.0), 0.0)
    rc  = _safe_float(frame.get("i_rdcost_accum", 0.0), 0.0)
    ci  = _safe_float(frame.get("i_intra_cost_accum", 0.0), 0.0)
    ce  = _safe_float(frame.get("i_inter_cost_accum", 0.0), 0.0)
    cc  = _safe_float(frame.get("cor_coef", 0.0), 0.0)

    row = [
        float(tid),
        float(bq),
        float(np.log1p(max(0.0, bp))),
        float(np.log1p(max(0.0, rc))),
        float(np.log1p(max(0.0, ci))),
        float(np.log1p(max(0.0, ce))),
        float(cc),
    ]
    base_q_int = int(round(bq))
    poc = int(_int(frame.get("poc", -1)))
    return row, base_q_int, poc


def build_mg_state(rq: Dict[str, Any], cfg) -> tuple[torch.Tensor, Dict[str, Any], List[int]]:
    """
    把 mg_*.rq.json 转成 (state_tensor, meta, poc_list)
    - state_tensor: [MG_MAX * 7] FloatTensor
    - meta: {mg_id, mg_size, mg_bits_tgt, gop_end_hint, base_q_list}
    - poc_list: 前 mg_size 个 POC（仅对参考统计有效）
    """
    MG_MAX = int(getattr(cfg, "mg_size", 16))
    qp_min = int(getattr(cfg, "qp_min", 0))
    qp_max = int(getattr(cfg, "qp_max", 255))
    qp_mid = int((qp_min + qp_max) // 2)

    frames = list(rq.get("frames", []) or [])
    n = len(frames)

    rows: list[list[float]] = []
    base_qs: list[int] = []
    pocs: list[int] = []

    if n <= 0:
        # 空 miniGOP 兜底：全 0
        rows = [[0.0] * MG_FEAT_DIM for _ in range(MG_MAX)]
        base_qs = [qp_mid] * MG_MAX
        pocs = []
    else:
        for f in frames:
            row, bqi, poc = _row_from_frame(f, qp_mid)
            rows.append(row)
            base_qs.append(max(qp_min, min(qp_max, int(bqi))))
            pocs.append(poc)

        # 不足 MG_MAX 时复制末帧补齐（行、base_q 均复制）
        if n < MG_MAX:
            last_row = rows[-1]
            last_bq  = base_qs[-1]
            add = MG_MAX - n
            rows.extend([list(last_row) for _ in range(add)])
            base_qs.extend([int(last_bq) for _ in range(add)])

    # 只把前 MG_MAX 帧进入状态（冗余截断）
    x = np.asarray(rows[:MG_MAX], dtype=np.float32)       # [MG_MAX, 7]
    s = torch.from_numpy(x.reshape(-1))                   # [MG_MAX * 7]

    mg_size = int(_int(rq.get("mg_size", n if n > 0 else MG_MAX)))
    mg_size = max(1, min(MG_MAX, mg_size))
    meta = {
        "mg_id": int(_int(rq.get("mg_id", -1))),
        "mg_size": mg_size,
        "mg_bits_tgt": float(_float(rq.get("mg_bits_tgt", 0.0))),
        "base_q_list": [int(b) for b in base_qs[:MG_MAX]],
    }
    # 仅返回前 mg_size 个 POC，用于 2-pass 参考均值统计
    poc_list = [int(p) for p in pocs[:mg_size]]
    return s, meta, poc_list


# ====== 可选：保留一个逐帧状态接口，避免旧调用崩溃（精简版） ======
FRAME_STATE_FIELDS: List[str] = [
    "temporal_id",
    "base_q",
    "log1p_bits_plan",
    "log1p_rdcost",
    "log1p_intra_cost",
    "log1p_inter_cost",
    "cor_coef",
]

def build_frame_state(rq_frame: Dict[str, Any], cfg) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    精简逐帧状态（仅为兼容旧路径；若只跑 miniGOP，可不用）
    返回：state_tensor([7]), meta(dict)
    """
    qp_min = int(getattr(cfg, "qp_min", 0))
    qp_max = int(getattr(cfg, "qp_max", 255))
    qp_mid = int((qp_min + qp_max) // 2)

    row, bqi, _ = _row_from_frame(rq_frame, qp_mid)
    s = torch.tensor(row, dtype=torch.float32)
    meta = {
        "base_q": int(max(qp_min, min(qp_max, bqi))),
        "poc": int(_int(rq_frame.get("poc", -1))),
        "doc": int(_int(rq_frame.get("doc", -1))),
        "temporal_id": int(_int(rq_frame.get("temporal_id", 1))),
    }
    return s, meta


__all__ = [
    "MG_FEATS", "MG_FEAT_DIM",
    "build_mg_state", "build_frame_state",
    "FRAME_STATE_FIELDS",
]
