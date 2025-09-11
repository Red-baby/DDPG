# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import math, torch
from typing import Dict, Any, Tuple, List
from utils import _float, _int

# === 状态字段 ===
# 0-5  tid_0..tid_5（temporal_id∈[1,6] → one-hot）
# 6    lookahead_feat
# 7    log_pred_bits_frame
# 8    log_mg_bits_tgt
# 9    log_mg_bits_rem
# 10   mg_progress
# 11   frames_left_mg
# 12   prev_qp_delta       （上一帧实际 QP - 上一帧 base_q）
# 13   prev_psnr_err       （上一帧 实际PSNR - 上一帧 预估PSNR）
# 14   prev_rel_err        （上一帧 实际bpf / 参考bpf - 1，夹[-1,1]）
# 15   log_thr_fb          （阈值帧比特的 log）
# 16   is_scene_cut        （场景切换标志，0/1）
STATE_FIELDS: List[str] = [
    "tid_0","tid_1","tid_2","tid_3","tid_4","tid_5",
    "lookahead_feat",
    "log_pred_bits_frame",
    "log_mg_bits_tgt","log_mg_bits_rem",
    "mg_progress","frames_left_mg",
    "prev_qp_delta","prev_psnr_err","prev_rel_err",
    "log_thr_fb",
    "is_scene_cut",
]

@dataclass
class RunningNorm:
    momentum: float = 0.01
    eps: float = 1e-6
    mean: torch.Tensor = field(default_factory=lambda: torch.zeros(len(STATE_FIELDS)))
    var:  torch.Tensor = field(default_factory=lambda: torch.ones(len(STATE_FIELDS)))
    def update(self, x: torch.Tensor):
        with torch.no_grad():
            self.mean.copy_((1-self.momentum)*self.mean + self.momentum*x)
            self.var.copy_((1-self.momentum)*self.var  + self.momentum*(x-self.mean)**2)
    def normalize(self, x: torch.Tensor, clip: float = 10.0):
        z = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(z, -clip, clip)
    def state_dict(self) -> dict:
        return {
            "momentum": float(self.momentum),
            "eps": float(self.eps),
            "mean": self.mean.detach().cpu(),
            "var":  self.var.detach().cpu(),
        }
    def load_state_dict(self, d: dict):
        if "momentum" in d: self.momentum = float(d["momentum"])
        if "eps" in d: self.eps = float(d["eps"])
        if "mean" in d:
            with torch.no_grad():
                self.mean.copy_(d["mean"].to(self.mean.dtype))
        if "var" in d:
            with torch.no_grad():
                self.var.copy_(d["var"].to(self.var.dtype))

@dataclass
class StateBuilder:
    cfg: Any
    use_norm: bool = True
    norm: RunningNorm = field(default_factory=lambda: RunningNorm(momentum=0.01))

    # 上一帧真实观测（由反馈更新）
    prev_bits: float = 0.0
    prev_psnr: float = 0.0
    prev_qp:   float = 0.0

    def reset(self):
        self.prev_bits = 0.0
        self.prev_psnr = 0.0
        self.prev_qp   = 0.0

    def _one_hot_tid(self, tid: int, n: int = 6) -> List[float]:
        oh = [0.0]*n
        if 1 <= tid <= n:
            oh[tid-1] = 1.0
        return oh

    def _ref_bpf(self, rq: Dict[str, Any], mg_rem: float, frames_left: int, pred_pf: float) -> float:
        thr_fb = _float(rq.get("threshold_frame_bits", 0.0))
        if thr_fb > 0.0:
            return max(1.0, thr_fb)
        if mg_rem > 0.0 and frames_left > 0:
            return max(1.0, mg_rem / float(frames_left))
        if pred_pf > 0.0:
            return max(1.0, pred_pf)
        return 1000.0

    def build(self, rq: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # --- 原始量 ---
        base_q  = _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max)/2))
        tid     = int(_int(rq.get("temporal_id", rq.get("update_type", 1))))
        pred_pf = _float(rq.get("bits_pred_frame", rq.get("bits_plan_frame", 0.0)))
        mg_tgt  = _float(rq.get("mg_bits_tgt", 0.0))
        mg_rem  = _float(rq.get("mg_bits_rem", 0.0))
        mg_size = int(_int(rq.get("mg_size", 16)))
        frames_left = int(_int(rq.get("frames_left_mg", 1)))
        lookahead_feat = _float(rq.get("lookahead_cost", rq.get("lookahead_feat", 0.0)))
        psnr_pred = _float(rq.get("psnr_pred", 0.0))

        # --- 派生量 ---
        mg_progress = 1.0 - (frames_left / max(1.0, float(mg_size)))
        log_pred_pf = math.log(max(1.0, pred_pf))
        log_mg_tgt  = math.log(max(1.0, mg_tgt))
        log_mg_rem  = math.log(max(1.0, mg_rem))
        thr_fb      = _float(rq.get("threshold_frame_bits", 0.0))
        log_thr_fb  = math.log(max(1.0, thr_fb))

        # 参考 bpf，用于构造 prev_rel_err
        ref_bpf = self._ref_bpf(rq, mg_rem, frames_left, pred_pf)
        prev_rel_err = (self.prev_bits / max(1.0, ref_bpf)) - 1.0
        prev_rel_err = max(-1.0, min(1.0, prev_rel_err))

        prev_qp_delta = float(self.prev_qp - base_q)
        prev_psnr_err = float(self.prev_psnr - psnr_pred)

        # 场景切换标志（0/1 → float）
        is_scene_cut = 1.0 if int(_int(rq.get("is_scene_cut", rq.get("scene_cut", 0)))) != 0 else 0.0

        # --- 组装状态向量 ---
        vec: List[float] = []
        vec += self._one_hot_tid(tid)                           # 6
        vec += [float(lookahead_feat)]                          # +1
        vec += [log_pred_pf, log_mg_tgt, log_mg_rem]            # +3
        vec += [mg_progress, float(frames_left)]                # +2
        vec += [prev_qp_delta, prev_psnr_err, prev_rel_err]     # +3
        vec += [log_thr_fb]                                     # +1
        vec += [float(is_scene_cut)]                            # +1  ← 新增

        nvec = torch.tensor(vec, dtype=torch.float32, device=self.cfg.device)
        if self.use_norm:
            self.norm.update(nvec)
            nvec = self.norm.normalize(nvec, clip=float(getattr(self.cfg, "feature_clip", 10.0)))

        # --- 打包 meta（便于 reward/日志） ---
        meta: Dict[str, Any] = {
            "doc": _int(rq.get("doc", -1)),
            "mg_id": _int(rq.get("mg_id", 0)),
            "mg_index": _int(rq.get("mg_index", 0)),
            "mg_size": int(mg_size),
            "frames_left_mg": int(frames_left),
            "base_q": int(base_q),
            "bits_pred_frame": float(pred_pf),
            "mg_bits_tgt": float(mg_tgt),
            "mg_bits_rem": float(mg_rem),
            "threshold_frame_bits": float(thr_fb),
            "is_scene_cut": int(is_scene_cut),
        }
        return nvec, meta

    def update_prev_meas(self, bits: float, psnr: float, qp: int):
        # 供反馈阶段更新上一帧真实观测
        self.prev_bits = float(bits)
        self.prev_psnr = float(psnr)
        self.prev_qp   = float(qp)
