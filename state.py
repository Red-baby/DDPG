# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import math, torch
from typing import Dict, Any, Tuple, List

# === 状态字段定义（已精简）===
#  0-5  tid_0..tid_5（temporal_id∈[1,6] → one-hot）
#  6    lookahead_feat（如 cost/poise/comp 的代表值，缺省0）
#  7    log_pred_bits_frame
#  8    log_mg_bits_tgt
#  9    log_mg_bits_rem
# 10    mg_progress   （[0,1]）
# 11    frames_left_mg
# 12    is_scene_cut  （0/1）
STATE_FIELDS: List[str] = [
    "tid_0","tid_1","tid_2","tid_3","tid_4","tid_5",
    "lookahead_feat",
    "log_pred_bits_frame",
    "log_mg_bits_tgt","log_mg_bits_rem",
    "mg_progress","frames_left_mg",
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
        return {"momentum": float(self.momentum),"eps": float(self.eps),
                "mean": self.mean.detach().cpu(),"var": self.var.detach().cpu()}
    def load_state_dict(self, d: dict):
        if "momentum" in d: self.momentum = float(d["momentum"])
        if "eps" in d: self.eps = float(d["eps"])
        if "mean" in d:
            with torch.no_grad(): self.mean.copy_(d["mean"].to(self.mean.dtype))
        if "var" in d:
            with torch.no_grad(): self.var.copy_(d["var"].to(self.var.dtype))

def _float(x, dv=0.0):
    try: return float(x)
    except: return float(dv)
def _int(x, dv=0):
    try: return int(x)
    except: return int(dv)

@dataclass
class StateBuilder:
    cfg: Any
    use_norm: bool = True
    norm: RunningNorm = field(default_factory=lambda: RunningNorm(momentum=0.01))

    # 上一帧真实观测（由反馈更新）
    prev_bits: float = 0.0
    prev_psnr: float = 0.0
    prev_qp:   float = 0.0
    prev_ref_bpf: float = 1000.0
    prev_psnr_pred: float = 0.0

    def reset(self):
        self.prev_bits = 0.0
        self.prev_psnr = 0.0
        self.prev_qp   = 0.0
        self.prev_ref_bpf = 1000.0
        self.prev_psnr_pred = 0.0

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
        base_q  = _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max)/2))
        tid     = int(_int(rq.get("temporal_id", rq.get("level", 1))))
        pred_pf = _float(rq.get("bits_pred_frame", rq.get("bits_plan_frame", 0.0)))
        mg_tgt  = _float(rq.get("mg_bits_tgt", 0.0))
        mg_rem  = _float(rq.get("mg_bits_rem", 0.0))
        mg_size = int(_int(rq.get("mg_size", 16)))
        frames_left = int(_int(rq.get("frames_left_mg", 1)))
        lookahead_feat = _float(rq.get("lookahead_cost", rq.get("poise", 0.0)))
        psnr_pred = _float(rq.get("psnr_pred", 0.0))
        thr_fb = _float(rq.get("threshold_frame_bits", 0.0))
        is_sc  = 1 if int(_int(rq.get("scene_cut", rq.get("is_scene_cut", 0)))) != 0 else 0

        mg_progress = 1.0 - (frames_left / max(1.0, float(mg_size)))
        log = lambda v: math.log(max(1.0, float(v)))

        # 仍计算 ref_bpf 以便元数据引用（但不再作为输入维度）
        ref_bpf = self._ref_bpf(rq, mg_rem, frames_left, pred_pf)

        # 组装精简后的状态向量
        x = []
        x += self._one_hot_tid(tid)
        x += [float(lookahead_feat)]
        x += [log(pred_pf)]
        x += [log(mg_tgt), log(mg_rem)]
        x += [float(mg_progress), float(frames_left)]
        x += [float(is_sc)]

        s = torch.tensor(x, dtype=torch.float32)
        if self.use_norm:
            self.norm.update(s)
            s = self.norm.normalize(s, clip=10.0)

        meta = dict(rq)  # 供 reward / runner 使用
        meta.update({
            "base_q": base_q,
            "ref_bpf_cur": ref_bpf,
            "poc": int(_int(rq.get("poc", -1))),
            "frames_left_mg": frames_left,
            "mg_size": mg_size,
            "mg_bits_tgt": mg_tgt,
            "threshold_frame_bits": thr_fb,
            "scene_cut": is_sc,
        })
        return s, meta

    def update_prev_meas(self, *, bits: float, psnr: float, qp: int,
                         ref_bpf_cur: float | None = None, psnr_pred: float | None = None):
        self.prev_bits = float(bits)
        self.prev_psnr = float(psnr)
        self.prev_qp   = int(qp)
        if ref_bpf_cur is not None:
            self.prev_ref_bpf = float(ref_bpf_cur)
        if psnr_pred is not None:
            self.prev_psnr_pred = float(psnr_pred)
