# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import math, torch
from typing import Dict, Any, List
from utils import _float, _int

"""
This StateBuilder is drop-in and backward-compatible in spirit:
- It keeps temporal_id one-hot, mg budget/progress, predicted bits, and previous
  frame deltas.
- It ADDS minimal cross-miniGOP smoothing context that the runner can provide
  even at inference time (no reward needed):
    * global_psnr_ema (from previous frames) → normalized diff to psnr_target_db
    * mg_frame_idx (1..mg_size) → normalized to [0,1]
    * has_sc_in_mg (0/1) and is_scene_cut (0/1)
If the runner does not provide these keys, safe fallbacks are used (zeros).
"""

TID_MAX = 6

@dataclass
class StateBuilder:
    cfg: Any
    # cached previous frame info for simple deltas
    prev_qp: float = 0.0
    prev_base_q: float = 0.0
    prev_ref_bpf: float = 1.0
    prev_psnr_pred: float = 0.0
    _inited: bool = False

    def reset(self):
        self.prev_qp = 0.0
        self.prev_base_q = 0.0
        self.prev_ref_bpf = 1.0
        self.prev_psnr_pred = 0.0
        self._inited = False

    def _maybe_cold_start(self, rq: Dict[str, Any]):
        if self._inited:
            return
        base_q0 = _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max)/2))
        bpf_t   = _float(rq.get("bits_pred_frame", rq.get("bits_plan_frame", 1000.0)))
        self.prev_qp   = float(base_q0)
        self.prev_base_q = float(base_q0)
        self.prev_ref_bpf = max(1.0, float(bpf_t))
        self.prev_psnr_pred = float(_float(rq.get("psnr_pred", 0.0)))
        self._inited = True

    def build(self, rq: Dict[str, Any]) -> torch.Tensor:
        self._maybe_cold_start(rq)

        # --- raw quantities ---
        base_q  = _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max)/2))
        tid     = int(_float(rq.get("temporal_id", rq.get("update_type", 1))))
        pred_pf = _float(rq.get("bits_pred_frame", rq.get("bits_plan_frame", self.prev_ref_bpf)))
        mg_tgt  = _float(rq.get("mg_bits_tgt", 0.0))
        mg_rem  = _float(rq.get("mg_bits_rem", 0.0))
        mg_size = int(_int(rq.get("mg_size", 16)))
        frames_left = int(_int(rq.get("frames_left_mg", 1)))
        mg_progress = 1.0 - (frames_left / max(1.0, float(mg_size)))
        lookahead_feat = _float(rq.get("lookahead_cost", rq.get("lookahead_feat", 0.0)))

        # --- previous actuals (must be fed by runner between frames) ---
        prev_qp = float(_float(rq.get("prev_qp", self.prev_qp)))
        prev_base_q = float(_float(rq.get("prev_base_q", self.prev_base_q)))
        prev_psnr_pred = float(_float(rq.get("prev_psnr_pred", self.prev_psnr_pred)))

        # --- cross-miniGOP smoothing context (fed by runner; safe defaults) ---
        global_psnr_ema = float(_float(rq.get("global_psnr_ema", 0.0)))  # previous EMA
        mg_frame_idx    = int(_int(rq.get("mg_frame_idx", max(1, mg_size - frames_left + 1))))
        has_sc_in_mg    = int(_int(rq.get("has_sc_in_mg", 0)))
        is_scene_cut    = int(_int(rq.get("scene_cut", rq.get("is_scene_cut", 0))))

        # --- derived / normalization ---
        tid_oh = [0.0]*TID_MAX
        if 1 <= tid <= TID_MAX: tid_oh[tid-1] = 1.0

        log_pred_pf = math.log(max(1.0, pred_pf))
        log_mg_tgt  = math.log(max(1.0, mg_tgt))
        log_mg_rem  = math.log(max(1.0, mg_rem))

        prev_qp_delta = float(prev_qp - prev_base_q)
        # previous psnr prediction error if present
        prev_psnr_err = float(_float(rq.get("prev_psnr", 0.0)) - prev_psnr_pred)

        # normalized global EMA diff to target
        psnr_target_db = float(getattr(self.cfg, "psnr_target_db", 40.5))
        if global_psnr_ema > 0.0:
            gema_diff = (global_psnr_ema - psnr_target_db) / 10.0
        else:
            gema_diff = 0.0

        # mg position in [0,1]
        mg_pos = (mg_frame_idx - 1) / max(1.0, float(mg_size - 1))

        # frame-level bit reference
        ref_bpf = mg_rem/ max(1, frames_left) if mg_rem>0 and frames_left>0 else max(1.0, pred_pf)
        ref_bpf = max(1.0, ref_bpf)
        log_ref_bpf = math.log(ref_bpf)

        # update caches for next call
        self.prev_qp = base_q
        self.prev_base_q = base_q
        self.prev_ref_bpf = ref_bpf
        self.prev_psnr_pred = prev_psnr_pred

        vec: List[float] = []
        vec += tid_oh                                      # 6
        vec += [float(lookahead_feat)]                     # +1
        vec += [log_pred_pf, log_mg_tgt, log_mg_rem]       # +3
        vec += [mg_progress, float(frames_left)]           # +2
        vec += [prev_qp_delta, prev_psnr_err]              # +2
        vec += [log_ref_bpf]                               # +1
        # new cross-mg smoothing features
        vec += [gema_diff, mg_pos, float(has_sc_in_mg), float(is_scene_cut)]  # +4

        return torch.tensor(vec, dtype=torch.float32, device=self.cfg.device)
