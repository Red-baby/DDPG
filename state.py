# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import math, torch
from utils import _float, _int

# === 状态字段：11 维 ===
# 0 base_q, 1 temporal_id, 2 lookahead_feat, 3 log_pred_bits_frame,
# 4 log_mg_bits_tgt, 5 log_mg_bits_rem, 6 mg_progress, 7 frames_left_mg,
# 8 prev_qp, 9 prev_psnr, 10 prev_rel_err
STATE_FIELDS = [
    "base_q", "temporal_id",
    "lookahead_feat",             # 新增：把 lookahead_cost 压缩后放进 state
    "log_pred_bits_frame",
    "log_mg_bits_tgt", "log_mg_bits_rem",
    "mg_progress", "frames_left_mg",
    "prev_qp", "prev_psnr", "prev_rel_err",
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

class StateBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.norm = RunningNorm(momentum=cfg.norm_momentum)
        # RL 端缓存（上一帧真实观测 & 参考 bpf）
        self.prev_bits = 0.0
        self.prev_psnr = 0.0
        self.prev_qp   = float((cfg.qp_min + cfg.qp_max) / 2)
        self.prev_ref_bpf = 1.0
        # 可选：上一帧的 lookahead_feat（默认不用）
        self.prev_lookahead_feat = 0.0

    def _maybe_cold_start(self, rq: dict):
        # 你的冷启动逻辑保持不变，略……
        # ……计算一个初始 bpf_t 以避免除0：
        mg_bits_rem = _float(rq.get("mg_bits_rem", 0.0))
        flm = max(1, _int(rq.get("frames_left_mg", 1)))
        bpf_t = mg_bits_rem / flm if flm > 0 else 1.0
        self.prev_bits = float(bpf_t)
        self.prev_psnr = 0.0
        self.prev_qp   = float(_float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max)/2)))
        self.prev_ref_bpf = max(1.0, float(bpf_t))
        self.prev_lookahead_feat = 0.0

    def build(self, rq: dict):
        self._maybe_cold_start(rq)

        # --- 原始量 ---
        base_q  = _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max)/2))
        tid     = _float(rq.get("temporal_id", rq.get("update_type", 0)))
        pred_pf = _float(rq.get("bits_pred_frame", 0.0))
        mg_tgt  = _float(rq.get("mg_bits_tgt", 0.0))
        mg_rem  = _float(rq.get("mg_bits_rem", 0.0))
        flm     = _float(rq.get("frames_left_mg", 0.0))

        # --- lookahead_cost 压缩到稳定量级：先缩放再 log1p ---
        lac_raw   = _float(rq.get("lookahead_cost", 0.0))
        lac_scale = float(getattr(self.cfg, "lookahead_scale", 1e5))  # 你的量级若是十万级，这里设 1e5
        lookahead_feat = math.log1p(max(0.0, lac_raw / max(1e-12, lac_scale)))

        # pred_pf 回退：无预测就用均摊
        if pred_pf <= 0.0:
            pred_pf = (mg_rem / max(1, int(flm))) if (mg_rem > 0 and flm > 0) else 0.0

        # --- 参考 bpf（用于下一帧 prev_rel_err 的“参照”）---
        cur_ref_bpf = max(1.0, float(pred_pf))

        # 上一帧超/欠（相对误差，夹 [-1,1]）
        if self.prev_ref_bpf > 0:
            prev_rel_err = max(-1.0, min(1.0, (self.prev_bits / self.prev_ref_bpf) - 1.0))
        else:
            prev_rel_err = 0.0

        # 预算进度
        mg_progress = 0.0 if mg_tgt <= 0 else max(0.0, min(1.0, (mg_tgt - mg_rem) / mg_tgt))

        # log1p 压缩大数
        log_pred_pf = math.log1p(max(0.0, pred_pf))
        log_mg_tgt  = math.log1p(max(0.0, mg_tgt))
        log_mg_rem  = math.log1p(max(0.0, mg_rem))

        # === 组装 11 维向量（不把 lookahead_cost 放进 meta）===
        s = {
            "base_q": float(base_q),
            "temporal_id": float(tid),
            "lookahead_feat": float(lookahead_feat),     # 2
            "log_pred_bits_frame": float(log_pred_pf),   # 3
            "log_mg_bits_tgt": float(log_mg_tgt),        # 4
            "log_mg_bits_rem": float(log_mg_rem),        # 5
            "mg_progress": float(mg_progress),           # 6
            "frames_left_mg": float(flm),                # 7
            "prev_qp": float(self.prev_qp),              # 8
            "prev_psnr": float(self.prev_psnr),          # 9
            "prev_rel_err": float(prev_rel_err),         #10
        }
        vec  = torch.tensor([s[k] for k in STATE_FIELDS], dtype=torch.float32)
        self.norm.update(vec)
        nvec = self.norm.normalize(vec, clip=self.cfg.feature_clip)

        # 更新用于下一帧的参照
        self.prev_ref_bpf = cur_ref_bpf
        # 如果你想“可选使用上一帧 lookahead”，这里缓存一下（默认不开启）
        self.prev_lookahead_feat = lookahead_feat

        # meta：只保留 reward/日志需要的键（不含 lookahead）
        meta = {
            "doc": _int(rq.get("doc", -1)),
            "mg_id": _int(rq.get("mg_id", 0)),
            "mg_index": _int(rq.get("mg_index", 0)),
            "frames_left_mg": _int(rq.get("frames_left_mg", 0)),
            "base_q": int(base_q),
            "bits_pred_frame": float(pred_pf),
            "mg_bits_tgt": float(mg_tgt),
            "mg_bits_rem": float(mg_rem),
        }
        return nvec, meta

    def update_prev_meas(self, bits: float, psnr: float, qp: int):
        self.prev_bits = float(bits)
        self.prev_psnr = float(psnr)
        self.prev_qp   = float(qp)
