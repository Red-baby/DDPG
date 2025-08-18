# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import torch
from utils import _float, _int

# 状态字段（注意：用 prev_psnr 替代 prev_mse）
STATE_FIELDS = [
    # GOP 维度（大账本）
    "gop_bits_rem", "frames_left_gop", "gop_credit",
    # mini-GOP 维度（小账本）
    "mg_bits_tgt", "mg_bits_rem", "frames_left_mg",
    # 当前帧上下文
    "base_q", "temporal_id", "lookahead_cost",
    # 上一帧真实观测（RL端缓存）
    "prev_bits", "prev_psnr", "prev_qp",
]

@dataclass
class RunningNorm:
    momentum: float = 0.01
    eps: float = 1e-6
    mean: torch.Tensor = field(default_factory=lambda: torch.zeros(len(STATE_FIELDS)))
    var:  torch.Tensor = field(default_factory=lambda: torch.ones(len(STATE_FIELDS)))

    def update(self, x: torch.Tensor):
        with torch.no_grad():
            m, v = self.mean, self.var
            m_new = (1 - self.momentum) * m + self.momentum * x
            v_new = (1 - self.momentum) * v + self.momentum * (x - m)**2
            self.mean.copy_(m_new); self.var.copy_(v_new)

    def normalize(self, x: torch.Tensor, clip: float = 10.0):
        z = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(z, -clip, clip)

class StateBuilder:
    """
    - 以 DOC 为主键（编码顺序）；
    - 上一帧缓存采用 PSNR，不再使用 MSE；
    - 默认跨 miniGOP/GOP 继承上一帧观测（可在 config 里关闭）。
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.norm = RunningNorm(momentum=cfg.norm_momentum)

        # RL 端缓存
        self.prev_bits = 0.0
        self.prev_psnr = 0.0
        self.prev_qp   = float((cfg.qp_min + cfg.qp_max) / 2)

    def _maybe_cold_start(self, rq: dict):
        """
        冷启动策略：
          1) 若是整段视频第一帧（poc==0），无条件重置上一帧观测（prev_*），确保首帧平滑项不罚。
          2) 否则，仅当对应 carry_* 开关为 False 且到达该边界时才重置：
             - GOP 首（doc==0）且 not carry_prev_across_gop
             - miniGOP 首（mg_index==0）且 not carry_prev_across_mg
          3) 默认你的配置是两个 carry_* 都为 True，因此跨 miniGOP/GOP 会继承上一帧观测。
        """
        doc = _int(rq.get("doc", -1))
        mg_index = _int(rq.get("mg_index", -1))
        poc = _int(rq.get("poc", -1))  # 由编码器写入；用来识别整段视频的首帧

        # 识别边界
        is_seq_first = (poc == 0)  # 整段视频开头
        is_gop_first = (doc == 0)  # 当前 GOP 开头（按你的写法 doc 从0开始）
        is_mg_first = (mg_index == 0)  # 当前 miniGOP 开头

        need_reset = False

        # 规则 1：整段视频第一帧时，强制重置（不受 carry_* 影响）
        if is_seq_first:
            need_reset = True
        else:
            # 规则 2：按开关决定是否在边界清零
            if (is_gop_first and not self.cfg.carry_prev_across_gop) or \
                    (is_mg_first and not self.cfg.carry_prev_across_mg):
                need_reset = True

        if need_reset:
            base_q = _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max) / 2))
            mg_bits_rem = _float(rq.get("mg_bits_rem", 0.0))
            frames_left_mg = max(1, _int(rq.get("frames_left_mg", 1)))
            # 用 bpf 作为上一帧 bits 的合理先验；prev_psnr=0 表示“平滑项不罚”
            bpf_t = mg_bits_rem / frames_left_mg if frames_left_mg > 0 else 0.0
            self.prev_bits = float(bpf_t)
            self.prev_psnr = 0.0
            self.prev_qp = float(base_q)

    def build(self, rq: dict):
        self._maybe_cold_start(rq)

        s = {
            "gop_bits_rem":   _float(rq.get("gop_bits_rem", 0.0))/15,
            "frames_left_gop": _float(rq.get("frames_left_gop", 0.0)),
            "gop_credit":     _float(rq.get("gop_credit", 0.0))/15,
            "mg_bits_tgt":    _float(rq.get("mg_bits_tgt", 0.0))/15,
            "mg_bits_rem":    _float(rq.get("mg_bits_rem", 0.0))/15,
            "frames_left_mg": _float(rq.get("frames_left_mg", 0.0)),
            "base_q":         _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max) / 2)),
            "temporal_id":    _float(rq.get("temporal_id", rq.get("update_type", 0))),
            "lookahead_cost": _float(rq.get("lookahead_cost", 0.0)),
            "prev_bits":      float(self.prev_bits),
            "prev_psnr":      float(self.prev_psnr),
            "prev_qp":        float(self.prev_qp),
        }
        vec  = torch.tensor([s[k] for k in STATE_FIELDS], dtype=torch.float32)
        self.norm.update(vec)
        nvec = self.norm.normalize(vec, clip=self.cfg.feature_clip)

        # 额外把 score_* 也放进 meta 供 reward 门控使用
        meta = {
            "doc": _int(rq.get("doc", -1)),
            "poc": _int(rq.get("poc", -1)),  # 兼容：若需要也提供
            "update_type": _int(rq.get("update_type", rq.get("temporal_id", 0))),
            "frames_left_mg": _int(rq.get("frames_left_mg", 0)),
            "frames_left_gop": _int(rq.get("frames_left_gop", 0)),
            "gop_credit": _float(rq.get("gop_credit", 0.0))/15,
            "gop_bits_rem": _float(rq.get("gop_bits_rem", 0.0))/15,
            "mg_index": _int(rq.get("mg_index", 0)),
            "base_q": int(s["base_q"]),
            "bits_plan_frame": _float(rq.get("bits_plan_frame", 0.0)),  # 可用于reward（可选）
            "score_max": _float(rq.get("score_max", 0.0)),
            "score_avg": _float(rq.get("score_avg", 0.0)),
            "score_min": _float(rq.get("score_min", 0.0)),
        }
        return nvec, meta

    def update_prev_meas(self, bits: float, psnr: float, qp: int):
        self.prev_bits = float(bits)
        self.prev_psnr = float(psnr)
        self.prev_qp   = float(qp)
