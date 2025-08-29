# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import math, torch
from utils import _float, _int

# === 状态字段（去掉 base_q；temporal_id → one-hot；prev_* 换为误差/偏差）===
# 次序：
# 0-5  tid_0..tid_5（共6维 one-hot，temporal_id∈[1,6] → 索引[0..5]）
# 6    lookahead_feat
# 5    log_pred_bits_frame
# 6    log_mg_bits_tgt
# 7    log_mg_bits_rem
# 8    mg_progress
# 9    frames_left_mg
# 10   prev_qp_delta      （上一帧实际 QP - 上一帧 base_q）
# 11   prev_psnr_err      （上一帧 实际PSNR - 上一帧 预估PSNR）
# 12   prev_rel_err       （上一帧 实际bpf / 参考bpf - 1，夹[-1,1]）
STATE_FIELDS = [
        "tid_0", "tid_1", "tid_2", "tid_3", "tid_4", "tid_5",
    "lookahead_feat",
    "log_pred_bits_frame",
    "log_mg_bits_tgt", "log_mg_bits_rem",
    "mg_progress", "frames_left_mg",
    "prev_qp_delta", "prev_psnr_err", "prev_rel_err",
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
        # 以 torch.save 友好的形式导出
        return {
            "momentum": float(self.momentum),
            "eps": float(self.eps),
            "mean": self.mean.detach().cpu(),
            "var":  self.var.detach().cpu(),
        }

    def load_state_dict(self, d: dict):
        # 兼容历史 ckpt 的健壮处理
        if "momentum" in d:
            self.momentum = float(d["momentum"])
        if "eps" in d:
            self.eps = float(d["eps"])
        if "mean" in d:
            # 保持原 dtype/shape
            with torch.no_grad():
                self.mean.copy_(d["mean"].to(self.mean.dtype))
        if "var" in d:
            with torch.no_grad():
                self.var.copy_(d["var"].to(self.var.dtype))


class StateBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.norm = RunningNorm(momentum=cfg.norm_momentum)
        # RL 端缓存（上一帧真实观测 & 参考 bpf）
        self.prev_bits = 0.0
        self.prev_psnr = 0.0
        self.prev_qp   = float((cfg.qp_min + cfg.qp_max) / 2)
        self.prev_ref_bpf = 1.0
        # 记录上一帧的 base_q（用于 prev_qp_delta）
        self.prev_base_q = float((cfg.qp_min + cfg.qp_max) / 2)
        # 可选：上一帧的 lookahead_feat（默认不用）
        self.prev_lookahead_feat = 0.0
        self.prev_psnr_pred = 0.0
        self._inited = False

    def _maybe_cold_start(self, rq: dict):
        has_mg_idx = ("mg_index" in rq)
        mg_idx = _int(rq.get("mg_index", -1))
        need_reset = (not self._inited) or (has_mg_idx and mg_idx == 0)
        if not need_reset:
            return
        # 冷启动：用 mini-GOP 余量均摊出一个 bpf_t 作为初始参考，避免除0；
        mg_bits_rem = _float(rq.get("mg_bits_rem", 0.0))
        flm = max(1, _int(rq.get("frames_left_mg", 1)))
        bpf_t = mg_bits_rem / flm if flm > 0 else 1.0
        self.prev_bits = float(bpf_t)
        self.prev_psnr = 0.0
        base_q0 = _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max)/2))
        self.prev_qp   = float(base_q0)
        self.prev_base_q = float(base_q0)
        self.prev_ref_bpf = max(1.0, float(bpf_t))
        self.prev_lookahead_feat = 0.0
        self.prev_psnr_pred = 0.0
        self._inited = True

    def build(self, rq: dict):
            self._maybe_cold_start(rq)

            # --- 原始量 ---
            base_q  = _float(rq.get("base_q", (self.cfg.qp_min + self.cfg.qp_max)/2))
            tid     = int(_float(rq.get("temporal_id", rq.get("update_type", 0))))
            pred_pf = _float(rq.get("bits_plan_frame", 0.0))
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

            # ---- 新特征：temporal_id one-hot（6 维） ----
            tid_onehot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                # temporal_id ∈ [1,6] → one-hot 索引 [0..5]
            if 1 <= tid <= 6:
                tid_onehot[tid - 1] = 1.0

            # ---- 新特征：上一帧 QP 相对偏差（相对上一帧 base_q） ----
            prev_qp_delta = float(self.prev_qp) - float(self.prev_base_q)

            # ---- 新特征：上一帧 PSNR 误差（实际 - 预估）----
            # 需求变更：rq.json 提供的是“当前帧的 PSNR 预估”，这里先用缓存的上一帧预估来计算误差，
            # 然后再把当前帧的预估缓存起来，供下一帧使用。
            prev_psnr_err = float(self.prev_psnr) - float(self.prev_psnr_pred)
            # 读取当前帧的 PSNR 预估并缓存
            psnr_pred_cur = _float(rq.get("psnr_pred", rq.get("psnr_est", rq.get("psnr_pred_cur", 0.0))))
            self.prev_psnr_pred = float(psnr_pred_cur)

            # === 组装向量 ===
            s = {
                "tid_0": float(tid_onehot[0]),
                "tid_1": float(tid_onehot[1]),
                "tid_2": float(tid_onehot[2]),
                    "tid_3": float(tid_onehot[3]),
                    "tid_4": float(tid_onehot[4]),
                    "tid_5": float(tid_onehot[5]),
                "lookahead_feat": float(lookahead_feat),
                "log_pred_bits_frame": float(log_pred_pf),
                "log_mg_bits_tgt": float(log_mg_tgt),
                "log_mg_bits_rem": float(log_mg_rem),
                "mg_progress": float(mg_progress),
                "frames_left_mg": float(flm),
                "prev_qp_delta": float(prev_qp_delta),
                "prev_psnr_err": float(prev_psnr_err),
                "prev_rel_err": float(prev_rel_err),
            }
            vec  = torch.tensor([s[k] for k in STATE_FIELDS], dtype=torch.float32)
            if str(getattr(self.cfg, "mode", "train")) == "train":
                self.norm.update(vec)
            nvec = self.norm.normalize(vec, clip=self.cfg.feature_clip)

            # 更新用于下一帧的参照
            self.prev_ref_bpf = cur_ref_bpf
            # 缓存当前帧的 base_q，供下一帧计算 prev_qp_delta
            self.prev_base_q = float(base_q)
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
        # 供反馈阶段更新上一帧真实观测
        self.prev_bits = float(bits)
        self.prev_psnr = float(psnr)
        self.prev_qp   = float(qp)
