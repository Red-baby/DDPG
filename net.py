# -*- coding: utf-8 -*-
"""
DDPG-based QP agent for frame-level rate control (miniGOP episode),
talking to the encoder via the file-based sync you integrated:
  - Encoder writes  frame_%08d.rq.json   (request/state)
  - Agent writes    frame_%08d.qp.txt    (action: QP)
  - Encoder writes  frame_%08d.fb.json   (feedback: bits/quality)
This script:
  - Watches the rl_dir, serves QP decisions in real time (blocking RPC).
  - Trains a DDPG agent online using (s, a, r, s', done) tuples.
  - Also supports inference-only mode.

Author: you + ChatGPT
"""

import os
import sys
import json
import time
import math
import glob
import argparse
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Config & Hyper-params
# =========================

@dataclass
class Config:
    rl_dir: str = "./rl_io"
    mode: str = "train"              # "train" | "infer"
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # DDPG
    gamma: float = 0.98
    tau: float = 0.005
    actor_lr: float = 2e-4
    critic_lr: float = 2e-4
    batch_size: int = 256
    replay_size: int = 400_000
    warmup_steps: int = 5_000
    train_steps_per_env_step: int = 1
    save_every_steps: int = 20_000
    ckpt_dir: str = "./checkpoints"

    # Exploration
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    ou_dt: float = 1.0
    action_eps_train: float = 0.10
    action_eps_infer: float = 0.00

    # ===== Reward shaping (new) =====
    # PSNR项
    psnr_norm: float = 45.0          # 把 PSNR 约到 ~[0,1] 的尺度：psnr/psnr_norm
    w_psnr: float = 1.0              # 质量收益权重

    # bits 偏差项（相对目标bpf的偏差，非对称惩罚）
    w_over: float = 1.2              # 超支惩罚
    w_under: float = 0.4             # 欠支惩罚
    min_bpf: float = 200.0           # 归一化下限，防止小目标导致数值爆

    # 质量平滑项（相邻帧PSNR变化的平方惩罚）
    w_smooth: float = 0.25
    smooth_ref_db: float = 5.0       # 用 (ΔPSNR / smooth_ref_db)^2 做无量纲化

    # GOP 信用对bpf的软调整：bpf += alpha * gop_credit / frames_left_gop
    alpha_credit_share: float = 0.5  # 把信用按“剩余帧数”均摊的系数
    w_gop_risk: float = 0.2          # GOP负信用惩罚（防止后程崩溃）

    # z-norm
    use_minigop: bool = True
    feature_clip: float = 10.0
    norm_momentum: float = 0.01

    # Logging
    print_every_sec: float = 2.0


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _safe_read_json(path: str, retries: int = 50, sleep_ms: int = 2):
    """Read JSON with a few retries to avoid partial write race."""
    for _ in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            time.sleep(sleep_ms / 1000.0)
    # last try raise
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_write_text(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def _try_remove(path: str):
    try:
        os.remove(path)
    except Exception:
        pass

def _now_ms() -> int:
    return int(time.time() * 1000)

def _int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def _float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


# =========================
# State builder
# =========================

STATE_FIELDS = [
    # GOP level
    "gop_bits_rem", "frames_left_gop", "gop_credit",
    # miniGOP level
    "mg_bits_tgt", "mg_bits_rem", "frames_left_mg",
    # current frame context
    "base_q", "min_q", "max_q", "temporal_id",
    "lookahead_cost",
    # prev frame real
    "prev_bits", "prev_mse",
]

@dataclass
class RunningNorm:
    """Very simple running mean/std tracker for normalization."""
    momentum: float = 0.01
    eps: float = 1e-6
    mean: torch.Tensor = field(default_factory=lambda: torch.zeros(len(STATE_FIELDS)))
    var: torch.Tensor  = field(default_factory=lambda: torch.ones(len(STATE_FIELDS)))

    def update(self, x: torch.Tensor):
        # x: [D]
        with torch.no_grad():
            m = self.mean
            v = self.var
            m_new = (1 - self.momentum) * m + self.momentum * x
            v_new = (1 - self.momentum) * v + self.momentum * (x - m) ** 2
            self.mean.copy_(m_new)
            self.var.copy_(v_new)

    def normalize(self, x: torch.Tensor, clip: float = 10.0):
        z = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(z, -clip, clip)

class StateBuilder:
    """Builds state vector from rq.json and cached running context."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.norm = RunningNorm(momentum=cfg.norm_momentum)

        # maintained across frames (agent-side cache; encoder maintains its own internal too)
        self.prev_bits = 0.0
        self.prev_mse  = 0.0

    def build(self, rq: dict) -> Tuple[torch.Tensor, dict]:
        # Pull fields with robust defaults
        s = {
            # GOP level
            "gop_bits_rem":   _float(rq.get("gop_bits_rem", 0.0)),
            "frames_left_gop": _float(rq.get("frames_left_gop", 0.0)),
            "gop_credit":     _float(rq.get("gop_credit", 0.0)),
            # miniGOP
            "mg_bits_tgt":    _float(rq.get("mg_bits_tgt", 0.0)),
            "mg_bits_rem":    _float(rq.get("mg_bits_rem", 0.0)),
            "frames_left_mg": _float(rq.get("frames_left_mg", 0.0)),
            # current frame
            "base_q":         _float(rq.get("base_q", 28)),
            "min_q":          _float(rq.get("min_q", 1)),
            "max_q":          _float(rq.get("max_q", 63)),
            "temporal_id":    _float(rq.get("temporal_id", rq.get("update_type", 0))),
            "lookahead_cost": _float(rq.get("lookahead_cost", 0.0)),
            # prev real (kept by agent between frames)
            "prev_bits":      float(self.prev_bits),
            "prev_mse":       float(self.prev_mse),
        }

        vec = torch.tensor([s[k] for k in STATE_FIELDS], dtype=torch.float32)
        self.norm.update(vec)
        nvec = self.norm.normalize(vec, clip=self.cfg.feature_clip)

        meta = {
            "poc": _int(rq.get("poc", -1)),
            "doc": _int(rq.get("doc", -1)),
            "update_type": _int(rq.get("update_type", _int(rq.get("temporal_id", 0)))),
            "frames_left_mg": int(s["frames_left_mg"]),
            "min_q": int(s["min_q"]), "max_q": int(s["max_q"]),
            "base_q": int(s["base_q"]),
        }
        return nvec, meta

    def update_prev_meas(self, bits: float, mse: float):
        self.prev_bits = bits
        self.prev_mse  = mse


# =========================
# DDPG Agent
# =========================

class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # output in (0,1) via sigmoid; scale to [min,max] outside
        x = self.net(s)
        return torch.sigmoid(x)  # [B,1]


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        # Q(s,a), a in (0,1)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a01: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a01], dim=1)
        return self.net(x)


class OUNoise:
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2, dt=1.0):
            self.mu = mu
            self.theta = theta
            self.sigma = sigma
            self.dt = dt
            self.x_prev = 0.0

    def reset(self): self.x_prev = 0.0

    def sample(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + \
            self.sigma * math.sqrt(self.dt) * np.random.randn()
        self.x_prev = x
        return x


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self._s  = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a  = np.zeros((capacity, 1), dtype=np.float32)
        self._r  = np.zeros((capacity, 1), dtype=np.float32)
        self._s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self._d  = np.zeros((capacity, 1), dtype=np.float32)
        self._n = 0
        self._p = 0

    def __len__(self): return self._n

    def push(self, s, a, r, s2, d):
        self._s[self._p]  = s
        self._a[self._p]  = a
        self._r[self._p]  = r
        self._s2[self._p] = s2
        self._d[self._p]  = d
        self._p = (self._p + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch: int):
        idx = np.random.randint(0, self._n, size=(batch,))
        s  = torch.from_numpy(self._s[idx])
        a  = torch.from_numpy(self._a[idx])
        r  = torch.from_numpy(self._r[idx])
        s2 = torch.from_numpy(self._s2[idx])
        d  = torch.from_numpy(self._d[idx])
        return s, a, r, s2, d


class DDPG:
    def __init__(self, state_dim: int, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(state_dim).to(self.device)
        self.actor_tgt = Actor(state_dim).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim).to(self.device)
        self.critic_tgt = Critic(state_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.buf = ReplayBuffer(cfg.replay_size, state_dim)
        self.noise = OUNoise(theta=cfg.ou_theta, sigma=cfg.ou_sigma, dt=cfg.ou_dt)

        self.total_env_steps = 0
        self.total_train_steps = 0

    @torch.no_grad()
    def select_action(self, s: torch.Tensor, min_q: int, max_q: int,
                      explore: bool) -> int:
        """
        s: [D] normalized state
        returns integer QP in [min_q, max_q]
        """
        self.actor.eval()
        a01 = self.actor(s.unsqueeze(0).to(self.device)).cpu().item()  # (0,1)
        if explore:
            a01 += self.noise.sample()
            a01 = float(np.clip(a01, 0.0, 1.0))
            if np.random.rand() < self.cfg.action_eps_train:
                a01 = np.random.rand()
        else:
            if self.cfg.action_eps_infer > 0 and np.random.rand() < self.cfg.action_eps_infer:
                a01 = np.random.rand()

        # scale to [min,max] and round to int
        qp = min_q + a01 * (max_q - min_q)
        qp = int(np.clip(round(qp), min_q, max_q))
        return qp

    def train_step(self):
        if len(self.buf) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return

        s, a, r, s2, d = self.buf.sample(self.cfg.batch_size)
        s  = s.to(self.device)
        a  = a.to(self.device)
        r  = r.to(self.device)
        s2 = s2.to(self.device)
        d  = d.to(self.device)

        # Critic loss
        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            q2 = self.critic_tgt(s2, a2)
            y  = r + self.cfg.gamma * (1.0 - d) * q2
        q = self.critic(s, a)
        loss_c = F.mse_loss(q, y)
        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        self.opt_c.step()

        # Actor loss (maximize Q => minimize -Q)
        a_pi = self.actor(s)
        loss_a = - self.critic(s, a_pi).mean()
        self.opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        self.opt_a.step()

        # Soft update
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        self.total_train_steps += 1

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "norm_mean": self._try_tensor(self_state=True, which="mean"),
            "norm_var":  self._try_tensor(self_state=True, which="var"),
            "total_env_steps": self.total_env_steps,
            "total_train_steps": self.total_train_steps,
        }, path)

    def load(self, path: str, state_builder: Optional[StateBuilder] = None):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        if state_builder is not None and "norm_mean" in ckpt and "norm_var" in ckpt:
            state_builder.norm.mean = ckpt["norm_mean"]
            state_builder.norm.var  = ckpt["norm_var"]
        self.total_env_steps = int(ckpt.get("total_env_steps", 0))
        self.total_train_steps = int(ckpt.get("total_train_steps", 0))

    def _try_tensor(self_state: bool, which: str):
        # this helper is just to keep save() simple if you pass a StateBuilder externally
        return None


# =========================
# RL-Encoder IO manager
# =========================

@dataclass
class Pending:
    state: torch.Tensor
    meta: dict
    action_a01: float     # action in (0,1) before scaling
    qp_used: int          # scaled & rounded, written to encoder
    next_state: Optional[torch.Tensor] = None
    done: bool = False

class RLRunner:
    """
    Watches rl_dir, handles (rq -> action) and (fb -> reward),
    and trains DDPG online.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sb = StateBuilder(cfg)
        self.agent = DDPG(state_dim=len(STATE_FIELDS), cfg=cfg)
        self.ckpt_path = os.path.join(cfg.ckpt_dir, "ddpg_best.pt")

        # pending cache by POC
        self.pending: Dict[int, Pending] = {}
        self.last_poc_in_mg: Optional[int] = None
        self.last_seen_print = 0.0

        os.makedirs(cfg.rl_dir, exist_ok=True)
        set_seed(cfg.seed)

    # ---------- reward function ----------
    def compute_reward(self, fb: dict, rq_for_meta: dict) -> float:
        """
        目标：高质量 + 码率守纪律 + 平滑，且允许在GOP层面用信用做软调整。
        r = w_psnr * (psnr/psnr_norm)  -  bit_dev_pen  -  smooth_pen  -  gop_risk_pen
        其中：
          - bit_dev_pen 通过本帧 bits 相对“动态目标 bpf”的偏差计算（超支/欠支非对称）。
          - smooth_pen 通过相邻帧 ΔPSNR 的平方计算（按 smooth_ref_db 归一）。
          - bpf = mg_bits_rem / frames_left_mg  +  alpha * gop_credit / frames_left_gop
        """
        cfg = self.cfg

        # ---------- 质量指标：优先用 PSNR_Y，若没有则用 mse_y 换算 ----------
        y_psnr = _float(fb.get("psnr_y", 0.0))
        if y_psnr <= 0.0:
            mse_y = _float(fb.get("mse_y", 0.0))
            if mse_y > 0.0:
                y_psnr = 10.0 * math.log10((255.0 * 255.0) / mse_y)

        # ---------- 本帧 bits ----------
        bits = _float(fb.get("bits", 0.0))

        # ---------- 动态目标 bpf（bits-per-frame） ----------
        mg_bits_rem = _float(rq_for_meta.get("mg_bits_rem", 0.0))
        frames_left_mg = _int(rq_for_meta.get("frames_left_mg", 0))
        gop_credit = _float(rq_for_meta.get("gop_credit", 0.0))
        frames_left_gop = _int(rq_for_meta.get("frames_left_gop", 0))
        gop_bits_rem = _float(rq_for_meta.get("gop_bits_rem", 0.0))

        # mini-GOP 动态目标（剩余可用比特 ÷ 剩余帧数）
        if mg_bits_rem > 0.0 and frames_left_mg >= 1:
            bpf_mg = mg_bits_rem / max(1, frames_left_mg)  # 当前帧上场前的“剩余摊销”
        else:
            bpf_mg = max(cfg.min_bpf, 0.5 * cfg.scale_bits)  # 兜底（极少发生）

        # GOP 信用均摊调整（正信用→放松，负信用→收紧）
        credit_share = 0.0
        if frames_left_gop >= 1 and abs(gop_credit) > 0.0:
            credit_share = cfg.alpha_credit_share * (gop_credit / max(1, frames_left_gop))

        # 组合后的动态目标
        bpf = max(cfg.min_bpf, bpf_mg + credit_share)

        # ---------- bits 偏差罚：超支>1 与 欠支<1 非对称 ----------
        norm = bits / max(bpf, 1.0)
        if norm >= 1.0:
            bit_dev_pen = cfg.w_over * (norm - 1.0) ** 2
        else:
            bit_dev_pen = cfg.w_under * (1.0 - norm) ** 2

        # ---------- 质量平滑罚（与上一帧PSNR差的平方） ----------
        prev_mse = float(self.sb.prev_mse)
        if prev_mse > 0.0:
            prev_psnr = 10.0 * math.log10((255.0 * 255.0) / prev_mse)
        else:
            prev_psnr = y_psnr  # 第一帧或未知时不罚
        d_psnr = (y_psnr - prev_psnr)
        smooth_pen = cfg.w_smooth * (d_psnr / max(1e-6, cfg.smooth_ref_db)) ** 2

        # ---------- GOP 风险罚（负信用越大越罚，按剩余GOP预算归一化） ----------
        if gop_credit < 0.0 and gop_bits_rem > 0.0:
            gop_risk_pen = cfg.w_gop_risk * (min(1.5, (-gop_credit / gop_bits_rem)) ** 2)
        else:
            gop_risk_pen = 0.0

        # ---------- 质量收益 ----------
        q_gain = cfg.w_psnr * (y_psnr / max(1e-6, cfg.psnr_norm))

        r = q_gain - bit_dev_pen - smooth_pen - gop_risk_pen
        return float(r)

    # ---------- main loop ----------
    def serve(self):
        print(f"[RL] Watching: {self.cfg.rl_dir} | mode={self.cfg.mode} | device={self.cfg.device}")
        last_save = _now_ms()
        last_print = _now_ms()
        while True:
            self.handle_requests()
            self.handle_feedbacks()
            # Train
            if self.cfg.mode == "train":
                for _ in range(self.cfg.train_steps_per_env_step):
                    self.agent.train_step()

            # periodic save & print
            now = _now_ms()
            if now - last_print > int(self.cfg.print_every_sec * 1000):
                print(f"[RL] steps env/train: {self.agent.total_env_steps}/{self.agent.total_train_steps} | "
                      f"replay={len(self.agent.buf)}")
                last_print = now
            if now - last_save > int(self.cfg.save_every_steps / max(1, self.agent.total_env_steps+1) * 1000):
                # guard: save roughly every save_every_steps env steps
                if self.agent.total_env_steps % self.cfg.save_every_steps < 3:
                    path = os.path.join(self.cfg.ckpt_dir, f"ddpg_step_{self.agent.total_env_steps}.pt")
                    self.agent.save(path)
                    print(f"[RL] saved checkpoint -> {path}")
                last_save = now

            time.sleep(0.003)

    # ---------- request (state -> action) ----------
    def handle_requests(self):
        rq_paths = sorted(glob.glob(os.path.join(self.cfg.rl_dir, "frame_*.rq.json")))
        for rq_path in rq_paths:
            try:
                rq = _safe_read_json(rq_path)
            except Exception as e:
                print(f"[RL][WARN] bad rq json {rq_path}: {e}")
                _try_remove(rq_path)
                continue

            s, meta = self.sb.build(rq)
            poc = int(meta["poc"])
            min_q, max_q = meta["min_q"], meta["max_q"]

            # select action
            explore = (self.cfg.mode == "train")
            qp = self.agent.select_action(s, min_q=min_q, max_q=max_q, explore=explore)
            # write qp
            qp_path = rq_path.replace(".rq.json", ".qp.txt")
            _safe_write_text(qp_path, f"{qp}\n")
            # remove request (optional)
            _try_remove(rq_path)

            # cache pending step
            a01 = float((qp - min_q) / max(1, (max_q - min_q)))  # save a in (0,1)
            self.pending[poc] = Pending(state=s, meta=meta, action_a01=a01, qp_used=qp)
            self.agent.total_env_steps += 1

            # link with previous for s' (next_state)
            if self.last_poc_in_mg is not None and self.last_poc_in_mg in self.pending:
                prev = self.pending[self.last_poc_in_mg]
                prev.next_state = s  # s_{t+1}
            self.last_poc_in_mg = poc

    # ---------- feedback (result -> reward, push replay) ----------
    def handle_feedbacks(self):
        fb_paths = sorted(glob.glob(os.path.join(self.cfg.rl_dir, "frame_*.fb.json")))
        for fb_path in fb_paths:
            try:
                fb = _safe_read_json(fb_path)
            except Exception as e:
                print(f"[RL][WARN] bad fb json {fb_path}: {e}")
                _try_remove(fb_path)
                continue

            poc = _int(fb.get("poc", -1))
            if poc not in self.pending:
                # Might be out-of-order; keep feedback until request seen? Here we just skip after logging.
                # In your pipeline (no frame parallel), this should not happen.
                print(f"[RL][WARN] feedback for unknown POC={poc}")
                _try_remove(fb_path)
                continue

            pend = self.pending[poc]
            # reward
            r = self.compute_reward(fb, rq_for_meta=pend.meta)

            # done flag: last frame in miniGOP?
            done = False
            flm = pend.meta.get("frames_left_mg", None)
            if self.cfg.use_minigop and (flm is not None) and int(flm) == 1:
                done = True

            # next_state: if not yet set (e.g., feedback arrives before next rq), use self-state for bootstrap
            if pend.next_state is None:
                # If terminal, s' is ignored. If non-terminal, we can delay pushing until s' arrives.
                if not done:
                    # delay: keep fb until next_state provided in handle_requests()
                    # We'll leave fb file, and revisit later. But to keep directory clean, we proceed anyway:
                    # use self-state as a weak bootstrap (less ideal but keeps pipeline moving).
                    pend.next_state = pend.state.clone()
                else:
                    pend.next_state = torch.zeros_like(pend.state)

            # push into replay
            self.agent.buf.push(
                pend.state.numpy(), np.array([[pend.action_a01]], dtype=np.float32),
                np.array([[r]], dtype=np.float32),
                pend.next_state.numpy(), np.array([[1.0 if done else 0.0]], dtype=np.float32)
            )

            # update prev measures for next state's features
            psnr_y = _float(fb.get("psnr_y", 0.0))
            mse = self._psnr_to_mse(psnr_y) if psnr_y > 0 else _float(fb.get("mse_y", 0.0))
            self.sb.update_prev_meas(bits=_float(fb.get("bits", 0.0)), mse=mse)

            # cleanup and forget this step if terminal
            _try_remove(fb_path)
            if done:
                # reset chain
                self.last_poc_in_mg = None
                # also clear any very old pending to avoid memory leak
                self._cleanup_pending()

    def _psnr_to_mse(self, psnr_y: float, peak: float = 255.0) -> float:
        return (peak * peak) / (10 ** (psnr_y / 10.0))

    def _cleanup_pending(self, keep_last: int = 2):
        # keep last few entries for safety
        if len(self.pending) <= keep_last:
            return
        # remove the oldest by key order
        for k in sorted(list(self.pending.keys()))[:-keep_last]:
            self.pending.pop(k, None)


# =========================
# Main
# =========================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", type=str, required=True, help="Directory shared with encoder (rl_set_dir)")
    ap.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    ap.add_argument("--ckpt", type=str, default="", help="Path to load actor/critic (optional)")
    ap.add_argument("--seed", type=int, default=2025)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = Config(rl_dir=args.rl_dir, mode=args.mode, seed=args.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    runner = RLRunner(cfg)
    if args.ckpt:
        try:
            runner.agent.load(args.ckpt, state_builder=runner.sb)
            print(f"[RL] loaded checkpoint: {args.ckpt}")
        except Exception as e:
            print(f"[RL][WARN] failed to load ckpt: {e}")

    try:
        runner.serve()
    except KeyboardInterrupt:
        print("\n[RL] bye.")

if __name__ == "__main__":
    main()
