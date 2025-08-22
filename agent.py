# -*- coding: utf-8 -*-
import math, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn
import os
from dataclasses import dataclass
from models import ActorNet, CriticNet


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self._s = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a = np.zeros((capacity, 1), dtype=np.float32)
        self._r = np.zeros((capacity, 1), dtype=np.float32)
        self._s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self._d = np.zeros((capacity, 1), dtype=np.float32)
        self._n = 0
        self._p = 0

    def __len__(self): return self._n

    def push(self, s, a, r, s2, d):
        self._s[self._p] = s
        self._a[self._p] = a
        self._r[self._p] = r
        self._s2[self._p] = s2
        self._d[self._p] = d
        self._p = (self._p + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch: int):
        idx = np.random.randint(0, self._n, size=(batch,))
        return (torch.from_numpy(self._s[idx]),
                torch.from_numpy(self._a[idx]),
                torch.from_numpy(self._r[idx]),
                torch.from_numpy(self._s2[idx]),
                torch.from_numpy(self._d[idx]))


class OUNoise:
    def __init__(self, theta=0.15, sigma=0.2, dt=1.0):
        self.theta, self.sigma, self.dt = theta, sigma, dt
        self.x_prev = 0.0

    def reset(self): self.x_prev = 0.0

    def sample(self):
        x = self.x_prev + self.theta * (0.0 - self.x_prev) * self.dt + self.sigma * math.sqrt(
            self.dt) * np.random.randn()
        self.x_prev = x
        return x


class DDPG:
    def __init__(self, state_dim: int, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = ActorNet(state_dim).to(self.device)
        self.actor_tgt = ActorNet(state_dim).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic = CriticNet(state_dim).to(self.device)
        self.critic_tgt = CriticNet(state_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.buf = ReplayBuffer(cfg.replay_size, state_dim)
        self.noise = OUNoise(theta=cfg.ou_theta, sigma=cfg.ou_sigma, dt=cfg.ou_dt)

        self.total_env_steps = 0
        self.total_train_steps = 0

        # 最近一次/EMA的损失（供打印）
        self.last_loss_c = None
        self.last_loss_a = None

    @torch.no_grad()
    def select_action(self, s: torch.Tensor, explore: bool) -> int:
        # s: [D]; 输出离散 QP（用 cfg.qp_min/max 映射）
        self.actor.eval()
        a01 = self.actor(s.unsqueeze(0).to(self.device)).cpu().item()
        if explore:
            a01 += self.noise.sample()
            a01 = float(np.clip(a01, 0.0, 1.0))
            if np.random.rand() < self.cfg.action_eps_train:
                a01 = np.random.rand()
        else:
            if self.cfg.action_eps_infer > 0 and np.random.rand() < self.cfg.action_eps_infer:
                a01 = np.random.rand()

        qp = self.cfg.qp_min + a01 * (self.cfg.qp_max - self.cfg.qp_min)
        return int(np.clip(round(qp), self.cfg.qp_min, self.cfg.qp_max))

    def train_step(self):
        if len(self.buf) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        s, a, r, s2, d = self.buf.sample(self.cfg.batch_size)
        s = s.to(self.device);
        a = a.to(self.device);
        r = r.to(self.device)
        s2 = s2.to(self.device);
        d = d.to(self.device)

        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            q2 = self.critic_tgt(s2, a2)
            y = r + self.cfg.gamma * (1.0 - d) * q2
        q = self.critic(s, a)
        loss_c = F.smooth_l1_loss(q, y)  # Huber，默认 δ=1
        self.opt_c.zero_grad(set_to_none=True);
        loss_c.backward();
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_c.step()

        a_pi = self.actor(s)
        loss_a = - self.critic(s, a_pi).mean()
        self.opt_a.zero_grad(set_to_none=True);
        loss_a.backward();
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_a.step()

        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        self.total_train_steps += 1

        # 记录最近损失（供外层EMA显示）
        self.last_loss_c = float(loss_c.item())
        self.last_loss_a = float(loss_a.item())
        return self.last_loss_c, self.last_loss_a

    # ====== 保存/加载，用于每epoch持久化 ======

    def save_checkpoint(self, path: str, extra: dict | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "algo": "ddpg",
            "actor": self.actor.state_dict(),
            "actor_tgt": self.actor_tgt.state_dict(),
            "critic": getattr(self, "critic", None).state_dict() if hasattr(self, "critic") else None,
            "critic_tgt": getattr(self, "critic_tgt", None).state_dict() if hasattr(self, "critic_tgt") else None,
            "opt_a": getattr(self, "opt_a", None).state_dict() if hasattr(self, "opt_a") else None,
            "opt_c": getattr(self, "opt_c", None).state_dict() if hasattr(self, "opt_c") else None,
            "env_steps": getattr(self, "total_env_steps", 0),
            "train_steps": getattr(self, "total_train_steps", 0),
            "cfg": getattr(self.cfg, "__dict__", None),
        }
        # Remove None entries
        ckpt = {k: v for k, v in ckpt.items() if v is not None}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_tgt.load_state_dict(ckpt["actor_tgt"])
        self.critic_tgt.load_state_dict(ckpt["critic_tgt"])
        self.opt_a.load_state_dict(ckpt["opt_a"])
        self.opt_c.load_state_dict(ckpt["opt_c"])
        self.total_env_steps = int(ckpt.get("env_steps", 0))
        self.total_train_steps = int(ckpt.get("train_steps", 0))
        return ckpt
        return ckpt
        return ckpt


class TD3:
    def __init__(self, state_dim: int, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # actor + targets
        self.actor = ActorNet(state_dim).to(self.device)
        self.actor_tgt = ActorNet(state_dim).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        # twin critics + targets
        self.critic1 = CriticNet(state_dim).to(self.device)
        self.critic2 = CriticNet(state_dim).to(self.device)
        self.critic1_tgt = CriticNet(state_dim).to(self.device)
        self.critic2_tgt = CriticNet(state_dim).to(self.device)
        self.critic1_tgt.load_state_dict(self.critic1.state_dict())
        self.critic2_tgt.load_state_dict(self.critic2.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr
        )

        self.buf = ReplayBuffer(cfg.replay_size, state_dim)
        self.total_env_steps = 0
        self.total_train_steps = 0
        self.last_loss_c = None
        self.last_loss_a = None

    @torch.no_grad()
    def select_action(self, s: torch.Tensor, explore: bool) -> int:
        self.actor.eval()
        a01 = self.actor(s.unsqueeze(0).to(self.device)).cpu().item()
        if explore:
            # 2) 仅在训练时加入探索噪声与epsilon随机
            a01 += np.random.randn() * float(getattr(self.cfg, "expl_noise_std", 0.15))
            if np.random.rand() < float(getattr(self.cfg, "action_eps_train", 0.10)):
                a01 = np.random.rand()
        else:
            # 验证/推理：强制确定性（不要任何随机项）
            pass

            # 3) 裁剪并离散到 QP
        a01 = float(np.clip(a01, 0.0, 1.0))
        qp = self.cfg.qp_min + a01 * (self.cfg.qp_max - self.cfg.qp_min)
        qp = int(np.clip(round(qp), self.cfg.qp_min, self.cfg.qp_max))
        return qp

    def _soft_update(self, src, tgt, tau):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def train_step(self):
        if len(self.buf) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        s, a, r, s2, d = self.buf.sample(self.cfg.batch_size)
        s = s.to(self.device);
        a = a.to(self.device);
        r = r.to(self.device)
        s2 = s2.to(self.device);
        d = d.to(self.device)

        # ---------- TD3 target: min(Q1’, Q2’) with target policy smoothing ----------
        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            pn = torch.randn_like(a2) * float(getattr(self.cfg, "policy_noise", 0.10))
            pn = torch.clamp(pn, -float(getattr(self.cfg, "noise_clip", 0.20)),
                             float(getattr(self.cfg, "noise_clip", 0.20)))
            a2 = torch.clamp(a2 + pn, 0.0, 1.0)

            # [可选] 让 target 动作也走一遍 QP 量化，再映回 a01，降低“训练目标”和真实环境的失配
            if bool(getattr(self.cfg, "target_discretize", True)):
                qp2 = self.cfg.qp_min + a2 * (self.cfg.qp_max - self.cfg.qp_min)
                qp2 = torch.clamp(torch.round(qp2), self.cfg.qp_min, self.cfg.qp_max)
                a2 = (qp2 - self.cfg.qp_min) / max(1, (self.cfg.qp_max - self.cfg.qp_min))

            q1_tgt = self.critic1_tgt(s2, a2)
            q2_tgt = self.critic2_tgt(s2, a2)
            q_tgt = torch.min(q1_tgt, q2_tgt)
            y = r + self.cfg.gamma * (1.0 - d) * q_tgt

        # ---------- update twin critics ----------
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        loss_c = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)
        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0)
        self.opt_c.step()

        # ---------- delayed policy update ----------
        loss_a_val = None
        if self.total_train_steps % int(getattr(self.cfg, "policy_delay", 2)) == 0:
            a_pi = self.actor(s)
            # 只用 Critic1 估计 policy gradient（TD3 约定）
            loss_a = - self.critic1(s, a_pi).mean()
            self.opt_a.zero_grad(set_to_none=True)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_a.step()
            loss_a_val = float(loss_a.item())

            # soft update all targets
            tau = self.cfg.tau
            self._soft_update(self.actor, self.actor_tgt, tau)
            self._soft_update(self.critic1, self.critic1_tgt, tau)
            self._soft_update(self.critic2, self.critic2_tgt, tau)

        self.total_train_steps += 1
        self.last_loss_c = float(loss_c.item())
        self.last_loss_a = float(loss_a_val) if loss_a_val is not None else self.last_loss_a
        return self.last_loss_c, (self.last_loss_a if self.last_loss_a is not None else 0.0)

    # ====== 保存/加载（与 DDPG 签名一致，供 main/RLRunner 调用） ======

    def save_checkpoint(self, path: str, extra: dict | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "algo": "td3",
            "actor": self.actor.state_dict(),
            "actor_tgt": self.actor_tgt.state_dict(),
            "critic": getattr(self, "critic", None).state_dict() if hasattr(self, "critic") else None,
            "critic_tgt": getattr(self, "critic_tgt", None).state_dict() if hasattr(self, "critic_tgt") else None,
            "opt_a": getattr(self, "opt_a", None).state_dict() if hasattr(self, "opt_a") else None,
            "opt_c": getattr(self, "opt_c", None).state_dict() if hasattr(self, "opt_c") else None,
            "env_steps": getattr(self, "total_env_steps", 0),
            "train_steps": getattr(self, "total_train_steps", 0),
            "cfg": getattr(self.cfg, "__dict__", None),
        }
        # Remove None entries
        ckpt = {k: v for k, v in ckpt.items() if v is not None}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)

        # ---- Actor ----
        self.actor.load_state_dict(ckpt["actor"])
        if "actor_tgt" in ckpt:
            self.actor_tgt.load_state_dict(ckpt["actor_tgt"])
        else:
            # 兼容旧格式：没有 target 则硬拷贝
            self.actor_tgt.load_state_dict(self.actor.state_dict())

        # ---- Critics ----
        if "critic1" in ckpt and "critic2" in ckpt:
            self.critic1.load_state_dict(ckpt["critic1"])
            self.critic2.load_state_dict(ckpt["critic2"])
        elif "critic" in ckpt:
            # 兼容 DDPG：把单 critic 权重灌到 critic1/critic2
            self.critic1.load_state_dict(ckpt["critic"])
            self.critic2.load_state_dict(ckpt["critic"])
        else:
            raise KeyError("checkpoint missing critic weights")

        if "critic1_tgt" in ckpt and "critic2_tgt" in ckpt:
            self.critic1_tgt.load_state_dict(ckpt["critic1_tgt"])
            self.critic2_tgt.load_state_dict(ckpt["critic2_tgt"])
        else:
            # 兼容旧格式：没有 target 就用在线权重
            self.critic1_tgt.load_state_dict(self.critic1.state_dict())
            self.critic2_tgt.load_state_dict(self.critic2.state_dict())

        # ---- Optims ----
        if "opt_a" in ckpt:
            self.opt_a.load_state_dict(ckpt["opt_a"])
        if "opt_c" in ckpt:
            self.opt_c.load_state_dict(ckpt["opt_c"])

        # ---- Counters ----
        self.total_env_steps = int(ckpt.get("env_steps", 0))
        self.total_train_steps = int(ckpt.get("train_steps", 0))
        return ckpt


class ReplayBufferDual:
    """Replay for dual-critic: (s, a01, rD, rR, s2, done)"""
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self._s = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a = np.zeros((capacity, 1), dtype=np.float32)
        self._rD = np.zeros((capacity, 1), dtype=np.float32)
        self._rR = np.zeros((capacity, 1), dtype=np.float32)
        self._s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self._d = np.zeros((capacity, 1), dtype=np.float32)
        self._n = 0
        self._p = 0
    def __len__(self): return self._n
    def push(self, s, a, rD, rR, s2, d):
        i = self._p
        self._s[i] = s; self._a[i] = a; self._rD[i] = rD; self._rR[i] = rR; self._s2[i] = s2; self._d[i] = d
        self._p = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)
    def sample(self, batch: int):
        idx = np.random.randint(0, self._n, size=(batch,))
        return (torch.from_numpy(self._s[idx]),
                torch.from_numpy(self._a[idx]),
                torch.from_numpy(self._rD[idx]),
                torch.from_numpy(self._rR[idx]),
                torch.from_numpy(self._s2[idx]),
                torch.from_numpy(self._d[idx]))


class DualCriticDDPG:
    """Two critics for different objectives: QD (distortion-to-go) and QR (rate deviation).
    Train critics continuously from RC; update actor only at mini-GOP terminal using QD or QR depending on budget.
    """
    def __init__(self, state_dim: int, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = ActorNet(state_dim).to(self.device)
        self.actor_tgt = ActorNet(state_dim).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic_d = CriticNet(state_dim).to(self.device)
        self.critic_r = CriticNet(state_dim).to(self.device)
        self.critic_d_tgt = CriticNet(state_dim).to(self.device)
        self.critic_r_tgt = CriticNet(state_dim).to(self.device)
        self.critic_d_tgt.load_state_dict(self.critic_d.state_dict())
        self.critic_r_tgt.load_state_dict(self.critic_r.state_dict())

        self.opt_qd = torch.optim.Adam(self.critic_d.parameters(), lr=cfg.critic_lr)
        self.opt_qr = torch.optim.Adam(self.critic_r.parameters(), lr=cfg.critic_lr)
        self.opt_a  = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.bufC = ReplayBufferDual(cfg.replay_size, state_dim)
        self._roll_states = []  # RA: noiseless states for terminal actor update

        self.total_env_steps = 0
        self.total_train_steps = 0
        self.last_loss_d = None
        self.last_loss_r = None
        self.last_loss_a = None

    @torch.no_grad()
    def select_action(self, s: torch.Tensor, base_qp: int, explore: bool) -> int:
        self.actor.eval()
        a01 = self.actor(s.unsqueeze(0).to(self.device)).cpu().item()
        if explore:
            a01 += np.random.randn() * float(getattr(self.cfg, "expl_noise_std", 0.15))
            if np.random.rand() < float(getattr(self.cfg, "action_eps_train", 0.10)):
                a01 = np.random.rand()
        a01 = float(np.clip(a01, 0.0, 1.0))
        # === 关键：以 baseQP 为中心、±窗口的直接映射 ===
        # 可在 config 里放一个开关；若没有配置，则默认 20
        W = int(getattr(self.cfg, "base_qp_window", 20))
        # 也可以区分训练/推理窗口（可选）：
        # W = int(getattr(self.cfg, "base_qp_window_train", 20) if explore
        #         else getattr(self.cfg, "base_qp_window_eval", 20))

        lo = max(self.cfg.qp_min, int(base_qp) - W)
        hi = min(self.cfg.qp_max, int(base_qp) + W)

        if hi <= lo:
            qp = int(lo)
        else:
            qp = int(round(lo + a01 * (hi - lo)))

        return qp

    def rollout_collect_state(self, s):
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        self._roll_states.append(s.astype('float32', copy=False))

    def train_step(self):
        """Update critics from RC; actor updated at terminals elsewhere."""
        if len(self.bufC) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        s, a, rD, rR, s2, d = self.bufC.sample(self.cfg.batch_size)
        s = s.to(self.device); a = a.to(self.device)
        rD = rD.to(self.device); rR = rR.to(self.device)
        s2 = s2.to(self.device); d = d.to(self.device)

        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            qd2 = self.critic_d_tgt(s2, a2)
            qr2 = self.critic_r_tgt(s2, a2)
            yD = rD + self.cfg.gamma * (1.0 - d) * qd2
            yR = rR + self.cfg.gamma * (1.0 - d) * qr2

        qd = self.critic_d(s, a)
        qr = self.critic_r(s, a)
        loss_d = F.smooth_l1_loss(qd, yD)
        loss_r = F.smooth_l1_loss(qr, yR)

        self.opt_qd.zero_grad(set_to_none=True); loss_d.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic_d.parameters(), 1.0); self.opt_qd.step()

        self.opt_qr.zero_grad(set_to_none=True); loss_r.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_r.parameters(), 1.0); self.opt_qr.step()

        # soft update targets
        tau = float(self.cfg.tau)
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)
            for p, pt in zip(self.critic_d.parameters(), self.critic_d_tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)
            for p, pt in zip(self.critic_r.parameters(), self.critic_r_tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

        self.total_train_steps += 1
        self.last_loss_d = float(loss_d.item())
        self.last_loss_r = float(loss_r.item())
        return 0.5 * (self.last_loss_d + self.last_loss_r), None

    def finish_rollout_and_update_actor(self, over_budget: bool):
        """At mini-GOP terminal: update actor using chosen critic (rate if over-budget, else distortion)."""
        if not self._roll_states:
            return None, None
        s = torch.from_numpy(np.stack(self._roll_states, axis=0)).to(self.device)
        self._roll_states.clear()
        self.actor.train()
        with torch.enable_grad():
            a_pi = self.actor(s)
            if over_budget:
                q = self.critic_r(s, a_pi)
                which = "rate"
            else:
                q = self.critic_d(s, a_pi)
                which = "dist"
            loss_a = - q.mean()
            self.opt_a.zero_grad(set_to_none=True); loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_a.step()
        self.last_loss_a = float(loss_a.item())
        return which, self.last_loss_a

    def save_checkpoint(self, path: str, extra: dict | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "algo": "dual",
            "actor": self.actor.state_dict(),
            "actor_tgt": self.actor_tgt.state_dict(),
            "critic_d": self.critic_d.state_dict(),
            "critic_r": self.critic_r.state_dict(),
            "critic_d_tgt": self.critic_d_tgt.state_dict(),
            "critic_r_tgt": self.critic_r_tgt.state_dict(),
            "opt_qd": self.opt_qd.state_dict(),
            "opt_qr": self.opt_qr.state_dict(),
            "opt_a": self.opt_a.state_dict(),
            "env_steps": getattr(self, "total_env_steps", 0),
            "train_steps": getattr(self, "total_train_steps", 0),
            "cfg": getattr(self.cfg, "__dict__", None),
        }
        if extra: ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_tgt.load_state_dict(ckpt.get("actor_tgt", ckpt["actor"]))
        self.critic_d.load_state_dict(ckpt["critic_d"])
        self.critic_r.load_state_dict(ckpt["critic_r"])
        self.critic_d_tgt.load_state_dict(ckpt.get("critic_d_tgt", ckpt["critic_d"]))
        self.critic_r_tgt.load_state_dict(ckpt.get("critic_r_tgt", ckpt["critic_r"]))
        if "opt_qd" in ckpt: self.opt_qd.load_state_dict(ckpt["opt_qd"])
        if "opt_qr" in ckpt: self.opt_qr.load_state_dict(ckpt["opt_qr"])
        if "opt_a"  in ckpt: self.opt_a.load_state_dict(ckpt["opt_a"])
        self.total_env_steps = int(ckpt.get("env_steps", 0))
        self.total_train_steps = int(ckpt.get("train_steps", 0))
        return ckpt

