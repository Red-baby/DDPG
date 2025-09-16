# -*- coding: utf-8 -*-
import math, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn
import os
from models import ActorNetMulti, CriticNetMulti

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._s  = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a  = np.zeros((capacity, action_dim), dtype=np.float32)
        self._r  = np.zeros((capacity, 1), dtype=np.float32)
        self._s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self._d  = np.zeros((capacity, 1), dtype=np.float32)
        self._n = 0; self._p = 0

    def __len__(self): return self._n

    def push(self, s, a, r, s2, d):
        i = self._p
        self._s[i,:]  = s.reshape(-1)
        self._a[i,:]  = a.reshape(-1)[:self.action_dim]
        self._r[i,:]  = r.reshape(1)
        self._s2[i,:] = s2.reshape(-1)
        self._d[i,:]  = d.reshape(1)
        self._p = (self._p + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self._n, size=(batch_size,))
        s  = torch.from_numpy(self._s[idx]).to(device)
        a  = torch.from_numpy(self._a[idx]).to(device)
        r  = torch.from_numpy(self._r[idx]).to(device)
        s2 = torch.from_numpy(self._s2[idx]).to(device)
        d  = torch.from_numpy(self._d[idx]).to(device)
        return s, a, r, s2, d

class TD3Multi:
    def __init__(self, state_dim: int, action_dim: int, cfg):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        hidden = int(getattr(cfg, "hidden_dim", 512))
        depth  = int(getattr(cfg, "depth", 4))

        self.actor     = ActorNetMulti(state_dim, action_dim, hidden=hidden, depth=depth).to(self.device)
        self.actor_tgt = ActorNetMulti(state_dim, action_dim, hidden=hidden, depth=depth).to(self.device)
        self.critic1   = CriticNetMulti(state_dim, action_dim, hidden=hidden, depth=depth).to(self.device)
        self.critic2   = CriticNetMulti(state_dim, action_dim, hidden=hidden, depth=depth).to(self.device)
        self.critic1_tgt= CriticNetMulti(state_dim, action_dim, hidden=hidden, depth=depth).to(self.device)
        self.critic2_tgt= CriticNetMulti(state_dim, action_dim, hidden=hidden, depth=depth).to(self.device)

        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic1_tgt.load_state_dict(self.critic1.state_dict())
        self.critic2_tgt.load_state_dict(self.critic2.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=float(getattr(cfg, "lr_actor", 1e-4)))
        self.opt_c = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                      lr=float(getattr(cfg, "lr_critic", 1e-4)))

        self.gamma = float(getattr(cfg, "gamma", 0.99))
        self.tau   = float(getattr(cfg, "tau", 0.005))
        self.policy_noise = float(getattr(cfg, "policy_noise", 0.05))  # noise in action space [0,1]
        self.noise_clip   = float(getattr(cfg, "noise_clip", 0.20))
        self.policy_delay = int(getattr(cfg, "policy_delay", 2))
        self.batch_size   = int(getattr(cfg, "batch_size", 64))

        rb_sz = int(getattr(cfg, "replay_size", 20000))
        self.buf = ReplayBuffer(rb_sz, self.state_dim, self.action_dim)

        self.total_env_steps = 0
        self.total_train_steps = 0

    # ==== in agent.py (类里) ====

    def _act_a01(self, s: torch.Tensor, explore: bool) -> torch.Tensor:
        """
        Get action in [0,1]^action_dim. Add exploration noise if training.
        """
        self.actor.eval()
        with torch.no_grad():
            a01 = self.actor(s.to(self.device).float().unsqueeze(0)).clamp(0.0, 1.0)  # [1, action_dim]
        a01 = a01.cpu().squeeze(0)
        if explore:
            eps = float(getattr(self.cfg, "explore_eps", 0.10))
            if eps > 0:
                noise = torch.randn_like(a01) * eps
                a01 = (a01 + noise).clamp(0.0, 1.0)
        return a01  # [action_dim]

    def _map_around_base(self, a01: torch.Tensor, base_q: torch.Tensor, qp_min: int, qp_max: int) -> torch.Tensor:
        """
        把 a01∈[0,1] 映射成以 base_q 为中心的绝对 QP：
          a01=0.5 -> ΔQP = 0（不变）
          a01<0.5 -> 负向调整；a01>0.5 -> 正向调整
          ΔQP 最大幅度由 cfg.delta_qp_max 控制（默认 20）
        """
        delta_max = int(getattr(self.cfg, "delta_qp_max", 20))
        # a01∈[0,1] -> [-1, +1] -> 乘以 Δmax -> 四舍五入成整数 ΔQP
        delta = torch.round((a01 - 0.5) * 2.0 * delta_max)
        qps = base_q.to(delta) + delta
        qps = torch.clamp(qps, min=qp_min, max=qp_max).to(torch.int32)
        return qps

    def select_action_vector(
            self,
            s: torch.Tensor,
            mg_size: int,
            qp_min: int,
            qp_max: int,
            base_q_vec=None,  # <--- 新增：来自 RQ 的每帧 base_q（长度=MG_MAX）
            explore: bool = True,
    ):
        """
        Return:
          a01_vec: np.ndarray [action_dim] in [0,1]
          qp_vec : np.ndarray [action_dim] absolute QP mapped around base_q
        说明：
          - 只写回前 mg_size 个 QP 到 .qp.txt；后面位置（补齐帧）不会被编码器使用
          - 若未提供 base_q_vec，则退化为以区间中点为基准的映射
        """
        a01 = self._act_a01(s, explore=explore)  # [action_dim]
        if base_q_vec is not None:
            base = torch.tensor(base_q_vec, dtype=torch.float32)  # [action_dim]
        else:
            # 兜底：没有 base_q 时用区间中点
            base_val = 0.5 * (float(qp_min) + float(qp_max))
            base = torch.full_like(a01, base_val)

        qps = self._map_around_base(a01, base, qp_min, qp_max)  # [action_dim] int32
        return a01.numpy(), qps.cpu().numpy()

    def train_step(self):
        warmup = int(getattr(self.cfg, "warmup_min_transitions", max(32, self.batch_size)))
        if len(self.buf) < warmup:
            return None

        s, a01, r, s2, d = self.buf.sample(self.batch_size, self.device)

        # target action with clipped noise
        with torch.no_grad():
            a2 = self.actor_tgt(s2).clamp(0.0, 1.0)
            if self.policy_noise > 0:
                n = (torch.randn_like(a2) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                a2 = (a2 + n).clamp(0.0, 1.0)
            q1t = self.critic1_tgt(s2, a2)
            q2t = self.critic2_tgt(s2, a2)
            qt  = torch.min(q1t, q2t)
            y   = r + (1.0 - d) * self.gamma * qt

        # critic update
        q1 = self.critic1(s, a01)
        q2 = self.critic2(s, a01)
        loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), 5.0)
        self.opt_c.step()

        # delayed policy update
        if (self.total_train_steps % self.policy_delay) == 0:
            a = self.actor(s).clamp(0.0, 1.0)
            # maximize Q => minimize -Q
            loss_a = - self.critic1(s, a).mean()
            self.opt_a.zero_grad(set_to_none=True)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.opt_a.step()

            # Polyak
            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
                for p, pt in zip(self.critic1.parameters(), self.critic1_tgt.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
                for p, pt in zip(self.critic2.parameters(), self.critic2_tgt.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        self.total_train_steps += 1
        return float(loss_c.detach().cpu().item()), float((loss_a.detach().cpu().item()) if 'loss_a' in locals() else 0.0)

    # ---------- checkpoint ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_tgt": self.actor_tgt.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic1_tgt": self.critic1_tgt.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic2_tgt": self.critic2_tgt.state_dict(),
            "opt_a": self.opt_a.state_dict(),
            "opt_c": self.opt_c.state_dict(),
            "env_steps": int(getattr(self, "total_env_steps", 0)),
            "train_steps": int(getattr(self, "total_train_steps", 0)),
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"]);     self.actor_tgt.load_state_dict(ckpt["actor_tgt"])
        self.critic1.load_state_dict(ckpt["critic1"]); self.critic1_tgt.load_state_dict(ckpt["critic1_tgt"])
        self.critic2.load_state_dict(ckpt["critic2"]); self.critic2_tgt.load_state_dict(ckpt["critic2_tgt"])
        self.opt_a.load_state_dict(ckpt["opt_a"]); self.opt_c.load_state_dict(ckpt["opt_c"])
        self.total_env_steps  = int(ckpt.get("env_steps", 0))
        self.total_train_steps= int(ckpt.get("train_steps", 0))
        return ckpt

def get_agent(state_dim: int, cfg, action_dim: int):
    return TD3Multi(state_dim, action_dim, cfg)
