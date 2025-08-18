# -*- coding: utf-8 -*-
import math, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
from models import ActorNet, CriticNet

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
        x = self.x_prev + self.theta*(0.0 - self.x_prev)*self.dt + self.sigma * math.sqrt(self.dt) * np.random.randn()
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
            return
        s, a, r, s2, d = self.buf.sample(self.cfg.batch_size)
        s  = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        s2 = s2.to(self.device); d = d.to(self.device)

        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            q2 = self.critic_tgt(s2, a2)
            y  = r + self.cfg.gamma * (1.0 - d) * q2
        q = self.critic(s, a)
        loss_c = F.mse_loss(q, y)
        self.opt_c.zero_grad(set_to_none=True); loss_c.backward(); self.opt_c.step()

        a_pi = self.actor(s)
        loss_a = - self.critic(s, a_pi).mean()
        self.opt_a.zero_grad(set_to_none=True); loss_a.backward(); self.opt_a.step()

        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        self.total_train_steps += 1
