# -*- coding: utf-8 -*-
import math, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn
import os
from models import ActorNet, CriticNet
from state import STATE_FIELDS

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self._s  = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a  = np.zeros((capacity, 1), dtype=np.float32)
        self._r  = np.zeros((capacity, 1), dtype=np.float32)
        self._s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self._d  = np.zeros((capacity, 1), dtype=np.float32)
        self._n = 0; self._p = 0
    def __len__(self): return self._n
    def push(self, s, a, r, s2, d):
        self._s[self._p]  = s;   self._a[self._p]  = a
        self._r[self._p]  = r;   self._s2[self._p] = s2
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

class TD3:
    def __init__(self, state_dim: int, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = ActorNet(state_dim).to(self.device)
        self.actor_tgt = ActorNet(state_dim).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic1 = CriticNet(state_dim).to(self.device)
        self.critic2 = CriticNet(state_dim).to(self.device)
        self.critic1_tgt = CriticNet(state_dim).to(self.device)
        self.critic2_tgt = CriticNet(state_dim).to(self.device)
        self.critic1_tgt.load_state_dict(self.critic1.state_dict())
        self.critic2_tgt.load_state_dict(self.critic2.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(list(self.critic1.parameters())+list(self.critic2.parameters()),
                                      lr=cfg.critic_lr)

        self.buf = ReplayBuffer(cfg.replay_size, state_dim)
        self.total_env_steps = 0
        self.total_train_steps = 0
        self._upd = 0
        self.last_loss_c = None
        self.last_loss_a = None

    @torch.no_grad()
    def select_action(self, s: torch.Tensor, base_q: int, explore: bool) -> int:
        self.actor.eval()
        a01 = self.actor(s.unsqueeze(0).to(self.device)).cpu().item()
        if explore:
            a01 += np.random.randn() * float(getattr(self.cfg, "expl_noise_std", 0.15))
            if np.random.rand() < float(getattr(self.cfg, "action_eps_train", 0.10)):
                a01 = np.random.rand()
        else:
            if float(getattr(self.cfg, "action_eps_infer", 0.0)) > 0 and \
               (np.random.rand() < float(self.cfg.action_eps_infer)):
                a01 = np.random.rand()
        a01 = float(np.clip(a01, 0.0, 1.0))
        delta_max = float(getattr(self.cfg, "delta_qp_max", 10))
        delta = (a01 - 0.5) * 2.0 * delta_max
        qp = int(np.clip(round(float(base_q) + delta), self.cfg.qp_min, self.cfg.qp_max))
        return qp

    def _soft_update(self, src, tgt, tau):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def train_step(self):
        if len(self.buf) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        s, a, r, s2, d = self.buf.sample(self.cfg.batch_size)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        s2 = s2.to(self.device); d = d.to(self.device)

        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            pn = torch.randn_like(a2) * float(getattr(self.cfg, "policy_noise", 0.10))
            pn = torch.clamp(pn, -float(getattr(self.cfg, "noise_clip", 0.20)),
                                float(getattr(self.cfg, "noise_clip", 0.20)))
            a2 = torch.clamp(a2 + pn, 0.0, 1.0)
            if bool(getattr(self.cfg, "target_discretize", True)):
                qp2 = self.cfg.qp_min + a2 * (self.cfg.qp_max - self.cfg.qp_min)
                qp2 = torch.clamp(torch.round(qp2), self.cfg.qp_min, self.cfg.qp_max)
                a2 = (qp2 - self.cfg.qp_min) / max(1, (self.cfg.qp_max - self.cfg.qp_min))
            q1_tgt = self.critic1_tgt(s2, a2)
            q2_tgt = self.critic2_tgt(s2, a2)
            y = r + self.cfg.gamma * (1.0 - d) * torch.min(q1_tgt, q2_tgt)

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        loss_c = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)
        self.opt_c.zero_grad(set_to_none=True); loss_c.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters())+list(self.critic2.parameters()), 1.0)
        self.opt_c.step()

        self._upd += 1
        if (self._upd % int(getattr(self.cfg, "policy_delay", 2))) == 0:
            a_pi = self.actor(s)
            loss_a = - self.critic1(s, a_pi).mean()
            self.opt_a.zero_grad(set_to_none=True); loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_a.step()
            self._soft_update(self.actor, self.actor_tgt, self.cfg.tau)
            self._soft_update(self.critic1, self.critic1_tgt, self.cfg.tau)
            self._soft_update(self.critic2, self.critic2_tgt, self.cfg.tau)
            self.last_loss_a = float(loss_a.item())
        self.last_loss_c = float(loss_c.item())
        self.total_train_steps += 1
        return self.last_loss_c, self.last_loss_a

    def save_checkpoint(self, path: str, extra: dict | None = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "algo": "td3",
            "actor": self.actor.state_dict(), "actor_tgt": self.actor_tgt.state_dict(),
            "critic1": self.critic1.state_dict(),"critic2": self.critic2.state_dict(),
            "critic1_tgt": self.critic1_tgt.state_dict(),"critic2_tgt": self.critic2_tgt.state_dict(),
            "opt_a": self.opt_a.state_dict(),"opt_c": self.opt_c.state_dict(),
            "env_steps": self.total_env_steps, "train_steps": self.total_train_steps,
        }
        if extra: ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"]);     self.actor_tgt.load_state_dict(ckpt["actor_tgt"])
        self.critic1.load_state_dict(ckpt["critic1"]); self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_tgt.load_state_dict(ckpt["critic1_tgt"])
        self.critic2_tgt.load_state_dict(ckpt["critic2_tgt"])
        self.opt_a.load_state_dict(ckpt["opt_a"]); self.opt_c.load_state_dict(ckpt["opt_c"])
        self.total_env_steps  = int(ckpt.get("env_steps", 0))
        self.total_train_steps= int(ckpt.get("train_steps", 0))
        return ckpt

def get_agent(state_dim: int, cfg):
    # TD3 与 TD3-Lagrangian 的“Lagrangian”在 reward/runner 中体现；
    # agent 结构相同，这里统一返回 TD3。
    return TD3(state_dim, cfg)
