# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim, hidden, pdrop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(pdrop)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.ln(x + h)

class FilmGate(nn.Module):
    """
    FiLM：用条件向量 z 产生尺度/平移，对主干特征做条件化。
    """
    def __init__(self, z_dim: int, in_dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, in_dim*2)
        )

    def forward(self, h, z):
        # h: [B, C]  z: [B, z_dim]
        gam_beta = self.fc(z)
        C = h.size(1)
        gamma, beta = gam_beta[:, :C], gam_beta[:, C:]
        return h * (1 + gamma) + beta

class ActorNet(nn.Module):
    """
    输入：标准化后的 state（不含 min/max）。
    输出：a01 in (0,1)。
    cond_idx：从 state 中抽取的条件字段索引，默认选取与当前帧/上一帧关联的部分。
    """
    def __init__(self, state_dim: int, hidden: int = 512, depth: int = 4,
                 cond_idx: tuple[int, ...] = (0, 1, 6, 8, 10)):
        super().__init__()
        self.fc_in = nn.Linear(state_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, hidden*2, pdrop=0.1) for _ in range(depth)])
        self.ln = nn.LayerNorm(hidden)
        self.cond_idx = cond_idx
        z_dim = (len(cond_idx) if cond_idx is not None else state_dim)
        self.film = FilmGate(z_dim, hidden, hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, s):
        h = F.gelu(self.fc_in(s))
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        z = s[:, list(self.cond_idx)] if self.cond_idx else s
        h = self.film(h, z)
        a = torch.sigmoid(self.fc_out(h))
        return a  # [B,1]

class CriticNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 512, depth: int = 3):
        super().__init__()
        self.fc_s = nn.Linear(state_dim, hidden)
        self.fc_a = nn.Linear(1, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, hidden*2, pdrop=0.1) for _ in range(depth)])
        self.ln = nn.LayerNorm(hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, s, a01):
        hs = F.gelu(self.fc_s(s))
        ha = F.gelu(self.fc_a(a01))
        h = hs + ha
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        q = self.fc_out(h)
        return q
