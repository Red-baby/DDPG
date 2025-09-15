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
        h = self.ln(h)
        return F.gelu(x + h)

class ActorNetMulti(nn.Module):
    """
    Actor: maps state->[0,1]^A (A=MG_MAX). We map to absolute QP in runner.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden=512, depth=4, pdrop=0.1):
        super().__init__()
        self.fc_in = nn.Linear(state_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, hidden*2, pdrop=pdrop) for _ in range(depth)])
        self.ln = nn.LayerNorm(hidden)
        self.fc_out = nn.Linear(hidden, action_dim)
    def forward(self, s):
        h = F.gelu(self.fc_in(s))
        for b in self.blocks:
            h = b(h)
        h = self.ln(h)
        a = torch.sigmoid(self.fc_out(h))  # [B, A] in [0,1]
        return a

class CriticNetMulti(nn.Module):
    """
    Critic: Q(s,a) with a in [0,1]^A (concatenate then MLP).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden=512, depth=4, pdrop=0.1):
        super().__init__()
        self.fc_in = nn.Linear(state_dim + action_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, hidden*2, pdrop=pdrop) for _ in range(depth)])
        self.ln = nn.LayerNorm(hidden)
        self.fc_out = nn.Linear(hidden, 1)
    def forward(self, s, a01):
        x = torch.cat([s, a01], dim=-1)
        h = F.gelu(self.fc_in(x))
        for b in self.blocks:
            h = b(h)
        h = self.ln(h)
        q = self.fc_out(h)
        return q
