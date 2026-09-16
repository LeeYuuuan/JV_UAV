"""Attention SAC with configurable identity features and shared-actor MAPPO."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


class AttentionEncoder(nn.Module):
    def __init__(self, sensors, token_dim, hidden, heads, layers):
        super().__init__()
        self.global_net = nn.Sequential(nn.Linear(sensors, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.token_net = nn.Linear(token_dim + hidden, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.0, batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, obs, actions=None):
        if not obs["mask"].any(dim=1).all():
            raise ValueError("attention received an empty nonterminal UAV set")
        global_ = self.global_net(obs["global"])
        tokens = obs["tokens"]
        if actions is not None:
            tokens = torch.cat([tokens, actions], dim=-1)
        repeated = global_[:, None, :].expand(-1, tokens.shape[1], -1)
        h = self.token_net(torch.cat([tokens, repeated], dim=-1))
        return self.attention(h, src_key_padding_mask=~obs["mask"]), global_


class AttentionActor(nn.Module):
    def __init__(self, sensors, hidden, heads, layers, token_dim=3):
        super().__init__()
        self.encoder = AttentionEncoder(sensors, token_dim, hidden, heads, layers)
        self.mean = nn.Linear(hidden, 2)
        self.log_std = nn.Linear(hidden, 2)

    def forward(self, obs, deterministic=False):
        h, _ = self.encoder(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        dist = torch.distributions.Normal(mean, log_std.exp())
        z = mean if deterministic else dist.rsample()
        actions = torch.tanh(z)
        # Stable tanh Jacobian, in the normalized [-1, 1] action domain.
        logp = (dist.log_prob(z) - 2 * (math.log(2) - z - F.softplus(-2 * z))).sum(dim=-1)
        actions = actions.masked_fill(~obs["mask"][:, :, None], 0)
        logp = logp.masked_fill(~obs["mask"], 0).sum(dim=1)
        return actions, logp


class AttentionQ(nn.Module):
    def __init__(self, sensors, hidden, heads, layers, fleet_size, token_dim=3):
        super().__init__()
        self.encoder = AttentionEncoder(sensors, token_dim + 2, hidden, heads, layers)
        self.fleet_size = fleet_size
        self.head = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, action):
        h, global_ = self.encoder(obs, action)
        mask = obs["mask"][:, :, None]
        count = mask.sum(dim=1)
        pooled = h.masked_fill(~mask, 0).sum(dim=1) / count.clamp_min(1)
        return self.head(torch.cat([pooled, global_, count / self.fleet_size], dim=-1)).squeeze(-1)


class TwinQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.q1 = AttentionQ(*args)
        self.q2 = AttentionQ(*args)

    def forward(self, obs, actions):
        return self.q1(obs, actions), self.q2(obs, actions)
