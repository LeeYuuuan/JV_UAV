"""Joint waypoint SAC with discounted physical-slot rewards and duration discounts."""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from jv_uav.rl.sac import backward_step


def hidden_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("activation must be relu or gelu")


class Encoder(nn.Module):
    def __init__(self, hidden, heads, action=False, activation="relu", uav_features=3):
        super().__init__()
        self.sensor = nn.Sequential(nn.Linear(3, hidden), hidden_activation(activation), nn.Linear(hidden, hidden))
        self.uav = nn.Linear(uav_features + (2 if action else 0), hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.0,
                                            batch_first=True, norm_first=True, activation=activation)
        self.team = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def forward(self, batch, action=None):
        sensors = self.sensor(batch["sensors"])
        tokens = batch["uavs"]
        if action is not None:
            tokens = torch.cat((tokens, action), -1)
        uav = self.uav(tokens)
        attended, _ = self.cross(uav, sensors, sensors, need_weights=False)
        h = self.norm(uav + attended)
        return self.team(h, src_key_padding_mask=~batch["context_mask"])


class Actor(nn.Module):
    def __init__(self, hidden, heads, activation="relu", uav_features=3):
        super().__init__()
        self.encoder = Encoder(hidden, heads, activation=activation, uav_features=uav_features)
        self.mean = nn.Linear(hidden, 2)
        self.log_std = nn.Linear(hidden, 2)

    def distribution(self, batch):
        h = self.encoder(batch)
        return self.mean(h), self.log_std(h).clamp(-5, 2)

    def forward(self, batch, deterministic=False):
        mean, log_std = self.distribution(batch)
        dist = torch.distributions.Normal(mean, log_std.exp())
        z = mean if deterministic else dist.rsample()
        action = z.tanh().masked_fill(~batch["mask"][..., None], 0)
        logp = (dist.log_prob(z) - 2 * (math.log(2) - z - F.softplus(-2 * z))).sum(-1)
        return action, logp.masked_fill(~batch["mask"], 0).sum(-1)


class Q(nn.Module):
    def __init__(self, hidden, heads, fleet, activation="relu", uav_features=3):
        super().__init__()
        self.encoder = Encoder(hidden, heads, action=True, activation=activation, uav_features=uav_features)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), hidden_activation(activation), nn.Linear(hidden, 1))
        self.fleet = fleet

    def forward(self, batch, action):
        h = self.encoder(batch, action)
        context_count = batch["context_mask"].sum(1, keepdim=True)
        h = h.masked_fill(~batch["context_mask"][..., None], 0).sum(1) / context_count
        return self.head(h).squeeze(-1)


class TwinQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.q1, self.q2 = Q(*args), Q(*args)

    def forward(self, batch, action):
        return self.q1(batch, action), self.q2(batch, action)


class PersistentSAC:
    def __init__(self, cfg, env, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.size = env.scene.world_size
        self.fleet = env.scene.num_uavs
        self.sensor_xy = env.scene.sensor_pos.copy() / self.size
        self.scale = env.cfg["observation_last_visit_scale_sec"]
        self.gamma = cfg["gamma_per_reference"]
        self.reference_sec = env.cfg["reward"]["reference_sec"]
        activation = cfg.get("activation", "relu")
        # Missing config key preserves historical three-feature checkpoints.
        self.include_active_status = cfg.get("include_active_status", False)
        self.uav_features = 4 if self.include_active_status else 3
        self.actor = Actor(cfg["hidden"], cfg["heads"], activation, self.uav_features).to(device)
        self.critic = TwinQ(cfg["hidden"], cfg["heads"], self.fleet, activation, self.uav_features).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"])
        self.log_alpha = torch.tensor(math.log(cfg["alpha"]), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["lr"])

    def pack(self, observations):
        n = len(observations)
        sensors = np.zeros((n, len(self.sensor_xy), 3), np.float32)
        sensors[:, :, :2] = self.sensor_xy
        uavs = np.zeros((n, self.fleet, self.uav_features), np.float32)
        mask = np.zeros((n, self.fleet), bool)
        for i, obs in enumerate(observations):
            sensors[i, :, 2] = obs["sensor_last_visit_sec"] / self.scale
            # Fixed token index=physical UAV ID. Inactive positions affect future
            # departures, so retain all fleet tokens as attention context.
            uavs[i, :, :2] = obs["all_positions"] / self.size
            uavs[i, :, 2] = np.arange(self.fleet) / max(self.fleet - 1, 1)
            mask[i, obs["ids"]] = True
            if self.include_active_status:
                # The same historical active IDs drive network input and legal
                # action masks. Replay never consults the live environment here.
                uavs[i, :, 3] = mask[i].astype(np.float32)
        return {key: torch.as_tensor(value, device=self.device) for key, value in
                dict(sensors=sensors, uavs=uavs, mask=mask, context_mask=np.ones_like(mask)).items()}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        action, _ = self.actor(self.pack([obs]), deterministic)
        ids = torch.as_tensor(obs["ids"], dtype=torch.long, device=self.device)
        return action[0, ids].cpu().numpy()

    @torch.no_grad()
    def diagnostics(self, obs):
        mean, log_std = self.actor.distribution(self.pack([obs]))
        ids = torch.as_tensor(obs["ids"], dtype=torch.long, device=self.device)
        mean = mean[0, ids]; sigma = log_std[0, ids].exp()
        return dict(mu=mean.cpu().tolist(), sigma=sigma.cpu().tolist(),
                    deterministic_action=mean.tanh().cpu().tolist(),
                    uav_is_active=np.isin(np.arange(self.fleet), obs["ids"]).astype(int).tolist(),
                    active_status_in_network=self.include_active_status)

    def update(self, transitions):
        obs, actions, rewards, future, done, duration = zip(*transitions)
        batch, next_batch = self.pack(obs), self.pack(future)
        padded = np.zeros((len(actions), self.fleet, 2), np.float32)
        for i, action in enumerate(actions):
            padded[i, obs[i]["ids"]] = action
        action = torch.as_tensor(padded, device=self.device)
        reward = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        discount = torch.as_tensor([self.gamma ** (dt / self.reference_sec) if not end else 0.0
                                    for dt, end in zip(duration, done)], dtype=torch.float32, device=self.device)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            # Terminal observations are valid non-empty sets, but do not bootstrap.
            nxt_action, logp = self.actor(next_batch)
            q1, q2 = self.target(next_batch, nxt_action)
            target = reward + discount * (torch.minimum(q1, q2) - alpha * logp)
        q1, q2 = self.critic(batch, action)
        critic_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        backward_step(critic_loss, self.critic_opt, self.critic.parameters(), self.cfg["max_grad_norm"])
        self.critic.requires_grad_(False)
        try:
            sampled, logp = self.actor(batch)
            qa, qb = self.critic(batch, sampled)
            q = torch.minimum(qa, qb)
            policy_loss = -q.mean()
            entropy_loss = (alpha * logp).mean()
            actor_loss = policy_loss + entropy_loss
            backward_step(actor_loss, self.actor_opt, self.actor.parameters(), self.cfg["max_grad_norm"])
        finally:
            self.critic.requires_grad_(True)
        alpha_loss = torch.zeros((), device=self.device)
        if self.cfg["auto_alpha"]:
            target_entropy = -2 * batch["mask"].sum(1)
            alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
            backward_step(alpha_loss, self.alpha_opt, [self.log_alpha], self.cfg["max_grad_norm"])
            with torch.no_grad():
                self.log_alpha.clamp_(-12, 5)
        with torch.no_grad():
            for p, target_p in zip(self.critic.parameters(), self.target.parameters()):
                target_p.lerp_(p, self.cfg["tau"])
        return dict(actor_loss=float(actor_loss.detach()), policy_loss=float(policy_loss.detach()),
                    entropy_loss=float(entropy_loss.detach()), critic_loss=float(critic_loss.detach()),
                    alpha=float(self.log_alpha.exp().detach()), alpha_loss=float(alpha_loss.detach()),
                    q_mean=float(q.detach().mean()), target_mean=float(target.mean()),
                    reward_mean=float(reward.mean()), discount_mean=float(discount.mean()),
                    input_last_visit_abs_max=float(batch["sensors"][:, :, 2].abs().max()))

    def state_dict(self):
        return {name: getattr(self, name).state_dict() for name in
                ("actor", "critic", "target", "actor_opt", "critic_opt", "alpha_opt")} | {
                    "log_alpha": self.log_alpha.detach().cpu()}

    def load_state_dict(self, state):
        for name in ("actor", "critic", "target", "actor_opt", "critic_opt", "alpha_opt"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))


def compact(obs):
    # Explicit observable-only replay schema; no simulation queues or clocks.
    return {k: copy.deepcopy(obs[k]) for k in ("sensor_last_visit_sec", "ids", "positions", "all_positions")}


class Replay:
    def __init__(self, capacity, seed):
        self.capacity, self.index = capacity, 0
        self.items = []
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, future, done, duration):
        row = (compact(obs), action.copy(), float(reward), compact(future), bool(done), float(duration))
        if len(self.items) < self.capacity:
            self.items.append(row)
        else:
            self.items[self.index] = row
        self.index = (self.index + 1) % self.capacity

    def sample(self, count):
        return [self.items[i] for i in self.rng.choice(len(self.items), count, replace=False)]

    def state_dict(self):
        return dict(items=self.items, index=self.index, rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.items, self.index = state["items"], state["index"]
        self.rng.bit_generator.state = state["rng"]
