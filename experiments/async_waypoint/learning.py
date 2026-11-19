"""Separate fixed-period upper PPO and event-duration lower SAC implementations."""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def optimize(loss, optimizer, parameters, max_norm):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def temporal_factors(duration, gamma, lam, reference):
    dt = np.asarray(duration, dtype=float)
    if np.any(dt < 0) or not np.isfinite(dt).all():
        raise ValueError("transition duration must be finite and nonnegative")
    return np.power(gamma, dt / reference), np.power(lam, dt / reference)


def duration_gae(rewards, values, dones, durations, last_value, cfg):
    """Terminal blocks both terms; rollout cut bootstraps without crossing resets."""
    gamma, lam = temporal_factors(durations, cfg["gamma"], cfg["gae_lambda"], cfg["reference_sec"])
    adv = np.zeros(len(rewards), np.float64)
    trace = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        nxt = last_value if i == len(rewards) - 1 else values[i + 1]
        live = 0.0 if dones[i] else 1.0
        delta = rewards[i] + live * gamma[i] * nxt - values[i]
        trace = delta + live * gamma[i] * lam[i] * trace
        adv[i] = trace
    return adv.astype(np.float32), (adv + np.asarray(values)).astype(np.float32)


def upper_observation(env):
    obs = env.observation()
    team, global_ = obs["team"], obs["global_"]
    n = env.n
    rows = []
    for uid in range(n):
        rows.append(np.concatenate(([env.soc[uid]], np.eye(4)[env.role[uid]], global_[2:4],
                                    [uid / max(n - 1, 1)], env.soc, env.return_cost(),
                                    [env.return_cost()[uid]], np.eye(8)[env.phase[uid]],
                                    [team[uid, 7]])))
    return dict(actor=np.asarray(rows, np.float32), critic=np.concatenate((team.ravel(), global_)).astype(np.float32))


def dense(input_dim, hidden, output_dim, layers):
    result = []
    for i in range(layers):
        result.extend((nn.Linear(input_dim if i == 0 else hidden, hidden), nn.ReLU()))
    result.append(nn.Linear(hidden, output_dim))
    return nn.Sequential(*result)


class UpperPPO:
    def __init__(self, cfg, env, device):
        self.cfg, self.device = cfg, torch.device(device)
        obs = upper_observation(env)
        self.actor = dense(obs["actor"].shape[1], cfg["hidden"], 2, 4).to(device)
        self.critic = dense(len(obs["critic"]), cfg["hidden"], 1, 3).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        x = torch.as_tensor(obs["actor"], device=self.device)
        dist = torch.distributions.Categorical(logits=self.actor(x))
        sampled = dist.logits.argmax(-1) if deterministic else dist.sample()
        # Preserve historical categorical0=charge; engine1=request.
        return (1 - sampled).cpu().numpy(), dist.log_prob(sampled).cpu().numpy(), self.value(obs), dist.probs[:, 0].cpu().numpy()

    @torch.no_grad()
    def value(self, obs):
        return float(self.critic(torch.as_tensor(obs["critic"], device=self.device)).squeeze())

    def update(self, rows, last_value):
        cfg = self.cfg
        adv, targets = duration_gae([r["reward"] for r in rows], [r["value"] for r in rows],
                                   [r["done"] for r in rows], [r["duration"] for r in rows], last_value, cfg)
        raw_adv = adv.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        def tensor(data, dtype=torch.float32):
            return torch.as_tensor(np.asarray(data), device=self.device, dtype=dtype)
        actors = tensor([r["obs"]["actor"] for r in rows]); critics = tensor([r["obs"]["critic"] for r in rows])
        actions = 1 - tensor([r["action"] for r in rows], torch.long)
        masks = tensor([r["mask"] for r in rows]); old_logp = tensor([r["logp"] for r in rows])
        adv, targets = tensor(adv), tensor(targets)
        metrics = []
        for _ in range(cfg["epochs"]):
            order = torch.randperm(len(rows), device=self.device)
            for start in range(0, len(rows), cfg["minibatch_frames"]):
                ids = order[start:start + cfg["minibatch_frames"]]
                dist = torch.distributions.Categorical(logits=self.actor(actors[ids]))
                log_ratio = dist.log_prob(actions[ids]) - old_logp[ids]
                ratio = log_ratio.exp()
                mask = masks[ids]; denominator = mask.sum().clamp_min(1)
                objective = torch.minimum(ratio * adv[ids, None], ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv[ids, None])
                policy = -(objective * mask).sum() / denominator
                entropy = (dist.entropy() * mask).sum() / denominator
                entropy_loss = -cfg["entropy_coef"] * entropy
                actor_loss = policy + entropy_loss
                actor_norm = optimize(actor_loss, self.actor_opt, self.actor.parameters(), cfg["actor_grad_norm"]) if mask.any() else 0.0
                values = self.critic(critics[ids]).squeeze(-1)
                value_loss = F.mse_loss(values, targets[ids])
                critic_norm = optimize(cfg["value_coef"] * value_loss, self.critic_opt, self.critic.parameters(), cfg["critic_grad_norm"])
                with torch.no_grad():
                    kl = (((ratio - 1) - log_ratio) * mask).sum() / denominator
                metrics.append([float(x.detach()) if torch.is_tensor(x) else float(x) for x in
                                (actor_loss, policy, entropy_loss, entropy, value_loss, kl, actor_norm, critic_norm)])
        means = np.mean(metrics, axis=0)
        names = ("upper_actor_loss", "upper_policy_loss", "upper_entropy_loss", "upper_entropy", "upper_critic_loss", "upper_approx_kl", "upper_actor_grad", "upper_critic_grad")
        return dict(zip(names, means.tolist())) | dict(upper_adv_mean=float(raw_adv.mean()),
                  upper_target_mean=float(targets.mean()), upper_rollout_T=len(rows), upper_rollout_sec=sum(r["duration"] for r in rows))

    def state_dict(self):
        return {key: getattr(self, key).state_dict() for key in ("actor", "critic", "actor_opt", "critic_opt")}

    def load_state_dict(self, state):
        for key, value in state.items():
            getattr(self, key).load_state_dict(value)


class Encoder(nn.Module):
    def __init__(self, hidden, heads, action=False):
        super().__init__()
        self.sensor = dense(3, hidden, hidden, 1)
        self.team = nn.Linear(26 + (2 if action else 0), hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=0, batch_first=True)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0, norm_first=True, batch_first=True)
        self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.global_net = nn.Linear(4, hidden)

    def forward(self, batch, action=None):
        selected = F.one_hot(batch["uid"], batch["team"].shape[1]).float()[..., None]
        inputs = [batch["team"], selected]
        if action is not None:
            inputs.append(selected * action[:, None, :])
        team = self.team(torch.cat(inputs, -1)) + self.global_net(batch["global_"])[:, None]
        sensor = self.sensor(batch["sensors"])
        contextual, _ = self.cross(team, sensor, sensor, need_weights=False)
        return self.attention(self.norm(team + contextual))


class LowerActor(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.encoder = Encoder(hidden, heads)
        self.mean, self.log_std = nn.Linear(hidden, 2), nn.Linear(hidden, 2)

    def distribution(self, batch):
        h = self.encoder(batch)[torch.arange(len(batch["uid"]), device=batch["uid"].device), batch["uid"]]
        return self.mean(h), self.log_std(h).clamp(-5, 2)

    def forward(self, batch, deterministic=False):
        mean, log_std = self.distribution(batch)
        dist = torch.distributions.Normal(mean, log_std.exp())
        z = mean if deterministic else dist.rsample()
        logp = (dist.log_prob(z) - 2 * (math.log(2) - z - F.softplus(-2 * z))).sum(-1)
        return z.tanh(), logp


class Q(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.encoder = Encoder(hidden, heads, action=True)
        self.head = dense(2 * hidden, hidden, 1, 1)

    def forward(self, batch, action):
        h = self.encoder(batch, action)
        selected = h[torch.arange(len(h), device=h.device), batch["uid"]]
        return self.head(torch.cat((h.mean(1), selected), -1)).squeeze(-1)


class LowerSAC:
    def __init__(self, cfg, env, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.xy = env.xy.astype(np.float32) / env.size
        self.scale = env.cfg["backlog_input_scale"]
        self.actor = LowerActor(cfg["hidden"], cfg["heads"]).to(device)
        self.q1, self.q2 = Q(cfg["hidden"], cfg["heads"]).to(device), Q(cfg["hidden"], cfg["heads"]).to(device)
        self.t1, self.t2 = copy.deepcopy(self.q1).requires_grad_(False), copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"])
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg["lr"])
        self.log_alpha = torch.tensor(math.log(cfg["alpha"]), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["lr"])

    def pack(self, obs):
        sensors = np.zeros((len(obs), len(self.xy), 3), np.float32)
        sensors[:, :, :2] = self.xy
        sensors[:, :, 2] = np.asarray([o["backlog"] for o in obs]) / self.scale
        values = dict(team=np.asarray([o["team"] for o in obs]), global_=np.asarray([o["global_"] for o in obs]),
                      uid=np.asarray([o["uid"] for o in obs], np.int64), sensors=sensors)
        return {key: torch.as_tensor(value, device=self.device) for key, value in values.items()}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        action, _ = self.actor(self.pack([obs]), deterministic)
        return action[0].cpu().numpy()

    @torch.no_grad()
    def probe(self, obs):
        mean, log_std = self.actor.distribution(self.pack([obs]))
        return dict(mu=mean[0].cpu().tolist(), sigma=log_std[0].exp().cpu().tolist())

    def update(self, rows):
        obs, actions, rewards, future, done, duration = zip(*rows)
        batch = self.pack(obs)
        action = torch.as_tensor(np.asarray(actions), device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        live = torch.as_tensor(np.logical_not(done), device=self.device)
        discount = torch.as_tensor(np.power(self.cfg["gamma"], np.asarray(duration) / self.cfg["reference_sec"]), dtype=torch.float32, device=self.device)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            target = rewards.clone()
            if live.any():
                nxt = self.pack([o for o, end in zip(future, done) if not end])
                a, logp = self.actor(nxt)
                target[live] += discount[live] * (torch.minimum(self.t1(nxt, a), self.t2(nxt, a)) - alpha * logp)
        q1, q2 = self.q1(batch, action), self.q2(batch, action)
        critic_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        critic_norm = optimize(critic_loss, self.critic_opt, parameters, self.cfg["max_grad_norm"])
        self.q1.requires_grad_(False); self.q2.requires_grad_(False)
        try:
            a, logp = self.actor(batch)
            value = torch.minimum(self.q1(batch, a), self.q2(batch, a))
            policy_loss, entropy_loss = -value.mean(), (alpha * logp).mean()
            actor_loss = policy_loss + entropy_loss
            actor_norm = optimize(actor_loss, self.actor_opt, self.actor.parameters(), self.cfg["max_grad_norm"])
        finally:
            self.q1.requires_grad_(True); self.q2.requires_grad_(True)
        if self.cfg["auto_alpha"]:
            # Exactly one UAV's2D action per decision, regardless of fleet size.
            alpha_loss = -(self.log_alpha * (logp.detach() - 2)).mean()
            optimize(alpha_loss, self.alpha_opt, [self.log_alpha], self.cfg["max_grad_norm"])
            with torch.no_grad():
                self.log_alpha.clamp_(-12, 5)
        with torch.no_grad():
            for net, target_net in ((self.q1, self.t1), (self.q2, self.t2)):
                for p, t in zip(net.parameters(), target_net.parameters()):
                    t.lerp_(p, self.cfg["tau"])
        return dict(lower_actor_loss=float(actor_loss.detach()), lower_policy_loss=float(policy_loss.detach()),
                    lower_entropy_loss=float(entropy_loss.detach()), lower_critic_loss=float(critic_loss.detach()),
                    lower_alpha=float(self.log_alpha.exp().detach()), lower_target_mean=float(target.mean()),
                    lower_q_mean=float(value.mean().detach()), lower_discount_mean=float(discount.mean()),
                    lower_zero_dt_fraction=float(np.mean(np.asarray(duration) == 0)),
                    lower_actor_grad=actor_norm, lower_critic_grad=critic_norm)

    def state_dict(self):
        names = ("actor", "q1", "q2", "t1", "t2", "actor_opt", "critic_opt", "alpha_opt")
        return {key: getattr(self, key).state_dict() for key in names} | {"log_alpha": self.log_alpha.detach().cpu()}

    def load_state_dict(self, state):
        for key, value in state.items():
            if key != "log_alpha":
                getattr(self, key).load_state_dict(value)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))


class Replay:
    def __init__(self, capacity, seed):
        self.capacity, self.index, self.rows = capacity, 0, []
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, future, done, duration):
        if duration < -EPSILON or not np.isfinite(reward):
            raise ValueError("invalid replay transition")
        row = copy.deepcopy((obs, np.asarray(action, np.float32), float(reward), future, bool(done), max(float(duration), 0)))
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.index] = row
        self.index = (self.index + 1) % self.capacity

    def sample(self, count):
        return [self.rows[i] for i in self.rng.choice(len(self.rows), count, replace=False)]

    def state_dict(self):
        return dict(rows=self.rows, index=self.index, rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.rows, self.index = state["rows"], state["index"]
        self.rng.bit_generator.state = state["rng"]


EPSILON = 1e-7
