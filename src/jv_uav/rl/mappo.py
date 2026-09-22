"""Shared Bernoulli actor, centralized team value, clipped PPO and GAE."""
from __future__ import annotations

import numpy as np
import torch

from .networks import mlp
from .sac import backward_step


def gae(rewards, values, dones, last_value, gamma, lam):
    advantages = np.zeros(len(rewards), np.float32)
    accumulator = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        live = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * live * next_value - values[t]
        accumulator = delta + gamma * lam * live * accumulator
        advantages[t] = accumulator
    return advantages, advantages + np.asarray(values, np.float32)


class MAPPO:
    def __init__(self, fleet, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.actor = mlp(2 * fleet + 7, cfg["hidden"], 1).to(device)
        self.critic = mlp(6 * fleet + 2, cfg["hidden"], 1).to(device)
        # Begin with unbiased request logits; requests are not slot allocations.
        torch.nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        torch.nn.init.zeros_(self.actor[-1].bias)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

    @torch.no_grad()
    def request_probabilities(self, obs):
        actors = torch.as_tensor(obs["actor"], device=self.device)
        return self.actor(actors).squeeze(-1).sigmoid().cpu().numpy()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        actors = torch.as_tensor(obs["actor"], device=self.device)
        dist = torch.distributions.Bernoulli(logits=self.actor(actors).squeeze(-1))
        action = (dist.probs >= 0.5).float() if deterministic else dist.sample()
        return action.cpu().numpy().astype(np.int8), dist.log_prob(action).cpu().numpy(), self.value(obs)

    @torch.no_grad()
    def value(self, obs):
        return float(self.critic(torch.as_tensor(obs["critic"], device=self.device)).squeeze())

    def entropy_coefficient(self, upper_steps):
        """Schedule uses total environment frames, including short episodes."""
        start = float(self.cfg["entropy_coef"])
        if "entropy_end_coef" not in self.cfg:
            return start  # Legacy checkpoints keep their constant coefficient.
        begin = self.cfg["entropy_decay_start_frame"]
        end = self.cfg["entropy_decay_end_frame"]
        fraction = float(np.clip((upper_steps - begin) / (end - begin), 0, 1))
        return start + fraction * (float(self.cfg["entropy_end_coef"]) - start)

    def update(self, rollout, last_value, *, upper_steps=0):
        cfg = self.cfg
        entropy_coef = self.entropy_coefficient(upper_steps)
        adv, returns = gae(
            [x["reward"] * cfg["reward_scale"] for x in rollout],
            [x["value"] for x in rollout], [x["done"] for x in rollout],
            last_value, cfg["gamma"], cfg["gae_lambda"],
        )
        adv = (adv - adv.mean()) / max(float(adv.std()), 1e-8)
        to_tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        actors = to_tensor([x["obs"]["actor"] for x in rollout])
        critics = to_tensor([x["obs"]["critic"] for x in rollout])
        actions = to_tensor([x["action"] for x in rollout])
        old_logp = to_tensor([x["logp"] for x in rollout])
        old_values = to_tensor([x["value"] for x in rollout])
        adv, returns = to_tensor(adv), to_tensor(returns)
        result = []
        for _ in range(cfg["epochs"]):
            order = torch.randperm(len(rollout), device=self.device)
            for start in range(0, len(order), cfg["minibatch_frames"]):
                ids = order[start:start + cfg["minibatch_frames"]]
                dist = torch.distributions.Bernoulli(logits=self.actor(actors[ids]).squeeze(-1))
                log_ratio = dist.log_prob(actions[ids]) - old_logp[ids]
                ratio = log_ratio.exp()
                unclipped = ratio * adv[ids, None]
                clipped = ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv[ids, None]
                actor_loss = -torch.minimum(unclipped, clipped).mean() - entropy_coef * dist.entropy().mean()
                backward_step(actor_loss, self.actor_opt, self.actor.parameters(), cfg["max_grad_norm"])
                value = self.critic(critics[ids]).squeeze(-1)
                value_error = (value - returns[ids]).square()
                if cfg.get("value_clip") is not None:
                    value_clipped = old_values[ids] + (value - old_values[ids]).clamp(-cfg["value_clip"], cfg["value_clip"])
                    value_error = torch.maximum(value_error, (value_clipped - returns[ids]).square())
                value_loss = 0.5 * value_error.mean()
                backward_step(value_loss, self.critic_opt, self.critic.parameters(), cfg["max_grad_norm"])
                result.append((float(actor_loss.detach()), float(value_loss.detach())))
        means = np.mean(result, axis=0)
        return {"mappo_actor_loss": float(means[0]), "mappo_value_loss": float(means[1]), "mappo_minibatches": len(result), "mappo_entropy_coef": entropy_coef}

    def state_dict(self):
        return {name: getattr(self, name).state_dict() for name in ("actor", "critic", "actor_opt", "critic_opt")}

    def load_state_dict(self, state):
        for name, value in state.items():
            getattr(self, name).load_state_dict(value)
