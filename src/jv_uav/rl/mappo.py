"""Shared Bernoulli actor, centralized team value, clipped PPO and GAE."""
from __future__ import annotations

import numpy as np
import torch

from .networks import mlp
from .sac import backward_step


def gae(rewards, values, dones, last_value, gamma, lam):
    if torch.is_tensor(rewards):
        advantages = torch.zeros_like(rewards)
    else:
        advantages = np.zeros(len(rewards), np.float32)
    accumulator = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        live = 1.0 - (dones[t] if torch.is_tensor(dones) else float(dones[t]))
        delta = rewards[t] + gamma * next_value * live - values[t]
        accumulator = delta + gamma * lam * live * accumulator
        advantages[t] = accumulator
    targets = values if torch.is_tensor(values) else np.asarray(values, np.float32)
    return advantages, advantages + targets


class MAPPO:
    def __init__(self, fleet, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.observation_mode = cfg.get("observation_mode", "current_soc_return_time")
        if self.observation_mode not in ("arrival_soc", "current_soc_return_time", "current_soc_return_soc"):
            raise ValueError("unknown upper observation mode")
        arrival = self.observation_mode == "arrival_soc"
        return_soc = self.observation_mode == "current_soc_return_soc"
        architecture = cfg.get("actor_architecture", "default")
        own_return = cfg.get("include_own_return_soc", False)
        if type(own_return) is not bool:
            raise ValueError("include_own_return_soc must be boolean")
        if own_return and (not return_soc or architecture != "paper1_notebook"):
            raise ValueError("own return SOC requires the Paper1 actor with current_soc_return_soc")
        self.paper1_ppo = cfg.get("paper1_ppo", False)
        if self.paper1_ppo and architecture != "paper1_notebook":
            raise ValueError("Paper1 PPO requires the Paper1 actor architecture")
        if architecture == "paper1_notebook":
            if not (arrival or return_soc):
                raise ValueError("Paper1 adapter requires arrival_soc or current_soc_return_soc observations")
            from .transfer import Paper1Actor
            self.actor = Paper1Actor(fleet, cfg["transfer_actor_hidden"], include_return_soc=return_soc, include_own_return_soc=own_return).to(device)
        elif architecture == "default":
            self.actor = mlp(2 * fleet + (6 if arrival or return_soc else 7) + (fleet if return_soc else 0), cfg["hidden"], 1).to(device)
        else:
            raise ValueError("unknown actor_architecture")
        critic_dim = (5 if arrival or return_soc else 6) * fleet + 2 + (fleet if return_soc else 0)
        hidden = cfg["hidden"]
        if self.paper1_ppo:
            self.critic = torch.nn.Sequential(
                torch.nn.Linear(critic_dim, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, 1),
            ).to(device)
        else:
            self.critic = mlp(critic_dim, hidden, 1).to(device)
        # Begin with unbiased request logits; requests are not slot allocations.
        if architecture == "default":
            torch.nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            torch.nn.init.zeros_(self.actor[-1].bias)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

    def _distribution(self, actors):
        if self.paper1_ppo:
            return torch.distributions.Categorical(logits=self.actor.categorical_logits(actors))
        return torch.distributions.Bernoulli(logits=self.actor(actors).squeeze(-1))

    @torch.no_grad()
    def request_probabilities(self, obs):
        actors = torch.as_tensor(obs["actor"], device=self.device)
        probs = self._distribution(actors).probs
        return (probs[..., 0] if self.paper1_ppo else probs).cpu().numpy()

    @torch.no_grad()
    def act(self, obs, deterministic=False, *, include_probabilities=False):
        actors = torch.as_tensor(obs["actor"], device=self.device)
        dist = self._distribution(actors)
        if self.paper1_ppo:
            sampled = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
            # Paper1 uses action 0 for charge, while the JV scheduler expects 1.
            action = 1 - sampled
            probabilities = dist.probs[..., 0]
        else:
            sampled = (dist.probs >= 0.5).float() if deterministic else dist.sample()
            action = sampled
            probabilities = dist.probs
        result = (action.cpu().numpy().astype(np.int8),
                  dist.log_prob(sampled).cpu().numpy(), self.value(obs))
        if include_probabilities:
            return (*result, probabilities.cpu().numpy())
        return result

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
        to_tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        rewards = [x["reward"] * cfg["reward_scale"] for x in rollout]
        values = [x["value"] for x in rollout]
        dones = [x["done"] for x in rollout]
        if self.paper1_ppo:
            rewards, values, dones = map(to_tensor, (rewards, values, dones))
        adv, returns = gae(rewards, values, dones, last_value, cfg["gamma"], cfg["gae_lambda"])
        actors = to_tensor([x["obs"]["actor"] for x in rollout])
        critics = to_tensor([x["obs"]["critic"] for x in rollout])
        actions = to_tensor([x["action"] for x in rollout])
        if self.paper1_ppo:
            actions = 1 - actions.long()
        old_logp = to_tensor([x["logp"] for x in rollout])
        old_values = to_tensor([x["value"] for x in rollout])
        if self.paper1_ppo:
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        else:
            adv = (adv - adv.mean()) / max(float(adv.std()), 1e-8)
            adv, returns = to_tensor(adv), to_tensor(returns)
        credit_enabled = cfg.get("credit_diagnostics", False)
        if credit_enabled:
            # Pure distribution queries: no samples, gradients or RNG changes.
            raw_adv, raw_targets = gae(rewards, values, dones, last_value, cfg["gamma"], cfg["gae_lambda"])
            with torch.no_grad():
                probabilities = self._distribution(actors).probs
                charge_before = (probabilities[..., 0] if self.paper1_ppo else probabilities).cpu().numpy()
        actor_grad_norm = cfg.get("max_grad_norm_actor", cfg["max_grad_norm"])
        critic_grad_norm = cfg.get("max_grad_norm_critic", cfg["max_grad_norm"])
        value_coef = cfg.get("value_coef", 0.5)
        result = []
        for _ in range(cfg["epochs"]):
            order = torch.randperm(len(rollout), device=self.device)
            for start in range(0, len(order), cfg["minibatch_frames"]):
                ids = order[start:start + cfg["minibatch_frames"]]
                dist = self._distribution(actors[ids])
                log_ratio = dist.log_prob(actions[ids]) - old_logp[ids]
                ratio = log_ratio.exp()
                unclipped = ratio * adv[ids, None]
                clipped = ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv[ids, None]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                entropy = dist.entropy().mean()
                actor_loss = policy_loss - entropy_coef * entropy
                backward_step(actor_loss, self.actor_opt, self.actor.parameters(), actor_grad_norm)
                value = self.critic(critics[ids]).squeeze(-1)
                if cfg.get("value_clip") is None:
                    value_loss = torch.nn.functional.mse_loss(value, returns[ids])
                else:
                    value_error = (value - returns[ids]).square()
                    value_clipped = old_values[ids] + (value - old_values[ids]).clamp(-cfg["value_clip"], cfg["value_clip"])
                    value_loss = torch.maximum(value_error, (value_clipped - returns[ids]).square()).mean()
                scaled_value_loss = value_coef * value_loss
                backward_step(scaled_value_loss, self.critic_opt, self.critic.parameters(), critic_grad_norm)
                logged_value_loss = value_loss if self.paper1_ppo else scaled_value_loss
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfrac = ((ratio - 1).abs() > cfg["clip"]).float().mean()
                result.append((float(actor_loss.detach()), float(logged_value_loss.detach()),
                               float(policy_loss.detach()), float(entropy.detach()),
                               float(approx_kl), float(clipfrac)))
        means = np.mean(result, axis=0)
        metrics = {"mappo_actor_loss": float(means[0]), "mappo_value_loss": float(means[1]), "mappo_minibatches": len(result), "mappo_entropy_coef": entropy_coef,
                "mappo_policy_loss": float(means[2]), "mappo_entropy": float(means[3]),
                "mappo_weighted_entropy": entropy_coef * float(means[3]),
                "mappo_approx_kl": float(means[4]), "mappo_clipfrac": float(means[5]),
                "mappo_value_target_mean": float(returns.mean()),
                "mappo_value_prediction_mean": float(old_values.mean())}
        if credit_enabled:
            with torch.no_grad():
                probabilities = self._distribution(actors).probs
                charge_after = (probabilities[..., 0] if self.paper1_ppo else probabilities).cpu().numpy()
            metrics["mappo_credit_trace"] = [dict(
                upper_steps=row.get("upper_steps"), reward=float(rewards[index]), done=bool(row["done"]),
                value=float(old_values[index]), value_target=float(raw_targets[index]),
                advantage_raw=float(raw_adv[index]), advantage_normalized=float(adv[index]),
                actions=row["action"].tolist(), battery_start=row.get("battery_start"),
                assigned_status=row.get("assigned_status"), minimum_service_soc=row.get("minimum_service_soc"),
                failure_records=row.get("failure_records", []),
                p_charge_before=charge_before[index].tolist(), p_charge_after=charge_after[index].tolist(),
            ) for index, row in enumerate(rollout)]
        return metrics

    def state_dict(self):
        return {name: getattr(self, name).state_dict() for name in ("actor", "critic", "actor_opt", "critic_opt")}

    def load_state_dict(self, state):
        for name, value in state.items():
            getattr(self, name).load_state_dict(value)
