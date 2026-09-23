"""Attention SAC with twin targets and active-dimension entropy tuning."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch.nn import functional as F

from .networks import AttentionActor, TwinQ
from .replay import pack_observations


def backward_step(loss, optimizer, parameters, max_norm):
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()


class SAC:
    def __init__(self, env_cfg, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        sensors, fleet = env_cfg["sensors"]["count"], env_cfg["uavs"]["count"]
        args = (sensors, cfg["hidden"], cfg["heads"], cfg["layers"])
        # Missing flag preserves the architecture of pre-identity checkpoints.
        self.identity_fleet_size = fleet if cfg.get("include_uav_id", False) else None
        token_dim = 4 if self.identity_fleet_size is not None else 3
        self.actor = AttentionActor(*args, token_dim).to(device)
        self.critic = TwinQ(*args, fleet, token_dim).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"])
        self.log_alpha = torch.tensor(float(np.log(cfg["alpha"])), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["lr"])
        self.world_size = env_cfg["world"]["size_m"]

    @torch.no_grad()
    def act(self, obs, rms, deterministic=False):
        batch = pack_observations([obs], rms, self.world_size, self.device, self.identity_fleet_size)
        action, _ = self.actor(batch, deterministic)
        return action[0, :len(obs["active_uav_ids"])].cpu().numpy()

    @torch.no_grad()
    def action_diagnostics(self, obs, rms):
        # Inspect distribution parameters without drawing samples or changing RNG.
        batch = pack_observations([obs], rms, self.world_size, self.device, self.identity_fleet_size)
        h, _ = self.actor.encoder(batch)
        mean = self.actor.mean(h)[batch["mask"]]
        sigma = self.actor.log_std(h).clamp(-5, 2).exp()[batch["mask"]]
        return dict(components=mean.numel(), mu_abs_mean=float(mean.abs().mean()),
                    mu_abs_max=float(mean.abs().max()), sigma_mean=float(sigma.mean()),
                    sigma_min=float(sigma.min()), sigma_max=float(sigma.max()),
                    deterministic_saturation_fraction=float((mean.tanh().abs() > .99).float().mean()))

    def update(self, replay, rms):
        cfg = self.cfg
        obs, action, reward, next_, done = replay.sample(cfg["batch_size"], rms, self.world_size, self.device, self.identity_fleet_size)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            target = reward * cfg["reward_scale"]
            alive = ~done
            # Terminal sets may be empty. Never evaluate their bootstrap branch.
            if alive.any():
                nxt = {key: value[alive] for key, value in next_.items()}
                a, logp = self.actor(nxt)
                q1, q2 = self.target(nxt, a)
                target[alive] += cfg["gamma"] * (torch.minimum(q1, q2) - alpha * logp)
        q1, q2 = self.critic(obs, action)
        critic_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        backward_step(critic_loss, self.critic_opt, self.critic.parameters(), cfg["max_grad_norm"])
        self.critic.requires_grad_(False)
        try:
            new_action, logp = self.actor(obs)
            qa, qb = self.critic(obs, new_action)
            actor_loss = (alpha * logp - torch.minimum(qa, qb)).mean()
            backward_step(actor_loss, self.actor_opt, self.actor.parameters(), cfg["max_grad_norm"])
        finally:
            self.critic.requires_grad_(True)
        alpha_loss = torch.zeros((), device=self.device)
        if cfg["auto_alpha"]:
            target_entropy = -2.0 * obs["mask"].sum(dim=1)
            alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
            backward_step(alpha_loss, self.alpha_opt, [self.log_alpha], cfg["max_grad_norm"])
            with torch.no_grad():
                self.log_alpha.clamp_(-12, 5)
        with torch.no_grad():
            for target_param, source in zip(self.target.parameters(), self.critic.parameters()):
                target_param.lerp_(source, cfg["tau"])
        return {"sac_actor_loss": float(actor_loss.detach()), "sac_critic_loss": float(critic_loss.detach()), "sac_alpha": float(self.log_alpha.exp().detach())}

    def state_dict(self):
        return {name: getattr(self, name).state_dict() for name in ("actor", "critic", "target", "actor_opt", "critic_opt", "alpha_opt")} | {"log_alpha": self.log_alpha.detach().cpu()}

    def load_state_dict(self, state):
        for name in ("actor", "critic", "target", "actor_opt", "critic_opt", "alpha_opt"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
