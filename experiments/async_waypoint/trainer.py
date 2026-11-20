"""One global lower decision stream; upper rollout spans episodes until full."""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import torch
from engine import Engine, Phase, RETURN_PHASES
from learning import UpperPPO, LowerSAC, Replay, upper_observation


class Trainer:
    def __init__(self, env_cfg, cfg, device="cpu"):
        self.env_cfg, self.cfg = copy.deepcopy(env_cfg), copy.deepcopy(cfg)
        # Checkpoints without this field keep their original per-transition cadence.
        self.cfg.setdefault("lower_update_mode", "per_transition")
        if self.cfg["lower_update_mode"] not in ("per_transition", "matched_upper"):
            raise ValueError("unknown lower update mode")
        self.device = torch.device(device)
        torch.manual_seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.env = Engine(env_cfg, cfg)
        self.env_cfg = copy.deepcopy(self.env.env_cfg)
        self.upper = UpperPPO(cfg["upper"], self.env, device)
        self.lower = LowerSAC(cfg["lower"], self.env, device)
        self.replay = Replay(cfg["lower"]["replay_capacity"], cfg["seed"] + 1)
        self.upper_steps = self.lower_steps = self.upper_updates = self.lower_updates = self.episodes = 0
        self.pending_upper = self.pending_lower = None
        self.rollout, self.history, self.loss_history = [], [], []
        self.last_losses = {}
        self.episode_over = False
        self.unattributed_lower_reward = 0.0

    def _close_lower(self, terminal):
        reward = self.env.take_lower_reward()
        if self.pending_lower is None:
            self.unattributed_lower_reward += reward
            return
        p = self.pending_lower
        self.replay.add(p["obs"], p["action"], reward, self.env.observation(), terminal, self.env.now - p["time"])
        self.pending_lower = None
        if self.cfg["lower_update_mode"] == "per_transition":
            self._update_lower()

    def _update_lower(self):
        cfg = self.cfg["lower"]
        starts = max(cfg["learning_starts"], cfg["random_steps"]) if self.cfg["lower_update_mode"] == "matched_upper" else cfg["learning_starts"]
        if self.lower_steps >= starts and len(self.replay.rows) >= cfg["batch_size"]:
            self.last_losses.update(self.lower.update(self.replay.sample(cfg["batch_size"])))
            self.lower_updates += 1
            return True
        return False

    def _close_upper(self, terminal):
        if self.pending_upper is None:
            raise RuntimeError("upper interval has no originating decision")
        p = self.pending_upper
        row = p | dict(reward=self.env.take_upper_reward(), done=bool(terminal), duration=self.env.now - p["time"])
        self.rollout.append(row)
        self.pending_upper = None
        self.upper_steps += 1
        updated = False
        lower_updated = False
        if len(self.rollout) == self.cfg["upper_rollout_steps"]:
            bootstrap = 0.0 if terminal else self.upper.value(upper_observation(self.env))
            self.last_losses.update(self.upper.update(self.rollout, bootstrap))
            self.rollout = []; self.upper_updates += 1; updated = True
            if self.cfg["lower_update_mode"] == "matched_upper":
                lower_updated = self._update_lower()
        elif len(self.rollout) > self.cfg["upper_rollout_steps"]:
            raise RuntimeError("upper rollout exceeded configured length")
        self.loss_history.append(dict(upper_steps=self.upper_steps, lower_steps=self.lower_steps,
                                      upper_updates=self.upper_updates, lower_updates=self.lower_updates, **self.last_losses))
        summary = None
        if terminal:
            self.episodes += 1
            summary = self.env.summary() | dict(episode=self.episodes, upper_steps=self.upper_steps,
                                               lower_steps=self.lower_steps, upper_updates=self.upper_updates,
                                               lower_updates=self.lower_updates)
            self.history.append(summary); self.episode_over = True
        return dict(upper_steps=self.upper_steps, lower_steps=self.lower_steps, upper_updates=self.upper_updates,
                    lower_updates=self.lower_updates, episode=self.episodes if terminal else self.episodes + 1,
                    interval_duration=row["duration"], upper_reward=row["reward"], terminal=terminal,
                    upper_updated=updated, rollout_T=len(self.rollout), simulation_sec=self.env.now,
                    lower_paired_update=lower_updated, lower_update_mode=self.cfg["lower_update_mode"],
                    mean_serving_so_far=self.env.serving_area / max(self.env.now, 1e-7),
                    time_mean_max_backlog=self.env.area / max(self.env.now, 1e-7),
                    request_probabilities=p["probabilities"].tolist(), sampled_actions=p["action"].tolist(),
                    policy_mask=p["mask"].tolist(), latest_allocation=self.env.allocations[-1],
                    losses=copy.deepcopy(self.last_losses), episode_summary=summary)

    def step(self):
        """Return after exactly one completed upper interval (possibly shortened)."""
        if self.episode_over:
            self.env.reset(); self.episode_over = False
        while True:
            if self.env.done:
                self._close_lower(True)
                return self._close_upper(True)
            if self.env.upper_due:
                if self.pending_upper is not None:
                    return self._close_upper(False)
                obs = upper_observation(self.env)
                action, logp, value, probabilities = self.upper.act(obs)
                mask = ~np.isin(self.env.phase, [*RETURN_PHASES, Phase.DEAD])
                self.pending_upper = dict(obs=obs, action=action.copy(), logp=logp.copy(), value=value,
                                          mask=mask.copy(), time=self.env.now, probabilities=probabilities.copy())
                self.env.apply_upper(action)
            elif self.env.current_uid is not None:
                self._close_lower(False)
                obs = self.env.observation()
                warmup = self.lower_steps < self.cfg["lower"]["random_steps"]
                action = self.rng.uniform(-1, 1, 2).astype(np.float32) if warmup else self.lower.act(obs)
                self.pending_lower = dict(obs=obs, action=action.copy(), time=self.env.now)
                planned = self.env.apply_lower(action)
                if not warmup and self.lower_steps % 1000 == 0:
                    planned.update(self.lower.probe(obs))
                self.lower_steps += 1
            else:
                self.env.advance()

    def evaluate(self, seed, output=None):
        env = Engine(self.env_cfg, self.cfg, seed=seed)
        upper_probes, lower_probes = [], []
        while not env.done:
            if env.upper_due:
                obs = upper_observation(env)
                action, _, _, probabilities = self.upper.act(obs, deterministic=True)
                row = env.apply_upper(action)
                upper_probes.append(row | dict(probabilities=probabilities.tolist()))
            elif env.current_uid is not None:
                obs = env.observation()
                action = self.lower.act(obs, deterministic=True)
                row = env.apply_lower(action)
                if output is not None:
                    lower_probes.append(row | self.lower.probe(obs))
            else:
                env.advance()
        summary = env.summary() | dict(seed=seed, upper_steps=self.upper_steps, lower_steps=self.lower_steps)
        if output is not None:
            from plots import dashboard, write_json
            output = Path(output); output.mkdir(parents=True, exist_ok=True)
            write_json(output / "summary.json", summary)
            write_json(output / "trace.json", dict(sensor_pos=env.xy.tolist(), backlog=env.backlog.tolist(),
                       upper=upper_probes, lower=lower_probes, collections=env.collections, failures=env.failures, segments=env.segments))
            dashboard(env, output / "dashboard.png")
        return summary

    def save(self, path):
        state = dict(format="async_waypoint_600s_v1", env_cfg=self.env_cfg, cfg=self.cfg,
                     env=self.env.state_dict(), upper=self.upper.state_dict(), lower=self.lower.state_dict(),
                     replay=self.replay.state_dict(), rng=self.rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
                     counters={k: getattr(self, k) for k in ("upper_steps", "lower_steps", "upper_updates", "lower_updates", "episodes")},
                     states={k: copy.deepcopy(getattr(self, k)) for k in ("pending_upper", "pending_lower", "rollout", "history", "loss_history", "last_losses", "episode_over", "unattributed_lower_reward")})
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp"); torch.save(state, temp); temp.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "async_waypoint_600s_v1":
            raise ValueError("checkpoint is not from this asynchronous experiment")
        obj = cls(state["env_cfg"], state["cfg"], device)
        obj.env.load_state_dict(state["env"]); obj.upper.load_state_dict(state["upper"])
        obj.lower.load_state_dict(state["lower"]); obj.replay.load_state_dict(state["replay"])
        for block in (state["counters"], state["states"]):
            for key, value in block.items():
                setattr(obj, key, value)
        obj.rng.bit_generator.state = state["rng"]; torch.set_rng_state(state["torch_rng"])
        if obj.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return obj
