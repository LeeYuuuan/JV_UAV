"""Joint training with a constant number of SAC updates per replay transition."""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from ..factory import build_smoke_test_runner
from .mappo import MAPPO
from .observations import RunningMeanStd, copy_lower, upper_observation
from .replay import Replay
from .sac import SAC
from .exploration import WaypointExploration


def validate_training(cfg):
    for key in ("upper_rollout_steps", "sac_updates_per_step"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    sac, ppo = cfg["sac"], cfg["mappo"]
    if sac.get("warmup_mode", "uniform") not in {"uniform", "waypoint"}:
        raise ValueError("warmup_mode must be uniform or waypoint")
    if not 0 <= sac.get("warmup_random_fraction", 0.2) <= 1:
        raise ValueError("warmup_random_fraction must be in [0, 1]")
    if not isinstance(sac.get("include_uav_id", False), bool):
        raise ValueError("sac.include_uav_id must be boolean")
    for block, keys in ((sac, ("batch_size", "replay_capacity", "hidden", "heads", "layers")), (ppo, ("hidden", "epochs", "minibatch_frames"))):
        if any(not isinstance(block[key], int) or block[key] <= 0 for key in keys):
            raise ValueError("network and batch sizes must be positive integers")
    if sac["replay_capacity"] < sac["batch_size"] or sac["hidden"] % sac["heads"]:
        raise ValueError("invalid replay capacity or attention head dimensions")
    if sac["random_steps"] < 0 or sac["learning_starts"] < 0:
        raise ValueError("warm-up steps cannot be negative")
    for block in (sac, ppo):
        if not 0 <= block["gamma"] <= 1 or block["reward_scale"] <= 0 or block["max_grad_norm"] <= 0:
            raise ValueError("invalid discount, reward scale or gradient limit")
    if not 0 < sac["tau"] <= 1 or sac["alpha"] <= 0 or sac["lr"] <= 0:
        raise ValueError("invalid SAC optimizer parameters")
    if not 0 <= ppo["gae_lambda"] <= 1 or min(ppo["clip"], ppo["actor_lr"], ppo["critic_lr"]) <= 0 or (ppo.get("value_clip") is not None and ppo["value_clip"] <= 0):
        raise ValueError("invalid PPO parameters")
    if not np.isfinite(ppo["entropy_coef"]) or ppo["entropy_coef"] < 0:
        raise ValueError("entropy_coef must be finite and nonnegative")
    if "entropy_end_coef" in ppo:
        if (not np.isfinite(ppo["entropy_end_coef"]) or ppo["entropy_end_coef"] < 0
                or not 0 <= ppo.get("entropy_decay_start_frame", -1) < ppo.get("entropy_decay_end_frame", -1)):
            raise ValueError("invalid entropy schedule")
    if cfg["normalization"]["clip"] <= 0 or cfg["normalization"]["min_std_sec"] <= 0:
        raise ValueError("normalization scales must be positive")


class JointTrainer:
    def __init__(self, env_cfg, cfg, device="cpu"):
        cfg = copy.deepcopy(cfg)
        # Old checkpoints keep their reward/replay data, but never restore the
        # removed frequency switch. Their former early rate remains constant.
        cfg.setdefault("sac_updates_per_step", cfg.get("early_sac_updates_per_step", 1))
        for key in ("switch_after_sac_updates", "early_sac_updates_per_step", "late_sac_updates_per_mappo"):
            cfg.pop(key, None)
        validate_training(cfg)
        self.env_cfg, self.cfg = copy.deepcopy(env_cfg), copy.deepcopy(cfg)
        self.device = torch.device(device)
        torch.manual_seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.env = build_smoke_test_runner(self.env_cfg).upper_env
        self.env.reset()
        self.exploration = WaypointExploration(
            self.env.scene, self.rng, cfg["sac"].get("warmup_random_fraction", 0.2)
        )
        self.rms = RunningMeanStd(cfg["normalization"]["clip"], cfg["normalization"]["min_std_sec"])
        self.sac = SAC(self.env_cfg, cfg["sac"], self.device)
        self.mappo = MAPPO(self.env.scene.num_uavs, cfg["mappo"], self.device)
        self.replay = Replay(cfg["sac"]["replay_capacity"], cfg["seed"] + 1)
        self.env.low_runner.low_policy = self._low_policy
        self.env.low_runner.on_step = self._low_step
        self.low_steps = self.upper_steps = self.sac_updates = self.mappo_updates = self.episodes = 0
        self.pending = None
        self.rollout = []
        self.last_losses = {}
        self.episode_over = False
        self.episode_return = 0.0

    def _sac_update(self):
        cfg = self.cfg["sac"]
        if self.low_steps < cfg["learning_starts"] or len(self.replay) < cfg["batch_size"]:
            return False
        self.last_losses.update(self.sac.update(self.replay, self.rms))
        self.sac_updates += 1
        return True

    def _transition_updates(self):
        for _ in range(self.cfg["sac_updates_per_step"]):
            self._sac_update()

    def _finish_pending(self, next_obs, done):
        if self.pending is not None:
            obs, action, reward = self.pending
            self.replay.add(obs, action, reward, next_obs, done)
            self.pending = None
            self._transition_updates()

    def _low_policy(self, obs):
        # Count each newly encountered decision observation once. Replaying data
        # or changing the active UAV count must not multiply sensor statistics.
        self.rms.update(obs["sensor_last_visit_sec"])
        # At a frame boundary this runs AFTER the next upper allocation.
        self._finish_pending(obs, False)
        if self.low_steps < self.cfg["sac"]["random_steps"]:
            if self.cfg["sac"].get("warmup_mode", "uniform") == "waypoint":
                return self.exploration(obs)
            return self.rng.uniform(-1, 1, (len(obs["active_uav_ids"]), 2)).astype(np.float32)
        return self.sac.act(obs, self.rms)

    def _low_step(self, obs, action, result):
        self.low_steps += 1
        dead = bool(result.dead_during_step.size or result.return_unsafe_ids.size)
        final = result.step_in_frame == self.env.scene.low_steps_per_frame - 1
        if final and not dead:
            # The reward already includes the final return check. It is not
            # inserted until the next allocation, or closed as terminal below.
            self.pending = (copy_lower(obs), action.copy(), result.reward)
        else:
            self.replay.add(obs, action, result.reward, self.env.scene.observe_lower(), dead)
            self._transition_updates()

    def step(self):
        if self.episode_over:
            self.env.reset(reset_rng=False)
            self.exploration.reset()
            self.episode_return = 0.0
            self.episode_over = False
        observation = upper_observation(self.env.scene, self.env.energy_model)
        action, logp, value = self.mappo.act(observation)
        _, reward, terminated, truncated, info = self.env.step(action)
        self.upper_steps += 1
        self.episode_return += reward
        done = terminated or truncated
        if done:
            # The configured horizon is finite: no bootstrap at truncation.
            self._finish_pending(self.env.scene.observe_lower(), True)
            self.episodes += 1
            self.episode_over = True
        self.rollout.append({"obs": observation, "action": action.copy(), "logp": logp.copy(), "value": value, "reward": reward, "done": done})
        updated = False
        if len(self.rollout) == self.cfg["upper_rollout_steps"]:
            last_value = 0.0 if done else self.mappo.value(upper_observation(self.env.scene, self.env.energy_model))
            self.last_losses.update(self.mappo.update(self.rollout, last_value, upper_steps=self.upper_steps))
            self.rollout.clear()
            self.mappo_updates += 1
            updated = True
        episode_metrics = None
        if done:
            backlog = np.asarray(self.env.trace.backlog_max_post_timeline[1:])
            episode_metrics = {
                "mean_max_backlog": float(backlog.mean()),
                "peak_max_backlog": float(backlog.max()),
                "low_steps": self.env.scene.low_step_total,
                "frames": len(self.env.trace.frames),
            }
        return {
            "upper_steps": self.upper_steps, "low_steps": self.low_steps,
            "sac_updates": self.sac_updates, "mappo_updates": self.mappo_updates,
            "phase": "joint", "episodes": self.episodes,
            "upper_reward": reward, "episode_return": self.episode_return,
            "episode_metrics": episode_metrics,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "mappo_updated": updated, "replay_size": len(self.replay),
            "pending_transition": self.pending is not None,
            "dead_ids": info["dead_ids_end"].tolist(),
            "system_max_backlog": float(self.env.scene.sensor_backlog.max()),
            "frame_low_steps": len(info["low_frame"].steps),
            "termination_reason": self.env.trace.frames[-1].termination_reason,
            "charge_requests": int(info["frame_plan"].requested_charge.sum()),
            "charging_assignments": int(np.sum(info["frame_plan"].assigned_status == 2)),
            "waiting_assignments": int(np.sum(info["frame_plan"].assigned_status == 1)),
            "charge_added_soc": float(np.sum(info["settlement"].get("charge_added_frac", 0.0))),
            "upper_reward_terms": dict(info["reward_terms"]),
            **self.last_losses,
        }

    def evaluate(self, seed, output=None, render=True):
        # Separate scene and frozen normalizer. Deterministic actions consume no
        # training RNG or replay; model and optimizer states are unchanged.
        cfg = copy.deepcopy(self.env_cfg)
        cfg["seeds"]["dynamics_seed"] = int(seed)
        env = build_smoke_test_runner(cfg).upper_env
        env.reset()
        frozen_rms = copy.deepcopy(self.rms)
        frozen_rms.training = False
        env.low_runner.low_policy = lambda obs: self.sac.act(obs, frozen_rms, deterministic=True)
        total_return = 0.0
        policy_trace = []
        charge_added = np.zeros(env.scene.num_uavs, dtype=np.float64)
        while True:
            obs = upper_observation(env.scene, env.energy_model)
            probabilities = self.mappo.request_probabilities(obs)
            action, _, _ = self.mappo.act(obs, deterministic=True)
            _, reward, terminated, truncated, info = env.step(action)
            plan = info["frame_plan"]
            added = info["settlement"].get("charge_added_frac", np.zeros(env.scene.num_uavs))
            charge_added += added
            policy_trace.append({
                "frame": len(policy_trace) + 1,
                "request_probabilities": probabilities.tolist(),
                "actions": action.tolist(),
                "assigned_status": plan.assigned_status.tolist(),
                "battery_start": plan.frame_start_battery.tolist(),
                "battery_end": env.scene.uav_battery.tolist(),
                "return_energy_frac": plan.return_energy_frac.tolist(),
                "charge_added_frac": added.tolist(),
                "low_steps": len(info["low_frame"].steps),
                "reward_terms": dict(info["reward_terms"]),
                "termination_reason": env.trace.frames[-1].termination_reason,
            })
            total_return += reward
            if terminated or truncated:
                break
        summary = {"seed": seed, "upper_return": total_return, "low_steps": env.scene.low_step_total, "frames": len(env.trace.frames), "dead_ids": env.scene.dead_ids().tolist(), "mean_max_backlog": float(np.mean(env.trace.backlog_max_post_timeline[1:])), "terminated": bool(terminated), "truncated": bool(truncated)}
        summary.update(
            configured_frames=env.max_frames,
            configured_low_steps=env.max_frames * env.scene.low_steps_per_frame,
            completed_horizon=bool(truncated),
            termination_reason=env.trace.frames[-1].termination_reason if terminated else "horizon_reached",
            policy_mode="deterministic",
            lower_identity_features=self.sac.identity_fleet_size is not None,
            charge_requests=sum(sum(row["actions"]) for row in policy_trace),
            charging_assignments=sum(row["assigned_status"].count(2) for row in policy_trace),
            waiting_assignments=sum(row["assigned_status"].count(1) for row in policy_trace),
            charge_added_soc_by_uav=charge_added.tolist(),
            charged_uav_ids=np.flatnonzero(charge_added > 0).tolist(),
            sensor_coverage_fraction=float(np.mean(env.trace.sensor_visit_count > 0)),
            trajectory_frames=len(env.trace.frames),
        )
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
            (output / "trace.json").write_text(json.dumps(env.trace.to_serializable(), allow_nan=False), encoding="utf-8")
            (output / "policy_trace.json").write_text(json.dumps(policy_trace, indent=2, allow_nan=False), encoding="utf-8")
            if render:
                env.render(show=False, save_path=output / "dashboard.png", trail_frames=len(env.trace.frames))
                env.render(show=False, view="heatmaps", save_path=output / "heatmaps.png")
        return summary

    def save(self, path):
        scene = self.env.scene
        arrays = ("sensor_pos", "sensor_backlog", "sensor_last_visit_sec", "uav_pos", "uav_battery", "uav_status")
        clocks = ("now_sec", "upper_step", "low_step_total", "step_in_frame")
        state = {
            "version": 1, "env_cfg": self.env_cfg, "train_cfg": self.cfg,
            "sac": self.sac.state_dict(), "mappo": self.mappo.state_dict(),
            "rms": self.rms.state_dict(), "replay": self.replay.state_dict(),
            "pending": self.pending, "rollout": self.rollout, "last_losses": self.last_losses,
            "exploration": self.exploration.state_dict(),
            "counters": {key: getattr(self, key) for key in ("low_steps", "upper_steps", "sac_updates", "mappo_updates", "episodes", "episode_over", "episode_return")},
            "scene": {key: getattr(scene, key).copy() for key in arrays} | {key: getattr(scene, key) for key in clocks},
            "trace": self.env.trace,
            "rng": self.rng.bit_generator.state, "scene_rng": scene.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        # This format contains Python objects. Load only this project's trusted
        # training checkpoints, never arbitrary third-party pickle files.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["version"] != 1:
            raise ValueError("unsupported checkpoint version")
        trainer = cls(state["env_cfg"], state["train_cfg"], device)
        if "switch_after_sac_updates" in state["train_cfg"]:
            warnings.warn("The retired SAC frequency switch is ignored; per-transition updates continue throughout training.", UserWarning)
        if state["env_cfg"]["reward"].get("upper_backlog_scale_packets") != 6000.0 or state["env_cfg"]["reward"].get("upper_backlog_mode", "linear") != "linear":
            warnings.warn("This checkpoint retains its saved reward configuration and replay. Start a fresh run to use the revised linear upper reward.", UserWarning)
        if trainer.sac.identity_fleet_size is None:
            warnings.warn(
                "This checkpoint uses legacy ID-free lower observations. Resume preserves "
                "that architecture; start a fresh run with include_uav_id=true for identity features.",
                UserWarning,
            )
        trainer.sac.load_state_dict(state["sac"])
        trainer.mappo.load_state_dict(state["mappo"])
        trainer.rms.load_state_dict(state["rms"])
        trainer.replay.load_state_dict(state["replay"])
        trainer.pending, trainer.rollout = state["pending"], state["rollout"]
        trainer.last_losses = state["last_losses"]
        if "exploration" in state:
            trainer.exploration.load_state_dict(state["exploration"])
        for key, value in state["counters"].items():
            setattr(trainer, key, value)
        for key, value in state["scene"].items():
            if isinstance(value, np.ndarray):
                getattr(trainer.env.scene, key)[:] = value
            else:
                setattr(trainer.env.scene, key, value)
        trainer.env.metrics.trace = state["trace"]
        trainer.rng.bit_generator.state = state["rng"]
        trainer.env.scene.rng.bit_generator.state = state["scene_rng"]
        torch.set_rng_state(state["torch_rng"])
        if trainer.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return trainer
