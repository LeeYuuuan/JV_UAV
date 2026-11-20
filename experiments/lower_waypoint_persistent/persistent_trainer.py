"""SAC on real-speed persistent intervals; mixed exploration, actor-only eval."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from persistent_env import PersistentEnv
from persistent_sac import PersistentSAC, Replay
from waypoint_sampler import WaypointSampler


class Trainer:
    def __init__(self, env_cfg, cfg, device="cpu"):
        self.env_cfg, self.cfg = copy.deepcopy(env_cfg), copy.deepcopy(cfg)
        self.device = torch.device(device)
        torch.manual_seed(cfg["seed"])
        self.env = PersistentEnv(env_cfg, cfg)
        self.sac = PersistentSAC(cfg["sac"], self.env, self.device)
        self.replay = Replay(cfg["sac"]["replay_capacity"], cfg["seed"] + 1)
        self.sampler = WaypointSampler(cfg["exploration"], env_cfg, cfg["seed"] + 3)
        self.steps = self.decisions = self.updates = self.episodes = 0
        self.history, self.loss_history, self.last_losses = [], [], {}

    def step(self):
        if self.env.done:
            self.env.reset()
        obs = self.env.observation()
        actor_action = self.sac.act(obs)
        action, proposal = self.sampler.propose(obs, actor_action, self.steps)
        future, reward, done, row = self.env.step(action)
        row.update(proposal)
        before = self.steps
        self.steps += row["physical_steps"]; self.decisions += 1
        self.replay.add(obs, action, reward, future, done, row["duration_sec"])
        sc = self.cfg["sac"]
        if self.steps >= sc["learning_starts"] and len(self.replay.items) >= sc["batch_size"]:
            for _ in range(self.cfg["updates_per_physical_step"] * row["physical_steps"]):
                self.last_losses = self.sac.update(self.replay.sample(sc["batch_size"]))
                self.updates += 1
            if self.steps // 50 != before // 50:
                self.loss_history.append(dict(training_steps=self.steps, **self.last_losses))
        summary = None
        if done:
            self.episodes += 1
            summary = self.env.summary() | dict(episode=self.episodes, training_steps=self.steps,
                                               sac_updates=self.updates, total_waypoint_decisions=self.decisions)
            counts = {key: sum(r["sources"].count(key) for r in self.env.decision_records) for key in ["actor", "random", "greedy"]}
            summary["behavior_counts"] = counts
            self.history.append(summary)
        return row | dict(training_steps=self.steps, waypoint_decisions=self.decisions, sac_updates=self.updates,
                          episode=self.episodes if done else self.episodes + 1, losses=self.last_losses,
                          episode_summary=summary)

    def evaluate(self, seed, output=None, mode="policy", detailed=True):
        env = PersistentEnv(self.env_cfg, self.cfg, seed=seed)
        rng = np.random.default_rng(seed + 10000)
        probes = []
        # Evaluation reference uses the same Top6-nearest+jitter rule but never
        # changes the training sampler/RNG. Policy eval does not call this sampler.
        reference_cfg = copy.deepcopy(self.cfg["exploration"])
        reference_cfg.update(epsilon_start=1.0, epsilon_end=1.0, random_probability=0.0,
                             greedy_probability=1.0, actor_probability=0.0, greedy_jitter_m=0.0)
        reference = WaypointSampler(reference_cfg, self.env_cfg, seed + 20000)
        while not env.done:
            obs = env.observation()
            if mode == "policy":
                action = self.sac.act(obs, deterministic=True)
                if output is not None and detailed:
                    probes.append(dict(physical_step=env.steps, ids=obs["ids"].tolist(), **self.sac.diagnostics(obs)))
            elif mode == "random":
                action = rng.uniform(-1, 1, (len(obs["ids"]), 2)).astype(np.float32)
            elif mode == "greedy_reference":
                action, _ = reference.propose(obs, np.zeros((len(obs["ids"]), 2), np.float32), 0)
            elif mode == "center":
                action = np.broadcast_to(env.scene.airship_pos / env.scene.world_size * 2 - 1, (len(obs["ids"]), 2)).copy()
            else:
                raise ValueError("unknown evaluation policy")
            env.step(action)
        summary = env.summary() | dict(seed=seed, training_steps=self.steps, sac_updates=self.updates,
                                      total_waypoint_decisions=self.decisions, policy_mode=mode)
        if output is not None:
            from persistent_plots import write_json, trajectories, fleet_overlay, dashboard
            output = Path(output); output.mkdir(parents=True, exist_ok=True)
            write_json(output / "summary.json", summary)
            trajectories(env, output / "trajectory.png"); fleet_overlay(env, output / "fleet_overlay.png")
            if detailed:
                write_json(output / "trace.json", dict(sensor_pos=env.scene.sensor_pos.tolist(),
                           sensor_backlog=env.scene.sensor_backlog.tolist(), rows=env.records,
                           decisions=env.decision_records, action_distribution=probes))
                dashboard(env, output / "dashboard.png")
        return summary

    def save(self, path):
        state = dict(format="persistent_waypoint_lastvisit_v2", env_cfg=self.env_cfg, cfg=self.cfg,
                     sac=self.sac.state_dict(), replay=self.replay.state_dict(), env=self.env.state_dict(),
                     steps=self.steps, decisions=self.decisions, updates=self.updates, episodes=self.episodes,
                     history=self.history, loss_history=self.loss_history, last_losses=self.last_losses,
                     sampler=self.sampler.state_dict(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp"); torch.save(state, temporary); temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "persistent_waypoint_lastvisit_v2":
            raise ValueError("not an observable last-visit v2 checkpoint; old backlog-input weights/replay are incompatible")
        obj = cls(state["env_cfg"], state["cfg"], device)
        obj.sac.load_state_dict(state["sac"]); obj.replay.load_state_dict(state["replay"])
        obj.env.load_state_dict(state["env"]); obj.sampler.load_state_dict(state["sampler"])
        for key in ("steps", "decisions", "updates", "episodes", "history", "loss_history", "last_losses"):
            setattr(obj, key, state[key])
        torch.set_rng_state(state["torch_rng"])
        if obj.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return obj
