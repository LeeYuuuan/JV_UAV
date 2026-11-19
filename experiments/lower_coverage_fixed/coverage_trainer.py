"""Fixed-time no-energy coverage training; only SAC."""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import torch
from coverage_env import CoverageEnv
from coverage_sac import CoverageSAC, Replay


class Trainer:
    def __init__(self, env_cfg, cfg, device="cpu"):
        self.env_cfg, self.cfg = copy.deepcopy(env_cfg), copy.deepcopy(cfg)
        self.device = torch.device(device)
        torch.manual_seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.env = CoverageEnv(env_cfg, cfg)
        self.sac = CoverageSAC(cfg["sac"], self.env, self.device)
        self.replay = Replay(cfg["sac"]["replay_capacity"], cfg["seed"] + 1)
        self.steps = self.updates = self.episodes = 0
        self.history, self.last_losses, self.loss_history = [], {}, []

    def step(self):
        if self.env.done:
            self.env.reset()
        obs = self.env.observation()
        warmup = self.steps < self.cfg["sac"]["random_steps"]
        action = self.rng.uniform(-1, 1, (len(obs["ids"]), 2)).astype(np.float32) if warmup else self.sac.act(obs)
        future, reward, done, row = self.env.step(action)
        self.replay.add(obs, action, reward, future, done, row["duration_sec"])
        self.steps += 1
        sc = self.cfg["sac"]
        if self.steps >= sc["learning_starts"] and len(self.replay.items) >= sc["batch_size"]:
            self.last_losses = self.sac.update(self.replay.sample(sc["batch_size"]))
            self.updates += 1
            if self.steps % 50 == 0:
                self.loss_history.append(dict(training_steps=self.steps, **self.last_losses))
        summary = None
        if done:
            self.episodes += 1
            summary = self.env.summary() | dict(episode=self.episodes, training_steps=self.steps, sac_updates=self.updates)
            self.history.append(summary)
        return row | dict(training_steps=self.steps, sac_updates=self.updates, episode=self.episodes if done else self.episodes + 1,
                          warmup=warmup, losses=self.last_losses, episode_summary=summary)

    def evaluate(self, seed, output=None, mode="policy", detailed=True):
        env = CoverageEnv(self.env_cfg, self.cfg, seed=seed)
        rng = np.random.default_rng(seed + 10000)
        probes = []
        cover = np.linalg.norm(env.scene.sensor_pos[:, None] - env.scene.sensor_pos[None, :], axis=-1) <= self.env_cfg["uavs"]["coverage_radius_m"]
        while not env.done:
            obs = env.observation()
            if mode == "random":
                action = rng.uniform(-1, 1, (len(obs["ids"]), 2)).astype(np.float32)
            elif mode == "center":
                action = np.broadcast_to(env.scene.airship_pos / env.scene.world_size * 2 - 1, (len(obs["ids"]), 2)).copy()
            elif mode == "greedy_reference":
                # Evaluation-only reference, never used for warmup or SAC data.
                backlog = obs["backlog"].astype(float)
                weights = (backlog / max(float(backlog.max()), 1.0)) ** 4
                points = []
                for _ in obs["ids"]:
                    best = int((cover @ weights).argmax())
                    points.append(env.scene.sensor_pos[best])
                    weights[cover[best]] = 0
                action = np.asarray(points) / env.scene.world_size * 2 - 1
            elif mode == "policy":
                action = self.sac.act(obs, deterministic=True)
                if output is not None and detailed:
                    probes.append(dict(frame=env.steps + 1, ids=obs["ids"].tolist(), **self.sac.diagnostics(obs)))
            else:
                raise ValueError("unknown evaluation mode")
            env.step(action)
        summary = env.summary() | dict(seed=seed, training_steps=self.steps, sac_updates=self.updates, policy_mode=mode)
        if output is not None:
            from coverage_plots import dashboard, trajectories, write_json
            output = Path(output); output.mkdir(parents=True, exist_ok=True)
            write_json(output / "summary.json", summary)
            trajectories(env, output / "trajectory.png")
            if detailed:
                write_json(output / "trace.json", dict(sensor_pos=env.scene.sensor_pos.tolist(),
                           sensor_backlog=env.scene.sensor_backlog.tolist(), rows=env.records, action_distribution=probes))
                dashboard(env, output / "dashboard.png")
        return summary

    def save(self, path):
        state = dict(format="coverage_fixed_v1", env_cfg=self.env_cfg, cfg=self.cfg,
                     sac=self.sac.state_dict(), replay=self.replay.state_dict(), env=self.env.state_dict(),
                     steps=self.steps, updates=self.updates, episodes=self.episodes, history=self.history,
                     last_losses=self.last_losses, loss_history=self.loss_history, rng=self.rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        torch.save(state, temp); temp.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "coverage_fixed_v1":
            raise ValueError("not a fixed-time coverage checkpoint")
        obj = cls(state["env_cfg"], state["cfg"], device)
        obj.sac.load_state_dict(state["sac"]); obj.replay.load_state_dict(state["replay"])
        obj.env.load_state_dict(state["env"])
        for key in ("steps", "updates", "episodes", "history", "last_losses", "loss_history"):
            setattr(obj, key, state[key])
        obj.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        if obj.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return obj
