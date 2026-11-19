"""Fixed60s global-waypoint collection; no flight/return/energy constraints."""
from __future__ import annotations
import copy
import numpy as np
from jv_uav.scene import Scene


class CoverageEnv:
    def __init__(self, env_cfg, cfg, seed=None):
        self.env_cfg, self.cfg = copy.deepcopy(env_cfg), copy.deepcopy(cfg)
        self.seed = int(cfg["seed"] if seed is None else seed)
        self.env_cfg["seeds"]["dynamics_seed"] = self.seed
        self.scene = Scene(self.env_cfg)
        self.selection_rng = np.random.default_rng(self.seed + 2)
        self.reset(reset_rng=True)

    def _sample_groups(self):
        count = int(self.selection_rng.integers(self.cfg["long_count_min"], self.cfg["long_count_max"] + 1))
        self.long_ids = np.sort(self.selection_rng.choice(self.scene.num_uavs, count, replace=False))
        self.short_ids = np.setdiff1d(np.arange(self.scene.num_uavs), self.long_ids)
        self.horizons = np.full(self.scene.num_uavs, self.cfg["short_steps"], dtype=np.int64)
        self.horizons[self.long_ids] = self.cfg["block_steps"]

    def reset(self, reset_rng=False):
        if reset_rng:
            self.selection_rng = np.random.default_rng(self.seed + 2)
        self.scene.reset(reset_rng=reset_rng)
        self.steps, self.phase, self.done = 0, 0, False
        self.records, self.seen = [], set()
        self._sample_groups()
        return self.observation()

    def observation(self):
        # A valid terminal token set is retained; terminal discount is exactly0.
        ids = np.flatnonzero(self.phase < self.horizons)
        if not len(ids):
            ids = self.long_ids.copy()
        s = self.scene
        return dict(sensor_pos=s.sensor_pos.copy(), backlog=s.sensor_backlog.copy(), ids=ids,
                    positions=s.uav_pos[ids].copy(),
                    service_remaining=np.maximum(self.horizons[ids] - self.phase, 0).astype(np.float32) / self.cfg["block_steps"],
                    remaining=max(0.0, 1.0 - self.steps / self.cfg["episode_steps"]))

    def step(self, action):
        if self.done:
            raise RuntimeError("reset completed coverage episode")
        s, cfg = self.scene, self.cfg
        ids = self.observation()["ids"]
        action = np.asarray(action, np.float32)
        if action.shape != (len(ids), 2) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError("invalid normalized absolute waypoints")
        long_ids, short_ids = self.long_ids.copy(), self.short_ids.copy()
        start = s.uav_pos.copy()
        target = (np.clip(action, -1, 1) + 1) * (s.world_size / 2)
        # Instant relocation is intentional: this experiment tests selection only.
        s.uav_pos[ids] = target
        arrivals = s.rng.poisson(s.arrival_rate * cfg["step_sec"])
        s.sensor_backlog += arrivals
        s.sensor_last_visit_sec += cfg["step_sec"]
        pre = s.sensor_backlog.copy()
        distances = np.linalg.norm(target[:, None, :] - s.sensor_pos[None, :, :], axis=2)
        covered = distances <= self.env_cfg["uavs"]["coverage_radius_m"]
        union = covered.any(axis=0)
        owner = np.full(s.num_sensors, -1, dtype=np.int64)
        if union.any():
            owner[union] = ids[np.where(covered[:, union], distances[:, union], np.inf).argmin(axis=0)]
        collected = np.bincount(owner[union], weights=pre[union], minlength=s.num_uavs)
        total = float(pre.sum())
        fraction = float(pre[union].sum()) / total if total > 0 else 0.0
        s.sensor_backlog[union] = 0
        s.sensor_last_visit_sec[union] = 0
        b_after = float(s.sensor_backlog.max())
        rw = cfg["reward"]
        terms = dict(backlog_post=-b_after / rw["backlog_scale"],
                     all_sensor_clear=rw["rho"] * b_after / rw["backlog_scale"] * fraction)
        self.seen.update(np.flatnonzero(union).tolist())
        self.steps += 1; self.phase += 1
        s.now_sec = self.steps * cfg["step_sec"]
        self.done = self.steps >= cfg["episode_steps"]
        outer = np.linalg.norm(s.sensor_pos - s.airship_pos, axis=1) > self.env_cfg["sensors"]["partition_radius_m"]
        row = dict(step=self.steps, block=(self.steps - 1) // cfg["block_steps"] + 1,
                   step_in_block=self.phase, active_ids=ids.tolist(), long_ids=long_ids.tolist(), short_ids=short_ids.tolist(),
                   action=action.tolist(), waypoints=target.tolist(), positions_start=start.tolist(), positions_end=s.uav_pos.tolist(),
                   covered_ids=np.flatnonzero(union).tolist(), collected_by_uav=collected.tolist(),
                   collected_packets=float(collected.sum()), pre_backlog_total=total,
                   pre_backlog_max=float(pre.max()), cleared_fraction=fraction,
                   reward=float(sum(terms.values())), reward_terms=terms,
                   max_backlog=b_after, mean_backlog=float(s.sensor_backlog.mean()),
                   sensor_coverage_fraction=len(self.seen) / s.num_sensors,
                   outer_visited=int(sum(outer[list(self.seen)])) if self.seen else 0,
                   duration_sec=float(cfg["step_sec"]), end_sec=float(s.now_sec), done=self.done)
        self.records.append(row)
        if self.phase == cfg["block_steps"] and not self.done:
            self.phase = 0
            self._sample_groups()
        return self.observation(), row["reward"], self.done, row

    def summary(self):
        rows = self.records
        duration = self.steps * self.cfg["step_sec"]
        terms = {key: sum(r["reward_terms"][key] for r in rows) for key in rows[0]["reward_terms"]}
        gamma = self.cfg["sac"]["gamma_per_reference"] ** (self.cfg["step_sec"] / self.cfg["reward"]["reference_sec"])
        positive = [r for r in rows if r["collected_packets"] > 0]
        return dict(steps=self.steps, simulation_sec=duration, lower_return=sum(r["reward"] for r in rows),
                    lower_discounted_return=sum(gamma ** i * r["reward"] for i, r in enumerate(rows)),
                    reward_terms=terms, mean_max_backlog=float(np.mean([r["max_backlog"] for r in rows])),
                    peak_max_backlog=max(r["max_backlog"] for r in rows),
                    mean_sensor_backlog=float(np.mean([r["mean_backlog"] for r in rows])),
                    sensor_coverage_fraction=len(self.seen) / self.scene.num_sensors, outer_visited=rows[-1]["outer_visited"],
                    collected_packets=sum(r["collected_packets"] for r in rows),
                    collection_rate_packets_sec=sum(r["collected_packets"] for r in rows) / duration,
                    collection_steps=len(positive),
                    collected_but_zero_bonus_steps=sum(r["reward_terms"]["all_sensor_clear"] == 0 for r in positive),
                    mean_clear_fraction=float(np.mean([r["cleared_fraction"] for r in rows])),
                    uav_decisions=sum(len(r["active_ids"]) for r in rows))

    def state_dict(self):
        return copy.deepcopy(dict(scene={k: getattr(self.scene, k) for k in
            ("sensor_pos", "sensor_backlog", "sensor_last_visit_sec", "uav_pos", "now_sec")},
            rng=self.scene.rng.bit_generator.state, selection_rng=self.selection_rng.bit_generator.state,
            long_ids=self.long_ids, short_ids=self.short_ids, horizons=self.horizons,
            steps=self.steps, phase=self.phase, done=self.done, records=self.records, seen=self.seen))

    def load_state_dict(self, state):
        for key, value in state["scene"].items():
            setattr(self.scene, key, copy.deepcopy(value))
        self.scene.rng.bit_generator.state = state["rng"]
        self.selection_rng.bit_generator.state = state["selection_rng"]
        for key in ("long_ids", "short_ids", "horizons", "steps", "phase", "done", "records", "seen"):
            setattr(self, key, copy.deepcopy(state[key]))
