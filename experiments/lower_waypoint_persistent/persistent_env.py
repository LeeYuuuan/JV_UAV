"""Real-speed60s slots under persistent waypoints; energy intentionally disabled."""
from __future__ import annotations
import copy
import numpy as np
from jv_uav.scene import Scene


class PhysicalSlots:
    def __init__(self, env_cfg, cfg, seed=None):
        self.env_cfg, self.cfg = copy.deepcopy(env_cfg), copy.deepcopy(cfg)
        self.seed = int(cfg["seed"] if seed is None else seed)
        self.env_cfg["seeds"]["dynamics_seed"] = self.seed
        self.scene = Scene(self.env_cfg)
        self.selection_rng = np.random.default_rng(self.seed + 2)
        self.reset(reset_rng=True)

    def _sample_groups(self):
        if self.cfg.get("disable_rotation", False):
            self.long_ids = np.arange(self.scene.num_uavs)
            self.short_ids = np.array([], dtype=np.int64)
            self.horizons = np.full(self.scene.num_uavs, self.cfg["block_steps"], dtype=np.int64)
            return
        count = int(self.selection_rng.integers(self.cfg["long_count_min"], self.cfg["long_count_max"] + 1))
        self.long_ids = np.sort(self.selection_rng.choice(self.scene.num_uavs, count, replace=False))
        self.short_ids = np.setdiff1d(np.arange(self.scene.num_uavs), self.long_ids)
        self.horizons = np.full(self.scene.num_uavs, self.cfg["short_steps"], dtype=np.int64)
        self.horizons[self.long_ids] = self.cfg["block_steps"]

    def reset(self, reset_rng=False):
        if reset_rng:
            self.selection_rng = np.random.default_rng(self.seed + 2)
        self.scene.reset(reset_rng=reset_rng)
        fixed = self.cfg.get("fixed_sensor_positions_m")
        if fixed is not None:
            positions = np.asarray(fixed, dtype=np.float32)
            if positions.shape != (self.scene.num_sensors, 2) or not np.isfinite(positions).all() or np.any(positions < 0) or np.any(positions > self.scene.world_size):
                raise ValueError("fixed sensor positions must match count and map bounds")
            self.scene.sensor_pos[:] = positions
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
        return dict(sensor_pos=s.sensor_pos.copy(), sensor_last_visit_sec=s.sensor_last_visit_sec.copy(), ids=ids,
                    positions=s.uav_pos[ids].copy(),
                    all_positions=s.uav_pos.copy())

    def _physical_step(self, action):
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
        # Each slot uses at most30s of real flight, then serves/hover for the rest.
        delta = target - start[ids]
        desired_distance = np.linalg.norm(delta, axis=1)
        limit = self.scene.speed_mps * min(self.scene.max_move_time_sec, cfg["step_sec"])
        factor = np.minimum(1, limit / np.maximum(desired_distance, 1e-12))
        s.uav_pos[ids] = start[ids] + delta * factor[:, None]
        actual_position = s.uav_pos[ids].copy()
        arrivals = s.rng.poisson(s.arrival_rate * cfg["step_sec"])
        s.sensor_backlog += arrivals
        s.sensor_last_visit_sec += cfg["step_sec"]
        pre = s.sensor_backlog.copy()
        distances = np.linalg.norm(actual_position[:, None, :] - s.sensor_pos[None, :, :], axis=2)
        covered = distances <= self.env_cfg["uavs"]["coverage_radius_m"]
        union = covered.any(axis=0)
        owner = np.full(s.num_sensors, -1, dtype=np.int64)
        if union.any():
            owner[union] = ids[np.where(covered[:, union], distances[:, union], np.inf).argmin(axis=0)]
        collected = np.bincount(owner[union], weights=pre[union], minlength=s.num_uavs)
        total = float(pre.sum())
        fraction = float(pre[union].sum()) / total if total > 0 else 0.0
        reward_fraction = fraction
        top_ids = None
        top_k = cfg["reward"].get("top_k")
        if top_k is not None:
            top_ids = np.argsort(-pre, kind="stable")[:min(top_k, s.num_sensors)]
            denominator = float(pre[top_ids].sum())
            reward_fraction = float(pre[top_ids][union[top_ids]].sum()) / denominator if denominator > 0 else 0.0
        s.sensor_backlog[union] = 0
        s.sensor_last_visit_sec[union] = 0
        b_after = float(s.sensor_backlog.max())
        rw = cfg["reward"]
        move = np.linalg.norm(actual_position - start[ids], axis=1)
        move_by_uav = np.zeros(s.num_uavs, dtype=np.float32)
        move_by_uav[ids] = move
        movement_cost = -float(rw.get("movement_weight", 0.0)) * float(move.mean()) / (s.world_size * np.sqrt(2))
        terms = dict(backlog_post=-float(rw.get("backlog_weight", 1.0)) * b_after / rw["backlog_scale"],
                     all_sensor_clear=rw["rho"] * b_after / rw["backlog_scale"] * reward_fraction,
                     movement=movement_cost)
        self.seen.update(np.flatnonzero(union).tolist())
        self.steps += 1; self.phase += 1
        s.now_sec = self.steps * cfg["step_sec"]
        self.done = not cfg.get("continuing_task", False) and self.steps >= cfg["episode_steps"]
        outer = np.linalg.norm(s.sensor_pos - s.airship_pos, axis=1) > self.env_cfg["sensors"]["partition_radius_m"]
        row = dict(step=self.steps, block=(self.steps - 1) // cfg["block_steps"] + 1,
                   step_in_block=self.phase, active_ids=ids.tolist(), long_ids=long_ids.tolist(), short_ids=short_ids.tolist(),
                   action=action.tolist(), waypoints=target.tolist(), executed_positions=actual_position.tolist(), positions_start=start.tolist(), positions_end=s.uav_pos.tolist(),
                   move_time_sec=(move / s.speed_mps).tolist(), service_time_sec=(cfg["step_sec"] - move / s.speed_mps).tolist(),
                   move_distance_by_uav_m=move_by_uav.tolist(), mean_move_m=float(move.mean()), total_move_m=float(move.sum()),
                   covered_ids=np.flatnonzero(union).tolist(), collected_by_uav=collected.tolist(),
                   collected_packets=float(collected.sum()), pre_backlog_total=total,
                   pre_backlog_max=float(pre.max()), cleared_fraction=fraction,
                   reward=float(sum(terms.values())), reward_terms=terms,
                   max_backlog=b_after, mean_backlog=float(s.sensor_backlog.mean()),
                   sensor_coverage_fraction=len(self.seen) / s.num_sensors,
                   outer_visited=int(sum(outer[list(self.seen)])) if self.seen else 0,
                   duration_sec=float(cfg["step_sec"]), end_sec=float(s.now_sec), done=self.done)
        if top_ids is not None:
            row.update(reward_clear_fraction=reward_fraction, reward_topk_sensor_ids=top_ids.tolist())
        self.records.append(row)
        if self.phase == cfg["block_steps"] and not self.done:
            self.phase = 0
            self._sample_groups()
        return self.observation(), row["reward"], self.done, row

    def summary(self):
        rows = self.records
        duration = self.steps * self.cfg["step_sec"]
        keys = sorted({key for r in rows for key in r["reward_terms"]})
        terms = {key: sum(r["reward_terms"].get(key, 0.0) for r in rows) for key in keys}
        moves = [np.linalg.norm(np.asarray(r["positions_end"])[r["active_ids"]] - np.asarray(r["positions_start"])[r["active_ids"]], axis=1) for r in rows]
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
                    mean_step_move_m=float(np.mean([x.mean() for x in moves])),
                    mean_uav_move_m=float(np.concatenate(moves).mean()),
                    total_move_m=float(sum(x.sum() for x in moves)),
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


class PersistentEnv(PhysicalSlots):
    """One replay transition per synchronous committed-waypoint interval.

    Early arrivals keep serving their endpoint. Replan when all arrive, active
    groups change, or the finite physical episode ends. No intermediate actor
    resampling or teleporting is allowed.
    """
    def reset(self, reset_rng=False):
        observation = super().reset(reset_rng=reset_rng)
        self.decision_records = []
        return observation

    def step(self, action):
        if self.done:
            raise RuntimeError("reset completed persistent-waypoint episode")
        ids = self.observation()["ids"].copy()
        action = np.asarray(action, np.float32)
        if action.shape != (len(ids), 2) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError("invalid normalized committed waypoints")
        targets = (np.clip(action, -1, 1) + 1) * (self.scene.world_size / 2)
        start = self.steps
        gamma = self.cfg["sac"]["gamma_per_reference"] ** (self.cfg["step_sec"] / self.cfg["reward"]["reference_sec"])
        reward, weight, terms, physical = 0.0, 1.0, {}, []
        while True:
            _, value, done, row = self._physical_step(action)
            physical.append(row)
            reward += weight * value
            for key, term in row["reward_terms"].items():
                terms[key] = terms.get(key, 0.0) + weight * term
            weight *= gamma
            reached = np.linalg.norm(self.scene.uav_pos[ids] - targets, axis=1) <= self.cfg["arrival_tolerance_m"]
            changed = not np.array_equal(ids, self.observation()["ids"]) or self.phase == 0
            limit = self.cfg.get("max_decision_physical_steps")
            decision_limit = limit is not None and len(physical) >= limit
            if done or changed or reached.all() or decision_limit:
                reason = "episode_end" if done else "group_change" if changed else "all_arrived" if reached.all() else "decision_limit"
                break
            if len(physical) > self.cfg["block_steps"]:
                raise RuntimeError("persistent interval missed a mandatory group boundary")
        result = dict(decision=len(self.decision_records) + 1, start_physical_step=start,
                      end_physical_step=self.steps, physical_steps=len(physical),
                      duration_sec=len(physical) * self.cfg["step_sec"],
                      active_ids=ids.tolist(), action=action.tolist(), target_positions=targets.tolist(),
                      reached_ids=ids[reached].tolist(), interrupted_ids=ids[~reached].tolist(),
                      stop_reason=reason, reward=float(reward), raw_reward_sum=sum(r["reward"] for r in physical),
                      reward_terms=terms, collected_packets=sum(r["collected_packets"] for r in physical),
                      done=self.done)
        self.decision_records.append(result)
        return self.observation(), float(reward), self.done, result

    def summary(self):
        return super().summary() | dict(waypoint_decisions=len(self.decision_records),
                   mean_commitment_steps=float(np.mean([r["physical_steps"] for r in self.decision_records])),
                   interrupted_targets=sum(len(r["interrupted_ids"]) for r in self.decision_records))

    def state_dict(self):
        return super().state_dict() | dict(decision_records=copy.deepcopy(self.decision_records))

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.decision_records = copy.deepcopy(state["decision_records"])
