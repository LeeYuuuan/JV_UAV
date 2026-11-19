"""Synchronous full-map waypoints, real sortie energy, persistent sensor queues."""
from __future__ import annotations

import copy
import numpy as np
from jv_uav.scene import Scene
from jv_uav.models import SimpleEnergyModel


class WaypointEnv:
    def __init__(self, env_cfg, cfg, seed=None):
        self.env_cfg = copy.deepcopy(env_cfg)
        self.cfg = copy.deepcopy(cfg)
        self.seed = int(cfg["seed"] if seed is None else seed)
        self.env_cfg["seeds"]["dynamics_seed"] = self.seed
        self.scene = Scene(self.env_cfg)
        self.energy = SimpleEnergyModel(self.env_cfg)
        self.selection_rng = np.random.default_rng(self.seed + 2)
        self.frames = 0
        self.done = False
        self.records = []
        self.seen = set()
        self.reset(reset_rng=True)

    def reset(self, reset_rng=False):
        if reset_rng:
            self.selection_rng = np.random.default_rng(self.seed + 2)
        self.scene.reset(reset_rng=reset_rng)
        self.frames, self.done, self.records, self.seen = 0, False, [], set()
        self._prepare_sortie()
        return self.observation()

    def _prepare_sortie(self):
        count = int(self.selection_rng.integers(self.cfg["active_count_min"], self.cfg["active_count_max"] + 1))
        self.ids = np.sort(self.selection_rng.choice(self.scene.num_uavs, count, replace=False))
        self.scene.uav_pos[:] = self.scene.airship_pos
        self.scene.uav_battery.fill(1.0)

    def observation(self):
        s = self.scene
        return dict(sensor_pos=s.sensor_pos.copy(), backlog=s.sensor_backlog.copy(),
                    ids=self.ids.copy(), positions=s.uav_pos[self.ids].copy(),
                    soc=s.uav_battery[self.ids].copy(),
                    remaining=max(0.0, 1.0 - s.now_sec / self.cfg["episode_sec"]))

    def _advance(self, duration):
        """Poisson counts; trapezoidal backlog integral on <=integration_sec bins.

        No collection occurs inside this interval. Integration convergence is
        separately tested with deterministic arrivals and analytic bounds.
        """
        s = self.scene
        if duration <= 0:
            return 0.0, 0.0
        count = max(1, int(np.ceil(duration / self.cfg["integration_sec"])))
        dt = duration / count
        arrivals = s.rng.poisson(s.arrival_rate[None, :] * np.full((count, 1), dt))
        states = s.sensor_backlog[None, :] + np.cumsum(arrivals, axis=0, dtype=np.float64)
        maxima = np.concatenate(([float(s.sensor_backlog.max())], states.max(axis=1)))
        integral = float(np.sum((maxima[:-1] + maxima[1:]) * (dt / 2)))
        peak = float(maxima.max())
        s.sensor_backlog[:] = states[-1]
        s.sensor_last_visit_sec += duration
        s.now_sec += duration
        return integral, peak

    def step(self, action):
        if self.done:
            raise RuntimeError("reset finished waypoint episode")
        s, e, cfg = self.scene, self.energy, self.cfg
        ids = self.ids.copy()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(ids), 2) or not np.isfinite(action).all():
            raise ValueError("waypoint action shape/values invalid")
        if np.any(np.abs(action) > 1.000001):
            raise ValueError("waypoints must be in normalized [-1, 1]")
        target = (np.clip(action, -1, 1) + 1) * (s.world_size / 2)
        start = s.uav_pos[ids].copy()
        distance = np.linalg.norm(target - start, axis=1)
        move_sec = distance / e.horizontal_speed_mps
        planned_service = float(move_sec.max() + cfg["service_sec"])
        time_left = max(0.0, cfg["episode_sec"] - s.now_sec)
        service_duration = min(planned_service, time_left)
        service_completed = service_duration >= planned_service - 1e-6
        executed_move_sec = np.minimum(move_sec, service_duration)
        hover_sec = service_duration - executed_move_sec
        start_soc = s.uav_battery[ids].copy()
        service_cost = e.serving_energy_fraction(executed_move_sec, hover_sec)
        service_ok = start_soc > service_cost
        remaining = np.maximum(start_soc - service_cost, 0)
        # A depleted aircraft cannot teleport to an unreachable target.
        flight_capacity = start_soc * e.capacity_wh * 3600 / e.horizontal_power_w
        fraction = np.minimum(1, np.minimum(flight_capacity, executed_move_sec) / np.maximum(move_sec, 1e-12))
        actual_pos = start + (target - start) * fraction[:, None]
        s.uav_pos[ids] = actual_pos
        s.uav_battery[ids] = remaining
        before_clock = float(s.now_sec)
        area_service, peak_service = self._advance(service_duration)
        pre = s.sensor_backlog.copy()
        distances = np.linalg.norm(s.sensor_pos[None, :, :] - actual_pos[:, None, :], axis=2)
        covered = (distances <= self.env_cfg["uavs"]["coverage_radius_m"]) & service_ok[:, None] & service_completed
        union = covered.any(axis=0)
        # Lowest distance owns overlap for diagnostics; union determines reward.
        ownership_dist = np.where(covered, distances, np.inf)
        owner = np.full(s.num_sensors, -1, dtype=np.int64)
        if union.any():
            owner[union] = ids[ownership_dist[:, union].argmin(axis=0)]
        collected = np.bincount(owner[union], weights=pre[union], minlength=s.num_uavs)
        k = min(int(cfg["reward"]["top_k"]), s.num_sensors)
        # Stable ties give reproducible sensor-ID ordering.
        top = np.argsort(-pre, kind="stable")[:k]
        denominator = float(pre[top].sum())
        h_k = float(pre[top][union[top]].sum()) / denominator if denominator > 0 else 0.0
        s.sensor_backlog[union] = 0
        s.sensor_last_visit_sec[union] = 0
        post_service_max = float(s.sensor_backlog.max())
        trip = e.return_trip(actual_pos, s.airship_pos)
        return_ok = service_ok & (remaining >= trip.energy_frac + s.return_reserve_frac)
        failed = ~return_ok
        planned_return = float(trip.time_sec[return_ok].max()) if return_ok.any() else 0.0
        return_duration = min(planned_return, max(0.0, cfg["episode_sec"] - s.now_sec)) if service_completed else 0.0
        executed_return_sec = np.where(return_ok, np.minimum(trip.time_sec, return_duration), 0.0)
        climb_sec = e.height_m / e.climb_speed_mps
        climb_executed = np.minimum(executed_return_sec, climb_sec)
        horizontal_executed = np.maximum(executed_return_sec - climb_sec, 0)
        paid_return = (e.climb_power_w * climb_executed + e.horizontal_power_w * horizontal_executed) / (3600 * e.capacity_wh)
        paid_return = paid_return.astype(np.float32)
        s.uav_battery[ids] = np.maximum(remaining - paid_return, 0)
        return_fraction = np.minimum(1, horizontal_executed / np.maximum(trip.distance_m / e.horizontal_speed_mps, 1e-12))
        s.uav_pos[ids] = actual_pos + (s.airship_pos - actual_pos) * return_fraction[:, None]
        # Successful UAVs land on arrival; no post-landing hover is charged.
        area_return, peak_return = self._advance(return_duration)
        total_duration = service_duration + return_duration
        rw = cfg["reward"]
        terms = dict(backlog_integral=-(area_service + area_return) / (rw["backlog_scale"] * rw["reference_sec"]),
                     topk_clear=rw["rho"] * post_service_max / rw["backlog_scale"] * h_k,
                     failure=-rw["failure_weight"] * int(failed.sum()),
                     failure_distance=-rw["failure_distance_weight"] * float(trip.distance_m[failed].sum()) / rw["distance_scale_m"])
        reward = float(sum(terms.values()))
        self.seen.update(np.flatnonzero(union).tolist())
        events = []
        for j, uid in enumerate(ids):
            events.append(dict(uid=int(uid), target=target[j].tolist(), position=actual_pos[j].tolist(),
                               start_soc=float(start_soc[j]), move_sec=float(executed_move_sec[j]),
                               planned_move_sec=float(move_sec[j]), hover_sec=float(hover_sec[j]),
                               service_soc=float(service_cost[j]), service_paid_soc=float(min(service_cost[j], start_soc[j])),
                               soc_before_return=float(remaining[j]),
                               return_required_soc=float(trip.energy_frac[j]), return_paid_soc=float(paid_return[j]),
                               soc_end=float(s.uav_battery[uid]), return_distance_m=float(trip.distance_m[j]),
                               return_sec=float(trip.time_sec[j]), return_executed_sec=float(executed_return_sec[j]),
                               return_completed=bool(return_ok[j] and executed_return_sec[j] >= trip.time_sec[j] - 1e-4),
                               success=bool(return_ok[j]),
                               service_survived=bool(service_ok[j])))
        self.frames += 1
        self.done = s.now_sec >= cfg["episode_sec"] - 1e-6
        if self.done:
            s.now_sec = float(cfg["episode_sec"])
        outer = np.linalg.norm(s.sensor_pos - s.airship_pos, axis=1) > self.env_cfg["sensors"]["partition_radius_m"]
        row = dict(frame=self.frames, active_ids=ids.tolist(), action=action.tolist(), waypoints=target.tolist(),
                   reward=reward, reward_terms=terms, h_k=h_k, topk_ids=top.tolist(), covered_ids=np.flatnonzero(union).tolist(),
                   collected_by_uav=collected.tolist(), collected_packets=float(collected.sum()),
                   service_sec=service_duration, service_completed=service_completed,
                   return_sec=return_duration, duration_sec=total_duration,
                   start_sec=before_clock, end_sec=float(s.now_sec), backlog_integral_packets_sec=area_service + area_return,
                   post_service_max=post_service_max, max_backlog=float(s.sensor_backlog.max()),
                   peak_max_backlog=max(peak_service, peak_return), return_events=events,
                   sensor_coverage_fraction=len(self.seen) / s.num_sensors,
                   outer_visited=int(sum(outer[list(self.seen)])) if self.seen else 0,
                   failures=int(failed.sum()), done=self.done)
        self.records.append(row)
        # No UAV identity or failed status leaks into the fresh sortie budget.
        if not self.done:
            self._prepare_sortie()
        return self.observation(), reward, self.done, row

    def summary(self):
        rows = self.records
        duration = sum(r["duration_sec"] for r in rows)
        terms = {key: sum(r["reward_terms"][key] for r in rows) for key in rows[0]["reward_terms"]}
        discount, discounted_return = 1.0, 0.0
        for row in rows:
            discounted_return += discount * row["reward"]
            discount *= self.cfg["sac"]["gamma_per_reference"] ** (row["duration_sec"] / self.cfg["reward"]["reference_sec"])
        return dict(steps=self.frames, lower_return=sum(r["reward"] for r in rows),
                    lower_discounted_return=discounted_return,
                    reward_terms=terms, simulation_sec=duration,
                    time_mean_max_backlog=sum(r["backlog_integral_packets_sec"] for r in rows) / duration,
                    mean_max_backlog=float(np.mean([r["max_backlog"] for r in rows])),
                    peak_max_backlog=max(r["peak_max_backlog"] for r in rows),
                    sensor_coverage_fraction=len(self.seen) / self.scene.num_sensors,
                    outer_visited=rows[-1]["outer_visited"], failures=sum(r["failures"] for r in rows),
                    sorties=sum(len(r["active_ids"]) for r in rows),
                    collected_packets=sum(r["collected_packets"] for r in rows),
                    collection_rate_packets_sec=sum(r["collected_packets"] for r in rows) / duration,
                    mean_duration_sec=duration / len(rows))

    def state_dict(self):
        return copy.deepcopy(dict(scene={k: getattr(self.scene, k) for k in
            ("sensor_pos", "sensor_backlog", "sensor_last_visit_sec", "uav_pos", "uav_battery", "now_sec")},
            rng=self.scene.rng.bit_generator.state, selection_rng=self.selection_rng.bit_generator.state,
            ids=self.ids, frames=self.frames, done=self.done, records=self.records, seen=self.seen))

    def load_state_dict(self, state):
        for key, value in state["scene"].items():
            setattr(self.scene, key, copy.deepcopy(value))
        self.scene.rng.bit_generator.state = state["rng"]
        self.selection_rng.bit_generator.state = state["selection_rng"]
        for key in ("ids", "frames", "done", "records", "seen"):
            setattr(self, key, copy.deepcopy(state[key]))
