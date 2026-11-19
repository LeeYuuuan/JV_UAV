"""Event-driven physics and slot allocation. No learner or optimizer dependencies."""
from __future__ import annotations

import copy
from enum import IntEnum
import numpy as np
from jv_uav.config import validate_config
from jv_uav.sensor_generator import generate_sensor_positions
from jv_uav.models import SimpleEnergyModel


class Role(IntEnum):
    SERVE = 0
    WAIT = 1
    CHARGE = 2
    DEAD = 3


class Phase(IntEnum):
    READY = 0
    MOVE = 1
    SERVICE = 2
    RETURN_CLIMB = 3
    RETURN_FLY = 4
    CHARGE = 5
    WAIT = 6
    DEAD = 7


RETURN_PHASES = (Phase.RETURN_CLIMB, Phase.RETURN_FLY)
POWERED_PHASES = (Phase.MOVE, Phase.SERVICE, Phase.RETURN_CLIMB, Phase.RETURN_FLY)
EPS = 1e-7


def allocate(action, role, phase, arrival_soc, charging_slots, waiting_slots):
    """Preserve existing dock/wait priorities; inbound reservations are locked."""
    n = len(role)
    action = np.asarray(action)
    if action.shape != (n,) or not np.isin(action, [0, 1]).all():
        raise ValueError("upper action must contain one binary request per UAV")
    locked = np.isin(phase, RETURN_PHASES)
    if (np.sum((phase == Phase.CHARGE) | (locked & (role == Role.CHARGE))) > charging_slots or
            np.sum((phase == Phase.WAIT) | (locked & (role == Role.WAIT))) > waiting_slots):
        raise RuntimeError("existing occupancy/reservations exceed capacity")
    alive = role != Role.DEAD
    requested = alive & (action == 1)
    assigned = np.where(alive, Role.SERVE, Role.DEAD).astype(np.int64)
    assigned[locked] = role[locked]
    def order(ids):
        return ids[np.lexsort((ids, arrival_soc[ids]))]
    retained_c = np.flatnonzero((locked & (role == Role.CHARGE)) |
                               (requested & (phase == Phase.CHARGE)))
    locked_w = np.flatnonzero(locked & (role == Role.WAIT))
    if len(retained_c) > charging_slots or len(locked_w) > waiting_slots:
        raise RuntimeError("inbound/retained reservations exceed capacity")
    remaining = requested & ~locked
    remaining[retained_c] = False
    waiters = order(np.flatnonzero(remaining & (phase == Phase.WAIT)))
    others = order(np.flatnonzero(remaining & (phase != Phase.WAIT)))
    new_c = np.concatenate((waiters, others))[:charging_slots - len(retained_c)]
    remaining[new_c] = False
    retained_w = order(np.flatnonzero(remaining & (phase == Phase.WAIT)))[:waiting_slots - len(locked_w)]
    remaining[retained_w] = False
    new_w = order(np.flatnonzero(remaining))[:waiting_slots - len(locked_w) - len(retained_w)]
    assigned[np.concatenate((retained_c, new_c))] = Role.CHARGE
    assigned[np.concatenate((locked_w, retained_w, new_w))] = Role.WAIT
    return assigned, alive & ~locked


class Engine:
    def __init__(self, env_cfg, cfg, seed=None):
        self.env_cfg, self.cfg = copy.deepcopy(env_cfg), copy.deepcopy(cfg)
        self.seed = int(cfg["seed"] if seed is None else seed)
        self.env_cfg["seeds"]["dynamics_seed"] = self.seed
        validate_config(self.env_cfg)
        self.xy = generate_sensor_positions(self.env_cfg).astype(np.float64)
        self.home = np.asarray(self.env_cfg["world"]["airship_xy_m"], np.float64)
        self.size, self.n = float(self.env_cfg["world"]["size_m"]), int(self.env_cfg["uavs"]["count"])
        self.rate = np.full(self.env_cfg["sensors"]["count"], self.env_cfg["sensors"]["arrival_rate_per_sec"], np.float64)
        self.energy = SimpleEnergyModel(env_cfg)
        self.rng = np.random.default_rng(self.seed)
        self.order_rng = np.random.default_rng(self.seed + 2)
        self.reset(reset_rng=True)

    def reset(self, reset_rng=False):
        if reset_rng:
            self.rng = np.random.default_rng(self.seed)
            self.order_rng = np.random.default_rng(self.seed + 2)
        self.backlog = self.rng.poisson(self.rate * self.env_cfg["sensors"]["cold_start_sec"]).astype(np.float64)
        self.initial_packets = float(self.backlog.sum())
        self.arrived_packets = 0.0
        self.pos = np.broadcast_to(self.home, (self.n, 2)).copy()
        self.target = self.pos.copy()
        self.soc = np.full(self.n, self.env_cfg["uavs"]["initial_battery_frac"], np.float64)
        self.role = np.full(self.n, Role.SERVE, np.int64)
        self.phase = np.full(self.n, Phase.READY, np.int64)
        self.remaining = np.zeros(self.n, np.float64)
        self.now = self.next_upper = 0.0
        self.done = False
        self.queue = []
        self.references = {}
        self.seen = set()
        self.failures, self.collections, self.allocations, self.paths, self.segments = [], [], [], [], []
        self.upper_acc = self.lower_acc = 0.0
        self.terms = dict(backlog=0.0, topk=0.0, failure=0.0, distance=0.0, tail=0.0)
        self.area = self.serving_area = self.collected = 0.0
        self.upper_total = self.lower_total = 0.0
        self.max_backlog = float(self.backlog.max())
        self.event_count = 0

    @property
    def upper_due(self):
        return not self.done and self.now >= self.next_upper - EPS

    @property
    def current_uid(self):
        return self.queue[0] if self.queue and not self.upper_due and not self.done else None

    def return_cost(self):
        cost = self.energy.return_trip(self.pos, self.home).energy_frac.astype(float)
        e = self.energy
        climb = self.phase == Phase.RETURN_CLIMB
        fly = self.phase == Phase.RETURN_FLY
        cost[climb] = (e.climb_power_w * self.remaining[climb] + e.horizontal_power_w *
                       np.linalg.norm(self.pos[climb] - self.home, axis=1) / e.horizontal_speed_mps) / (3600 * e.capacity_wh)
        cost[fly] = e.horizontal_power_w * self.remaining[fly] / (3600 * e.capacity_wh)
        cost[np.isin(self.phase, [Phase.CHARGE, Phase.WAIT, Phase.DEAD])] = 0
        return cost

    def _lower_add(self, key, value):
        value = float(value)
        self.lower_acc += value; self.lower_total += value; self.terms[key] += value

    def _upper_add(self, value):
        self.upper_acc += float(value); self.upper_total += float(value)

    def take_lower_reward(self):
        value, self.lower_acc = self.lower_acc, 0.0
        return value

    def take_upper_reward(self):
        value, self.upper_acc = self.upper_acc, 0.0
        return value

    def _fill_queue(self):
        if self.queue:
            raise RuntimeError("cannot reorder unresolved same-time decisions")
        ready = np.flatnonzero((self.role == Role.SERVE) & (self.phase == Phase.READY))
        self.queue = self.order_rng.permutation(ready).tolist() if not self.done else []

    def assert_invariants(self):
        if not all(np.isfinite(x).all() for x in (self.pos, self.target, self.soc, self.backlog, self.remaining)):
            raise RuntimeError("nonfinite physical state")
        if np.any(self.soc < -EPS) or np.any(self.soc > 1 + EPS) or np.any(self.backlog < -EPS):
            raise RuntimeError("energy or packet bounds violated")
        if np.any(self.pos < -EPS) or np.any(self.pos > self.size + EPS):
            raise RuntimeError("physical position outside map")
        if np.sum(self.role == Role.CHARGE) > self.env_cfg["zones"]["charging_slots"] or np.sum(self.role == Role.WAIT) > self.env_cfg["zones"]["waiting_slots"]:
            raise RuntimeError("slot reservations overbooked")
        if not np.isclose(self.initial_packets + self.arrived_packets - self.collected, self.backlog.sum(), atol=1e-4, rtol=1e-10):
            raise RuntimeError("packet conservation violated")
        if len(self.queue) != len(set(self.queue)) or any(self.phase[i] != Phase.READY or self.role[i] != Role.SERVE for i in self.queue):
            raise RuntimeError("invalid same-time decision queue")
        if not np.array_equal(self.phase == Phase.DEAD, self.role == Role.DEAD):
            raise RuntimeError("death role/phase mismatch")
        for phase, role in ((Phase.CHARGE, Role.CHARGE), (Phase.WAIT, Role.WAIT), (Phase.READY, Role.SERVE), (Phase.MOVE, Role.SERVE), (Phase.SERVICE, Role.SERVE)):
            if np.any(self.role[self.phase == phase] != role):
                raise RuntimeError("role/phase mismatch")
        if np.any(~np.isin(self.role[np.isin(self.phase, RETURN_PHASES)], [Role.CHARGE, Role.WAIT])):
            raise RuntimeError("return has no reserved destination")

    def _reference(self, uid):
        settings, e = self.cfg["responsibility"], self.energy
        duration = min(self.cfg["upper_period_sec"], self.cfg["episode_sec"] - self.now)
        vec = self.home - self.pos[uid]
        distance = float(np.linalg.norm(vec))
        direction = vec / distance if distance else np.zeros(2)
        advance = settings["inward_fraction"] * min(distance, e.horizontal_speed_mps * duration)
        end = self.pos[uid] + direction * advance
        optimistic = self.pos[uid] + direction * min(distance, e.horizontal_speed_mps * duration)
        move = advance / e.horizontal_speed_mps
        reserve = self.env_cfg["uavs"]["return_reserve_frac"]
        service = (move * e.horizontal_power_w + (duration - move) * e.hover_power_w) / (3600 * e.capacity_wh)
        required = service + float(e.return_trip(end[None], self.home).energy_frac[0]) + reserve + settings["margin_soc"]
        lower_bound = min(e.horizontal_power_w, e.hover_power_w) * duration / (3600 * e.capacity_wh)
        lower_bound += float(e.return_trip(optimistic[None], self.home).energy_frac[0]) + reserve
        return dict(start_sec=self.now, start_soc=float(self.soc[uid]), start_xy=self.pos[uid].tolist(),
                    duration_sec=duration, reference_required_soc=required, optimistic_required_soc=lower_bound,
                    reference_end=end.tolist())

    def _die(self, uid, cause):
        if self.phase[uid] == Phase.DEAD:
            raise RuntimeError("duplicate failure event")
        ref = copy.deepcopy(self.references.get(int(uid), {}))
        if cause in ("service_depletion", "return_depletion"):
            upper_weight, category = 1.0, cause
        elif not ref or ref["start_soc"] + EPS < ref["optimistic_required_soc"]:
            upper_weight, category = 1.0, "upper_infeasible"
        elif ref["start_soc"] + EPS >= ref["reference_required_soc"]:
            upper_weight, category = 0.0, "lower_trajectory"
        else:
            upper_weight, category = self.cfg["responsibility"]["shared_upper_weight"], "shared_margin"
        lower_weight = 1 - upper_weight
        distance = float(np.linalg.norm(self.pos[uid] - self.home))
        record = dict(time=self.now, uid=int(uid), cause=cause, category=category,
                      upper_weight=upper_weight, lower_weight=lower_weight, reference=ref,
                      position=self.pos[uid].tolist(), soc=float(self.soc[uid]), return_distance_m=distance)
        self.failures.append(record)
        rw = self.cfg["lower_reward"]
        self._lower_add("failure", -rw["death_weight"] * lower_weight)
        self._lower_add("distance", -rw["distance_weight"] * lower_weight * distance / rw["distance_scale_m"])
        ur = self.cfg["upper_reward"]
        self._upper_add(-ur["death_weight"] * upper_weight / ur["divisor"])
        self.phase[uid], self.role[uid], self.remaining[uid] = Phase.DEAD, Role.DEAD, 0

    def _finish_if_needed(self):
        if self.failures or self.now >= self.cfg["episode_sec"] - EPS:
            if not self.done and self.failures and self.cfg["lower_reward"]["terminal_tail"]:
                # Absorbing no-service tail, charged once and responsibility weighted.
                rest = max(self.cfg["episode_sec"] - self.now, 0.0)
                weight = float(np.mean([f["lower_weight"] for f in self.failures]))
                rw = self.cfg["lower_reward"]
                tail = (float(self.backlog.max()) * rest + .5 * float(self.rate.max()) * rest**2)
                self._lower_add("tail", -weight * tail / (rw["backlog_scale"] * rw["reference_sec"]))
            self.done = True; self.queue = []

    def apply_upper(self, action):
        if not self.upper_due or self.queue:
            raise RuntimeError("upper allocation out of event order")
        prior_role, prior_phase = self.role.copy(), self.phase.copy()
        trip = self.energy.return_trip(self.pos, self.home)
        cost = self.return_cost()
        assigned, mask = allocate(action, self.role, self.phase, self.soc - cost,
                                  self.env_cfg["zones"]["charging_slots"], self.env_cfg["zones"]["waiting_slots"])
        for uid in range(self.n):
            if prior_phase[uid] in RETURN_PHASES or prior_phase[uid] == Phase.DEAD:
                continue
            self.role[uid] = assigned[uid]
            if assigned[uid] == Role.SERVE:
                if prior_phase[uid] in (Phase.CHARGE, Phase.WAIT):
                    self.phase[uid], self.remaining[uid] = Phase.READY, 0
                    self.target[uid] = self.pos[uid]
                self.references[uid] = self._reference(uid)
            elif prior_phase[uid] in (Phase.CHARGE, Phase.WAIT):
                self.phase[uid] = Phase.CHARGE if assigned[uid] == Role.CHARGE else Phase.WAIT
                self.remaining[uid] = 0
            else:
                if self.soc[uid] + EPS < cost[uid] + self.env_cfg["uavs"]["return_reserve_frac"]:
                    self._die(uid, "return_infeasible")
                else:
                    self.target[uid] = self.home
                    self.phase[uid] = Phase.RETURN_CLIMB
                    self.remaining[uid] = self.energy.height_m / self.energy.climb_speed_mps
                    if self.remaining[uid] <= EPS:
                        self.phase[uid] = Phase.RETURN_FLY
                        self.remaining[uid] = float(trip.distance_m[uid]) / self.energy.horizontal_speed_mps
        row = dict(time=self.now, requests=np.asarray(action).astype(int).tolist(), assigned=assigned.tolist(),
                   phase_before=prior_phase.tolist(), role_before=prior_role.tolist(), policy_mask=mask.tolist(),
                   reserved_charging=int(np.sum(self.role == Role.CHARGE)), reserved_waiting=int(np.sum(self.role == Role.WAIT)),
                   return_cost=cost.tolist(), soc=self.soc.tolist(), positions=self.pos.tolist(),
                   phase_after=self.phase.tolist(), actual_charging=int(np.sum(self.phase == Phase.CHARGE)),
                   actual_waiting=int(np.sum(self.phase == Phase.WAIT)),
                   returning=int(np.sum(np.isin(self.phase, RETURN_PHASES))))
        self.allocations.append(row)
        self.next_upper += self.cfg["upper_period_sec"]
        self._process_zero_returns()
        self._finish_if_needed()
        if not self.done:
            self._fill_queue()
        self.assert_invariants()
        return row

    def _process_zero_returns(self):
        for uid in np.flatnonzero(np.isin(self.phase, RETURN_PHASES) & (self.remaining <= EPS)):
            if self.phase[uid] == Phase.RETURN_CLIMB:
                self.phase[uid] = Phase.RETURN_FLY
                self.remaining[uid] = np.linalg.norm(self.pos[uid] - self.home) / self.energy.horizontal_speed_mps
            if self.phase[uid] == Phase.RETURN_FLY and self.remaining[uid] <= EPS:
                self.pos[uid] = self.home
                self.phase[uid] = Phase.CHARGE if self.role[uid] == Role.CHARGE else Phase.WAIT
                self.remaining[uid] = 0

    def apply_lower(self, action):
        uid = self.current_uid
        if uid is None:
            raise RuntimeError("no lower decision available")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1 + EPS):
            raise ValueError("lower action must be finite normalized xy")
        self.queue.pop(0)
        self.target[uid] = (np.clip(action, -1, 1) + 1) * self.size / 2
        duration = np.linalg.norm(self.target[uid] - self.pos[uid]) / self.energy.horizontal_speed_mps
        self.phase[uid] = Phase.MOVE if duration > EPS else Phase.SERVICE
        self.remaining[uid] = duration if duration > EPS else self.cfg["service_sec"]
        row = dict(time=self.now, uid=uid, origin=self.pos[uid].tolist(), target=self.target[uid].tolist(),
                   soc=float(self.soc[uid]), flight_sec=float(duration), action=action.tolist())
        self.paths.append(row)
        self.assert_invariants()
        return row

    def _queues_advance(self, dt):
        bins = max(1, int(np.ceil(dt / self.cfg["integration_sec"])))
        step = dt / bins
        increments = self.rng.poisson(np.broadcast_to(self.rate * step, (bins, len(self.rate))))
        self.arrived_packets += float(increments.sum())
        states = self.backlog[None] + np.cumsum(increments, axis=0)
        maxima = np.concatenate(([float(self.backlog.max())], states.max(axis=1)))
        area = float(np.sum(maxima[:-1] + maxima[1:]) * step / 2)
        self.backlog[:] = states[-1]
        self.max_backlog = max(self.max_backlog, float(maxima.max()))
        self.area += area
        rw = self.cfg["lower_reward"]
        self._lower_add("backlog", -area / (rw["backlog_scale"] * rw["reference_sec"]))

    def _collect(self, ids):
        if not len(ids):
            return
        distances = np.linalg.norm(self.xy[:, None] - self.pos[ids][None], axis=2)
        nearest = distances.argmin(axis=1)
        union = distances[np.arange(len(self.xy)), nearest] <= self.env_cfg["uavs"]["coverage_radius_m"]
        pre = self.backlog.copy()
        rw = self.cfg["lower_reward"]
        top = np.argsort(-pre, kind="stable")[:min(rw["top_k"], len(pre))]
        denominator = float(pre[top].sum())
        fraction = float(pre[top][union[top]].sum()) / denominator if denominator else 0.0
        by_uid = np.zeros(self.n)
        np.add.at(by_uid, ids[nearest[union]], pre[union])
        self.backlog[union] = 0
        bonus = rw["rho"] * float(self.backlog.max()) / rw["backlog_scale"] * fraction
        self._lower_add("topk", bonus)
        self.collected += float(by_uid.sum())
        self.seen.update(np.flatnonzero(union).tolist())
        self.collections.append(dict(time=self.now, uids=ids.tolist(), positions=self.pos[ids].tolist(), sensor_ids=np.flatnonzero(union).tolist(),
                                     by_uid=by_uid.tolist(), h_k=fraction, topk_bonus=bonus,
                                     post_max=float(self.backlog.max())))

    def advance(self):
        if self.done or self.upper_due or self.queue:
            raise RuntimeError("resolve decisions before advancing physical time")
        e = self.energy
        power = np.zeros(self.n)
        power[np.isin(self.phase, [Phase.MOVE, Phase.RETURN_FLY])] = e.horizontal_power_w
        power[self.phase == Phase.SERVICE] = e.hover_power_w
        power[self.phase == Phase.RETURN_CLIMB] = e.climb_power_w
        durations = [self.next_upper - self.now, self.cfg["episode_sec"] - self.now]
        timed = np.isin(self.phase, POWERED_PHASES)
        if timed.any():
            durations.extend(self.remaining[timed].tolist())
            durations.extend((self.soc[timed] * 3600 * e.capacity_wh / power[timed]).tolist())
        dt = float(min(durations))
        if dt < -EPS:
            raise RuntimeError("event clock went backwards")
        dt = max(dt, 0.0)
        segment = dict(start=self.now, positions_start=self.pos.tolist(), soc_start=self.soc.tolist(), phase_start=self.phase.tolist())
        self._queues_advance(dt)
        ur = self.cfg["upper_reward"]
        serving = int(np.sum(self.role == Role.SERVE))
        self.serving_area += serving * dt
        self._upper_add(ur["serving_weight"] * serving * dt / ur["reference_sec"] / ur["divisor"])
        moving = np.isin(self.phase, [Phase.MOVE, Phase.RETURN_FLY])
        vec = self.target[moving] - self.pos[moving]
        length = np.linalg.norm(vec, axis=1)
        fraction = np.minimum(1, dt * e.horizontal_speed_mps / np.maximum(length, 1e-12))
        self.pos[moving] += vec * fraction[:, None]
        self.soc[:] = np.maximum(self.soc - power * dt / (3600 * e.capacity_wh), 0)
        charging = self.phase == Phase.CHARGE
        self.soc[charging] = np.minimum(1, self.soc[charging] + e.charging_power_w * dt / (3600 * e.capacity_wh))
        self.remaining[timed] = np.maximum(0, self.remaining[timed] - dt)
        self.now += dt; self.event_count += 1
        # Landing exactly at zero SOC is allowed by the existing return budget
        # convention. Service/mid-flight zero SOC still fails before collection.
        self._process_zero_returns()
        for uid in np.flatnonzero(np.isin(self.phase, POWERED_PHASES) & (self.soc <= EPS)):
            self._die(uid, "return_depletion" if self.phase[uid] in RETURN_PHASES else "service_depletion")
        completed = np.flatnonzero((self.phase == Phase.SERVICE) & (self.remaining <= EPS))
        self._collect(completed)
        self.phase[completed], self.remaining[completed] = Phase.READY, 0
        arrivals = np.flatnonzero((self.phase == Phase.MOVE) & (self.remaining <= EPS))
        self.pos[arrivals] = self.target[arrivals]
        self.phase[arrivals], self.remaining[arrivals] = Phase.SERVICE, self.cfg["service_sec"]
        self._process_zero_returns()
        self._finish_if_needed()
        if not self.done and not self.upper_due:
            self._fill_queue()
        self.segments.append(segment | dict(end=self.now, positions_end=self.pos.tolist(), soc_end=self.soc.tolist(), phase_end=self.phase.tolist(),
                                            max_backlog=float(self.backlog.max()), cumulative_covered=len(self.seen)))
        if dt <= EPS and not self.done and not self.upper_due and not self.queue:
            raise RuntimeError("event failed to make progress")
        self.assert_invariants()
        return dict(time=self.now, dt=dt, completed_service=completed.tolist(), arrivals=arrivals.tolist(), done=self.done)

    def observation(self):
        e = self.energy
        work = self.pos.copy()
        traveling = np.isin(self.phase, [Phase.MOVE, Phase.RETURN_CLIMB, Phase.RETURN_FLY])
        work[traveling] = self.target[traveling]
        rank = np.zeros(self.n)
        for index, uid in enumerate(self.queue):
            rank[uid] = (len(self.queue) - index) / self.n
        eta = self.remaining.copy()
        climb = self.phase == Phase.RETURN_CLIMB
        eta[climb] += np.linalg.norm(self.pos[climb] - self.home, axis=1) / e.horizontal_speed_mps
        reference = np.zeros((self.n, 4))
        for uid, ref in self.references.items():
            if self.role[uid] != Role.SERVE:
                continue
            reference[uid] = [ref["start_soc"], ref["reference_required_soc"], ref["optimistic_required_soc"], (self.now - ref["start_sec"]) / 600]
        team = np.column_stack((self.pos / self.size, work / self.size, self.soc, self.return_cost(),
                                np.arange(self.n) / max(self.n - 1, 1), eta / self.cfg["upper"]["reference_sec"],
                                rank, np.eye(4)[self.role], np.eye(8)[self.phase], reference)).astype(np.float32)
        global_ = np.array([max(0, 1 - self.now / self.cfg["episode_sec"]),
                            max(0, self.next_upper - self.now) / self.cfg["upper_period_sec"],
                            np.sum(self.role == Role.CHARGE) / max(self.env_cfg["zones"]["charging_slots"], 1),
                            np.sum(self.role == Role.WAIT) / max(self.env_cfg["zones"]["waiting_slots"], 1)], np.float32)
        return dict(team=team, global_=global_, backlog=self.backlog.astype(np.float32).copy(),
                    uid=int(self.current_uid if self.current_uid is not None else 0))

    def summary(self):
        outer = np.linalg.norm(self.xy - self.home, axis=1) > self.env_cfg["sensors"]["partition_radius_m"]
        return dict(simulation_sec=self.now, survived=not bool(self.failures),
                    mean_serving=self.serving_area / max(self.now, EPS),
                    time_mean_max_backlog=self.area / max(self.now, EPS), peak_max_backlog=self.max_backlog,
                    sensor_coverage_fraction=len(self.seen) / len(self.xy),
                    outer_visited=int(sum(outer[list(self.seen)])) if self.seen else 0,
                    collected_packets=self.collected, lower_return=self.lower_total, upper_return=self.upper_total,
                    arrived_packets=self.arrived_packets, initial_packets=self.initial_packets,
                    final_packets=float(self.backlog.sum()),
                    collection_rate_packets_sec=self.collected / max(self.now, EPS),
                    upper_reward_terms=dict(serving=self.cfg["upper_reward"]["serving_weight"] * self.serving_area /
                                            self.cfg["upper_reward"]["reference_sec"] / self.cfg["upper_reward"]["divisor"],
                                            death=-sum(f["upper_weight"] for f in self.failures) * self.cfg["upper_reward"]["death_weight"] /
                                            self.cfg["upper_reward"]["divisor"]),
                    lower_reward_terms=copy.deepcopy(self.terms), failures=copy.deepcopy(self.failures),
                    lower_decisions=len(self.paths), upper_decisions=len(self.allocations),
                    charge_requests=sum(sum(r["requests"]) for r in self.allocations),
                    charge_assignments=sum(sum(x == Role.CHARGE for x in r["assigned"]) for r in self.allocations))

    def state_dict(self):
        excluded = {"energy", "rng", "order_rng"}
        return dict(values=copy.deepcopy({k: v for k, v in vars(self).items() if k not in excluded}),
                    rng=self.rng.bit_generator.state, order_rng=self.order_rng.bit_generator.state)

    def load_state_dict(self, state):
        self.__dict__.update(copy.deepcopy(state["values"]))
        self.rng.bit_generator.state = state["rng"]
        self.order_rng.bit_generator.state = state["order_rng"]
