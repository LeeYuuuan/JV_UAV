"""Independent displacement experiment; never invokes the legacy Scene runner."""
from __future__ import annotations

import copy
from enum import IntEnum
import numpy as np
from jv_uav.models import SimpleEnergyModel, FullClearNearestService
from jv_uav.sensor_generator import generate_sensor_positions


class Status(IntEnum):
    SERVING = 0
    RETURNING = 1
    CHARGING = 2
    WAITING = 3
    IDLE = 4
    STOPPED = 5


class Environment:
    def __init__(self, world, cfg, seed):
        self.world, self.cfg = copy.deepcopy(world), copy.deepcopy(cfg)
        self.rng = np.random.default_rng(seed)
        self.energy = SimpleEnergyModel(world)
        self.n = world['uavs']['count']
        self.sensor_xy = generate_sensor_positions(world)
        self.m = len(self.sensor_xy)
        self.size = float(world['world']['size_m'])
        self.home = np.array(world['world']['airship_xy_m'], dtype=np.float32)
        self.dt = float(world['time']['low_step_sec'])
        self.speed = float(world['uavs']['speed_mps'])
        self.max_distance = self.speed * world['uavs']['max_move_time_sec']
        self.radius = float(world['uavs']['coverage_radius_m'])
        self.collector = FullClearNearestService(self.n, self.radius)
        self.capacity = 3600 * self.energy.capacity_wh
        self.reserve = float(world['uavs']['return_reserve_frac'])
        self.reset()

    def reset(self):
        self.step_count = 0
        self.xy = np.tile(self.home, (self.n, 1))
        self.soc = np.ones(self.n, dtype=np.float32)
        self.status = np.full(self.n, int(Status.IDLE), dtype=np.int8)
        self.climb_left = np.zeros(self.n)
        self.failure_charged = np.zeros(self.n, dtype=bool)
        cold = float(self.world['sensors']['cold_start_sec'])
        self.age = np.full(self.m, cold, dtype=np.float32)
        self.backlog = self.rng.poisson(self.world['sensors']['arrival_rate_per_sec'] * cold,
                                       size=self.m).astype(np.float32)
        self.waiting = []
        self.rows = []
        self.dispatch()
        return self.observation()

    def ids(self):
        return np.flatnonzero((self.status == Status.SERVING) & (self.soc > 0))

    def targets(self):
        # Stable sensor index resolves equal-age ties. No queue values enter observations.
        return np.lexsort((np.arange(self.m), -self.age))[:min(self.cfg['target_count'], self.m)]

    def observation(self):
        target = self.targets()
        return dict(ids=self.ids(), xy=self.xy.copy(), soc=self.soc.copy(), status=self.status.copy(),
                    age=self.age.copy(), targets=self.sensor_xy[target].copy(), target_age=self.age[target].copy(),
                    phase=np.float32((self.step_count % self.cfg['check_every']) / self.cfg['check_every']))

    def dispatch(self):
        desired = int(self.rng.integers(self.cfg['service_count_min'], self.cfg['service_count_max'] + 1))
        eligible = np.flatnonzero((self.status != Status.RETURNING) & (self.status != Status.STOPPED)
                                 & (self.soc > 0))
        chosen = self.rng.choice(eligible, min(desired, len(eligible)), replace=False)
        old = self.ids()
        leaving = np.setdiff1d(old, chosen)
        self.status[leaving] = Status.RETURNING
        self.climb_left[leaving] = self.energy.height_m / self.energy.climb_speed_mps
        newly_serving = chosen[self.status[chosen] != Status.SERVING]
        self.failure_charged[newly_serving] = False
        self.status[chosen] = Status.SERVING
        self.waiting = [i for i in self.waiting if i not in chosen]
        self.desired_count = desired
        self.selected_ids = chosen.tolist()

    def return_requirement(self, xy):
        return self.energy.return_trip(np.asarray(xy).reshape(-1, 2), self.home).energy_frac + self.reserve

    def failure_cost(self, ids):
        if not len(ids):
            return 0.0
        rw = self.cfg['reward']
        distance = np.linalg.norm(self.xy[ids] - self.home, axis=1)
        return float(np.sum(rw['return_failure_per_uav'] + rw['return_distance_weight'] *
                            distance / rw['return_distance_scale_m']))

    def _reference_safe(self, pos, battery):
        delta = self.home - pos
        distance = float(np.linalg.norm(delta))
        movement = min(distance, self.max_distance)
        endpoint = pos + delta * (movement / max(distance, 1e-8))
        mt = movement / self.speed
        used = float(self.energy.serving_energy_fraction(np.array([mt]), np.array([self.dt-mt]))[0])
        return battery - used >= float(self.return_requirement(endpoint)[0])

    def _return(self, ids):
        arrivals, failures = [], []
        for i in ids:
            elapsed = 0.0
            # Flight order matches SimpleEnergyModel: vertical climb, then horizontal.
            for climb in (True, False):
                needed = self.climb_left[i] if climb else float(np.linalg.norm(self.xy[i]-self.home))/self.energy.horizontal_speed_mps
                power = self.energy.climb_power_w if climb else self.energy.horizontal_power_w
                allowed = min(needed, self.dt-elapsed, max(0., float(self.soc[i]))*self.capacity/power)
                self.soc[i] = max(0., self.soc[i] - allowed*power/self.capacity)
                elapsed += allowed
                if climb:
                    self.climb_left[i] = max(0., self.climb_left[i]-allowed)
                else:
                    direction = self.home-self.xy[i]
                    length = float(np.linalg.norm(direction))
                    self.xy[i] += direction * min(1., allowed*self.energy.horizontal_speed_mps/max(length, 1e-8))
                if allowed + 1e-7 < needed:
                    break
            arrived = self.climb_left[i] < 1e-5 and np.linalg.norm(self.xy[i]-self.home) < 1e-3
            if arrived:
                self.xy[i] = self.home
                self.status[i] = Status.WAITING
                arrivals.append((elapsed, int(i)))
            elif self.soc[i] <= 1e-7:
                self.status[i] = Status.STOPPED
                if not self.failure_charged[i]:
                    failures.append(int(i))
                    self.failure_charged[i] = True
        return arrivals, failures

    def _charge(self, arrivals):
        # Integrate arrival and charge-completion events inside this physical step.
        slots = self.cfg['charging_slots']
        slots = self.n if slots is None else slots
        rate = self.energy.charging_power_w / self.capacity
        events = sorted(arrivals, key=lambda pair: (pair[0], pair[1]))
        t = 0.0
        def promote():
            available = slots - int(np.count_nonzero(self.status == Status.CHARGING))
            while available > 0 and self.waiting:
                i = self.waiting.pop(0)
                if self.status[i] == Status.WAITING:
                    self.status[i] = Status.CHARGING
                    available -= 1
        promote()
        while t < self.dt - 1e-8:
            while events and events[0][0] <= t + 1e-8:
                _, i = events.pop(0)
                self.waiting.append(i)
            promote()
            charging = np.flatnonzero(self.status == Status.CHARGING)
            full = charging[self.soc[charging] >= 1-1e-7]
            self.soc[full] = 1
            self.status[full] = Status.IDLE
            if len(full):
                promote()
                charging = np.flatnonzero(self.status == Status.CHARGING)
            next_arrival = events[0][0] if events else self.dt
            next_full = t + float(np.min((1-self.soc[charging])/rate)) if len(charging) else self.dt
            end = min(self.dt, next_arrival, next_full)
            if end <= t + 1e-9:
                end = min(self.dt, t+1e-6)
            self.soc[charging] = np.minimum(1., self.soc[charging]+(end-t)*rate)
            t = end
        # Exact-boundary arrivals do not receive an extra step of charging.
        for _, i in events:
            self.waiting.append(i)
        full = np.flatnonzero((self.status == Status.CHARGING) & (self.soc >= 1-1e-7))
        self.soc[full] = 1
        self.status[full] = Status.IDLE
        promote()

    def step(self, action):
        if self.step_count >= self.cfg['episode_steps']:
            raise RuntimeError('reset required after truncation')
        ids, targets = self.ids(), self.targets()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(ids), 2) or not np.isfinite(action).all():
            raise ValueError('action rows must match current serving IDs')
        before, soc_before, status_before = self.xy.copy(), self.soc.copy(), self.status.copy()
        returning = np.flatnonzero(self.status == Status.RETURNING)
        delta = np.clip(action, -1, 1)*self.max_distance
        delta *= np.minimum(1., self.max_distance/np.maximum(np.linalg.norm(delta, axis=1), 1e-8))[:, None]
        intended = self.xy[ids] + delta
        bounded = np.clip(intended, 0, self.size)
        oob = np.linalg.norm(intended-bounded, axis=1) > 1e-5
        viable, depleted, avoidable = [], [], []
        for k, i in enumerate(ids):
            direction = bounded[k]-self.xy[i]
            requested_time = float(np.linalg.norm(direction))/self.speed
            move_time = min(requested_time, float(self.soc[i])*self.capacity/self.energy.horizontal_power_w)
            self.xy[i] += direction * (move_time/max(requested_time, 1e-8))
            self.soc[i] = max(0., self.soc[i]-move_time*self.energy.horizontal_power_w/self.capacity)
            hover_time = self.dt-requested_time
            enough = move_time+1e-7 >= requested_time and self.soc[i]*self.capacity > hover_time*self.energy.hover_power_w + 1e-5
            if enough:
                self.soc[i] -= hover_time*self.energy.hover_power_w/self.capacity
                viable.append(int(i))
            else:
                self.soc[i] = 0
                self.status[i] = Status.STOPPED
                depleted.append(int(i))
                if self._reference_safe(before[i], soc_before[i]) and not self.failure_charged[i]:
                    avoidable.append(int(i))
                    self.failure_charged[i] = True
        arrivals, return_failed = self._return(returning)
        self._charge(arrivals)
        self.backlog += self.rng.poisson(self.world['sensors']['arrival_rate_per_sec']*self.dt, size=self.m)
        self.age += self.dt
        pre = self.backlog.copy()
        result = self.collector.collect(self.sensor_xy, pre, self.xy, np.array(viable, dtype=int))
        covered = result.owner >= 0
        self.backlog[covered] = 0
        self.age[covered] = 0
        rw = self.cfg['reward']
        hits = int(np.count_nonzero(covered[targets]))
        values = pre[covered].astype(np.float64)
        # Stable p-norm, including empty unions.
        maximum = float(values.max()) if len(values) else 0.
        pnorm = maximum*float(np.sum((values/maximum)**rw['pnorm_p'])**(1/rw['pnorm_p'])) if maximum else 0.
        self.step_count += 1
        checked, unsafe = [], []
        if self.step_count % self.cfg['check_every'] == 0:
            checked = self.ids().tolist()
            unsafe = [i for i in checked if self.soc[i] < self.return_requirement(self.xy[i])[0]
                      and not self.failure_charged[i]]
            self.failure_charged[unsafe] = True
        penalty_ids = sorted(set(avoidable + return_failed + unsafe))
        terms = dict(target=rw['target_bonus']*hits,
                     auxiliary=rw['auxiliary_max']*pnorm/(pnorm+rw['auxiliary_reference_packets']),
                     boundary=-rw['boundary_per_uav']*int(oob.sum()),
                     return_failure=-self.failure_cost(penalty_ids))
        row = dict(step=self.step_count, reward=float(sum(terms.values())), terms=terms,
                   position_before=before.tolist(), position_after=self.xy.tolist(),
                   soc_before=soc_before.tolist(), soc_after=self.soc.tolist(),
                   status_before=status_before.tolist(), status_after=self.status.tolist(),
                   action_ids=ids.tolist(), action=action.tolist(), desired_count=self.desired_count,
                   target_ids=targets.tolist(), target_hits=hits, covered_ids=np.flatnonzero(covered).tolist(),
                   collected=float(result.collected_per_sensor.sum()), pnorm=pnorm,
                   max_backlog=float(self.backlog.max()), mean_backlog=float(self.backlog.mean()),
                   backlog=self.backlog.tolist(), checked_ids=checked, unsafe_ids=unsafe,
                   return_failed_ids=return_failed, depleted_ids=depleted,
                   avoidable_depleted_ids=avoidable, penalty_ids=penalty_ids)
        cycle_reset = self.step_count % self.cfg['check_every'] == 0 and self.cfg['cycle_reset']
        if cycle_reset:
            self.xy[:] = self.home
            self.soc[:] = 1
            self.status[:] = Status.IDLE
            self.climb_left[:] = 0
            self.failure_charged[:] = False
            self.waiting.clear()
        if self.step_count % self.cfg['dispatch_every'] == 0:
            self.dispatch()
        row.update(next_status=self.status.tolist(), next_soc=self.soc.tolist(),
                   next_positions=self.xy.tolist(), cycle_reset=bool(cycle_reset))
        self.rows.append(row)
        return self.observation(), row['reward'], False, self.step_count >= self.cfg['episode_steps'], row

    def summary(self):
        rows = self.rows
        return dict(episode_reward=float(sum(r['reward'] for r in rows)), steps=len(rows),
                    auxiliary_max=float(self.cfg['reward']['auxiliary_max']),
                    observed_auxiliary_step_max=float(max(r['terms']['auxiliary'] for r in rows)),
                    cumulative_auxiliary_bound=float(len(rows)*self.cfg['reward']['auxiliary_max']),
                    mean_max_backlog=float(np.mean([r['max_backlog'] for r in rows])),
                    worst_max_backlog=float(max(r['max_backlog'] for r in rows)),
                    final_max_backlog=float(self.backlog.max()),
                    mean_serving=float(np.mean([len(r['action_ids']) for r in rows])),
                    coverage_fraction=len({i for r in rows for i in r['covered_ids']})/self.m,
                    collected=float(sum(r['collected'] for r in rows)),
                    depletion_count=sum(len(r['depleted_ids'])+len(r['return_failed_ids']) for r in rows),
                    penalty_count=sum(len(r['penalty_ids']) for r in rows),
                    **{f'reward_{k}':float(sum(r['terms'][k] for r in rows)) for k in rows[0]['terms']})

    def state_dict(self):
        return {k:copy.deepcopy(v) for k,v in self.__dict__.items() if k not in
                ('energy', 'collector', 'rng')} | {'rng_state':copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        self.rng.bit_generator.state = state.pop('rng_state')
        self.__dict__.update(state)
