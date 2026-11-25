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
        self.round_visited = np.zeros(self.m,dtype=bool)
        self.completed_rounds = 0
        self.round_start_step = 0
        self.xy = np.tile(self.home, (self.n, 1))
        self.soc = np.ones(self.n, dtype=np.float32)
        self.status = np.full(self.n, int(Status.IDLE), dtype=np.int8)
        self.climb_left = np.zeros(self.n)
        self.failure_charged = np.zeros(self.n, dtype=bool)
        self.energy_deployed = np.zeros(self.n,dtype=bool)
        self.return_started = np.zeros(self.n,dtype=bool)
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

    def reward_targets(self):
        # Expand ties at the K-th oldest age. This is independent of assignment.
        k=min(self.cfg['target_count'], self.m)
        threshold=float(np.partition(self.age,self.m-k)[self.m-k])
        return np.flatnonzero(self.age>=threshold),threshold

    def assignment(self):
        """Oldest first, nearest within ties, one unique target per serving UAV."""
        ids=self.ids()
        order=np.roll(ids,-((self.step_count//self.cfg['dispatch_every'])%max(1,len(ids))))
        assigned=np.full(self.n,-1,dtype=np.int64)
        pending=~self.round_visited
        available=pending.copy()
        for i in order:
            if not pending.any():break
            if not available.any():available=pending.copy()  # cooperate on final remaining goals
            oldest=self.age[available].max()
            choices=np.flatnonzero(available & (self.age==oldest))
            distance=np.linalg.norm(self.sensor_xy[choices]-self.xy[i],axis=1)
            selected=int(choices[np.argmin(distance)])
            assigned[i]=selected;available[selected]=False
        return assigned,order

    def observation(self):
        assigned,order=self.assignment()
        # Fixed-width target context with an explicit mask; no fake (0,0) targets.
        width=self.cfg['target_count']
        targets=np.zeros((width,2),dtype=np.float32)
        target_age=np.zeros(width,dtype=np.float32)
        target_mask=np.zeros(width,dtype=bool)
        target_ids=np.full(width,-1,dtype=np.int64)
        own_xy=np.zeros((self.n,2),dtype=np.float32)
        own_age=np.zeros(self.n,dtype=np.float32)
        valid=assigned>=0
        own_xy[valid]=self.sensor_xy[assigned[valid]]
        own_age[valid]=self.age[assigned[valid]]
        ordered=assigned[order];ordered=ordered[ordered>=0]
        if len(ordered)>width:
            raise ValueError('target_count must cover all serving UAVs')
        targets[:len(ordered)]=self.sensor_xy[ordered]
        target_age[:len(ordered)]=self.age[ordered]
        target_mask[:len(ordered)]=True;target_ids[:len(ordered)]=ordered
        return dict(ids=self.ids(), xy=self.xy.copy(), soc=self.soc.copy(), status=self.status.copy(),
                    age=self.age.copy(), targets=targets, target_age=target_age,target_mask=target_mask,
                    target_ids=target_ids,assigned_ids=assigned,assigned_xy=own_xy,assigned_age=own_age,
                    assigned_valid=valid,assignment_order=order.copy(),
                    phase=np.float32((self.step_count % self.cfg['check_every']) / self.cfg['check_every']),
                    round_visited=self.round_visited.copy(),round_progress=np.float32(self.round_visited.mean()))

    def dispatch(self):
        desired = int(self.rng.integers(self.cfg['service_count_min'], self.cfg['service_count_max'] + 1))
        eligible = np.flatnonzero((self.status != Status.RETURNING) & (self.status != Status.STOPPED)
                                 & (self.soc > 0))
        chosen = self.rng.choice(eligible, min(desired, len(eligible)), replace=False)
        old = self.ids()
        leaving = np.setdiff1d(old, chosen)
        self.status[leaving] = Status.RETURNING
        self.return_started[leaving]=True
        self.climb_left[leaving] = self.energy.height_m / self.energy.climb_speed_mps
        newly_serving = chosen[self.status[chosen] != Status.SERVING]
        self.failure_charged[newly_serving] = False
        self.energy_deployed[chosen]=True
        self.return_started[chosen]=False
        self.status[chosen] = Status.SERVING
        self.waiting = [i for i in self.waiting if i not in chosen]
        self.desired_count = desired
        self.selected_ids = chosen.tolist()

    def return_requirement(self, xy):
        return self.energy.return_trip(np.asarray(xy).reshape(-1, 2), self.home).energy_frac + self.reserve

    def potential(self, obs=None):
        """Bounded round progress + linear goal closeness; no backlog magnitude."""
        o=self.observation() if obs is None else obs
        ids=o['ids'];valid=ids[o['assigned_valid'][ids]]
        dmax=max(1.,float(np.sqrt(2)*self.size-self.radius))
        if len(valid):
            gap=np.maximum(np.linalg.norm(o['assigned_xy'][valid]-o['xy'][valid],axis=1)-self.radius,0.)
            closeness=float(np.clip(1-gap/dmax,0,1).sum()/self.n)
        else:closeness=0.
        rw=self.cfg['reward']
        return float(rw['progress_weight']*o['round_progress']+rw['navigation_weight']*closeness)

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

    def linear_energy_cost(self):
        """Charge only energy shortfall, without a terminal or unsafe-state cutoff."""
        required=self.return_requirement(self.xy).astype(np.float64)
        in_return=self.return_started | (self.status==Status.RETURNING)
        distance=np.linalg.norm(self.xy-self.home,axis=1)
        # During an actual return, only unfinished climb may be counted.
        required[in_return]=(self.energy.horizontal_power_w*distance[in_return]/self.energy.horizontal_speed_mps
                             +self.energy.climb_power_w*self.climb_left[in_return])/self.capacity+self.reserve
        active=(self.status==Status.SERVING)|(self.status==Status.RETURNING)|(
            (self.status==Status.STOPPED)&self.energy_deployed)
        required[~active]=0.
        deficit=np.where(active,np.maximum(0.,required-self.soc),0.)
        rw=self.cfg['reward']
        cost=rw['energy_deficit_weight']*deficit/rw['energy_deficit_scale_soc']
        return required,deficit,cost

    def _return(self, ids):
        arrivals, failures = [], []
        self.return_started[ids]=True
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
                self.energy_deployed[i]=False
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
        obs_before = self.observation()
        phi_before=self.potential(obs_before)
        ids=self.ids()
        targets,threshold=self.reward_targets()
        assigned,assignment_order=self.assignment()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(ids), 2) or not np.isfinite(action).all():
            raise ValueError('action rows must match current serving IDs')
        before, soc_before, status_before = self.xy.copy(), self.soc.copy(), self.status.copy()
        returning = np.flatnonzero(self.status == Status.RETURNING)
        self.energy_deployed[ids]=True
        self.energy_deployed[returning]=True
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
        new_ids=np.flatnonzero(covered & ~self.round_visited)
        self.round_visited[covered]=True
        progress_after_collection=float(self.round_visited.mean())
        rw = self.cfg['reward']
        hits = int(np.count_nonzero(covered[targets]))
        values = pre[covered].astype(np.float64)
        # Stable p-norm, including empty unions.
        maximum = float(values.max()) if len(values) else 0.
        pnorm = maximum*float(np.sum((values/maximum)**rw['pnorm_p'])**(1/rw['pnorm_p'])) if maximum else 0.
        self.step_count += 1
        round_completed=bool(self.round_visited.all())
        round_duration=None
        if round_completed:
            self.completed_rounds+=1
            round_duration=self.step_count-self.round_start_step
            self.round_start_step=self.step_count
            # No overflow credits into the next round from the SAME collection.
            self.round_visited[:]=False
        checked, unsafe = [], []
        if self.step_count % self.cfg['check_every'] == 0:
            checked = self.ids().tolist()
            unsafe = [i for i in checked if self.soc[i] < self.return_requirement(self.xy[i])[0]
                      and not self.failure_charged[i]]
            self.failure_charged[unsafe] = True
        penalty_ids = sorted(set(avoidable + return_failed + unsafe))
        required_soc,deficit_soc,energy_cost=self.linear_energy_cost()
        terms = dict(round=rw['round_reward']*int(round_completed),
                     boundary=-rw['boundary_per_uav']*int(oob.sum()),
                     energy_deficit=-float(energy_cost.sum()),
                     return_failure=-self.failure_cost(penalty_ids))
        row = dict(step=self.step_count, reward=float(sum(terms.values())), terms=terms,
                   position_before=before.tolist(), position_after=self.xy.tolist(),
                   soc_before=soc_before.tolist(), soc_after=self.soc.tolist(),
                   status_before=status_before.tolist(), status_after=self.status.tolist(),
                   action_ids=ids.tolist(), action=action.tolist(), desired_count=self.desired_count,
                   target_ids=targets.tolist(), target_age_threshold_sec=threshold,
                   eligible_target_count=len(targets),assigned_target_ids=assigned.tolist(),
                   assignment_order=assignment_order.tolist(),
                   assigned_hits=len(set(assigned[assigned>=0].tolist()) & set(np.flatnonzero(covered).tolist())),
                   target_hits=hits, covered_ids=np.flatnonzero(covered).tolist(),
                   collected=float(result.collected_per_sensor.sum()), pnorm=pnorm,
                   max_backlog=float(self.backlog.max()), mean_backlog=float(self.backlog.mean()),
                   backlog=self.backlog.tolist(), checked_ids=checked, unsafe_ids=unsafe,
                   return_failed_ids=return_failed, depleted_ids=depleted,
                   avoidable_depleted_ids=avoidable, penalty_ids=penalty_ids)
        row.update(required_return_soc=required_soc.tolist(),energy_deficit_soc=deficit_soc.tolist(),
                   energy_cost_by_uav=energy_cost.tolist(),
                   energy_deficit_uav_steps=int(np.count_nonzero(deficit_soc>0)),
                   energy_deficit_scale_soc=rw['energy_deficit_scale_soc'],
                   energy_deficit_weight=rw['energy_deficit_weight'])
        cycle_reset = self.step_count % self.cfg['check_every'] == 0 and self.cfg['cycle_reset']
        if cycle_reset:
            self.xy[:] = self.home
            self.soc[:] = 1
            self.status[:] = Status.IDLE
            self.climb_left[:] = 0
            self.failure_charged[:] = False
            self.energy_deployed[:]=False
            self.return_started[:]=False
            self.waiting.clear()
        if self.step_count % self.cfg['dispatch_every'] == 0:
            self.dispatch()
        row.update(next_status=self.status.tolist(), next_soc=self.soc.tolist(),
                   next_positions=self.xy.tolist(), cycle_reset=bool(cycle_reset))
        nxt = self.observation()
        phi_after=self.potential(nxt)
        row['terms']['shaping']=self.cfg['sac']['gamma']*phi_after-phi_before
        row['reward']=float(sum(row['terms'].values()))
        row.update(round_new_sensor_ids=new_ids.tolist(),round_completed=round_completed,
                   round_duration_steps=round_duration,completed_rounds=self.completed_rounds,
                   round_progress_before=float(obs_before['round_progress']),
                   round_progress_after_collection=progress_after_collection,
                   round_progress_next=float(self.round_visited.mean()),
                   round_visited_next=self.round_visited.tolist(),
                   potential_before=phi_before,potential_after=phi_after,shaping_gamma=self.cfg['sac']['gamma'])
        row.update(assigned_target_xy=obs_before['assigned_xy'].tolist(),
                   assigned_distance_before_m=np.linalg.norm(obs_before['assigned_xy'][ids]-before[ids],axis=1).tolist(),
                   assigned_distance_after_m=np.linalg.norm(obs_before['assigned_xy'][ids]-np.asarray(row['position_after'])[ids],axis=1).tolist())
        self.rows.append(row)
        return nxt, row['reward'], False, self.step_count % self.cfg['episode_steps'] == 0, row

    def summary(self):
        rows = self.rows
        return dict(episode_reward=float(sum(r['reward'] for r in rows)), steps=len(rows),
                    completed_rounds=sum(r['round_completed'] for r in rows),
                    cumulative_completed_rounds=self.completed_rounds,
                    round_progress=float(self.round_visited.mean()),
                    round_elapsed_steps=self.step_count-self.round_start_step,
                    completed_round_durations=[r['round_duration_steps'] for r in rows if r['round_completed']],
                    round_new_sensor_count=sum(len(r['round_new_sensor_ids']) for r in rows),
                    episode_base_reward=float(sum(r['reward']-r['terms']['shaping'] for r in rows)),
                    shaping_telescoping_error=float(sum(self.cfg['sac']['gamma']**j*r['terms']['shaping'] for j,r in enumerate(rows))
                        -(-rows[0]['potential_before']+self.cfg['sac']['gamma']**len(rows)*rows[-1]['potential_after'])),
                    discounted_return=float(sum(self.cfg['sac']['gamma']**j*r['reward'] for j,r in enumerate(rows))),
                    mean_max_backlog=float(np.mean([r['max_backlog'] for r in rows])),
                    worst_max_backlog=float(max(r['max_backlog'] for r in rows)),
                    final_max_backlog=float(self.backlog.max()),
                    mean_serving=float(np.mean([len(r['action_ids']) for r in rows])),
                    coverage_fraction=len({i for r in rows for i in r['covered_ids']})/self.m,
                    collected=float(sum(r['collected'] for r in rows)),
                    mean_eligible_targets=float(np.mean([r['eligible_target_count'] for r in rows])),
                    assigned_hits=sum(r['assigned_hits'] for r in rows),
                    energy_deficit_uav_steps=sum(r['energy_deficit_uav_steps'] for r in rows),
                    peak_energy_deficit_soc=float(max(max(r['energy_deficit_soc']) for r in rows)),
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
