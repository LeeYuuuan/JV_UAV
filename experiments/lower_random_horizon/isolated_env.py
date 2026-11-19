"""Real-energy lower-only episodes with random long/short serving groups."""
from __future__ import annotations
import copy
import numpy as np
from jv_uav.scene import Scene
from jv_uav.models import SimpleEnergyModel
from jv_uav.metrics import EpisodeMetrics
from jv_uav.low_env import LowEnv
from jv_uav.types import UAVStatus
from jv_uav.rl.training_env import TopKReward
from jv_uav.rl.exploration import WaypointExploration


class IsolatedEnv:
    def __init__(self, env_cfg, cfg, *, seed=None):
        self.cfg=copy.deepcopy(cfg)
        self.env_cfg=copy.deepcopy(env_cfg)
        settings=cfg["isolated"]
        self.long_steps=int(settings["long_steps"])
        self.short_steps=int(settings["short_steps"])
        if not 0<self.short_steps<self.long_steps:
            raise ValueError("require 0 < short steps < long steps")
        if int(env_cfg["uavs"]["count"])!=6:
            raise ValueError("the isolated experiment requires six UAVs")
        self.env_cfg["time"].update(low_steps_per_frame=self.long_steps,episode_upper_frames=1)
        if seed is not None:self.env_cfg["seeds"]["dynamics_seed"]=int(seed)
        self.scene=Scene(self.env_cfg)
        self.energy_model=SimpleEnergyModel(self.env_cfg)
        self.metrics=EpisodeMetrics(self.env_cfg,self.scene.sensor_pos)
        self.low_env=LowEnv(self.scene,self.energy_model,self.metrics,self.env_cfg)
        reward=cfg["lower_reward"]
        self.low_env.reward_model=TopKReward(self.env_cfg,reward["k"],reward["rho"],reward["scale"],reward["offset"])
        self.selection_seed=int(settings["selection_seed"] if seed is None else seed+2)
        self.selection_rng=np.random.default_rng(self.selection_seed)
        self.return_penalty=bool(settings["return_penalty"])
        self.reset(reset_rng=True)

    def reset(self, *, reset_rng=False, long_count=None):
        self.scene.reset(reset_rng=reset_rng)
        self.scene.uav_status[:]=int(UAVStatus.SERVING)
        self.metrics.reset(self.scene.sensor_backlog)
        if reset_rng:self.selection_rng=np.random.default_rng(self.selection_seed)
        count=int(self.selection_rng.integers(3,5)) if long_count is None else int(long_count)
        if count not in (3,4):raise ValueError("long group count must be three or four")
        self.long_ids=np.sort(self.selection_rng.choice(6,count,replace=False))
        self.short_ids=np.setdiff1d(np.arange(6),self.long_ids)
        self.horizons=np.full(6,self.short_steps,dtype=np.int64)
        self.horizons[self.long_ids]=self.long_steps
        self.steps=0;self.done=False;self.return_events=[];self.step_records=[]
        self.paths={i:[self.scene.uav_pos[i].astype(float).tolist()] for i in range(6)}
        self.soc_history=[self.scene.uav_battery.astype(float).tolist()]
        return self.observation()

    def observation(self):return self.scene.observe_lower()

    def _return(self, ids, group):
        if not len(ids):return np.zeros(0,np.int64)
        if np.any(self.scene.uav_status[ids] != int(UAVStatus.SERVING)):
            raise RuntimeError("automatic return can settle each serving UAV only once")
        trip=self.energy_model.return_trip(self.scene.uav_pos[ids],self.scene.airship_pos)
        reserve=self.scene.return_reserve_frac
        failed=ids[self.scene.uav_battery[ids]<trip.energy_frac+reserve]
        for index,uid in enumerate(ids):
            uid=int(uid);before=float(self.scene.uav_battery[uid]);success=uid not in failed
            # Required energy is always logged. Only feasible automatic returns settle it.
            event=dict(step=self.steps,uav_id=uid,group=group,success=success,
                service_endpoint=self.scene.uav_pos[uid].astype(float).tolist(),soc_before_return=before,
                required_return_soc=float(trip.energy_frac[index]),return_time_sec=float(trip.time_sec[index]),
                distance_m=float(trip.distance_m[index]),return_margin_soc=before-float(trip.energy_frac[index])-reserve,
                return_energy_paid=float(trip.energy_frac[index]) if success else 0.)
            if success:
                self.scene.uav_battery[uid]-=trip.energy_frac[index]
                self.scene.uav_pos[uid]=self.scene.airship_pos
                self.scene.uav_status[uid]=int(UAVStatus.WAITING)
            else:self.scene.uav_status[uid]=int(UAVStatus.DEAD)
            e=self.energy_model
            horizontal_budget=(before-reserve)*3600*e.capacity_wh-e.climb_power_w*e.height_m/e.climb_speed_mps
            event["safe_return_distance_m"]=max(0.,horizontal_budget/e.horizontal_power_w*e.horizontal_speed_mps)
            event["return_from_airship_feasible"]=horizontal_budget>=0
            event["soc_after_return"]=float(self.scene.uav_battery[uid])
            self.return_events.append(event)
        return failed

    def step(self, action):
        if self.done:raise RuntimeError("reset the finished episode before stepping")
        result=self.low_env.step(action)
        self.steps+=1
        for uid in result.serving_ids:self.paths[int(uid)].append(result.position_after[uid].astype(float).tolist())
        reason=None
        if result.dead_during_step.size:
            self.done=True;reason="battery_depleted_during_service"
        elif self.steps==self.short_steps:
            failed=self._return(self.short_ids,"short")
            if failed.size:
                result.return_unsafe_ids=failed.copy();self.done=True;reason="short_return_infeasible"
        elif self.steps==self.long_steps:
            # Physical return feasibility is unchanged by the reward toggle.
            ids=self.scene.serving_ids()
            trip=self.energy_model.return_trip(self.scene.uav_pos[ids],self.scene.airship_pos)
            failed=ids[self.scene.uav_battery[ids]<trip.energy_frac+self.scene.return_reserve_frac]
            self.low_env.apply_final_return_penalty(result,failed,
                lower_failure_weights=None if self.return_penalty else np.zeros(len(failed)))
            self._return(ids,"long")
            self.done=True;reason="long_return_infeasible" if len(failed) else "horizon_reached"
        self.soc_history.append(self.scene.uav_battery.astype(float).tolist())
        self.step_records.append(dict(step=self.steps,serving_ids=result.serving_ids.tolist(),
            positions=result.position_after.astype(float).tolist(),reward=result.reward,reward_terms=dict(result.reward_terms),
            collected_by_uav=result.collected_by_uav.astype(float).tolist(),
            owned_max_pre_service=result.per_uav_owned_max_pre_service.astype(float).tolist(),
            covered_sensor_ids=np.flatnonzero(result.sensor_owner>=0).tolist(),
            battery_after_settlement=self.scene.uav_battery.astype(float).tolist(),
            return_unsafe_ids=result.return_unsafe_ids.tolist()))
        return self.observation(),result,self.done,dict(termination_reason=reason,
            long_ids=self.long_ids.tolist(),short_ids=self.short_ids.tolist())

    def trace(self):
        return dict(long_ids=self.long_ids.tolist(),short_ids=self.short_ids.tolist(),paths=self.paths,
                    soc_history=self.soc_history,return_events=self.return_events,steps=self.step_records,
                    metrics=self.metrics.trace.to_serializable())


class HorizonExploration:
    """Waypoint warmup with a per-UAV retirement horizon; random20% stay unguarded."""
    def __init__(self, env, rng, cfg):
        self.env=env;self.rng=rng;self.cfg=cfg
        self.base=WaypointExploration(env.scene,rng,0.,energy_guard=False,energy_model=env.energy_model)

    @property
    def counts(self):return self.base.counts
    def reset(self):self.base.reset()
    def state_dict(self):return self.base.state_dict()
    def load_state_dict(self,state):self.base.load_state_dict(state)

    def _margin(self,position,soc,first,remaining):
        scene=self.env.scene;energy=self.env.energy_model;position=position.astype(float).copy()
        for step in range(remaining):
            action=first if step==0 else self.base._home_action(position)
            endpoint=np.clip(position+action*scene.max_move_distance_m,0,scene.world_size)
            move_time=np.linalg.norm(endpoint-position)/scene.speed_mps
            soc-=float(energy.serving_energy_fraction(move_time,scene.low_step_sec-move_time))
            position=endpoint
        trip=energy.return_trip(position[None],scene.airship_pos)
        return soc-float(trip.energy_frac[0])-max(scene.return_reserve_frac,self.cfg.get("warmup_return_reserve_frac",.02))

    def __call__(self,obs):
        action=self.base(obs)
        for row,uid in enumerate(obs["active_uav_ids"]):
            position=obs["active_uav_features"][row,:2];soc=float(obs["active_uav_features"][row,2])
            remaining=int(self.env.horizons[uid])-self.env.steps
            random=self.rng.random()<self.cfg.get("warmup_random_fraction",.2)
            if random:
                angle=self.rng.uniform(0,2*np.pi)
                action[row]=np.sqrt(self.rng.random())*np.array([np.cos(angle),np.sin(angle)])
                self.counts["random_actions"]+=1
                self.counts["random_infeasible_forecasts"]+=int(self._margin(position,soc,action[row],remaining)<0)
            else:
                self.counts["guided_actions"]+=1
                if self.cfg.get("warmup_energy_guard",True):
                    if not self.base.returning[uid] and self._margin(position,soc,action[row],remaining)<0:
                        self.base.returning[uid]=True;self.counts["turnbacks"]+=1
                    if self.base.returning[uid]:action[row]=self.base._home_action(position)
                    self.counts["guided_infeasible_forecasts"]+=int(self._margin(position,soc,action[row],remaining)<0)
            endpoint=np.clip(position+action[row]*self.env.scene.max_move_distance_m,0,self.env.scene.world_size)
            action[row]=(endpoint-position)/self.env.scene.max_move_distance_m
        return action
