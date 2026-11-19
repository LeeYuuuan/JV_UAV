"""Attention SAC only: one replay insertion/update per physical lower step."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
from jv_uav.rl.sac import SAC
from jv_uav.rl.replay import Replay
from jv_uav.rl.observations import RunningMeanStd
from jv_uav.rl.diagnostics import EpisodeDiagnostics
from isolated_env import IsolatedEnv,HorizonExploration


class LowerTrainer:
    def __init__(self,env_cfg,cfg,device="cpu"):
        self.env_cfg=copy.deepcopy(env_cfg);self.cfg=copy.deepcopy(cfg);self.device=torch.device(device)
        torch.manual_seed(cfg["seed"]);self.rng=np.random.default_rng(cfg["seed"])
        self.env=IsolatedEnv(env_cfg,cfg)
        self.sac=SAC(env_cfg,cfg["sac"],self.device)
        self.rms=RunningMeanStd(clip=cfg["normalization"]["clip"],min_std=cfg["normalization"]["min_std_sec"])
        self.replay=Replay(cfg["sac"]["replay_capacity"],cfg["seed"]+1)
        self.exploration=HorizonExploration(self.env,self.rng,cfg["sac"])
        self.low_steps=self.sac_updates=self.episodes=0
        self.last_losses={};self.history=[];self.global_seen=set()
        self.diagnostics=EpisodeDiagnostics(cfg["sac"]["gamma"])
        self.last_episode=None

    def step(self):
        if self.env.done:
            self.env.reset();self.exploration.reset();self.diagnostics=EpisodeDiagnostics(self.cfg["sac"]["gamma"])
        obs=self.env.observation();battery=self.env.scene.uav_battery.copy()
        self.rms.update(obs["sensor_last_visit_sec"])
        warmup=self.low_steps<self.cfg["sac"]["random_steps"]
        action=self.exploration(obs) if warmup else self.sac.act(obs,self.rms)
        if not warmup and self.low_steps%100==0:self.diagnostics.probes.append(self.sac.action_diagnostics(obs,self.rms))
        future,result,done,info=self.env.step(action)
        self.low_steps+=1
        self.diagnostics.record(result,action,self.env.scene.airship_pos,self.env.scene.sensor_pos,warmup=warmup)
        self.global_seen.update(np.flatnonzero(result.sensor_owner>=0).tolist())
        # At step10, future already excludes the automatically retired group.
        # At step20 the transition is terminal and future may be an empty set.
        self.replay.add(obs,action,result.reward,future,done)
        if self.low_steps>=self.cfg["sac"]["learning_starts"] and len(self.replay)>=self.cfg["sac"]["batch_size"]:
            self.last_losses=self.sac.update(self.replay,self.rms);self.sac_updates+=1
        summary=None
        if done:
            self.episodes+=1
            summary=self.episode_summary()
            summary.update(episode=self.episodes,low_steps=self.low_steps,sac_updates=self.sac_updates,
                           global_sensor_coverage_fraction=len(self.global_seen)/self.env.scene.num_sensors,
                           **self.last_losses,
                           warmup_exploration_counts_total={k:int(v) for k,v in self.exploration.counts.items()})
            self.last_episode=summary;self.history.append(summary)
        return dict(low_steps=self.low_steps,sac_updates=self.sac_updates,episode=self.episodes if done else self.episodes+1,
            step=self.env.steps,done=done,warmup=warmup,termination_reason=info["termination_reason"],reward=result.reward,
            reward_terms=dict(result.reward_terms),collected_by_uav=result.collected_by_uav.astype(float).tolist(),
            active_ids=obs["active_uav_ids"].tolist(),
            next_active_ids=future["active_uav_ids"].tolist(),long_ids=self.env.long_ids.tolist(),short_ids=self.env.short_ids.tolist(),
            battery_start=battery.tolist(),battery_end=self.env.scene.uav_battery.tolist(),
            sensor_coverage_fraction=len(self.diagnostics.sensor_ids)/self.env.scene.num_sensors,
            return_events=[r for r in self.env.return_events if r["step"]==self.env.steps],
            episode_summary=summary,**self.last_losses)

    def episode_summary(self):
        return summarize(self.env,self.diagnostics)

    def evaluate(self,seed,output=None):
        env=IsolatedEnv(self.env_cfg,self.cfg,seed=int(seed))
        # The fixed evaluation suite explicitly includes both3- and4-UAV long groups.
        env.reset(reset_rng=True,long_count=3 if seed%2 else 4)
        rms=copy.deepcopy(self.rms);rms.training=False
        diagnostics=EpisodeDiagnostics(self.cfg["sac"]["gamma"])
        while not env.done:
            obs=env.observation();action=self.sac.act(obs,rms,deterministic=True)
            if env.steps%5==0:diagnostics.probes.append(self.sac.action_diagnostics(obs,rms))
            _,result,_,_=env.step(action)
            diagnostics.record(result,action,env.scene.airship_pos,env.scene.sensor_pos)
        summary=summarize(env,diagnostics);summary.update(seed=int(seed),low_steps=self.low_steps,sac_updates=self.sac_updates,policy_mode="deterministic")
        if output is not None:
            path=Path(output);path.mkdir(parents=True,exist_ok=True)
            write_json(path/"summary.json",summary);write_json(path/"trace.json",env.trace())
            from isolated_plots import dashboard
            dashboard(env,path/"dashboard.png")
        return summary

    def save(self,path):
        scene=self.env.scene
        arrays=("sensor_pos","sensor_backlog","sensor_last_visit_sec","uav_pos","uav_battery","uav_status")
        clocks=("now_sec","upper_step","low_step_total","step_in_frame")
        state=dict(format="isolated_lower_v1",env_cfg=self.env_cfg,cfg=self.cfg,sac=self.sac.state_dict(),rms=self.rms.state_dict(),
            replay=self.replay.state_dict(),exploration=self.exploration.state_dict(),diagnostics=self.diagnostics.state_dict(),
            counters=dict(low_steps=self.low_steps,sac_updates=self.sac_updates,episodes=self.episodes),last_losses=self.last_losses,
            history=self.history,global_seen=sorted(self.global_seen),last_episode=self.last_episode,
            scene={k:getattr(scene,k).copy() for k in arrays}|{k:getattr(scene,k) for k in clocks},
            env={k:copy.deepcopy(getattr(self.env,k)) for k in ("long_ids","short_ids","horizons","steps","done","return_events","step_records","paths","soc_history")},
            trace=self.env.metrics.trace,selection_rng=self.env.selection_rng.bit_generator.state,
            scene_rng=scene.rng.bit_generator.state,rng=self.rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=="cuda" else None)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(".tmp")
        torch.save(state,temporary);temporary.replace(path)

    @classmethod
    def load(cls,path,device="cpu"):
        state=torch.load(path,map_location="cpu",weights_only=False)
        if state.get("format")!="isolated_lower_v1":raise ValueError("not an isolated-lower checkpoint")
        obj=cls(state["env_cfg"],state["cfg"],device)
        obj.sac.load_state_dict(state["sac"]);obj.rms.load_state_dict(state["rms"]);obj.replay.load_state_dict(state["replay"])
        obj.exploration.load_state_dict(state["exploration"]);obj.diagnostics.load_state_dict(state["diagnostics"])
        for k,v in state["counters"].items():setattr(obj,k,v)
        obj.last_losses=state["last_losses"];obj.history=state["history"];obj.global_seen=set(state["global_seen"]);obj.last_episode=state["last_episode"]
        for k,v in state["scene"].items():
            if isinstance(v,np.ndarray):getattr(obj.env.scene,k)[:]=v
            else:setattr(obj.env.scene,k,v)
        for k,v in state["env"].items():setattr(obj.env,k,v)
        obj.env.metrics.trace=state["trace"]
        obj.env.selection_rng.bit_generator.state=state["selection_rng"];obj.env.scene.rng.bit_generator.state=state["scene_rng"]
        obj.rng.bit_generator.state=state["rng"];torch.set_rng_state(state["torch_rng"])
        if obj.device.type=="cuda" and state["cuda_rng"] is not None:torch.cuda.set_rng_state_all(state["cuda_rng"])
        return obj


def write_json(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8");temporary.replace(path)


def summarize(env,diagnostics):
    result=diagnostics.summary(env.scene.low_step_total)
    maxima=env.metrics.trace.backlog_max_post_timeline[1:]
    returns=env.return_events
    result.update(topk_positive_steps=sum(r["reward_terms"].get("topk_clear",0)>0 for r in env.step_records),
        collected_by_uav=np.sum([r["collected_by_uav"] for r in env.step_records],axis=0).tolist(),
        max_owned_pre_service_by_uav=np.max([r["owned_max_pre_service"] for r in env.step_records],axis=0).tolist(),
        steps=env.steps,long_ids=env.long_ids.tolist(),short_ids=env.short_ids.tolist(),
        mean_max_backlog=float(np.mean(maxima)),peak_max_backlog=float(np.max(maxima)),
        sensor_coverage_fraction=len(diagnostics.sensor_ids)/env.scene.num_sensors,
        outer_sensor_visited=len(diagnostics.outer_sensor_ids),
        long_return_success=sum(r["success"] for r in returns if r["group"]=="long"),
        long_return_count=sum(r["group"]=="long" for r in returns),
        short_return_success=sum(r["success"] for r in returns if r["group"]=="short"),
        short_return_count=sum(r["group"]=="short" for r in returns),return_events=copy.deepcopy(returns),
        horizon_completed=env.steps==env.long_steps,return_penalty_enabled=env.return_penalty)
    return result
