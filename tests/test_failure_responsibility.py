"""Physical counterfactual responsibility, reward accounting and replay timing."""
from pathlib import Path
import tempfile
import copy
import numpy as np
import torch
import yaml
from test_core import ROOT
from jv_uav.rl.training_env import build_training_env
from jv_uav.rl.trainer import JointTrainer
from jv_uav.failure_responsibility import validate_failure_records


def configs():
    env=yaml.safe_load((ROOT/"configs/default.yaml").read_text())
    train=yaml.safe_load((ROOT/"configs/training_joint_responsibility_v4.yaml").read_text())
    env["reward"].update(train["upper_reward_overrides"])
    return env,train


def run_case(batteries, *, tail=True, charge=False):
    ec,tc=configs();tc["lower_reward"]["terminal_tail"]=tail
    env=build_training_env(ec,tc);env.reset()
    for uid,battery in enumerate(batteries):env.scene.uav_battery[uid]=battery
    moving=set(range(len(batteries)))
    def policy(obs):
        return np.array([[1.,0.] if int(uid) in moving else [0.,0.] for uid in obs["active_uav_ids"]],np.float32)
    env.low_runner.low_policy=policy
    callbacks=[]
    env.low_runner.on_step=lambda obs,action,result:callbacks.append((result.reward,copy.deepcopy(result.reward_terms),copy.deepcopy(result.failure_records)))
    action=np.zeros(6,np.int8)
    if charge:
        action[0]=1;env.scene.uav_pos[0]=env.scene.airship_pos+[1800.,0.]
    transition=env.step(action)
    return env,transition,callbacks


def test_pure_upper_shared_and_pure_lower_return_failure_match_reward_tables():
    cases=[]
    for battery,upper,lower in [(.395,1.,0.),(.42,.5,.5),(.5,0.,1.)]:
        env,(_,reward,terminated,truncated,info),callbacks=run_case([battery])
        assert terminated and not truncated and env.scene.low_step_total==10
        events=info["failure_records"];assert len(events)==1
        event=events[0];assert event["upper_weight"]==upper and event["lower_weight"]==lower
        assert event["failure_type"]=="final_return_infeasible"
        assert np.isclose(info["reward_terms"]["dead"],-upper)
        last=info["low_frame"].steps[-1]
        assert last.reward_terms["return_failure"]==-500*lower
        assert np.isclose(last.reward_terms["return_distance"],-500*lower,atol=.001)
        assert info["failure_tail"]["lower_weight"]==lower
        assert np.isclose(last.reward_terms["terminal_future_backlog"],info["failure_tail"]["unweighted_cost"]*lower)
        assert len(callbacks)==10 and len([r for _,_,r in callbacks if r])==1
        assert callbacks[-1][0]==last.reward
        assert callbacks[-1][2][0]==event
        assert np.isclose(reward,sum(info["reward_terms"].values()))
        for step in info["low_frame"].steps:assert np.isclose(step.reward,sum(step.reward_terms.values()))
        cases.append(info["failure_tail"]["unweighted_cost"])
    np.testing.assert_allclose(cases,[cases[0]]*3)


def test_multiple_deaths_have_one_event_each_and_one_mean_weighted_tail():
    env,(_,_,terminated,_,info),callbacks=run_case([.395,.5])
    assert terminated and len(info["failure_records"])==2
    assert {r["uav_id"] for r in info["failure_records"]}=={0,1}
    assert sum(r["upper_weight"] for r in info["failure_records"])==1
    assert sum(r["lower_weight"] for r in info["failure_records"])==1
    assert info["reward_terms"]["dead"]==-1
    terms=info["low_frame"].steps[-1].reward_terms
    assert terms["return_failure"]==-500
    assert np.isclose(terms["return_distance"],-500,atol=.001)
    assert info["failure_tail"]["lower_weight"]==.5
    assert np.isclose(terms["terminal_future_backlog"],.5*info["failure_tail"]["unweighted_cost"])
    _,(_,_,_,_,two_lower),_=run_case([.5,.5])
    assert two_lower["failure_tail"]["lower_weight"]==1
    assert two_lower["low_frame"].steps[-1].reward_terms["terminal_future_backlog"]==two_lower["failure_tail"]["unweighted_cost"]


def test_midstep_upper_death_and_settlement_upper_death_do_not_penalize_lower():
    for battery,charge,cause in [(.1,False,"battery_depleted_during_low_step"),(.1,True,"return_failed_during_settlement")]:
        env,(_,_,terminated,_,info),callbacks=run_case([battery],charge=charge)
        assert terminated and len(info["failure_records"])==1
        event=info["failure_records"][0]
        assert event["failure_type"]==cause and event["upper_weight"]==1 and event["lower_weight"]==0
        assert info["reward_terms"]["dead"]==-1
        assert len(callbacks)==env.scene.low_step_total
        terms=callbacks[-1][1]
        assert terms["return_failure"]==0 and terms["return_distance"]==0 and terms["terminal_future_backlog"]==0
        assert callbacks[-1][2][0]==event


def test_no_failure_no_failure_cost_or_tail_even_at_horizon():
    ec,tc=configs();ec["time"]["episode_upper_frames"]=1
    env=build_training_env(ec,tc);env.reset()
    env.low_runner.low_policy=lambda obs:np.zeros((len(obs["active_uav_ids"]),2),np.float32)
    _,reward,terminated,truncated,info=env.step(np.zeros(6,np.int8))
    assert truncated and not terminated and info["failure_records"]==[]
    assert info["reward_terms"]["dead"]==0 and "failure_tail" not in info
    assert all("terminal_future_backlog" not in r.reward_terms for r in info["low_frame"].steps)


def test_reference_path_matches_physical_execution_and_thresholds():
    ec,tc=configs();env=build_training_env(ec,tc);env.reset()
    env.scene.uav_pos[0]=env.scene.airship_pos+[1500.,0.]
    env.scene.uav_battery[0]=.55
    def policy(obs):return np.array([[-75/180,0.] if int(uid)==0 else [0.,0.] for uid in obs["active_uav_ids"]],np.float32)
    env.low_runner.low_policy=policy
    _,_,terminated,_,info=env.step(np.zeros(6,np.int8))
    assert not terminated
    assessor=env.failure_assessor;ref=assessor.references[0]
    np.testing.assert_allclose(ref["reference_end_xy"],env.scene.uav_pos[0],atol=1e-4)
    assert np.isclose(.55-env.scene.uav_battery[0],ref["reference_service_soc"],atol=1e-6)
    actual_return=env.energy_model.return_trip(env.scene.uav_pos[:1],env.scene.airship_pos).energy_frac[0]
    assert np.isclose(actual_return,ref["reference_return_soc"],atol=1e-7)
    assert np.isclose(ref["reference_required_soc"],.5057778,atol=1e-6)
    # Explicit inclusive equality and distinct gray/impossible intervals.
    ref["frame_start_soc"]=ref["reference_required_soc"]
    assert assessor.assess(0,"final_return_infeasible")["upper_weight"]==0
    ref["frame_start_soc"]=ref["reference_required_soc"]-1e-5
    assert assessor.assess(0,"final_return_infeasible")["upper_weight"]==.5
    ref["frame_start_soc"]=ref["optimistic_required_soc"]
    assert assessor.assess(0,"final_return_infeasible")["upper_weight"]==.5
    ref["frame_start_soc"]=ref["optimistic_required_soc"]-1e-5
    assert assessor.assess(0,"final_return_infeasible")["upper_weight"]==1
    # Physical return reserve and the reference margin each enter exactly once.
    ec,tc=configs();ec["uavs"]["return_reserve_frac"]=.02
    other=build_training_env(ec,tc);other.reset()
    plan=other.scheduler.make_plan(other.scene,np.zeros(6,np.int8))
    other.failure_assessor.begin_frame(plan,other.scene)
    rr=other.failure_assessor.references[0]
    assert np.isclose(rr["reference_required_soc"],rr["reference_service_soc"]+rr["reference_return_soc"]+.02+.01)


def test_responsibility_is_final_before_terminal_replay_and_resume_is_exact():
    ec,tc=configs()
    tc["sac"].update(hidden=16,heads=2,layers=1,batch_size=2,replay_capacity=64,random_steps=0,learning_starts=99999)
    tc["mappo"].update(hidden=16,transfer_actor_hidden=16,epochs=1,minibatch_frames=2)
    tc["upper_rollout_steps"]=2
    for battery in [.395,.42,.5]:
        trainer=JointTrainer(ec,tc)
        trainer.env.scene.uav_battery[0]=battery
        trainer.mappo.act=lambda obs,**kwargs:(np.zeros(6,np.int8),np.zeros(6),0.,np.zeros(6))
        trainer.sac.act=lambda obs,rms:np.array([[1.,0.] if int(uid)==0 else [0.,0.] for uid in obs["active_uav_ids"]],np.float32)
        row=trainer.step()
        assert row["terminated"] and len(trainer.replay)==10 and trainer.pending is None
        assert sum(bool(item[4]) for item in trainer.replay.items)==1 and trainer.replay.items[-1][4]
        assert np.isclose(sum(item[2] for item in trainer.replay.items),row["lower_reward"])
        assert row["episode_metrics"]["failure_events"]==row["failure_records"]
    # Use the real stochastic policy for exact pending/rollout checkpoint continuation.
    tc["sac"]["learning_starts"]=0
    trainer=JointTrainer(ec,tc);trainer.step()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/"state.pt";trainer.save(path)
        expected=trainer.step()
        resumed=JointTrainer.load(path)
        assert resumed.step()==expected


def test_duplicate_events_and_legacy_extra_penalties_are_rejected():
    record=dict(uav_id=0,upper_weight=.5,lower_weight=.5)
    try:validate_failure_records([record,record],[0])
    except RuntimeError:pass
    else:raise AssertionError("duplicate failure accepted")
    ec,tc=configs();ec["reward"]["upper_lower_return_failure_weight"]=15
    try:build_training_env(ec,tc)
    except ValueError:pass
    else:raise AssertionError("overlapping legacy failure penalty accepted")


def test_reference_checks_do_not_consume_dynamics_rng_or_change_trajectories():
    ec,tc=configs();new=build_training_env(ec,tc);new.reset()
    old_cfg=copy.deepcopy(tc);old_cfg.pop("failure_responsibility")
    old=build_training_env(ec,old_cfg);old.reset()
    for env in [new,old]:
        env.low_runner.low_policy=lambda obs:np.zeros((len(obs["active_uav_ids"]),2),np.float32)
        env.step(np.zeros(6,np.int8))
    assert new.scene.rng.bit_generator.state==old.scene.rng.bit_generator.state
    for field in ["uav_pos","uav_battery","sensor_backlog"]:
        np.testing.assert_array_equal(getattr(new.scene,field),getattr(old.scene,field))


def test_failure_on_last_episode_frame_has_zero_remaining_tail_and_one_event():
    ec,tc=configs();ec["time"]["episode_upper_frames"]=1
    env=build_training_env(ec,tc);env.reset();env.scene.uav_battery[0]=.42
    env.low_runner.low_policy=lambda obs:np.array([[1.,0.] if int(uid)==0 else [0.,0.] for uid in obs["active_uav_ids"]],np.float32)
    _,_,terminated,truncated,info=env.step(np.zeros(6,np.int8))
    assert terminated and not truncated and len(info["failure_records"])==1
    assert info["failure_tail"]["unweighted_cost"]==0
    assert info["low_frame"].steps[-1].reward_terms["terminal_future_backlog"]==0


def test_last_step_midframe_depletion_is_not_also_counted_as_return_failure():
    env,(_,_,terminated,_,info),callbacks=run_case([.388])
    assert terminated and env.scene.low_step_total==10
    assert len(info["failure_records"])==1
    assert info["failure_records"][0]["failure_type"]=="battery_depleted_during_low_step"
    assert info["reward_terms"]["dead"]==-1
    assert callbacks[-1][1]["return_failure"]==0 and callbacks[-1][1]["terminal_future_backlog"]==0


def test_settlement_and_final_failures_are_settled_before_callback_without_tail_wrapper():
    for battery,charge,lower in [(.42,False,.5),(.1,True,0.)]:
        env,(_,_,terminated,_,info),callbacks=run_case([battery],tail=False,charge=charge)
        assert terminated and len(callbacks)==10
        assert callbacks[-1][2]==info["failure_records"]
        assert callbacks[-1][1]["return_failure"]==-500*lower
        assert "terminal_future_backlog" not in callbacks[-1][1]


def test_reference_snapshot_is_recomputed_for_each_new_frame():
    ec,tc=configs();env=build_training_env(ec,tc);env.reset()
    env.low_runner.low_policy=lambda obs:np.zeros((len(obs["active_uav_ids"]),2),np.float32)
    env.step(np.zeros(6,np.int8))
    before=env.scene.uav_battery[0].item()
    env.low_runner.low_policy=lambda obs:np.array([[1.,0.] if int(uid)==0 else [0.,0.] for uid in obs["active_uav_ids"]],np.float32)
    _,_,terminated,_,info=env.step(np.zeros(6,np.int8))
    assert terminated and len(info["failure_records"])==1
    record=info["failure_records"][0]
    assert record["event_id"]=="frame:1:uav:0"
    assert record["frame_start_soc"]==before and before<1
