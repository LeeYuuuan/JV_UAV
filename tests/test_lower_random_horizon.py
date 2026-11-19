"""Physical20/10 retirement, variable masks, reward control and exact resume."""
from pathlib import Path
import copy
import sys
import tempfile
import numpy as np
import torch
import yaml
from test_core import ROOT
sys.path.insert(0,str(ROOT/"experiments/lower_random_horizon"))
from isolated_env import IsolatedEnv,HorizonExploration
from isolated_trainer import LowerTrainer


def configs():
    env=yaml.safe_load((ROOT/"configs/default.yaml").read_text())
    cfg=yaml.safe_load((ROOT/"configs/training_lower_isolated.yaml").read_text())
    return env,cfg


def zero(env):return np.zeros((len(env.observation()["active_uav_ids"]),2),np.float32)


def test_six_then_three_or_four_serving_and_short_return_energy_paid_once():
    ec,cfg=configs();env=IsolatedEnv(ec,cfg);env.reset(reset_rng=True,long_count=3)
    short=env.short_ids.copy();long=env.long_ids.copy();short_soc=None
    for step in range(1,21):
        _,result,done,_=env.step(zero(env))
        assert len(result.serving_ids)==(6 if step<=10 else 3)
        assert done==(step==20)
        if step==10:
            assert len(env.observation()["active_uav_ids"])==3
            short_soc=env.scene.uav_battery[short].copy()
            assert len(env.return_events)==3
            assert all(r["success"] and r["group"]=="short" for r in env.return_events)
            np.testing.assert_allclose(short_soc,1-10*326*60/450000-4000/450000,atol=1e-6)
            try:env._return(short,"short")
            except RuntimeError:pass
            else:raise AssertionError("duplicate automatic return was accepted")
        if step>10:np.testing.assert_array_equal(env.scene.uav_battery[short],short_soc)
    assert len(env.return_events)==6 and len(env.observation()["active_uav_ids"])==0
    assert all(len(env.paths[int(i)])==11 for i in short)
    assert all(len(env.paths[int(i)])==21 for i in long)
    np.testing.assert_allclose(env.scene.uav_battery[long],1-20*326*60/450000-4000/450000,atol=1e-6)
    assert not any(r.return_unsafe_ids.size for r in [result])


def test_long_return_penalty_toggle_changes_reward_only_not_physics_or_episode_length():
    ec,cfg=configs();a=IsolatedEnv(ec,cfg);cfg2=copy.deepcopy(cfg);cfg2["isolated"]["return_penalty"]=False;b=IsolatedEnv(ec,cfg2)
    for step in range(1,21):
        ra=a.step(np.ones((len(a.observation()["active_uav_ids"]),2),np.float32))
        rb=b.step(np.ones((len(b.observation()["active_uav_ids"]),2),np.float32))
        assert ra[2]==rb[2]==(step==20)
    result=ra[1];control=rb[1]
    assert len(result.return_unsafe_ids)==len(a.long_ids)
    assert result.reward_terms["return_failure"]==-500*len(a.long_ids)
    assert control.reward_terms["return_failure"]==control.reward_terms["return_distance"]==0
    assert np.isclose(result.reward-control.reward,result.reward_terms["return_failure"]+result.reward_terms["return_distance"])
    assert "terminal_future_backlog" not in result.reward_terms
    for field in ["uav_pos","uav_battery","sensor_backlog","uav_status"]:
        np.testing.assert_array_equal(getattr(a.scene,field),getattr(b.scene,field))
    assert all(r["success"] for r in a.return_events if r["group"]=="short")
    assert a.scene.rng.bit_generator.state==b.scene.rng.bit_generator.state


def test_random_membership_is_reproducible_and_samples_both_counts():
    ec,cfg=configs();a=IsolatedEnv(ec,cfg);b=IsolatedEnv(ec,cfg);counts=set();seen=set()
    for _ in range(30):
        np.testing.assert_array_equal(a.long_ids,b.long_ids);counts.add(len(a.long_ids));seen.update(a.long_ids.tolist())
        a.reset();b.reset()
    assert counts=={3,4} and seen==set(range(6))


def small_trainer():
    ec,cfg=configs();cfg["sac"].update(hidden=16,heads=2,layers=1,batch_size=2,replay_capacity=64,random_steps=0,learning_starts=0)
    return LowerTrainer(ec,cfg)


def test_lower_only_replay_crosses_retirement_and_terminal_masks_correctly():
    t=small_trainer()
    assert not hasattr(t,"mappo")
    for _ in range(20):t.step()
    assert len(t.replay)==20
    boundary=t.replay.items[9]
    assert len(boundary[0]["active_uav_ids"])==6 and len(boundary[3]["active_uav_ids"]) in (3,4)
    assert not boundary[4]
    assert t.replay.items[-1][4] and not len(t.replay.items[-1][3]["active_uav_ids"])
    assert sum(r[4] for r in t.replay.items)==1
    assert np.isclose(sum(r[2] for r in t.replay.items),t.last_episode["lower_return"])
    assert t.sac_updates==19 and t.episodes==1


def test_resume_before_retirement_and_at_reset_matches_exactly():
    for steps in [9,20]:
        t=small_trainer()
        for _ in range(steps):t.step()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"checkpoint.pt";t.save(path)
            expected=t.step();weights=copy.deepcopy(t.sac.actor.state_dict())
            loaded=LowerTrainer.load(path)
            assert loaded.step()==expected
            for key,value in weights.items():torch.testing.assert_close(value,loaded.sac.actor.state_dict()[key],rtol=0,atol=0)
            np.testing.assert_array_equal(t.env.long_ids,loaded.env.long_ids)


def test_isolated_evaluation_freezes_training_rng_normalizer_and_models():
    t=small_trainer();t.step();state=torch.get_rng_state().clone();stats=copy.deepcopy(t.rms.state_dict());scene=copy.deepcopy(t.env.trace());steps=t.low_steps
    first=t.evaluate(1001);second=t.evaluate(1001)
    assert first==second and len(first["long_ids"])==3
    assert len(t.evaluate(1002)["long_ids"])==4
    assert t.rms.state_dict()==stats and t.env.trace()==scene and t.low_steps==steps
    torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)


def test_real_energy_makes20_steps_possible_but30_steps_impossible_before_return():
    ec,cfg=configs();env=IsolatedEnv(ec,cfg);e=env.energy_model
    step=float(e.serving_energy_fraction(np.array([30.]),np.array([30.]))[0])
    assert 20*step<1 and 30*step>1
    assert 20*326*60/450000<1


def test_far_sensor_collection_then_inward_service_and_real_energy_return_is_feasible():
    for radius,outward_steps,energy_bound in [(2400.,13,.90),(2000*np.sqrt(2),15,.97)]:
        ec,cfg=configs();env=IsolatedEnv(ec,cfg)
        uid=int(env.long_ids[0]);direction=np.array([1.,1.],np.float32)/np.sqrt(2)
        # A stress-test sensor; this does not modify the training map configuration.
        env.scene.sensor_pos[0]=env.scene.airship_pos+direction*radius
        env.scene.sensor_backlog[0]=100
        collected=0.
        for step in range(20):
            ids=env.observation()["active_uav_ids"]
            actions=np.array([direction*(1 if step<outward_steps else -1) if int(i)==uid else [0.,0.] for i in ids],np.float32)
            _,result,done,_=env.step(actions)
            collected+=result.collected_per_sensor[0]
        event=next(r for r in env.return_events if r["uav_id"]==uid)
        assert collected>=100 and event["success"] and event["group"]=="long"
        assert event["soc_after_return"]>0
        assert 1-event["soc_after_return"]<energy_bound
        assert env.steps==20 and done
        np.testing.assert_allclose(env.scene.uav_pos[uid],env.scene.airship_pos)
