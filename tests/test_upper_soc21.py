"""Dedicated own-return SOC feature and checkpoint/transfer compatibility."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import yaml
from test_core import ROOT, build_env
from jv_uav.rl.observations import upper_observation
from jv_uav.rl.transfer import Paper1Actor, read_paper1_actor
from jv_uav.rl.trainer import JointTrainer


def test_own_return_soc_uses_the_correct_uav_and_preserves_first20_features():
    _, scene, energy, _, _ = build_env()
    scene.uav_pos[:] = scene.airship_pos + np.array([[0,0],[100,0],[0,200],[300,0],[0,400],[500,0]])
    scene.uav_status[1]=1
    obs=torch.from_numpy(upper_observation(scene,energy,"current_soc_return_soc")["actor"])
    old=Paper1Actor(6,16,include_return_soc=True)
    new=Paper1Actor(6,16,include_return_soc=True,include_own_return_soc=True)
    features=new.old_features(obs)
    assert features.shape==(6,21)
    torch.testing.assert_close(features[:,:20],old.old_features(obs),rtol=0,atol=0)
    torch.testing.assert_close(features[:,-1],obs[0,-6:],rtol=0,atol=0)
    assert features[1,-1]==0
    permutation=torch.tensor([5,2,0,4,1,3])
    torch.testing.assert_close(new.old_features(obs[permutation])[:,-1],obs[0,-6:][permutation],rtol=0,atol=0)


def test_old14_feature_actor_transfer_into21_preserves_probabilities():
    _,scene,energy,_,_=build_env()
    obs=torch.from_numpy(upper_observation(scene,energy,"current_soc_return_soc")["actor"])
    old=Paper1Actor(6,16)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/"old.pt"
        torch.save({"actor":old.state_dict()},path)
        state,meta=read_paper1_actor(path,6,"current_soc_return_soc",include_own_return_soc=True)
    assert state["net.0.weight"].shape==(16,21)
    assert torch.count_nonzero(state["net.0.weight"][:,14:])==0
    new=Paper1Actor(6,16,include_return_soc=True,include_own_return_soc=True)
    new.load_state_dict(state,strict=True)
    torch.testing.assert_close(old.categorical_logits(obs[:,:18]),new.categorical_logits(obs))
    assert meta["actor_input_dim"]==21 and meta["actor_input_order"][-1]=="own_return_energy_soc"


def test_soc21_joint_updates_and_resumes_exactly_without_changing_critic():
    env=yaml.safe_load((ROOT/"configs/default.yaml").read_text())
    cfg=yaml.safe_load((ROOT/"configs/training_joint_topk_soc21_v3.yaml").read_text())
    env["reward"].update(cfg["upper_reward_overrides"])
    env["time"]["episode_upper_frames"]=3
    cfg["upper_rollout_steps"]=2
    cfg["sac"].update(hidden=16,heads=2,layers=1,batch_size=2,replay_capacity=64,random_steps=0,learning_starts=0)
    cfg["mappo"].update(hidden=16,transfer_actor_hidden=16,epochs=1,minibatch_frames=2)
    trainer=JointTrainer(env,cfg)
    assert trainer.mappo.actor.net[0].in_features==21
    assert trainer.mappo.critic[0].in_features==38
    trainer.step()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/"state.pt"
        trainer.save(path)
        expected=trainer.step()
        loaded=JointTrainer.load(path)
        assert loaded.mappo.actor.net[0].in_features==21
        assert loaded.step()==expected


def test_missing_own_feature_flag_preserves20_dimension_checkpoint_architecture():
    from jv_uav.rl.mappo import MAPPO
    cfg=yaml.safe_load((ROOT/"configs/training_joint_topk_soc_v2.yaml").read_text())["mappo"]
    model=MAPPO(6,cfg,"cpu")
    assert model.actor.net[0].in_features==20 and model.critic[0].in_features==38
