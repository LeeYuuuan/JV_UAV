"""Separate battery and return-energy observations with legacy compatibility."""
from pathlib import Path
import copy
import tempfile
import numpy as np
import torch
import yaml
from test_core import ROOT, build_env
from jv_uav.rl.observations import upper_observation
from jv_uav.rl.transfer import Paper1Actor, read_paper1_actor
from jv_uav.rl.trainer import JointTrainer


def test_current_soc_and_return_soc_keep_physical_units_and_zone_zero_cost():
    cfg, scene, energy, _, _ = build_env()
    scene.uav_pos[2] = scene.airship_pos + [1000., 0.]
    scene.uav_status[:2] = [1, 2]
    scene.uav_battery[:] = [.8, .7, .6, .5, .4, .3]
    obs = upper_observation(scene, energy, "current_soc_return_soc")
    assert obs["actor"].shape == (6, 24) and obs["critic"].shape == (38,)
    cost = energy.return_trip(scene.uav_pos, scene.airship_pos).energy_frac
    cost[:2] = 0
    np.testing.assert_array_equal(obs["actor"][:, :6], np.tile(scene.uav_battery, (6, 1)))
    np.testing.assert_array_equal(obs["actor"][:, -6:], np.tile(cost, (6, 1)))
    np.testing.assert_array_equal(obs["critic"][-6:], cost)
    np.testing.assert_array_equal(obs["critic"][:30].reshape(6, 5)[:, 0], scene.uav_battery)
    model = Paper1Actor(6, 16, include_return_soc=True)
    features = model.old_features(torch.from_numpy(obs["actor"]))
    assert features.shape == (6, 20)
    torch.testing.assert_close(features[:, -6:], torch.from_numpy(np.tile(cost, (6,1))))
    torch.testing.assert_close(features[:, 0], torch.from_numpy(scene.uav_battery))
    torch.testing.assert_close(features[:, 5:7], torch.from_numpy(obs["actor"][:,16:18]))


def test_equal_arrival_margin_states_are_distinguishable_in_new_mode():
    _, scene, energy, _, _ = build_env()
    scene.uav_pos[0] = scene.airship_pos + [400.,0.]
    first_cost = energy.return_trip(scene.uav_pos,scene.airship_pos).energy_frac[0]
    scene.uav_battery[0] = .5 + first_cost
    old_a = upper_observation(scene,energy,"arrival_soc")
    new_a = upper_observation(scene,energy,"current_soc_return_soc")
    scene.uav_pos[0] = scene.airship_pos + [1400.,0.]
    second_cost = energy.return_trip(scene.uav_pos,scene.airship_pos).energy_frac[0]
    scene.uav_battery[0] = .5 + second_cost
    old_b = upper_observation(scene,energy,"arrival_soc")
    new_b = upper_observation(scene,energy,"current_soc_return_soc")
    np.testing.assert_allclose(old_a["actor"],old_b["actor"],atol=1e-7)
    assert not np.array_equal(new_a["actor"],new_b["actor"])


def test_old_actor_transfer_zero_columns_preserves_current_soc_policy():
    _, scene, energy, _, _ = build_env()
    obs = upper_observation(scene,energy,"current_soc_return_soc")
    actor = Paper1Actor(6,16)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/"old.pt"
        torch.save({"actor":actor.state_dict()},path)
        state,metadata = read_paper1_actor(path,6,"current_soc_return_soc")
    assert state["net.0.weight"].shape == (16,20)
    assert torch.count_nonzero(state["net.0.weight"][:,14:]) == 0
    new_actor = Paper1Actor(6,16,include_return_soc=True)
    new_actor.load_state_dict(state,strict=True)
    raw = torch.from_numpy(obs["actor"])
    torch.testing.assert_close(actor.categorical_logits(raw[:,:18]),new_actor.categorical_logits(raw))
    assert metadata["soc_mapping"] == "current_soc_plus_separate_return_cost"
    assert metadata["actor_input_dim"] == 20
    new_actor.categorical_logits(raw).sum().backward()
    assert new_actor.net[0].weight.grad[:,14:].abs().sum() > 0


def test_soc_return_trainer_updates_and_resumes_exactly_with_top20():
    env = yaml.safe_load((ROOT/"configs/default.yaml").read_text())
    cfg = yaml.safe_load((ROOT/"configs/training_joint_topk_soc_v2.yaml").read_text())
    env["reward"].update(cfg["upper_reward_overrides"])
    env["time"]["episode_upper_frames"] = 3
    cfg["upper_rollout_steps"] = 2
    cfg["sac"].update(hidden=16,heads=2,layers=1,batch_size=2,replay_capacity=64,random_steps=0,learning_starts=0)
    cfg["mappo"].update(hidden=16,transfer_actor_hidden=16,epochs=1,minibatch_frames=2)
    trainer = JointTrainer(env,cfg)
    assert trainer.env.low_runner.low_env.reward_model.k == 20
    assert trainer.mappo.actor.net[0].in_features == 20
    assert trainer.mappo.critic[0].in_features == 38
    row=trainer.step()
    assert row["upper_input_soc"] == row["battery_start"]
    assert len(row["upper_input_return_soc"]) == 6
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/"state.pt"
        trainer.save(path)
        expected=trainer.step()
        resumed=JointTrainer.load(path)
        assert resumed.step() == expected
        for key,value in trainer.mappo.actor.state_dict().items():
            torch.testing.assert_close(value,resumed.mappo.actor.state_dict()[key],rtol=0,atol=0)


def test_credit_diagnostics_preserves_optimizer_updates_and_rng():
    from test_rl import trainer_config
    env, cfg = trainer_config()
    a = JointTrainer(env,cfg)
    a.step()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/"a.pt"
        a.save(path)
        expected=a.step()
        expected_rng=torch.get_rng_state().clone()
        b=JointTrainer.load(path)
        b.cfg["mappo"]["credit_diagnostics"]=True
        b.mappo.cfg["credit_diagnostics"]=True
        actual=b.step()
        credits=actual.pop("upper_credit_trace")
        assert len(credits)==2 and credits[0]["upper_steps"]==1
        assert credits[1]["upper_steps"]==2
        assert "advantage_normalized" in credits[0]
        expected.pop("upper_credit_trace")
        assert actual==expected
        torch.testing.assert_close(expected_rng,torch.get_rng_state(),rtol=0,atol=0)
        for group in ("actor","critic"):
            for key,value in getattr(a.mappo,group).state_dict().items():
                torch.testing.assert_close(value,getattr(b.mappo,group).state_dict()[key],rtol=0,atol=0)
