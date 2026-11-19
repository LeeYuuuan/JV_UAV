"""Reward termination and staged joint updates across real frame boundaries."""
import copy
import tempfile
from pathlib import Path
import numpy as np
import torch
from test_rl import trainer_config
from jv_uav.rl.trainer import JointTrainer
from jv_uav.rl.training_env import build_training_env


def config():
    env, cfg = trainer_config()
    env["reward"].update(upper_reward_mode="paper1_serving_backlog", upper_reward_divisor=10.)
    cfg["lower_reward"] = dict(mode="topk", k=5, rho=.5, scale=300., offset=0., terminal_tail=True)
    cfg["update_schedule"] = dict(mode="staged", switch_after_sac_updates=2, late_sac_updates_per_mappo=1)
    return env, cfg


def test_late_updates_only_at_upper_update_and_resume_exactly():
    env, cfg = config()
    trainer = JointTrainer(env, cfg)
    first = trainer.step()
    assert trainer.sac_updates == 2 and not first["mappo_updated"]
    second = trainer.step()
    assert trainer.sac_updates == 3 and second["mappo_updated"]
    trainer.step()
    assert trainer.sac_updates == 3
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "checkpoint.pt"
        trainer.save(path)
        expected = trainer.step()
        expected_params = copy.deepcopy(trainer.sac.actor.state_dict())
        resumed = JointTrainer.load(path)
        actual = resumed.step()
        assert actual["upper_reward"] == expected["upper_reward"]
        assert actual["lower_reward"] == expected["lower_reward"]
        assert resumed.sac_updates == trainer.sac_updates == 4
        assert resumed.mappo_updates == trainer.mappo_updates
        for key, value in expected_params.items():
            torch.testing.assert_close(resumed.sac.actor.state_dict()[key], value, rtol=0, atol=0)


def test_failure_tail_is_in_callback_once_without_extra_physical_steps_or_rng():
    env_cfg, cfg = config()
    env_cfg["uavs"]["initial_battery_frac"] = .001
    env_cfg["time"]["episode_upper_frames"] = 100
    env = build_training_env(env_cfg, cfg)
    env.reset()
    rewards = []
    env.low_runner.on_step = lambda obs, act, r: rewards.append((r.reward, dict(r.reward_terms)))
    _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.int8))
    assert terminated and not truncated and env.scene.low_step_total == 1
    assert len(rewards) == 1
    value, terms = rewards[0]
    assert terms["terminal_future_backlog"] < -100
    assert value == sum(terms.values())
    assert value == info["low_frame"].steps[0].reward
    cfg["lower_reward"]["terminal_tail"] = False
    control = build_training_env(env_cfg, cfg)
    control.reset()
    control.step(np.zeros(6, dtype=np.int8))
    assert env.scene.rng.bit_generator.state == control.scene.rng.bit_generator.state
    np.testing.assert_array_equal(env.scene.sensor_backlog, control.scene.sensor_backlog)


def test_topk_actual_collection_has_no_overlap_double_count():
    from types import SimpleNamespace
    from jv_uav.rl.training_env import TopKReward
    env, _ = config()
    reward = TopKReward(env, k=2, rho=.5, scale=300.)
    result = SimpleNamespace(packets_pre_service=np.array([100., 80., 30.]),
        collected_per_sensor=np.array([100., 0., 30.]), system_max_post_service=80.,
        per_uav_owned_max_pre_service=np.zeros(6), oob_mask=np.zeros(6, bool))
    value, terms = reward(result)
    assert np.isclose(terms["topk_clear"], .5*80*(100/180)/300)
    assert np.isclose(value, -80/300 + .5*80*(100/180)/300)
    assert "covered_pnorm" not in terms


def test_joint_topk_evaluation_freezes_rng_replay_and_normalization():
    env, cfg = config()
    trainer = JointTrainer(env, cfg)
    trainer.step()
    before = (copy.deepcopy(trainer.rms.state_dict()), torch.get_rng_state().clone(),
              len(trainer.replay), trainer.sac_updates, trainer.mappo_updates)
    first = trainer.evaluate(1001, render=False)
    second = trainer.evaluate(1001, render=False)
    assert first == second
    assert first["lower_diagnostics_complete"]
    assert np.isclose(first["lower_return"], sum(first["lower_reward_terms_sum"].values()))
    assert "topk_clear" in first["lower_reward_terms_sum"]
    assert before[0] == trainer.rms.state_dict()
    torch.testing.assert_close(before[1], torch.get_rng_state(), rtol=0, atol=0)
    assert before[2:] == (len(trainer.replay), trainer.sac_updates, trainer.mappo_updates)


def test_final_return_failure_tail_and_penalty_are_in_terminal_replay():
    env, cfg = config()
    env["time"]["episode_upper_frames"] = 100
    cfg["sac"]["learning_starts"] = 99999
    trainer = JointTrainer(env, cfg)
    trainer.env.scene.uav_battery.fill(.44)
    trainer.mappo.act = lambda obs, **kwargs: (np.zeros(6, np.int8), np.zeros(6), 0., np.zeros(6))
    trainer.sac.act = lambda obs, rms: np.zeros((len(obs["active_uav_ids"]), 2), np.float32)
    row = trainer.step()
    assert row["terminated"] and row["frame_low_steps"] == 10
    assert row["lower_reward_terms"]["return_failure"] == -3000.
    assert row["lower_reward_terms"]["terminal_future_backlog"] < 0.
    assert trainer.pending is None and len(trainer.replay) == 10
    assert trainer.replay.items[-1][4]
    assert not any(item[4] for item in trainer.replay.items[:-1])
    assert np.isclose(sum(item[2] for item in trainer.replay.items), row["lower_reward"])
