"""Reward incentives, terminal accounting and spatial warm-up regressions."""
import copy
from pathlib import Path

import numpy as np
import torch

from test_core import build_env, ROOT
from test_rl import trainer_config
from jv_uav.upper_env import UpperReward
from jv_uav.rl.exploration import WaypointExploration
from jv_uav.rl.trainer import JointTrainer


def test_bounded_backlog_monotonic_scale_and_no_negative_survival_tail():
    cfg, *_ = build_env()
    reward = UpperReward(cfg)
    assert reward.backlog_reference == 3000
    values = [reward(2, b, 0)[0] for b in [0, 300, 3000, 30000, 1e12]]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert min(values) > 1
    assert reward(2, 3000, 0)[1]['frame_mean_system_max_post'] == -0.5
    # Even the highest instantaneous serving benefit cannot turn death into
    # positive return. Any safe continuation under this default fleet is > 0.
    for deaths in [0, 1, 6]:
        assert reward(6, 0, deaths, episode_failed=True)[0] < 0
    changed = copy.deepcopy(cfg)
    changed['sensors']['arrival_rate_per_sec'] *= 2
    assert UpperReward(changed)(2, 6000, 0)[0] == reward(2, 3000, 0)[0]
    changed['reward']['upper_backlog_mode'] = 'linear'
    assert UpperReward(changed)(2, 3000, 0)[0] == -2998


def test_episode_failure_applies_to_lower_failure_not_timeout_or_twice():
    _, scene, _, _, env = build_env()
    scene.uav_battery.fill(0.44)
    _, reward, terminated, _, info = env.step(np.zeros(6, dtype=np.int8))
    assert terminated and reward < 0
    assert info['reward_terms']['dead'] == 0
    assert info['reward_terms']['episode_failure'] == -1000
    _, scene, _, _, env = build_env()
    scene.uav_battery[0] = 0.001
    _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.int8))
    assert terminated
    assert info['reward_terms']['dead'] == -1000
    assert info['reward_terms']['episode_failure'] == 0
    _, scene, _, _, env = build_env()
    env.max_frames = 1
    _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.int8))
    assert truncated and not terminated
    assert info['reward_terms']['episode_failure'] == 0


def test_return_penalty_grows_with_failed_distance_and_only_on_final_step():
    totals = []
    for distance, battery in [(0, 0.44), (900, 0.50), (1800, 0.55)]:
        _, scene, _, _, env = build_env()
        scene.uav_pos[0] = scene.airship_pos + [distance, 0]
        scene.uav_battery[0] = battery
        _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.int8))
        assert terminated
        steps = info['low_frame'].steps
        assert len(steps) == 10
        assert all(s.reward_terms['return_distance'] == 0 for s in steps[:-1])
        np.testing.assert_array_equal(steps[-1].return_unsafe_ids, [0])
        terms = steps[-1].reward_terms
        assert terms['return_distance'] == -1000 * distance / 1800
        totals.append(terms['return_failure'] + terms['return_distance'])
    assert totals == [-1000, -1500, -2000]
    _, scene, _, _, env = build_env()
    scene.uav_pos[0] = scene.airship_pos + [1800, 0]
    _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.int8))
    assert not terminated
    assert all(s.reward_terms['return_distance'] == 0 for s in info['low_frame'].steps)


def test_waypoint_exploration_spreads_uavs_and_preserves_action_mapping():
    _, scene, *_ = build_env()
    explorer = WaypointExploration(scene, np.random.default_rng(123), random_fraction=0)
    obs = scene.observe_lower()
    action = explorer(obs)
    assert action.shape == (6, 2)
    assert np.all(np.linalg.norm(action, axis=1) <= 1.000001)
    angles = np.unwrap(np.arctan2(action[:, 1], action[:, 0]))
    assert np.ptp(angles) > np.pi
    targets = explorer.targets.copy()
    obs['active_uav_ids'] = np.array([5, 0, 2])
    obs['active_uav_features'] = obs['active_uav_features'][[5, 0, 2]]
    action = explorer(obs)
    assert action.shape == (3, 2)
    np.testing.assert_array_equal(explorer.valid, [1, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(explorer.targets[[5, 0, 2]], targets[[5, 0, 2]])
    delta = targets[[5, 0, 2]] - obs['active_uav_features'][:, :2]
    assert np.all((delta * action).sum(axis=1) > 0)
    explorer.reset()
    assert not explorer.valid.any() and explorer.phase is None


def test_checkpoint_during_waypoint_warmup_resumes_exactly():
    env, cfg = trainer_config()
    cfg['sac'].update(random_steps=1000, warmup_mode='waypoint')
    trainer = JointTrainer(env, cfg)
    trainer.step()
    path = ROOT / 'tests/_waypoint_resume.pt'
    try:
        trainer.save(path)
        expected = trainer.step()
        expected_weights = copy.deepcopy(trainer.sac.actor.state_dict())
        resumed = JointTrainer.load(path)
        assert resumed.step() == expected
        for key, value in resumed.sac.actor.state_dict().items():
            torch.testing.assert_close(value, expected_weights[key], atol=0, rtol=0)
        np.testing.assert_array_equal(resumed.exploration.targets, trainer.exploration.targets)
        assert resumed.env.trace.to_serializable() == trainer.env.trace.to_serializable()
    finally:
        path.unlink(missing_ok=True)


def test_old_schedule_settings_cannot_restore_frequency_reduction():
    env, cfg = trainer_config()
    cfg.pop('sac_updates_per_step', None)
    cfg.update(switch_after_sac_updates=1, early_sac_updates_per_step=1, late_sac_updates_per_mappo=1)
    trainer = JointTrainer(env, cfg)
    trainer.step()
    assert trainer.sac_updates > 1
    assert 'switch_after_sac_updates' not in trainer.cfg
