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


def test_linear_backlog_scale_has_no_cap():
    cfg, *_ = build_env()
    reward = UpperReward(cfg)
    for backlog, cost in [(0, 0), (300, -0.15), (3000, -1.5), (15000, -7.5), (30000, -15), (60000, -30)]:
        value, terms = reward(2, backlog, 0, charging_count=2, waiting_count=2)
        assert np.isclose(terms['frame_mean_system_max_post'], cost)
        assert np.isclose(value, 10 + cost)
    changed = copy.deepcopy(cfg)
    changed['sensors']['arrival_rate_per_sec'] *= 2
    assert UpperReward(changed)(2, 6000, 0)[1]['frame_mean_system_max_post'] == -3
    # Old saved configs retain the previous bounded calculation.
    changed['reward'].update(upper_backlog_mode='bounded', upper_backlog_reference_frames=10)
    assert UpperReward(changed)(2, 6000, 0)[1]['frame_mean_system_max_post'] == -0.5


def test_upper_deaths_and_shared_return_failures_have_separate_costs():
    _, scene, _, _, env = build_env()
    scene.uav_battery.fill(0.44)
    _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.int8))
    assert terminated
    assert info['reward_terms']['dead'] == 0
    assert info['reward_terms']['lower_return_failure'] == -90
    assert info['reward_terms']['episode_failure'] == 0
    assert info['low_frame'].steps[-1].reward_terms['return_failure'] < 0
    _, scene, _, _, env = build_env()
    scene.uav_battery[0] = 0.001
    _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.int8))
    assert terminated
    assert info['reward_terms']['dead'] == -20
    assert info['reward_terms']['lower_return_failure'] == 0
    assert info['low_frame'].steps[-1].reward_terms['return_failure'] == 0
    assert info['low_frame'].steps[-1].reward_terms['return_distance'] == 0
    assert info['reward_terms']['episode_failure'] == 0
    _, scene, _, _, env = build_env()
    env.max_frames = 1
    _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.int8))
    assert truncated and not terminated
    assert info['reward_terms']['lower_return_failure'] == 0
    assert info['reward_terms']['episode_failure'] == 0


def test_occupancy_cost_uses_assigned_roles_not_requests_or_released_roles():
    _, _, _, _, env = build_env()
    # Nonzero costs remain supported for old checkpoints and custom experiments.
    env.upper_reward.charging_weight = 1.0
    env.upper_reward.waiting_weight = 0.5
    _, _, _, _, info = env.step(np.ones(6, dtype=np.int8))
    assert info['reward_terms']['serving'] == 10
    assert info['reward_terms']['charging'] == -2
    assert info['reward_terms']['waiting'] == -1


def test_default_charging_and_waiting_have_no_direct_penalty():
    _, _, _, _, env = build_env()
    _, reward, _, _, info = env.step(np.ones(6, dtype=np.int8))
    terms = info['reward_terms']
    assert terms['charging'] == 0 and terms['waiting'] == 0
    assert terms['serving'] == 10
    assert np.isclose(reward, sum(terms.values()))


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
        assert info['reward_terms']['lower_return_failure'] == -15
        assert info['reward_terms']['dead'] == 0
        terms = steps[-1].reward_terms
        assert terms['return_distance'] == -500 * distance / 1800
        totals.append(terms['return_failure'] + terms['return_distance'])
    assert totals == [-500, -750, -1000]
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


def test_lower_reward_units_and_legacy_compatibility():
    from types import SimpleNamespace
    from jv_uav.low_env import LowReward
    cfg, *_ = build_env()
    result = SimpleNamespace(per_uav_owned_max_pre_service=np.array([3000., 3000., 0, 0, 0, 0]),
                             collected_per_sensor=np.array([3000.,3000.,1500.]),
                             system_max_post_service=3000., oob_mask=np.array([True, False]))
    reward, terms = LowReward(cfg)(result, final_return_failure_count=1, final_return_distance_sum_m=1800)
    assert np.isclose(terms['collected_packets'], 7500 / 180)
    assert terms['system_max_post_service'] == -2.25
    assert terms['oob'] == -5
    assert terms['return_failure'] + terms['return_distance'] == -1000
    assert np.isclose(reward, 7500 / 180 - 2.25 - 5 - 1000)
    result.system_max_post_service = 1e12
    assert LowReward(cfg)(result)[1]['system_max_post_service'] == -(1e12 / 2000) ** 2
    cfg['reward'].pop('low_collection_mode')
    cfg['reward'].pop('low_collection_weight')
    cfg['reward'].pop('low_collection_scale_packets')
    cfg['reward'].pop('low_backlog_mode')
    cfg['reward'].pop('low_backlog_scale_packets')
    cfg['reward'].update(low_covered_max_sum_weight=0.05, low_system_max_after_weight=1)
    result.system_max_post_service = 3000
    _, terms = LowReward(cfg)(result)
    assert terms['covered_max_sum'] == 50
    assert terms['system_max_post_service'] == -3000


def test_default_entropy_stays_constant():
    env, cfg = trainer_config()
    trainer = JointTrainer(env, cfg)
    for frame in (0, 10000, 55000, 100000, 400000):
        assert trainer.mappo.entropy_coefficient(frame) == .05


def test_entropy_schedule_is_frame_based_and_resume_preserves_it():
    env, cfg = trainer_config()
    # Explicit legacy schedule remains supported for saved checkpoints.
    cfg['mappo'].update(entropy_end_coef=.005, entropy_decay_start_frame=10000,
                        entropy_decay_end_frame=100000)
    trainer = JointTrainer(env, cfg)
    for frame, expected in [(0, .05), (10000, .05), (55000, .0275), (100000, .005), (400000, .005)]:
        assert np.isclose(trainer.mappo.entropy_coefficient(frame), expected)
    trainer.upper_steps = 54999
    trainer.step()
    path = ROOT / 'tests/_entropy_resume.pt'
    try:
        trainer.save(path)
        expected = trainer.step()
        resumed = JointTrainer.load(path)
        assert resumed.step() == expected
        assert np.isclose(expected['mappo_entropy_coef'], trainer.mappo.entropy_coefficient(55001))
    finally:
        path.unlink(missing_ok=True)


def test_512_rollout_crosses_resets_without_resetting_at_update():
    from jv_uav.rl.mappo import gae
    env, cfg = trainer_config()
    env['time']['episode_upper_frames'] = 100
    cfg['upper_rollout_steps'] = 512
    cfg['sac'].update(random_steps=10000, learning_starts=10000)
    cfg['mappo'].update(epochs=3, minibatch_frames=128)
    trainer = JointTrainer(env, cfg)
    # Stationary service with recharge resets isolates rollout boundary behavior.
    trainer.env.low_runner.low_policy = lambda obs: np.zeros((len(obs['active_uav_ids']), 2), np.float32)
    def stationary_upper(obs):
        probabilities = trainer.mappo.request_probabilities(obs)
        return np.zeros(6, np.int8), np.log1p(-probabilities), trainer.mappo.value(obs)
    trainer.mappo.act = stationary_upper
    for frame in range(1, 513):
        trainer.env.scene.uav_battery.fill(1)
        row = trainer.step()
        if frame == 511:
            assert trainer.mappo_updates == 0 and len(trainer.rollout) == 511
            assert sum(x['done'] for x in trainer.rollout) == 5
    assert trainer.episodes == 5 and trainer.env.scene.upper_step == 12
    assert not trainer.episode_over and not trainer.rollout
    assert row['mappo_minibatches'] == 12 and trainer.mappo_updates == 1
    assert trainer.sac_updates == 0
    # Rewards after a reset must not leak into the previous episode's GAE.
    adv, _ = gae([1, 100], [0, 0], [True, False], 0, .99, .95)
    assert adv[0] == 1
