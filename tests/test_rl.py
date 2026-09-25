"""Behavioral checks for the confirmed hierarchy and training schedule."""
from pathlib import Path
import copy

import numpy as np
import torch
import yaml

from test_core import build_env, ROOT
from jv_uav.rl.observations import RunningMeanStd, upper_observation
from jv_uav.rl.replay import pack_observations
from jv_uav.rl.trainer import JointTrainer
from jv_uav.rl.mappo import gae

torch.set_num_threads(1)


def trainer_config():
    env, *_ = build_env()
    cfg = yaml.safe_load((ROOT / "configs/training.yaml").read_text())
    cfg.update(upper_rollout_steps=2)
    cfg["sac"].update(hidden=16, heads=2, layers=1, batch_size=2, replay_capacity=64, random_steps=0, learning_starts=0)
    cfg["mappo"].update(hidden=16, epochs=1, minibatch_frames=2)
    env["time"]["episode_upper_frames"] = 3
    return env, cfg


def raw_obs(ids):
    ids = np.array(ids, np.int64)
    return {"active_uav_ids": ids, "active_uav_features": np.array([[100+i*30, 200+i*10, 0.7] for i in ids], np.float32).reshape(-1, 3), "sensor_last_visit_sec": np.arange(yaml.safe_load((ROOT / "configs/default.yaml").read_text())["sensors"]["count"], dtype=np.float32)}


def test_running_stats_are_shared_and_frozen_for_evaluation():
    rms = RunningMeanStd(min_std=1)
    rms.update([0, 2])
    rms.update([4, 6])
    assert rms.count == 4 and rms.mean == 3 and rms.var == 5
    np.testing.assert_allclose(rms.normalize([3]), [0])
    state = copy.deepcopy(rms.state_dict())
    rms.training = False
    rms.update([1000])
    assert rms.count == state['count'] and rms.mean == state['mean']


def test_upper_observation_contains_only_confirmed_features():
    _, scene, energy, _, _ = build_env()
    scene.uav_status[0] = 2
    scene.uav_status[1] = 1
    scene.uav_pos[2] = scene.airship_pos + [1800, 0]
    scene.uav_battery[2] = .1
    obs = upper_observation(scene, energy)
    assert obs['actor'].shape == (6, 18)
    assert obs['critic'].shape == (32,)
    expected = scene.uav_battery - energy.return_trip(scene.uav_pos, scene.airship_pos).energy_frac
    expected[:2] = scene.uav_battery[:2]
    np.testing.assert_array_equal(obs['actor'][:, :6], np.tile(expected, (6, 1)))
    np.testing.assert_array_equal(obs['critic'][:30].reshape(6, 5)[:, 0], expected)
    assert expected[2] < 0
    np.testing.assert_array_equal(obs['actor'][:, 6:12], np.eye(6))
    np.testing.assert_array_equal(obs['actor'][0, 12:16], [0, 0, 1, 0])
    np.testing.assert_allclose(obs['actor'][:, 16:18], 0.5)
    scene.sensor_backlog += 999
    for key, value in upper_observation(scene, energy).items():
        np.testing.assert_array_equal(value, obs[key])


def test_attention_ignores_padding_and_is_permutation_equivariant():
    env, cfg = trainer_config()
    trainer = JointTrainer(env, cfg)
    obs = raw_obs([0, 2, 5])
    batch = pack_observations([obs], trainer.rms, 2400, 'cpu', trainer.sac.identity_fleet_size)
    with torch.no_grad():
        action, _ = trainer.sac.actor(batch, True)
        q = trainer.sac.critic(batch, action)[0]
        permutation = [2, 0, 1]
        permuted = {k: (v[:, permutation] if k in ['tokens','mask','ids'] else v) for k,v in batch.items()}
        perm_action, _ = trainer.sac.actor(permuted, True)
        np.testing.assert_allclose(perm_action.numpy(), action[:, permutation].numpy(), atol=1e-6)
        torch.testing.assert_close(trainer.sac.critic(permuted, perm_action)[0], q)
        padded = pack_observations([obs, raw_obs([0,1,2,3,4,5])], trainer.rms, 2400, 'cpu', trainer.sac.identity_fleet_size)
        padded['tokens'][0, 3:] = 10000
        padded_action, _ = trainer.sac.actor(padded, True)
        torch.testing.assert_close(padded_action[0,:3], action[0], atol=1e-6, rtol=1e-5)
        assert not padded_action[0,3:].any()


def test_raw_replay_masks_and_terminal_empty_next_set():
    env, cfg = trainer_config()
    trainer = JointTrainer(env, cfg)
    obs, next_ = raw_obs([0,2,5]), raw_obs([1,4])
    trainer.replay.add(obs, np.zeros((3,2)), 1.0, next_, False)
    trainer.replay.add(next_, np.zeros((2,2)), -1000, raw_obs([]), True)
    obs['active_uav_features'][:] = -999
    assert trainer.replay.items[0][0]['active_uav_features'][0,0] == 100
    batch, actions, reward, future, done = trainer.replay.sample(2, trainer.rms, 2400, 'cpu')
    assert batch['mask'].shape == (2,3) and future['mask'].shape == (2,2)
    assert sorted(batch['mask'].sum(1).tolist()) == [2,3]
    assert sorted(future['mask'].sum(1).tolist()) == [0,2]
    assert set(batch['ids'][batch['mask']].tolist()) == {0,1,2,4,5}
    losses = trainer.sac.update(trainer.replay, trainer.rms)
    assert all(np.isfinite(v) for v in losses.values())
    assert trainer.rms.count == 0


def test_frame_boundary_waits_for_next_charging_allocation():
    env, cfg = trainer_config()
    cfg['sac']['learning_starts'] = 99999
    cfg['upper_rollout_steps'] = 100
    trainer = JointTrainer(env, cfg)
    actions = iter([np.array([1,1,0,0,0,0]), np.array([0,0,1,1,0,0])])
    trainer.mappo.act = lambda obs: (next(actions), np.zeros(6), 0.0)
    trainer.step()
    assert trainer.pending is not None and len(trainer.replay) == 9
    old_state = trainer.pending[0]['active_uav_features'].copy()
    trainer.step()
    boundary = trainer.replay.items[9]
    np.testing.assert_array_equal(boundary[0]['active_uav_ids'], [2,3,4,5])
    np.testing.assert_array_equal(boundary[3]['active_uav_ids'], [0,1,4,5])
    np.testing.assert_array_equal(boundary[0]['active_uav_features'], old_state)
    assert not boundary[4]


def test_horizon_and_return_death_never_bootstrap():
    for battery in [1.0, 0.44]:
        env, cfg = trainer_config()
        env['time']['episode_upper_frames'] = 1
        cfg['sac']['learning_starts'] = 99999
        trainer = JointTrainer(env, cfg)
        trainer.env.scene.uav_battery.fill(battery)
        trainer.mappo.act = lambda obs: (np.zeros(6, np.int8), np.zeros(6), 0.0)
        # Fixed zero actions make the exact return-failure energy case deterministic.
        trainer.sac.act = lambda obs, rms: np.zeros((len(obs['active_uav_ids']),2),np.float32)
        row = trainer.step()
        assert trainer.pending is None
        assert trainer.replay.items[-1][4]
        assert len(trainer.replay) == 10
        assert not any(x[4] for x in trainer.replay.items[:-1])
        if battery == 0.44:
            assert row['terminated'] and not row['truncated']
            assert row['episode_metrics']['lower_reward_terms_sum']['return_failure'] == -3000
            assert trainer.replay.items[-1][2] <= -3000
        else:
            assert row['truncated'] and not row['terminated']


def test_schedule_updates_each_ready_transition_without_late_reduction():
    env, cfg = trainer_config()
    trainer = JointTrainer(env, cfg)
    trainer.sac_updates = 250000
    previous = trainer.sac_updates
    for _ in range(4):
        before = len(trainer.replay)
        row = trainer.step()
        inserted = len(trainer.replay) - before
        # Only the first transition is skipped while the batch is unavailable.
        expected = inserted - int(before == 0)
        assert trainer.sac_updates - previous == expected
        previous = trainer.sac_updates
        assert row['phase'] == 'joint'
    assert trainer.mappo_updates == 2
    cfg['sac']['learning_starts'] = 99999
    trainer = JointTrainer(env, cfg)
    trainer.step()
    assert trainer.sac_updates == 0


def test_gae_stops_at_episode_boundaries_and_bootstraps_rollout_only():
    adv, returns = gae([1,2,3], [0,0,0], [False,True,False], 4, 1, 1)
    np.testing.assert_array_equal(returns, [3,2,7])


def test_checkpoint_resumes_pending_transition_and_partial_rollout_exactly():
    env, cfg = trainer_config()
    trainer = JointTrainer(env, cfg)
    trainer.step()
    # The path is controlled by the test caller's writable project test directory.
    path = ROOT / 'tests' / '_rl_resume_test.pt'
    try:
        trainer.save(path)
        expected = trainer.step()
        expected_weights = copy.deepcopy(trainer.sac.actor.state_dict())
        resumed = JointTrainer.load(path)
        assert resumed.pending is not None and len(resumed.rollout) == 1
        actual = resumed.step()
        assert actual == expected
        for key, value in resumed.sac.actor.state_dict().items():
            torch.testing.assert_close(value, expected_weights[key], atol=0, rtol=0)
        assert resumed.rms.state_dict() == trainer.rms.state_dict()
        assert resumed.env.trace.to_serializable() == trainer.env.trace.to_serializable()
    finally:
        path.unlink(missing_ok=True)


def test_evaluation_freezes_statistics_rng_and_training_state():
    env, cfg = trainer_config()
    trainer = JointTrainer(env, cfg)
    trainer.step()
    stats = copy.deepcopy(trainer.rms.state_dict())
    state = torch.get_rng_state().clone()
    trace = copy.deepcopy(trainer.env.trace.to_serializable())
    weights = copy.deepcopy(trainer.sac.actor.state_dict())
    a = trainer.evaluate(555, render=False)
    b = trainer.evaluate(555, render=False)
    assert a == b
    assert trainer.rms.state_dict() == stats
    assert trainer.env.trace.to_serializable() == trace
    torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
    for key, value in trainer.sac.actor.state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
