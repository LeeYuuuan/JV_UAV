"""Identity remains stable across role changes and breaks identical-token symmetry."""
import numpy as np
import torch

from test_rl import trainer_config, raw_obs
from jv_uav.rl.trainer import JointTrainer
from jv_uav.rl.replay import pack_observations


def test_identity_uses_global_fleet_ids_and_preserves_battery():
    env, cfg = trainer_config()
    trainer = JointTrainer(env, cfg)
    first, second = raw_obs([0, 2, 5]), raw_obs([5])
    batch = pack_observations([first, second], trainer.rms, 2400, 'cpu', 6)
    np.testing.assert_allclose(batch['tokens'][0, :, 3], [0, 0.4, 1])
    assert batch['tokens'][1, 0, 3] == 1
    np.testing.assert_allclose(batch['tokens'][0, :, 2], 0.7)
    assert not batch['tokens'][1, 1:].any()


def test_identity_distinguishes_colocated_equal_battery_uavs():
    env, cfg = trainer_config()
    trainer = JointTrainer(env, cfg)
    obs = trainer.env.scene.observe_lower()
    action = trainer.sac.act(obs, trainer.rms, deterministic=True)
    assert np.max(np.abs(action - action[0])) > 1e-4
    permutation = np.array([5, 0, 2, 1, 4, 3])
    reordered = {key: value.copy() for key, value in obs.items()}
    for key in ['active_uav_ids', 'active_uav_features']:
        reordered[key] = reordered[key][permutation]
    actual = trainer.sac.act(reordered, trainer.rms, deterministic=True)
    np.testing.assert_allclose(actual, action[permutation], atol=1e-6)


def test_legacy_checkpoint_configuration_keeps_three_feature_architecture():
    env, cfg = trainer_config()
    cfg['sac'].pop('include_uav_id', None)
    original = JointTrainer(env, cfg)
    restored = JointTrainer(env, cfg)
    restored.sac.load_state_dict(original.sac.state_dict())
    assert restored.sac.identity_fleet_size is None
    obs = original.env.scene.observe_lower()
    a = original.sac.act(obs, original.rms, deterministic=True)
    b = restored.sac.act(obs, restored.rms, deterministic=True)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, np.broadcast_to(a[0], a.shape), atol=1e-6)
