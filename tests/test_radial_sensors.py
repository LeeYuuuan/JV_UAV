"""Radial map quotas and deterministic generation are part of the environment."""
import copy
import numpy as np
from test_core import ROOT
from jv_uav import load_config
from jv_uav.scene import Scene
from jv_uav.sensor_generator import generate_sensor_positions, generate_radial_sensor_distribution


def test_radial_quotas_bounds_and_local_scatter_across_seeds():
    cfg = load_config(ROOT / 'configs/default.yaml')
    for seed in range(20):
        cfg['seeds']['map_seed'] = seed
        data = generate_radial_sensor_distribution(cfg)
        p = data['positions']
        radius = np.linalg.norm(p.astype(float) - cfg['world']['airship_xy_m'], axis=1)
        assert p.shape == (150, 2) and p.dtype == np.float32
        assert np.count_nonzero(radius <= 1800) == 100
        assert np.count_nonzero(radius > 1800) == 50
        assert np.all((radius[~data['inside']] >= 2000) & (radius[~data['inside']] <= 2400))
        assert np.all((p >= 0) & (p <= 4000))
        assert data['is_scattered'].sum() == 9
        assert np.sum(data['is_scattered'] & data['inside']) == 6
        assert np.sum(data['is_scattered'] & ~data['inside']) == 3
        assert np.all(data['cluster_ids'][data['is_scattered']] == -1)
        np.testing.assert_array_equal(np.bincount(data['cluster_ids'][~data['is_scattered']]), [19,19,19,19,18,24,23])
        assert np.max(np.linalg.norm(p[~data['is_scattered']]-data['centers'][data['cluster_ids'][~data['is_scattered']]], axis=1)) <= 300.001
        np.testing.assert_array_equal(p, generate_sensor_positions(cfg))
    np.testing.assert_array_equal(Scene(cfg).sensor_pos, generate_sensor_positions(cfg))


def test_missing_distribution_preserves_legacy_mode():
    cfg = load_config(ROOT / 'configs/default.yaml')
    cfg['sensors'].pop('distribution')
    implicit = generate_sensor_positions(cfg)
    explicit = copy.deepcopy(cfg)
    explicit['sensors']['distribution'] = 'cluster_uniform'
    np.testing.assert_array_equal(implicit, generate_sensor_positions(explicit))


def test_seed42_manual_adjustment_changes_only_two_outer_scatter_points():
    cfg = load_config(ROOT / 'configs/default.yaml')
    adjusted = generate_radial_sensor_distribution(cfg)
    cfg['sensors'].pop('position_overrides_by_seed')
    original = generate_radial_sensor_distribution(cfg)
    changed = np.flatnonzero(np.any(adjusted['positions'] != original['positions'], axis=1))
    np.testing.assert_array_equal(changed, [42, 106])
    np.testing.assert_array_equal(adjusted['positions'][changed], [[3300, 3600], [3700, 3200]])
    assert adjusted['is_scattered'][changed].all()
    assert not adjusted['inside'][changed].any()


def test_invalid_radial_bounds_fail_instead_of_hanging():
    cfg = load_config(ROOT / 'configs/default.yaml')
    cfg['sensors']['outer_radius_range_m'] = [1700, 2400]
    try:
        generate_sensor_positions(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('outer radius must exceed the partition radius')
