from dataclasses import replace
import copy

import numpy as np

from test_core import build_env
from jv_uav.low_env import LowEnv
from jv_uav.models import FullClearNearestService, LowestArrivalBatteryScheduler
from jv_uav.types import UAVStatus


def test_normalized_mapping_radial_limit_and_integer_rows():
    _, scene, _, _, env = build_env()
    scene.uav_status[:] = int(UAVStatus.WAITING)
    ids = np.array([0, 2, 5], dtype=np.int64)
    scene.uav_status[ids] = int(UAVStatus.SERVING)
    before = scene.uav_pos.copy()
    result = env.low_runner.low_env.step(np.array([[1, 1], [-0.5, 0], [0, 2]]))
    expected = np.array([[180 / np.sqrt(2), 180 / np.sqrt(2)], [-90, 0], [0, 180]])
    np.testing.assert_array_equal(result.serving_ids, ids)
    assert np.issubdtype(result.serving_ids.dtype, np.integer)
    np.testing.assert_allclose(result.requested_displacement[ids], expected, atol=1e-4)
    np.testing.assert_allclose(result.position_after[ids] - before[ids], expected, atol=1e-4)
    np.testing.assert_array_equal(result.position_after[[1, 3, 4]], before[[1, 3, 4]])
    np.testing.assert_allclose(result.move_time_sec[ids], [30, 15, 30], atol=1e-4)
    np.testing.assert_allclose(result.service_time_sec[ids], [30, 45, 30], atol=1e-4)
    np.testing.assert_allclose(result.normalized_action[ids], [[1, 1], [-0.5, 0], [0, 1]])


def test_boundary_clipping_distinguishes_requested_and_executed():
    _, scene, _, _, env = build_env()
    scene.uav_pos[0] = [scene.world_size - 10, 1200]
    actions = np.zeros((scene.num_uavs, 2))
    actions[0] = [1, 0]
    result = env.low_runner.low_env.step(actions)
    np.testing.assert_allclose(result.requested_displacement[0], [180, 0])
    np.testing.assert_allclose(result.executed_displacement[0], [10, 0])
    assert result.oob_mask[0]
    np.testing.assert_allclose(result.oob_overflow_distance_m[0], 170)


def test_poisson_arrivals_match_independent_vector_draws():
    _, scene, _, _, env = build_env()
    scene.arrival_rate[:] = np.linspace(0, 0.2, scene.num_sensors)
    reference = np.random.default_rng()
    reference.bit_generator.state = copy.deepcopy(scene.rng.bit_generator.state)
    before = scene.sensor_backlog.copy()
    expected = reference.poisson(scene.arrival_rate * scene.low_step_sec)
    result = env.low_runner.low_env.step(np.zeros((scene.num_uavs, 2)))
    np.testing.assert_array_equal(result.arrivals, expected)
    np.testing.assert_array_equal(result.packets_pre_service, before + expected)
    assert np.unique(result.arrivals).size > 1
    np.testing.assert_array_equal(
        result.packets_post_service + result.collected_per_sensor,
        result.packets_pre_service,
    )


def test_nearest_ownership_noncontiguous_ids_and_accounting():
    positions = np.zeros((6, 2), dtype=np.float32)
    positions[[0, 2, 5]] = [[0, 0], [10, 0], [20, 0]]
    sensors = np.array([[1, 0], [9, 0], [19, 0], [100, 0], [5, 0]])
    packets = np.array([3, 7, 11, 13, 17], dtype=np.float32)
    service = FullClearNearestService(6, 6)
    result = service.collect(sensors, packets, positions, np.array([0, 2, 5]))
    np.testing.assert_array_equal(result.owner, [0, 2, 5, -1, 0])
    np.testing.assert_array_equal(result.collected_per_sensor, [3, 7, 11, 0, 17])
    np.testing.assert_array_equal(result.collected_by_uav, [20, 0, 7, 0, 0, 11])
    np.testing.assert_array_equal(result.per_uav_max, [17, 0, 7, 0, 0, 11])
    assert result.collected_per_sensor.sum() == result.collected_by_uav.sum()
    empty = service.collect(sensors, packets, positions, np.array([], dtype=np.int64))
    assert np.all(empty.owner == -1)
    assert empty.collected_per_sensor.sum() == 0


def test_midframe_depletion_stops_without_lower_death_penalty():
    _, scene, _, _, env = build_env()
    env.reset()
    scene.uav_battery[0] = 0.01
    _, _, terminated, _, info = env.step(np.zeros(scene.num_uavs, dtype=np.int8))
    frame = info['low_frame']
    assert terminated and len(frame.steps) == 1
    assert not frame.completed_all_steps
    assert frame.termination_reason == 'battery_depleted_during_low_step'
    result = frame.steps[0]
    np.testing.assert_array_equal(result.dead_during_step, [0])
    assert result.reward_terms['return_failure'] == 0
    assert 'death' not in result.reward_terms
    assert result.return_unsafe_ids.size == 0
    assert not np.any(result.sensor_owner == 0)
    assert scene.now_sec == 60
    assert info['reward_terms']['dead'] == -scene.cfg['reward']['upper_dead_weight']


def test_final_return_penalty_only_on_last_step():
    cfg, scene, _, _, env = build_env()
    env.reset()
    scene.uav_battery.fill(0.44)
    _, _, terminated, _, info = env.step(np.zeros(scene.num_uavs, dtype=np.int8))
    frame = info['low_frame']
    assert terminated and len(frame.steps) == 10
    assert frame.termination_reason == 'cannot_return_to_airship'
    assert all(s.reward_terms['return_failure'] == 0 for s in frame.steps[:-1])
    last = frame.steps[-1]
    assert all(s.dead_during_step.size == 0 for s in frame.steps)
    np.testing.assert_array_equal(last.return_unsafe_ids, np.arange(scene.num_uavs))
    assert last.reward_terms['return_failure'] == -cfg['reward']['low_death_weight'] * scene.num_uavs
    assert frame.reward_sum == sum(s.reward for s in frame.steps)
    assert info['reward_terms']['dead'] == 0
    np.testing.assert_array_equal(info['dead_ids_end'], np.arange(scene.num_uavs))


def test_scheduler_return_failure_is_not_a_lower_penalty():
    _, scene, energy, _, env = build_env()
    plan = LowestArrivalBatteryScheduler(scene.cfg, energy).make_plan(
        scene, np.zeros(scene.num_uavs, dtype=np.int8)
    )
    plan = replace(plan, return_failure_mask=np.ones(scene.num_uavs, dtype=bool))
    scene.begin_upper_frame(plan)
    frame = env.low_runner.run_frame(plan)
    assert not frame.terminated
    assert len(frame.steps) == 10
    assert all(s.reward_terms['return_failure'] == 0 for s in frame.steps)


def test_seeded_heuristic_rollouts_are_reproducible():
    from jv_uav import build_smoke_test_runner
    cfg, _, _, _, _ = build_env()
    cfg['time']['episode_upper_frames'] = 3
    runner = build_smoke_test_runner(cfg)
    first = runner.run_episode(reset_rng=True).to_serializable()
    second = runner.run_episode(reset_rng=True).to_serializable()
    assert first == second
    assert len(first['frames']) == 3
    assert len(first['backlog_max_post_timeline']) == 31

