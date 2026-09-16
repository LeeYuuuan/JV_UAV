import json
import numpy as np
from test_core import build_env


def test_step_metrics_match_reports_and_preserve_full_fleet_columns():
    _, scene, _, metrics, env = build_env()
    env.reset()
    scene.uav_status[:] = 1
    scene.uav_status[[0, 2, 5]] = 0
    scene.arrival_rate.fill(0)
    scene.sensor_backlog.fill(0)
    scene.sensor_pos[:3] = [[100, 100], [500, 500], [900, 900]]
    scene.uav_pos[[0, 2, 5]] = scene.sensor_pos[:3]
    scene.sensor_backlog[:3] = [3, 7, 11]
    result = env.low_runner.low_env.step(np.zeros((3, 2)))
    trace = metrics.trace
    assert trace.serving_ids_timeline == [[0, 2, 5]]
    np.testing.assert_array_equal(trace.per_uav_owned_max_pre_service_timeline, [[3, 0, 7, 0, 0, 11]])
    assert trace.covered_max_pre_service_timeline == [11]
    np.testing.assert_array_equal(trace.per_uav_owned_max_pre_service_timeline[0], result.per_uav_owned_max_pre_service)
    result.per_uav_owned_max_pre_service[:] = -1
    result.serving_ids[:] = -1
    assert trace.serving_ids_timeline == [[0, 2, 5]]
    assert trace.per_uav_owned_max_pre_service_timeline[0][5] == 11
    saved = json.loads(json.dumps(trace.to_serializable(), allow_nan=False))
    assert saved['covered_max_pre_service_timeline'] == [11]
    assert len(saved['backlog_max_post_timeline']) == 2
    env.reset()
    assert env.trace.covered_max_pre_service_timeline == []
    assert env.trace.per_uav_owned_max_pre_service_timeline == []
    assert env.trace.serving_ids_timeline == []


def test_step_metrics_cover_early_and_final_termination():
    for battery, expected_steps in [(0.01, 1), (0.44, 10)]:
        _, scene, _, _, env = build_env()
        env.reset()
        scene.uav_battery.fill(battery)
        _, _, terminated, _, info = env.step(np.zeros(scene.num_uavs, dtype=np.int8))
        assert terminated
        trace = env.trace
        assert len(trace.covered_max_pre_service_timeline) == expected_steps
        assert len(trace.per_uav_owned_max_pre_service_timeline) == expected_steps
        assert len(trace.serving_ids_timeline) == expected_steps
        assert len(trace.backlog_max_post_timeline) == expected_steps + 1
        for index, result in enumerate(info['low_frame'].steps):
            assert trace.covered_max_pre_service_timeline[index] == result.covered_max_pre_service
            np.testing.assert_array_equal(trace.per_uav_owned_max_pre_service_timeline[index], result.per_uav_owned_max_pre_service)
            np.testing.assert_array_equal(trace.serving_ids_timeline[index], result.serving_ids)
        json.dumps(trace.to_serializable(), allow_nan=False)
