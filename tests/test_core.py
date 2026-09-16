from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jv_uav import (  # noqa: E402
    EpisodeMetrics,
    LowEnv,
    LowFrameRunner,
    LowestArrivalBatteryScheduler,
    Scene,
    SimpleEnergyModel,
    UpperEnv,
    load_config,
)
from jv_uav.low_env import ZeroLowPolicy  # noqa: E402


def build_env():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    scene = Scene(cfg)
    energy = SimpleEnergyModel(cfg)
    metrics = EpisodeMetrics(cfg, scene.sensor_pos)
    low_env = LowEnv(scene, energy, metrics, cfg)
    low_runner = LowFrameRunner(low_env, ZeroLowPolicy(), energy)
    scheduler = LowestArrivalBatteryScheduler(cfg, energy)
    return cfg, scene, energy, metrics, UpperEnv(
        cfg, scene, scheduler, energy, low_runner, metrics
    )


def test_observations_hide_backlog_and_lower_is_variable_length():
    _, scene, _, _, _ = build_env()
    upper = scene.observe_upper()
    lower = scene.observe_lower()
    assert "sensor_backlog" not in upper
    assert "sensor_backlog" not in lower
    assert lower["active_uav_features"].shape == (len(scene.serving_ids()), 3)


def test_one_upper_step_is_ten_minutes_and_records_metrics():
    _, scene, _, metrics, env = build_env()
    env.reset()
    action = np.zeros(scene.num_uavs, dtype=np.int8)
    _, _, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert scene.now_sec == 600.0
    assert scene.upper_step == 1
    assert len(info["low_frame"].steps) == 10
    assert len(metrics.trace.backlog_max_post_timeline) == 11
    assert len(metrics.trace.frames[0].uavs[0].trajectory_xy_m) == 11


def test_charged_uav_release_is_visible_in_next_observation():
    _, scene, _, _, env = build_env()
    env.reset()
    scene.uav_battery[0] = 0.5
    action = np.zeros(scene.num_uavs, dtype=np.int8)
    action[0] = 1
    next_obs, _, terminated, _, info = env.step(action)
    assert not terminated
    assert info["serving_count_during_frame"] == scene.num_uavs - 1
    assert info["frame_plan"].assigned_status[0] == 2
    # With 500 W for almost the whole frame, UAV 0 reaches full and is released.
    assert scene.uav_status[0] == 0
    assert next_obs["charging_occupancy"][0] == 0


def test_packet_accounting_for_a_low_step():
    _, scene, energy, metrics, _ = build_env()
    low = LowEnv(scene, energy, metrics, scene.cfg)
    before = float(scene.sensor_backlog.sum())
    result = low.step(np.zeros((len(scene.serving_ids()), 2), dtype=np.float32))
    after = float(scene.sensor_backlog.sum())
    expected = before + float(result.arrivals.sum()) - float(result.collected_per_sensor.sum())
    assert np.isclose(after, expected)


def test_continuing_charger_is_not_displaced_by_new_request():
    _, scene, energy, _, _ = build_env()
    scheduler = LowestArrivalBatteryScheduler(scene.cfg, energy)
    scene.uav_status[0] = 2
    scene.uav_battery[0] = 0.8
    scene.uav_battery[1] = 0.1
    action = np.zeros(scene.num_uavs, dtype=np.int8)
    action[[0, 1]] = 1
    plan = scheduler.make_plan(scene, action)
    assert plan.assigned_status[0] == 2


def test_full_clear_uses_nearest_owner_and_pre_post_metrics():
    _, scene, energy, metrics, _ = build_env()
    scene.arrival_rate.fill(0.0)
    scene.sensor_pos[0] = scene.airship_pos
    scene.sensor_backlog.fill(0.0)
    scene.sensor_backlog[0] = 12.0
    low = LowEnv(scene, energy, metrics, scene.cfg)
    result = low.step(np.zeros((len(scene.serving_ids()), 2), dtype=np.float32))
    assert result.sensor_owner[0] == 0  # deterministic tie: lowest active ID
    assert result.packets_pre_service[0] == 12.0
    assert result.packets_post_service[0] == 0.0
    assert result.per_uav_owned_max_pre_service[0] >= 12.0


def test_last_low_step_marks_return_unsafe_and_penalizes_it():
    _, scene, _, _, env = build_env()
    env.reset()
    scene.uav_battery.fill(0.44)
    action = np.zeros(scene.num_uavs, dtype=np.int8)
    _, _, terminated, _, info = env.step(action)
    last = info["low_frame"].steps[-1]
    assert len(info["low_frame"].steps) == 10
    assert terminated
    assert last.return_unsafe_ids.size == scene.num_uavs
    assert last.reward_terms["return_failure"] < 0.0
