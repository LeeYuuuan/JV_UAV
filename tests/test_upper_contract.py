import numpy as np
from test_core import build_env
from jv_uav.models import LowestArrivalBatteryScheduler
from jv_uav.types import UAVStatus


def test_unfilled_charger_and_waiter_can_leave_and_free_slots():
    _, scene, energy, _, _ = build_env()
    scene.uav_status[:] = [2, 2, 1, 1, 0, 0]
    scene.uav_battery[:] = [0.4, 0.5, 0.6, 0.7, 0.2, 0.3]
    plan = LowestArrivalBatteryScheduler(scene.cfg, energy).make_plan(scene, np.array([0, 1, 0, 1, 1, 1]))
    np.testing.assert_array_equal(plan.assigned_status, [0, 2, 0, 2, 1, 1])
    assert plan.return_energy_frac[0] == 0
    assert plan.return_energy_frac[2] == 0


def test_occupancy_priority_and_rejected_requests():
    _, scene, energy, _, _ = build_env()
    scene.uav_status[:] = [2, 2, 1, 1, 0, 0]
    scene.uav_battery[:] = [0.8, 0.9, 0.7, 0.6, 0.2, 0.1]
    plan = LowestArrivalBatteryScheduler(scene.cfg, energy).make_plan(scene, np.ones(6))
    np.testing.assert_array_equal(plan.assigned_status, [2, 2, 1, 1, 0, 0])
    np.testing.assert_array_equal(plan.return_energy_frac[4:], [0, 0])
    np.testing.assert_array_equal(plan.return_time_sec[4:], [0, 0])


def test_new_requests_order_by_arrival_battery_and_id():
    _, scene, energy, _, _ = build_env()
    scene.uav_battery.fill(0.8)
    scene.uav_pos[5] += [600, 0]
    plan = LowestArrivalBatteryScheduler(scene.cfg, energy).make_plan(scene, np.ones(6))
    np.testing.assert_array_equal(np.flatnonzero(plan.assigned_status == 2), [0, 5])
    np.testing.assert_array_equal(np.flatnonzero(plan.assigned_status == 1), [1, 2])
    np.testing.assert_array_equal(plan.serving_ids, [3, 4])


def test_return_charge_settlement_and_waiter_promotion():
    _, scene, energy, _, env = build_env()
    env.reset()
    scene.uav_status[:] = [2, 2, 1, 1, 0, 0]
    scene.uav_battery[:] = [0.9, 0.2, 0.3, 0.4, 1, 1]
    obs, _, terminated, _, info = env.step(np.array([1, 1, 1, 1, 0, 0]))
    assert not terminated
    plan = info['frame_plan']
    assert plan.charging_time_sec[0] == 600
    assert plan.waiting_time_sec[2] == 600
    np.testing.assert_array_equal(info['settlement']['released_ids'], [0])
    np.testing.assert_array_equal(info['settlement']['promoted_ids'], [2])
    np.testing.assert_allclose(scene.uav_battery[2], 0.3)
    assert info['settlement']['charge_added_frac'][2] == 0
    assert obs['charging_occupancy'][0] == 2
    assert obs['waiting_occupancy'][0] == 1
    record = env.trace.frames[0].uavs[2]
    assert record.role_during_frame == 'WAITING' and record.status_at_frame_end == 'CHARGING'


def test_return_time_and_charge_energy_from_distant_position():
    _, scene, energy, _, env = build_env()
    env.reset()
    scene.uav_pos[0] += [600, 0]
    scene.uav_battery[0] = 0.4
    _, _, terminated, _, info = env.step(np.array([1, 0, 0, 0, 0, 0]))
    assert not terminated
    plan = info['frame_plan']
    np.testing.assert_allclose(plan.return_time_sec[0], 110)
    np.testing.assert_allclose(plan.return_energy_frac[0], (257 * 100 + 400 * 10) / 450000)
    np.testing.assert_allclose(plan.charging_time_sec[0], 490)
    np.testing.assert_allclose(scene.uav_battery[0], 0.4 - 29700 / 450000 + 500 * 490 / 450000)
    np.testing.assert_array_equal(scene.uav_pos[0], scene.airship_pos)


def test_worst_map_return_fits_frame_and_safe_battery_is_accepted():
    _, scene, energy, _, _ = build_env()
    corners = np.array([[0, 0], [0, 2400], [2400, 0], [2400, 2400]])
    trip = energy.return_trip(corners, scene.airship_pos)
    assert np.max(trip.time_sec) < 600
    scene.uav_pos[:4] = corners
    trips = energy.return_trip(scene.uav_pos, scene.airship_pos)
    scene.uav_battery[:] = trips.energy_frac + 0.01
    assert scene.mark_unable_to_return_dead(energy).size == 0
    plan = LowestArrivalBatteryScheduler(scene.cfg, energy).make_plan(scene, np.ones(6))
    assert not plan.return_failure_mask.any()


def test_all_binary_requests_preserve_capacity_and_scene_until_begin():
    from itertools import product
    _, scene, energy, _, _ = build_env()
    scheduler = LowestArrivalBatteryScheduler(scene.cfg, energy)
    for statuses in ([0, 0, 0, 0, 0, 0], [2, 2, 1, 1, 0, 0], [2, 0, 1, 0, 0, 0]):
        scene.uav_status[:] = statuses
        scene.uav_battery[:] = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        before_status = scene.uav_status.copy()
        before_battery = scene.uav_battery.copy()
        before_pos = scene.uav_pos.copy()
        for actions in product([0, 1], repeat=6):
            actions = np.asarray(actions, dtype=np.int8)
            plan = scheduler.make_plan(scene, actions)
            assert np.count_nonzero(plan.assigned_status == 2) <= 2
            assert np.count_nonzero(plan.assigned_status == 1) <= 2
            assert len(plan.serving_ids) >= 2
            assert np.all(plan.assigned_status[actions == 0] == 0)
            assert np.all(plan.return_energy_frac[plan.serving_ids] == 0)
            np.testing.assert_array_equal(scene.uav_status, before_status)
            np.testing.assert_array_equal(scene.uav_battery, before_battery)
            np.testing.assert_array_equal(scene.uav_pos, before_pos)
