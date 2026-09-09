from __future__ import annotations

import numpy as np


def clip_displacements_by_norm(
    actions: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"actions must have shape [num_uavs, 2], got {actions.shape}")

    norm = np.linalg.norm(actions, axis=1, keepdims=True)
    scale = np.minimum(1.0, float(max_distance) / np.maximum(norm, 1e-8))
    return (actions * scale).astype(np.float32)


def move_serving_uavs(
    uav_pos: np.ndarray,
    actions: np.ndarray,
    serving_ids: np.ndarray,
    world_size: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | int | float]]:
    old_pos = np.asarray(uav_pos, dtype=np.float32)
    new_pos = old_pos.copy()
    serving_ids = np.asarray(serving_ids, dtype=np.int64)

    if serving_ids.size == 0:
        return new_pos, {
            "proposed": np.zeros((0, 2), np.float32),
            "executed": np.zeros((0, 2), np.float32),
            "oob_mask": np.zeros(0, dtype=bool),
            "oob_count": 0,
            "overflow_dist": 0.0,
        }

    proposed = old_pos[serving_ids] + actions[serving_ids]
    executed = np.clip(proposed, 0.0, float(world_size)).astype(np.float32)
    overflow = proposed - executed
    oob_mask = np.any(np.abs(overflow) > 1e-6, axis=1)
    new_pos[serving_ids] = executed

    return new_pos, {
        "proposed": proposed.astype(np.float32),
        "executed": executed,
        "oob_mask": oob_mask,
        "oob_count": int(oob_mask.sum()),
        "overflow_dist": float(np.linalg.norm(overflow[oob_mask], axis=1).sum())
        if oob_mask.any()
        else 0.0,
    }


def compute_step_timing(
    old_pos: np.ndarray,
    new_pos: np.ndarray,
    serving_ids: np.ndarray,
    *,
    speed: float,
    low_step_sec: float,
    max_move_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_uavs = int(old_pos.shape[0])
    distance = np.zeros(num_uavs, np.float32)
    move_time = np.zeros(num_uavs, np.float32)
    service_time = np.zeros(num_uavs, np.float32)

    if serving_ids.size:
        distance[serving_ids] = np.linalg.norm(
            new_pos[serving_ids] - old_pos[serving_ids], axis=1
        )
        move_time[serving_ids] = np.minimum(
            distance[serving_ids] / float(speed), float(max_move_sec)
        )
        service_time[serving_ids] = float(low_step_sec) - move_time[serving_ids]

    return distance, move_time, service_time


def compute_serving_energy_frac(
    move_time: np.ndarray,
    service_time: np.ndarray,
    serving_ids: np.ndarray,
    *,
    p_horiz: float,
    p_hover: float,
    battery_capacity_wh: float,
) -> np.ndarray:
    energy_wh = np.zeros_like(move_time, dtype=np.float32)
    if serving_ids.size:
        energy_wh[serving_ids] = (
            float(p_horiz) * move_time[serving_ids]
            + float(p_hover) * service_time[serving_ids]
        ) / 3600.0
    return (energy_wh / float(battery_capacity_wh)).astype(np.float32)


def compute_return_energy_frac(
    uav_pos: np.ndarray,
    airship_pos: np.ndarray,
    *,
    speed: float,
    p_horiz: float,
    battery_capacity_wh: float,
) -> np.ndarray:
    distance = np.linalg.norm(
        np.asarray(uav_pos) - np.asarray(airship_pos)[None, :], axis=1
    )
    return_time = distance / float(speed)
    energy_wh = float(p_horiz) * return_time / 3600.0
    return (energy_wh / float(battery_capacity_wh)).astype(np.float32)


def assign_sensors_to_nearest_serving_uav(
    sensor_pos: np.ndarray,
    uav_pos: np.ndarray,
    serving_ids: np.ndarray,
    coverage_radius: float,
) -> np.ndarray:
    sensor_pos = np.asarray(sensor_pos, dtype=np.float32)
    owner = np.full(sensor_pos.shape[0], -1, dtype=np.int64)
    if sensor_pos.shape[0] == 0 or serving_ids.size == 0:
        return owner

    diff = (
        sensor_pos[:, None, :]
        - np.asarray(uav_pos)[serving_ids][None, :, :]
    )
    dist2 = np.sum(diff * diff, axis=-1)
    covered = dist2 <= float(coverage_radius) ** 2
    covered_any = covered.any(axis=1)

    if covered_any.any():
        nearest_local = np.where(covered, dist2, np.inf).argmin(axis=1)
        owner[covered_any] = serving_ids[nearest_local[covered_any]]
    return owner


def apply_full_clear(
    sensor_pkts: np.ndarray,
    sensor_last_visit: np.ndarray,
    owner_sensor_to_uav: np.ndarray,
    max_uavs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_pkts = np.asarray(sensor_pkts, np.float32).copy()
    new_last_visit = np.asarray(sensor_last_visit, np.float32).copy()
    owner = np.asarray(owner_sensor_to_uav, np.int64)
    collected = np.zeros((max_uavs, new_pkts.size), dtype=np.float32)

    covered_ids = np.flatnonzero(owner >= 0)
    if covered_ids.size:
        uav_ids = owner[covered_ids]
        collected[uav_ids, covered_ids] = new_pkts[covered_ids]
        new_pkts[covered_ids] = 0.0
        new_last_visit[covered_ids] = 0.0

    return new_pkts, new_last_visit, collected