"""A frozen role schedule and an optional top-K lower reward."""
from __future__ import annotations

import numpy as np

from jv_uav.low_env import LowReward
from jv_uav.types import FramePlan, UAVStatus


class FixedScheduler:
    def __init__(self, cfg, energy, roles):
        self.energy = energy
        self.roles = np.asarray(roles, dtype=np.int8)
        self.frame_sec = cfg["time"]["low_step_sec"] * cfg["time"]["low_steps_per_frame"]
        fleet = cfg["uavs"]["count"]
        if self.roles.shape != (cfg["time"]["episode_upper_frames"], fleet):
            raise ValueError("Schedule shape must match the full episode and fleet.")
        if not np.isin(self.roles, [0, 1, 2]).all():
            raise ValueError("Schedule must contain serving, waiting, or charging roles.")
        if np.any((self.roles == 2).sum(1) > cfg["zones"]["charging_slots"]):
            raise ValueError("Schedule exceeds charging capacity.")
        if np.any((self.roles == 1).sum(1) > cfg["zones"]["waiting_slots"]):
            raise ValueError("Schedule exceeds waiting capacity.")
        if not (self.roles == 0).any(1).all():
            raise ValueError("Each frame must have a serving UAV.")

    def make_plan(self, scene, unused_action):
        assigned = self.roles[scene.upper_step].copy()
        assigned[scene.uav_status == int(UAVStatus.DEAD)] = int(UAVStatus.DEAD)
        returning = np.isin(assigned, [1, 2])
        trip = self.energy.return_trip(scene.uav_pos, scene.airship_pos)
        needs_trip = returning & ~np.isin(scene.uav_status, [1, 2])
        distance = np.where(needs_trip, trip.distance_m, 0.0).astype(np.float32)
        time = np.where(needs_trip, trip.time_sec, 0.0).astype(np.float32)
        cost = np.where(needs_trip, trip.energy_frac, 0.0).astype(np.float32)
        arrival = scene.uav_battery - cost
        remaining = np.maximum(0.0, self.frame_sec - time)
        return FramePlan(
            requested_charge=returning,
            assigned_status=assigned,
            frame_start_status=scene.uav_status.copy(),
            frame_start_pos=scene.uav_pos.copy(),
            frame_start_battery=scene.uav_battery.copy(),
            return_distance_m=distance, return_time_sec=time,
            return_energy_frac=cost, arrival_battery_frac=arrival,
            charging_time_sec=np.where(assigned == 2, remaining, 0).astype(np.float32),
            waiting_time_sec=np.where(assigned == 1, remaining, 0).astype(np.float32),
            return_failure_mask=returning & ((arrival < 0) | (time > self.frame_sec)),
        )


class TopKReward:
    def __init__(self, cfg, k=5, rho=0.5, scale=300.0, offset=0.0):
        if k < 1 or not 0 <= rho < 1 or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Require K >= 1, 0 <= rho < 1 and a positive reward scale.")
        if not np.isfinite(offset):
            raise ValueError("Reward offset must be finite.")
        self.k, self.rho, self.scale, self.offset = k, rho, scale, offset
        self.constraints = LowReward(cfg)

    def __call__(self, result, **kwargs):
        _, old = self.constraints(result, **kwargs)
        pre = np.asarray(result.packets_pre_service, dtype=np.float64)
        # Stable sensor-index tie break. Actual full-clear collection is counted once.
        ids = np.argsort(-pre, kind="stable")[:self.k]
        denominator = pre[ids].sum()
        h = float(result.collected_per_sensor[ids].sum() / denominator) if denominator else 0.0
        b = float(result.system_max_post_service)
        terms = {
            "offset": self.offset / self.scale,
            "system_max_post_service": -b / self.scale,
            "topk_clear": self.rho * b * h / self.scale,
            "oob": old["oob"],
            "return_failure": old["return_failure"],
            "return_distance": old["return_distance"],
        }
        return float(sum(terms.values())), terms
