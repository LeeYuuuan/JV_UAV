from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from .config import section
from .types import FramePlan, ReturnTrip, UAVStatus, UpperAction


class EnergyModel(Protocol):
    def serving_energy_fraction(
        self, move_time_sec: np.ndarray, hover_time_sec: np.ndarray
    ) -> np.ndarray: ...

    def return_trip(
        self, positions_xy_m: np.ndarray, airship_xy_m: np.ndarray
    ) -> ReturnTrip: ...

    def charging_energy_fraction(self, charging_time_sec: np.ndarray) -> np.ndarray: ...


class SimpleEnergyModel:
    """Replaceable first-version energy model; all energy units end as SOC."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        energy = section(cfg, "energy")
        self.capacity_wh = float(energy["battery_capacity_wh"])
        self.horizontal_power_w = float(energy["horizontal_power_w"])
        self.hover_power_w = float(energy["hover_power_w"])
        self.climb_power_w = float(energy["climb_power_w"])
        self.charging_power_w = float(energy["charging_power_w"])
        self.horizontal_speed_mps = float(energy["horizontal_speed_mps"])
        self.climb_speed_mps = float(energy["climb_speed_mps"])
        self.height_m = float(energy["service_to_airship_height_m"])

    def _joules_to_fraction(self, joules: np.ndarray) -> np.ndarray:
        return (joules / (3600.0 * self.capacity_wh)).astype(np.float32)

    def serving_energy_fraction(
        self, move_time_sec: np.ndarray, hover_time_sec: np.ndarray
    ) -> np.ndarray:
        joules = (
            self.horizontal_power_w * np.asarray(move_time_sec)
            + self.hover_power_w * np.asarray(hover_time_sec)
        )
        return self._joules_to_fraction(joules)

    def return_trip(
        self, positions_xy_m: np.ndarray, airship_xy_m: np.ndarray
    ) -> ReturnTrip:
        positions = np.asarray(positions_xy_m, dtype=np.float32)
        distance = np.linalg.norm(positions - airship_xy_m, axis=-1)
        horizontal_time = distance / self.horizontal_speed_mps
        climb_time = np.full_like(distance, self.height_m / self.climb_speed_mps)
        time_sec = horizontal_time + climb_time
        joules = (
            self.horizontal_power_w * horizontal_time
            + self.climb_power_w * climb_time
        )
        return ReturnTrip(
            distance_m=distance.astype(np.float32),
            time_sec=time_sec.astype(np.float32),
            energy_frac=self._joules_to_fraction(joules),
        )

    def charging_energy_fraction(self, charging_time_sec: np.ndarray) -> np.ndarray:
        joules = self.charging_power_w * np.asarray(charging_time_sec)
        return self._joules_to_fraction(joules)


class ChargingScheduler(Protocol):
    def make_plan(self, scene: Any, upper_action: np.ndarray) -> FramePlan: ...


class LowestArrivalBatteryScheduler:
    """Whole-frame slot allocation; no mid-frame handoff in this version."""

    def __init__(self, cfg: Mapping[str, Any], energy_model: EnergyModel) -> None:
        zones = section(cfg, "zones")
        time = section(cfg, "time")
        self.energy_model = energy_model
        self.charging_slots = int(zones["charging_slots"])
        self.waiting_slots = int(zones["waiting_slots"])
        self.frame_sec = float(time["low_step_sec"]) * int(time["low_steps_per_frame"])

    def make_plan(self, scene: Any, upper_action: np.ndarray) -> FramePlan:
        action = np.asarray(upper_action, dtype=np.int8)
        if action.shape != (scene.num_uavs,):
            raise ValueError(f"upper_action must have shape ({scene.num_uavs},)")
        alive = scene.uav_status != int(UAVStatus.DEAD)
        if not np.isin(action[alive], [int(UpperAction.SERVE), int(UpperAction.REQUEST_CHARGE)]).all():
            raise ValueError("upper action must be SERVE(0) or REQUEST_CHARGE(1)")

        requested = alive & (action == int(UpperAction.REQUEST_CHARGE))
        assigned = np.full(scene.num_uavs, int(UAVStatus.SERVING), dtype=np.int8)
        assigned[~alive] = int(UAVStatus.DEAD)

        trip = self.energy_model.return_trip(scene.uav_pos, scene.airship_pos)
        already_in_zone = np.isin(
            scene.uav_status, [int(UAVStatus.WAITING), int(UAVStatus.CHARGING)]
        )
        return_distance = trip.distance_m.copy()
        return_time = trip.time_sec.copy()
        return_energy = trip.energy_frac.copy()
        return_distance[already_in_zone] = 0.0
        return_time[already_in_zone] = 0.0
        return_energy[already_in_zone] = 0.0
        arrival_battery = scene.uav_battery - return_energy

        def low_battery_order(ids: np.ndarray) -> np.ndarray:
            return ids[np.lexsort((ids, arrival_battery[ids]))]

        # Occupancy visible in the upper observation must have actual meaning:
        # a continuing charger keeps its dock, and a continuing waiter keeps a
        # zone place.  Only vacancies are allocated to new requests.
        retained_charging = np.flatnonzero(
            requested & (scene.uav_status == int(UAVStatus.CHARGING))
        )
        retained_charging = low_battery_order(retained_charging)[: self.charging_slots]
        remaining = requested.copy()
        remaining[retained_charging] = False

        waiting_for_dock = np.flatnonzero(
            remaining & (scene.uav_status == int(UAVStatus.WAITING))
        )
        other_for_dock = np.flatnonzero(
            remaining & (scene.uav_status != int(UAVStatus.WAITING))
        )
        dock_candidates = np.concatenate(
            [low_battery_order(waiting_for_dock), low_battery_order(other_for_dock)]
        )
        new_charging = dock_candidates[: self.charging_slots - len(retained_charging)]
        charging_ids = np.concatenate([retained_charging, new_charging])
        remaining[new_charging] = False

        retained_waiting = np.flatnonzero(
            remaining & (scene.uav_status == int(UAVStatus.WAITING))
        )
        retained_waiting = low_battery_order(retained_waiting)[: self.waiting_slots]
        remaining[retained_waiting] = False
        new_waiting_candidates = low_battery_order(np.flatnonzero(remaining))
        new_waiting = new_waiting_candidates[: self.waiting_slots - len(retained_waiting)]
        waiting_ids = np.concatenate([retained_waiting, new_waiting])
        assigned[charging_ids] = int(UAVStatus.CHARGING)
        assigned[waiting_ids] = int(UAVStatus.WAITING)

        returning = np.isin(assigned, [int(UAVStatus.CHARGING), int(UAVStatus.WAITING)])
        # Rejected requests remain in the field and therefore have no return cost.
        return_distance[~returning] = 0.0
        return_time[~returning] = 0.0
        return_energy[~returning] = 0.0
        arrival_battery = scene.uav_battery - return_energy
        return_failure = returning & (
            (arrival_battery < 0.0) | (return_time > self.frame_sec)
        )
        charging_time = np.zeros(scene.num_uavs, dtype=np.float32)
        waiting_time = np.zeros(scene.num_uavs, dtype=np.float32)
        charging_time[charging_ids] = np.maximum(
            0.0, self.frame_sec - return_time[charging_ids]
        )
        waiting_time[waiting_ids] = np.maximum(
            0.0, self.frame_sec - return_time[waiting_ids]
        )

        return FramePlan(
            requested_charge=requested,
            assigned_status=assigned,
            frame_start_status=scene.uav_status.copy(),
            frame_start_pos=scene.uav_pos.copy(),
            frame_start_battery=scene.uav_battery.copy(),
            return_distance_m=return_distance,
            return_time_sec=return_time,
            return_energy_frac=return_energy,
            arrival_battery_frac=arrival_battery.astype(np.float32),
            charging_time_sec=charging_time,
            waiting_time_sec=waiting_time,
            return_failure_mask=return_failure,
        )


@dataclass(frozen=True)
class FullClearResult:
    owner: np.ndarray
    collected_per_sensor: np.ndarray
    collected_by_uav: np.ndarray
    per_uav_max: np.ndarray


class FullClearNearestService:
    def __init__(self, num_uavs: int, coverage_radius_m: float) -> None:
        self.num_uavs = num_uavs
        self.radius = float(coverage_radius_m)

    def collect(
        self,
        sensor_pos: np.ndarray,
        packets: np.ndarray,
        uav_pos: np.ndarray,
        service_ids: np.ndarray,
    ) -> FullClearResult:
        owner = np.full(len(sensor_pos), -1, dtype=np.int64)
        if service_ids.size:
            distances = np.linalg.norm(
                sensor_pos[:, None, :] - uav_pos[service_ids][None, :, :], axis=2
            )
            nearest_local = np.argmin(distances, axis=1)
            nearest_distance = distances[np.arange(len(sensor_pos)), nearest_local]
            covered = nearest_distance <= self.radius
            owner[covered] = service_ids[nearest_local[covered]]

        collected_per_sensor = np.where(owner >= 0, packets, 0.0).astype(np.float32)
        collected_by_uav = np.zeros(self.num_uavs, dtype=np.float32)
        per_uav_max = np.zeros(self.num_uavs, dtype=np.float32)
        for uav_id in service_ids:
            mask = owner == uav_id
            if np.any(mask):
                collected_by_uav[uav_id] = float(packets[mask].sum())
                per_uav_max[uav_id] = float(packets[mask].max())
        return FullClearResult(owner, collected_per_sensor, collected_by_uav, per_uav_max)
