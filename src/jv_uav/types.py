from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class UAVStatus(IntEnum):
    SERVING = 0
    WAITING = 1
    CHARGING = 2
    DEAD = 3


class UpperAction(IntEnum):
    SERVE = 0
    REQUEST_CHARGE = 1


@dataclass(frozen=True)
class ReturnTrip:
    distance_m: np.ndarray
    time_sec: np.ndarray
    energy_frac: np.ndarray


@dataclass(frozen=True)
class FramePlan:
    requested_charge: np.ndarray
    assigned_status: np.ndarray
    frame_start_status: np.ndarray
    frame_start_pos: np.ndarray
    frame_start_battery: np.ndarray
    return_distance_m: np.ndarray
    return_time_sec: np.ndarray
    return_energy_frac: np.ndarray
    arrival_battery_frac: np.ndarray
    charging_time_sec: np.ndarray
    waiting_time_sec: np.ndarray
    return_failure_mask: np.ndarray

    @property
    def serving_ids(self) -> np.ndarray:
        return np.flatnonzero(self.assigned_status == int(UAVStatus.SERVING))


@dataclass
class LowStepResult:
    step_in_frame: int
    serving_ids: np.ndarray
    position_before: np.ndarray
    position_after: np.ndarray
    requested_displacement: np.ndarray
    executed_displacement: np.ndarray
    move_distance_m: np.ndarray
    move_time_sec: np.ndarray
    service_time_sec: np.ndarray
    energy_used_frac: np.ndarray
    arrivals: np.ndarray
    sensor_owner: np.ndarray
    packets_pre_service: np.ndarray
    packets_post_service: np.ndarray
    collected_per_sensor: np.ndarray
    collected_by_uav: np.ndarray
    per_uav_owned_max_pre_service: np.ndarray
    covered_max_pre_service: float
    system_max_pre_service: float
    system_max_post_service: float
    system_mean_post_service: float
    oob_mask: np.ndarray
    oob_overflow_distance_m: np.ndarray
    dead_during_step: np.ndarray
    return_unsafe_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    deferred_return_failure_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    reward: float = 0.0
    reward_terms: dict[str, float] = field(default_factory=dict)


@dataclass
class LowFrameResult:
    steps: list[LowStepResult]
    completed_all_steps: bool
    terminated: bool
    termination_reason: str | None

    @property
    def reward_sum(self) -> float:
        return float(sum(step.reward for step in self.steps))

    @property
    def mean_system_max_post_service(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.system_max_post_service for s in self.steps]))


@dataclass
class UAVFrameRecord:
    uav_id: int
    role_during_frame: str
    status_at_frame_end: str
    trajectory_xy_m: list[list[float]]
    battery_start_frac: float
    battery_end_frac: float
    return_time_sec: float
    return_energy_frac: float
    charging_time_sec: float
    waiting_time_sec: float
    collected_packets: float
    mean_owned_max_pre_service: float


@dataclass
class FrameRecord:
    frame_index: int
    duration_sec: float
    serving_count: int
    dead_ids_end: list[int]
    system_max_post_service_mean: float
    system_mean_post_service_mean: float
    uavs: list[UAVFrameRecord]


@dataclass
class EpisodeTrace:
    frames: list[FrameRecord] = field(default_factory=list)
    backlog_max_post_timeline: list[float] = field(default_factory=list)
    backlog_mean_post_timeline: list[float] = field(default_factory=list)
    sensor_visit_count: np.ndarray | None = None
    sensor_collected_packets: np.ndarray | None = None
    coverage_grid_step_count: np.ndarray | None = None
    uav_presence_grid_count: np.ndarray | None = None

    def to_serializable(self) -> dict[str, Any]:
        from dataclasses import asdict

        data = asdict(self)
        return _arrays_to_lists(data)


def _arrays_to_lists(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _arrays_to_lists(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_arrays_to_lists(v) for v in value]
    return value
