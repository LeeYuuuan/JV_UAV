from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import section, validate_config
from .models import EnergyModel, FullClearNearestService
from .sensor_generator import generate_sensor_positions
from .types import FramePlan, LowStepResult, UAVStatus


class Scene:
    """Single source of truth for observable and hidden simulation state."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        validate_config(cfg)
        self.cfg = cfg
        world = section(cfg, "world")
        sensors = section(cfg, "sensors")
        uavs = section(cfg, "uavs")
        time = section(cfg, "time")
        seeds = section(cfg, "seeds")

        self.world_size = float(world["size_m"])
        self.airship_pos = np.asarray(world["airship_xy_m"], dtype=np.float32)
        self.num_sensors = int(sensors["count"])
        self.num_uavs = int(uavs["count"])
        self.low_step_sec = float(time["low_step_sec"])
        self.low_steps_per_frame = int(time["low_steps_per_frame"])
        self.speed_mps = float(uavs["speed_mps"])
        self.max_move_time_sec = float(uavs["max_move_time_sec"])
        self.max_move_distance_m = self.speed_mps * self.max_move_time_sec
        self.return_reserve_frac = float(uavs["return_reserve_frac"])
        self.arrival_rate = np.full(
            self.num_sensors, float(sensors["arrival_rate_per_sec"]), np.float32
        )
        self.sensor_pos = generate_sensor_positions(cfg)

        self._dynamics_seed = int(seeds["dynamics_seed"])
        self._initial_battery = float(uavs["initial_battery_frac"])
        self._initial_serving_ids = np.asarray(uavs["initial_serving_ids"], dtype=np.int64)
        self.rng = np.random.default_rng(self._dynamics_seed)

        self.sensor_backlog = np.zeros(self.num_sensors, dtype=np.float32)
        self.sensor_last_visit_sec = np.zeros(self.num_sensors, dtype=np.float32)
        self.uav_pos = np.zeros((self.num_uavs, 2), dtype=np.float32)
        self.uav_battery = np.zeros(self.num_uavs, dtype=np.float32)
        self.uav_status = np.zeros(self.num_uavs, dtype=np.int8)
        self.now_sec = 0.0
        self.upper_step = 0
        self.low_step_total = 0
        self.step_in_frame = 0
        self.reset()

    def reset(self, *, reset_rng: bool = True) -> None:
        if reset_rng:
            self.rng = np.random.default_rng(self._dynamics_seed)
        cold_start = float(section(self.cfg, "sensors")["cold_start_sec"])
        self.sensor_backlog[:] = self.rng.poisson(self.arrival_rate * cold_start)
        self.sensor_last_visit_sec.fill(cold_start)
        self.uav_pos[:] = self.airship_pos
        self.uav_battery.fill(self._initial_battery)
        self.uav_status.fill(int(UAVStatus.WAITING))
        self.uav_status[self._initial_serving_ids] = int(UAVStatus.SERVING)
        self.now_sec = 0.0
        self.upper_step = 0
        self.low_step_total = 0
        self.step_in_frame = 0

    def observe_lower(self) -> dict[str, np.ndarray]:
        ids = self.serving_ids()
        # Backlog is intentionally hidden; last-visit remains globally observable.
        return {
            "active_uav_ids": ids,
            "active_uav_features": np.column_stack(
                [self.uav_pos[ids], self.uav_battery[ids]]
            ).astype(np.float32),
            "sensor_last_visit_sec": self.sensor_last_visit_sec.copy(),
        }

    def observe_upper(self) -> dict[str, np.ndarray]:
        # This is a temporary global raw view. Agent-specific MAPPO views stay open.
        one_hot = np.eye(len(UAVStatus), dtype=np.float32)[self.uav_status]
        return {
            "uav_features": np.column_stack(
                [self.uav_pos, self.uav_battery, one_hot]
            ).astype(np.float32),
            "charging_occupancy": np.asarray([len(self.charging_ids())], np.float32),
            "waiting_occupancy": np.asarray([len(self.waiting_ids())], np.float32),
        }

    def begin_upper_frame(self, plan: FramePlan) -> None:
        """Freeze each UAV's role before any of the ten low steps run."""
        if self.step_in_frame != 0:
            raise RuntimeError("cannot begin a frame while another frame is active")
        alive = self.uav_status != int(UAVStatus.DEAD)
        self.uav_status[alive] = plan.assigned_status[alive]

    def evolve_low_step(
    self,
    actions: np.ndarray,
    energy_model: EnergyModel,
    service_model: FullClearNearestService,
) -> LowStepResult:
        """Evolve one lower-level step using actions measured in meters."""

        serving_ids = self.serving_ids()
        actions = np.asarray(actions, dtype=np.float32)

        if actions.shape != (len(serving_ids), 2):
            raise ValueError(
                f"actions must have shape ({len(serving_ids)}, 2)"
            )

        pos_before = self.uav_pos.copy()

        norms = np.linalg.norm(actions, axis=1)
        scale = np.minimum(
            1.0,
            self.max_move_distance_m / np.maximum(norms, 1e-12),
        )
        actions = actions * scale[:, None]

        intended = self.uav_pos[serving_ids] + actions
        bounded = np.clip(intended, 0.0, self.world_size)

        overflow = np.linalg.norm(intended - bounded, axis=1)
        oob_active = overflow > 1e-6

        self.uav_pos[serving_ids] = bounded

        requested = np.zeros((self.num_uavs, 2), dtype=np.float32)
        requested[serving_ids] = actions

        executed = np.zeros((self.num_uavs, 2), dtype=np.float32)
        executed[serving_ids] = bounded - pos_before[serving_ids]

        move_distance = np.zeros(self.num_uavs, dtype=np.float32)
        move_distance[serving_ids] = np.linalg.norm(
            executed[serving_ids],
            axis=1,
        )

        move_time = move_distance / self.speed_mps

        service_time = np.zeros(self.num_uavs, dtype=np.float32)
        service_time[serving_ids] = (self.low_step_sec - move_time[serving_ids])

        energy_used = energy_model.serving_energy_fraction(move_time, service_time)

        self.uav_battery[serving_ids] -= energy_used[serving_ids]

        dead_during = serving_ids[
            self.uav_battery[serving_ids] <= 0.0
        ]
        self.uav_battery[dead_during] = 0.0
        self.uav_status[dead_during] = int(UAVStatus.DEAD)

        arrivals = self.rng.poisson(
            self.arrival_rate * self.low_step_sec
        ).astype(np.float32)

        self.sensor_backlog += arrivals
        self.sensor_last_visit_sec += self.low_step_sec

        service_ids = serving_ids[
            (self.uav_status[serving_ids] == int(UAVStatus.SERVING))
            & (service_time[serving_ids] > 0.0)
        ]

        pre_service = self.sensor_backlog.copy()

        service = service_model.collect(
            self.sensor_pos,
            pre_service,
            self.uav_pos,
            service_ids,
        )

        covered = service.owner >= 0
        self.sensor_backlog[covered] = 0.0
        self.sensor_last_visit_sec[covered] = 0.0

        post_service = self.sensor_backlog.copy()

        oob_mask = np.zeros(self.num_uavs, dtype=bool)
        oob_mask[serving_ids] = oob_active

        overflow_full = np.zeros(self.num_uavs, dtype=np.float32)
        overflow_full[serving_ids] = overflow

        self.now_sec += self.low_step_sec
        self.low_step_total += 1

        result = LowStepResult(
            step_in_frame=self.step_in_frame,
            serving_ids=serving_ids.copy(),
            position_before=pos_before,
            position_after=self.uav_pos.copy(),
            requested_displacement=requested,
            executed_displacement=executed,
            move_distance_m=move_distance,
            move_time_sec=move_time,
            service_time_sec=service_time,
            energy_used_frac=energy_used,
            arrivals=arrivals,
            sensor_owner=service.owner,
            packets_pre_service=pre_service,
            packets_post_service=post_service,
            collected_per_sensor=service.collected_per_sensor,
            collected_by_uav=service.collected_by_uav,
            per_uav_owned_max_pre_service=service.per_uav_max,
            covered_max_pre_service=float(pre_service[covered].max(initial=0.0)),
            system_max_pre_service=float(pre_service.max(initial=0.0)),
            system_max_post_service=float(post_service.max(initial=0.0)),
            system_mean_post_service=float(post_service.mean()),
            oob_mask=oob_mask,
            oob_overflow_distance_m=overflow_full,
            dead_during_step=dead_during.copy(),
        )

        self.step_in_frame += 1
        return result

    def mark_unable_to_return_dead(self, energy_model: EnergyModel) -> np.ndarray:
        ids = self.serving_ids()
        if not ids.size:
            return ids
        trip = energy_model.return_trip(self.uav_pos[ids], self.airship_pos)
        unsafe = ids[
            self.uav_battery[ids] < trip.energy_frac + self.return_reserve_frac
        ]
        self.uav_status[unsafe] = int(UAVStatus.DEAD)
        return unsafe

    def complete_upper_frame(
        self,
        plan: FramePlan,
        energy_model: EnergyModel,
    ) -> dict[str, np.ndarray]:
        """Settle return/charge, then release/promote for the next observation."""
        if self.step_in_frame != self.low_steps_per_frame:
            raise RuntimeError("a full set of low steps must finish before settlement")
        nonserving = np.isin(
            plan.assigned_status, [int(UAVStatus.WAITING), int(UAVStatus.CHARGING)]
        )
        failed = nonserving & plan.return_failure_mask
        successful = nonserving & ~failed & (self.uav_status != int(UAVStatus.DEAD))
        self.uav_battery[successful] -= plan.return_energy_frac[successful]
        self.uav_battery[failed] = 0.0
        self.uav_status[failed] = int(UAVStatus.DEAD)
        self.uav_pos[successful] = self.airship_pos

        charging = successful & (plan.assigned_status == int(UAVStatus.CHARGING))
        charge_added = np.zeros(self.num_uavs, dtype=np.float32)
        charge_added[charging] = energy_model.charging_energy_fraction(
            plan.charging_time_sec[charging]
        )
        room = 1.0 - self.uav_battery
        charge_added = np.minimum(charge_added, np.maximum(room, 0.0))
        self.uav_battery += charge_added
        self.uav_battery[:] = np.clip(self.uav_battery, 0.0, 1.0)

        release_threshold = float(section(self.cfg, "zones")["release_battery_frac"])
        released = np.flatnonzero(
            (self.uav_status == int(UAVStatus.CHARGING))
            & (self.uav_battery >= release_threshold)
        )
        self.uav_status[released] = int(UAVStatus.SERVING)

        slot_count = int(section(self.cfg, "zones")["charging_slots"])
        vacancy = slot_count - len(self.charging_ids())
        waiting = self.waiting_ids()
        if vacancy > 0 and waiting.size:
            order = waiting[np.lexsort((waiting, self.uav_battery[waiting]))]
            promoted = order[:vacancy]
            self.uav_status[promoted] = int(UAVStatus.CHARGING)
        else:
            promoted = np.zeros(0, dtype=np.int64)

        self.upper_step += 1
        self.step_in_frame = 0
        return {
            "charge_added_frac": charge_added,
            "released_ids": released,
            "promoted_ids": promoted,
            "failed_return_ids": np.flatnonzero(failed),
        }

    def serving_ids(self) -> np.ndarray:
        return np.flatnonzero(self.uav_status == int(UAVStatus.SERVING)).astype(np.int64)

    def waiting_ids(self) -> np.ndarray:
        return np.flatnonzero(self.uav_status == int(UAVStatus.WAITING)).astype(np.int64)

    def charging_ids(self) -> np.ndarray:
        return np.flatnonzero(self.uav_status == int(UAVStatus.CHARGING)).astype(np.int64)

    def dead_ids(self) -> np.ndarray:
        return np.flatnonzero(self.uav_status == int(UAVStatus.DEAD)).astype(np.int64)
