from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import section
from .low_runner import LowFrameRunner
from .metrics import EpisodeMetrics
from .models import ChargingScheduler, EnergyModel
from .scene import Scene
from .types import (
    EpisodeTrace,
    FramePlan,
    FrameRecord,
    UAVFrameRecord,
    UAVStatus,
)


class UpperReward:
    """Minimal replaceable upper reward; evaluation metrics remain independent."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        reward = section(cfg, "reward")
        self.serving_weight = float(reward["upper_serving_weight"])
        self.backlog_weight = float(reward["upper_frame_max_backlog_weight"])
        self.dead_weight = float(reward["upper_dead_weight"])

    def __call__(
        self, serving_count: int, frame_max_mean: float, dead_count: int
    ) -> tuple[float, dict[str, float]]:
        terms = {
            "serving": self.serving_weight * serving_count,
            "frame_mean_system_max_post": -self.backlog_weight * frame_max_mean,
            "dead": -self.dead_weight * dead_count,
        }
        return float(sum(terms.values())), terms


class UpperEnv:
    """One upper step owns planning, ten lower steps, settlement, and release."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        scene: Scene,
        scheduler: ChargingScheduler,
        energy_model: EnergyModel,
        low_runner: LowFrameRunner,
        metrics: EpisodeMetrics,
    ) -> None:
        self.cfg = cfg
        self.scene = scene
        self.scheduler = scheduler
        self.energy_model = energy_model
        self.low_runner = low_runner
        self.metrics = metrics
        self.upper_reward = UpperReward(cfg)
        self.max_frames = int(section(cfg, "time")["episode_upper_frames"])
        self.frame_sec = scene.low_step_sec * scene.low_steps_per_frame

    @property
    def trace(self) -> EpisodeTrace:
        return self.metrics.trace

    def reset(self, *, reset_rng: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.scene.reset(reset_rng=reset_rng)
        self.metrics.reset(self.scene.sensor_backlog)
        return self.scene.observe_upper(), {"trace": self.trace}

    def step(
        self, upper_action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        plan = self.scheduler.make_plan(self.scene, upper_action)
        self.scene.begin_upper_frame(plan)
        low_frame = self.low_runner.run_frame(plan)

        settlement: dict[str, np.ndarray] = {}
        if low_frame.completed_all_steps:
            settlement = self.scene.complete_upper_frame(plan, self.energy_model)

        dead_ids = self.scene.dead_ids()
        serving_count = int(np.sum(plan.assigned_status == int(UAVStatus.SERVING)))
        # Final-step return infeasibility is penalized only at the lower level.
        lower_failure_ids = {
            int(uav_id)
            for step in low_frame.steps
            for uav_id in step.return_unsafe_ids
        }
        upper_dead_count = sum(int(uav_id) not in lower_failure_ids for uav_id in dead_ids)
        reward, reward_terms = self.upper_reward(
            serving_count, low_frame.mean_system_max_post_service, upper_dead_count
        )
        self.trace.frames.append(self._make_frame_record(plan, low_frame))

        terminated = bool(low_frame.terminated or dead_ids.size)
        truncated = bool(self.scene.upper_step >= self.max_frames and not terminated)
        observation = self.scene.observe_upper()
        info = {
            "frame_plan": plan,
            "low_frame": low_frame,
            "settlement": settlement,
            "reward_terms": reward_terms,
            "serving_count_during_frame": serving_count,
            "dead_ids_end": dead_ids.copy(),
            "trace": self.trace,
        }
        return observation, reward, terminated, truncated, info

    def render(self, **kwargs):
        """Draw the current scene and trace; see render_env for options."""
        from .render import render_env

        return render_env(self, **kwargs)

    def _make_frame_record(self, plan: FramePlan, low_frame: Any) -> FrameRecord:
        records: list[UAVFrameRecord] = []
        for uav_id in range(self.scene.num_uavs):
            role = UAVStatus(int(plan.assigned_status[uav_id])).name
            trajectory = [plan.frame_start_pos[uav_id].astype(float).tolist()]
            if role == "SERVING":
                trajectory.extend(
                    step.position_after[uav_id].astype(float).tolist()
                    for step in low_frame.steps
                )
            elif (
                low_frame.completed_all_steps
                and not plan.return_failure_mask[uav_id]
                and role in {"WAITING", "CHARGING"}
            ):
                end = self.scene.airship_pos.astype(float).tolist()
                if trajectory[-1] != end:
                    trajectory.append(end)

            collected = float(sum(s.collected_by_uav[uav_id] for s in low_frame.steps))
            owned_max = [s.per_uav_owned_max_pre_service[uav_id] for s in low_frame.steps]
            records.append(
                UAVFrameRecord(
                    uav_id=uav_id,
                    role_during_frame=role,
                    status_at_frame_end=UAVStatus(int(self.scene.uav_status[uav_id])).name,
                    trajectory_xy_m=trajectory,
                    battery_start_frac=float(plan.frame_start_battery[uav_id]),
                    battery_end_frac=float(self.scene.uav_battery[uav_id]),
                    return_time_sec=float(plan.return_time_sec[uav_id]),
                    return_energy_frac=float(plan.return_energy_frac[uav_id]),
                    charging_time_sec=float(plan.charging_time_sec[uav_id]),
                    waiting_time_sec=float(plan.waiting_time_sec[uav_id]),
                    collected_packets=collected,
                    mean_owned_max_pre_service=float(np.mean(owned_max)) if owned_max else 0.0,
                )
            )
        max_values = [s.system_max_post_service for s in low_frame.steps]
        mean_values = [s.system_mean_post_service for s in low_frame.steps]
        return FrameRecord(
            frame_index=len(self.trace.frames),
            duration_sec=len(low_frame.steps) * self.scene.low_step_sec,
            serving_count=int(np.sum(plan.assigned_status == int(UAVStatus.SERVING))),
            dead_ids_end=self.scene.dead_ids().tolist(),
            system_max_post_service_mean=float(np.mean(max_values)) if max_values else 0.0,
            system_mean_post_service_mean=float(np.mean(mean_values)) if mean_values else 0.0,
            uavs=records,
            settled=low_frame.completed_all_steps,
            termination_reason=low_frame.termination_reason or (
                "return_failed_during_settlement"
                if np.any(plan.return_failure_mask) and low_frame.completed_all_steps
                else None
            ),
        )
