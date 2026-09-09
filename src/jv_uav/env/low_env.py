from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dynamics import (
    apply_full_clear,
    assign_sensors_to_nearest_serving_uav,
    clip_displacements_by_norm,
    compute_return_energy_frac,
    compute_serving_energy_frac,
    compute_step_timing,
    move_serving_uavs,
)
from .scene import Scene, UAVStatus


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing config section: {name}")
    return value


@dataclass
class StepMetrics:
    owner_user_to_uav: np.ndarray
    covered_any: np.ndarray
    pkts_before_collect: np.ndarray
    pkts_after_collect: np.ndarray
    collected_by_uav: np.ndarray
    per_uav_collected_sum: np.ndarray
    per_uav_collected_max: np.ndarray
    team_max_sum: float
    max_before: float
    max_after: float


class VariableUAVLowEnv:
    """Distance-dependent lower environment."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.scene = Scene(cfg)

        # Alias retained for easier integration with the previous joint code.
        self.world = self.scene
        self.max_uavs = self.scene.max_uavs
        self.num_users = self.scene.num_users

        time_cfg = _section(cfg, "time")
        uav_cfg = _section(cfg, "uav")
        energy_cfg = _section(cfg, "energy")

        self.high_steps = int(time_cfg["high_steps"])
        self.low_steps_per_high = int(time_cfg["low_steps_per_high"])
        self.low_step_sec = float(time_cfg.get("low_step_sec", 60.0))
        self.max_move_sec = float(time_cfg.get("max_move_sec", 30.0))
        self.total_low_steps = self.high_steps * self.low_steps_per_high

        self.speed = float(uav_cfg.get("speed", 6.0))
        self.move_limit = float(
            uav_cfg.get("move_limit", self.speed * self.max_move_sec)
        )
        self.coverage_radius = float(uav_cfg["coverage_radius"])

        self.p_horiz = float(energy_cfg.get("P_horiz", 257.0))
        self.p_hover = float(energy_cfg.get("P_hover", 326.0))
        self.battery_capacity_wh = float(
            energy_cfg.get("battery_capacity_Wh", 125.0)
        )
        self.deduct_energy = bool(energy_cfg.get("deduct_in_low_env", True))
        self.reserve_frac = float(energy_cfg.get("reserve_frac", 0.0))

        if not 0.0 < self.max_move_sec <= self.low_step_sec:
            raise ValueError("max_move_sec must be in (0, low_step_sec]")
        if self.speed <= 0.0:
            raise ValueError("uav.speed must be positive")
        if self.move_limit > self.speed * self.max_move_sec + 1e-6:
            raise ValueError("move_limit cannot exceed speed * max_move_sec")

        self.traj: list[list[np.ndarray]] = []
        self.status_history: list[np.ndarray] = []
        self.hist_max_pkt: list[float] = []
        self.hist_reward: list[float] = []

    def reset(
        self,
        *,
        initial_status: np.ndarray | None = None,
        reset_dyn_rng: bool = False,
    ) -> dict[str, np.ndarray]:
        self.scene.reset(reset_dyn_rng=reset_dyn_rng, cold_start=True)

        if initial_status is None:
            initial_status = self._default_initial_status()
        self.scene.set_uav_status(initial_status)

        self.traj = [
            [self.scene.uav_pos[i].copy()] for i in range(self.max_uavs)
        ]
        self.status_history = [self.scene.uav_status.copy()]
        self.hist_max_pkt = []
        self.hist_reward = []
        return self._build_obs()

    def step(
        self,
        actions: np.ndarray,
        target_status: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if target_status is not None:
            self.scene.set_uav_status(target_status)

        serving_ids = self.scene.serving_ids()
        old_pos = self.scene.uav_pos.copy()
        self.scene.uav_prev_pos[:] = old_pos

        actions = np.asarray(actions, np.float32)
        if actions.shape != (self.max_uavs, 2):
            raise ValueError(
                f"actions must have shape ({self.max_uavs}, 2), got {actions.shape}"
            )

        actions = clip_displacements_by_norm(actions, self.move_limit)
        new_pos, move_info = move_serving_uavs(
            old_pos,
            actions,
            serving_ids,
            self.scene.world_size,
        )
        self.scene.uav_pos[:] = new_pos

        move_dist, move_time, service_time = compute_step_timing(
            old_pos,
            new_pos,
            serving_ids,
            speed=self.speed,
            low_step_sec=self.low_step_sec,
            max_move_sec=self.max_move_sec,
        )
        energy_used = compute_serving_energy_frac(
            move_time,
            service_time,
            serving_ids,
            p_horiz=self.p_horiz,
            p_hover=self.p_hover,
            battery_capacity_wh=self.battery_capacity_wh,
        )

        if self.deduct_energy and serving_ids.size:
            self.scene.uav_battery[serving_ids] -= energy_used[serving_ids]
            dead_ids = serving_ids[
                self.scene.uav_battery[serving_ids] <= 0.0
            ]
            np.maximum(
                self.scene.uav_battery,
                0.0,
                out=self.scene.uav_battery,
            )
            self.scene.uav_status[dead_ids] = int(UAVStatus.DEAD)

        self.scene.update_packets(self.low_step_sec)
        service_ids = serving_ids[
            (service_time[serving_ids] > 1e-6)
            & (
                self.scene.uav_status[serving_ids]
                == int(UAVStatus.SERVING)
            )
        ]
        metrics = self._collect_data(service_ids)

        return_cost = self._return_energy_frac()
        return_margin = (
            self.scene.uav_battery - return_cost - self.reserve_frac
        )
        reward = self._compute_reward(metrics, move_info, return_margin)

        self.scene.low_step += 1
        self._record_step(metrics, reward)
        done = self.scene.low_step >= self.total_low_steps
        obs = self._build_obs()

        info = {
            "time": float(self.scene.now),
            "low_step": int(self.scene.low_step),
            "high_step": int(
                self.scene.low_step // self.low_steps_per_high
            ),
            "serving_ids": self.scene.serving_ids().copy(),
            "uav_status": self.scene.uav_status.copy(),
            "uav_battery": self.scene.uav_battery.copy(),
            "move_distance": move_dist,
            "move_time_sec": move_time,
            "service_time_sec": service_time,
            "energy_used_frac": energy_used,
            "return_energy_frac": return_cost,
            "return_margin": return_margin.astype(np.float32),
            "covered_count": int(metrics.covered_any.sum()),
            "owner_user_to_uav": metrics.owner_user_to_uav.copy(),
            "pkts_before_collect": metrics.pkts_before_collect.copy(),
            "pkts_after_collect": metrics.pkts_after_collect.copy(),
            "collected_by_uav": metrics.collected_by_uav.copy(),
            "per_uav_collected_sum": metrics.per_uav_collected_sum.copy(),
            "per_uav_collected_max": metrics.per_uav_collected_max.copy(),
            "team_max_sum": float(metrics.team_max_sum),
            "max_before": float(metrics.max_before),
            "max_after": float(metrics.max_after),
            "oob_count": int(move_info["oob_count"]),
            "oob_overflow_dist": float(move_info["overflow_dist"]),
            "reward": float(reward),
        }
        return obs, float(reward), bool(done), info

    @property
    def is_high_boundary(self) -> bool:
        return self.scene.low_step % self.low_steps_per_high == 0

    def set_uav_status(
        self,
        status: np.ndarray,
    ) -> dict[str, np.ndarray]:
        self.scene.set_uav_status(status)
        return self._build_obs()

    def _collect_data(self, service_ids: np.ndarray) -> StepMetrics:
        owner = assign_sensors_to_nearest_serving_uav(
            self.scene.user_pos,
            self.scene.uav_pos,
            service_ids,
            self.coverage_radius,
        )
        pkts_before = self.scene.user_pkts.copy()
        pkts_after, last_visit, collected = apply_full_clear(
            self.scene.user_pkts,
            self.scene.user_last_visit,
            owner,
            self.max_uavs,
        )
        self.scene.user_pkts[:] = pkts_after
        self.scene.user_last_visit[:] = last_visit

        per_uav_sum = collected.sum(axis=1).astype(np.float32)
        per_uav_max = (
            collected.max(axis=1).astype(np.float32)
            if collected.size
            else np.zeros(self.max_uavs, np.float32)
        )
        return StepMetrics(
            owner_user_to_uav=owner,
            covered_any=owner >= 0,
            pkts_before_collect=pkts_before,
            pkts_after_collect=pkts_after.copy(),
            collected_by_uav=collected,
            per_uav_collected_sum=per_uav_sum,
            per_uav_collected_max=per_uav_max,
            team_max_sum=float(per_uav_max.sum()),
            max_before=float(pkts_before.max()) if pkts_before.size else 0.0,
            max_after=float(pkts_after.max()) if pkts_after.size else 0.0,
        )

    def _return_energy_frac(self) -> np.ndarray:
        return compute_return_energy_frac(
            self.scene.uav_pos,
            self.scene.airship_pos,
            speed=self.speed,
            p_horiz=self.p_horiz,
            battery_capacity_wh=self.battery_capacity_wh,
        )

    def _compute_reward(
        self,
        metrics: StepMetrics,
        move_info: dict[str, np.ndarray | int | float],
        return_margin: np.ndarray,
    ) -> float:
        cfg = _section(self.cfg, "reward")
        serving_ids = self.scene.serving_ids()
        threshold = float(cfg.get("safety_threshold", 0.0))
        violation = (
            np.maximum(0.0, threshold - return_margin[serving_ids]).sum()
            if serving_ids.size
            else 0.0
        )
        return float(
            float(cfg.get("collect_weight", 1.0)) * metrics.team_max_sum
            - float(cfg.get("backlog_weight", 1.0)) * metrics.max_after
            - float(cfg.get("oob_penalty", 500.0))
            * int(move_info["oob_count"])
            - float(cfg.get("oob_penalty_scale", 5.0))
            * float(move_info["overflow_dist"])
            - float(cfg.get("safety_weight", 0.0)) * float(violation)
        )

    def _build_obs(self) -> dict[str, np.ndarray]:
        cfg = self.cfg.get("obs", {})
        last_visit = self.scene.user_last_visit.astype(
            np.float32, copy=True
        )
        if bool(cfg.get("normalize_last_visit", False)):
            last_visit /= float(cfg.get("last_visit_norm", 1000.0))

        ids = self.scene.serving_ids()
        pos = self.scene.uav_pos[ids].astype(np.float32, copy=True)
        if bool(cfg.get("normalize_positions", False)):
            pos /= self.scene.world_size

        battery = self.scene.uav_battery[ids, None].astype(np.float32)
        return_cost = self._return_energy_frac()[ids, None]
        return_margin = battery - return_cost - self.reserve_frac
        features = [pos, battery, return_cost, return_margin]

        if bool(cfg.get("include_uav_id", True)):
            denom = max(1, self.max_uavs - 1)
            features.append((ids.astype(np.float32) / denom)[:, None])

        return {
            "global_obs": last_visit,
            "active_uav_obs": np.concatenate(features, axis=1).astype(
                np.float32
            ),
            "active_uav_ids": ids,
            "uav_status": self.scene.uav_status.copy(),
        }

    def _record_step(
        self,
        metrics: StepMetrics,
        reward: float,
    ) -> None:
        for i in range(self.max_uavs):
            self.traj[i].append(self.scene.uav_pos[i].copy())
        self.status_history.append(self.scene.uav_status.copy())
        self.hist_max_pkt.append(float(metrics.max_after))
        self.hist_reward.append(float(reward))

    def _default_initial_status(self) -> np.ndarray:
        ids = np.asarray(
            self.cfg.get("env", {}).get(
                "initial_serving_ids", [0, 1]
            ),
            dtype=np.int64,
        )
        status = np.full(
            self.max_uavs,
            int(UAVStatus.WAITING),
            dtype=np.int8,
        )
        status[ids] = int(UAVStatus.SERVING)
        return status
