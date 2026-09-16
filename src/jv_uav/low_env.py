from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import section
from .metrics import EpisodeMetrics
from .models import EnergyModel, FullClearNearestService
from .scene import Scene
from .types import LowStepResult


class LowReward:
    """The requested first-version lower reward, kept separate from metrics."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        reward = section(cfg, "reward")
        uavs = section(cfg, "uavs")
        self.covered_weight = float(reward["low_covered_max_sum_weight"]) / float(uavs["count"])
        self.system_weight = float(reward["low_system_max_after_weight"])
        self.oob_weight = float(reward["low_oob_uav_weight"])
        self.death_weight = float(reward["low_death_weight"])

    def __call__(
        self,
        result: LowStepResult, 
        *,
        final_return_failure_count: int = 0,
    ) -> tuple[float, dict[str, float]]:

        covered_term = self.covered_weight * float(result.per_uav_owned_max_pre_service.sum())
        system_term = (-self.system_weight * result.system_max_post_service)
        oob_term = (-self.oob_weight * float(result.oob_mask.sum()))
        return_failure_term = (-self.death_weight * final_return_failure_count)

        terms = {
            "covered_max_sum": covered_term,
            "system_max_post_service": system_term,
            "oob": oob_term,
            "return_failure": return_failure_term,
        }

        return float(sum(terms.values())), terms


class LowEnv:
    """Thin lower-level RL interface around Scene.evolve_low_step."""

    def __init__(
        self,
        scene: Scene,
        energy_model: EnergyModel,
        metrics: EpisodeMetrics,
        cfg: Mapping[str, Any],
    ) -> None:
        self.scene = scene
        self.energy_model = energy_model
        self.metrics = metrics
        self.service_model = FullClearNearestService(
            scene.num_uavs, float(section(cfg, "uavs")["coverage_radius_m"])
        )
        self.reward_model = LowReward(cfg)

    def observation(self) -> dict[str, np.ndarray]:
        return self.scene.observe_lower()

    # low_env.py
    def step(self, active_actions: np.ndarray) -> LowStepResult:
        active_ids = self.scene.serving_ids()
        normalized = np.asarray(active_actions, dtype=np.float32)

        expected_shape = (len(active_ids), 2)
        if normalized.shape != expected_shape:
            raise ValueError(f"normalized low action must have shape {expected_shape}")

        normalized = np.clip(normalized, -1.0, 1.0)
        # [-1, 1] → physical displacement in metres
        active_displacements_m = (normalized * self.scene.max_move_distance_m)

        result = self.scene.evolve_low_step(
            active_displacements_m, self.energy_model, self.service_model,
        )

        full_normalized_action = np.zeros((self.scene.num_uavs, 2), dtype=np.float32)
        full_normalized_action[active_ids] = normalized
        result.normalized_action = full_normalized_action

        result.reward, result.reward_terms = self.reward_model(result)
        self.metrics.record_low_step(result)
        return result


    def apply_final_return_penalty(
        self,
        result: LowStepResult,
        unsafe_serving_ids: np.ndarray,
    ) -> None:
        result.return_unsafe_ids = unsafe_serving_ids.copy()

        result.reward, result.reward_terms = self.reward_model(
            result,
            final_return_failure_count=len(unsafe_serving_ids.copy()),
        )


class ZeroLowPolicy:
    """Smoke-test policy. A learned attention policy can use the same callable API."""

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros((len(observation["active_uav_ids"]), 2), dtype=np.float32)
