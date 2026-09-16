from __future__ import annotations

from typing import Callable

import numpy as np

from .low_env import LowEnv
from .models import EnergyModel
from .scene import Scene
from .types import FramePlan, LowFrameResult, LowStepResult

LowPolicy = Callable[[dict[str, np.ndarray]], np.ndarray]


class LowFrameRunner:
    """Runs exactly one upper frame of trajectory decisions."""

    def __init__(
        self,
        low_env: LowEnv,
        low_policy: LowPolicy,
        energy_model: EnergyModel,
        on_step: Callable[[dict[str, np.ndarray], np.ndarray, LowStepResult], None] | None = None,
    ) -> None:
        self.low_env = low_env
        self.low_policy = low_policy
        self.energy_model = energy_model
        self.on_step = on_step

    @property
    def scene(self) -> Scene:
        return self.low_env.scene

    def run_frame(self, plan: FramePlan) -> LowFrameResult:
        steps = []
        reason: str | None = None
        total = self.scene.low_steps_per_frame
        for index in range(total):
            obs = self.low_env.observation()
            action = self.low_policy(obs)
            result = self.low_env.step(action)
            steps.append(result)

            if result.dead_during_step.size:
                reason = "battery_depleted_during_low_step"

            elif index == total - 1:
                unsafe = self.scene.mark_unable_to_return_dead(self.energy_model)
                self.low_env.apply_final_return_penalty(result, unsafe)
                if unsafe.size:
                    reason = "cannot_return_to_airship"

            if self.on_step is not None:
                self.on_step(obs, action, result)
            if reason is not None:
                break

        completed = len(steps) == total
        return LowFrameResult(
            steps=steps,
            completed_all_steps=completed,
            terminated=reason is not None,
            termination_reason=reason,
        )
