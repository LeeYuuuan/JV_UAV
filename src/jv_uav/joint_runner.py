from __future__ import annotations

from typing import Callable

import numpy as np

from .types import EpisodeTrace
from .upper_env import UpperEnv

UpperPolicy = Callable[[dict[str, np.ndarray]], np.ndarray]


class JointEpisodeRunner:
    """Final-flow entry point. RL code only has to replace the two policies."""

    def __init__(self, upper_env: UpperEnv, upper_policy: UpperPolicy) -> None:
        self.upper_env = upper_env
        self.upper_policy = upper_policy

    def run_episode(self, *, reset_rng: bool = True) -> EpisodeTrace:
        obs, _ = self.upper_env.reset(reset_rng=reset_rng)
        while True:
            action = self.upper_policy(obs)
            obs, _, terminated, truncated, _ = self.upper_env.step(action)
            if terminated or truncated:
                return self.upper_env.trace


class LowestBatteryRequestPolicy:
    """Simple upper smoke-test policy, not an RL baseline."""

    def __init__(self, requests_per_frame: int = 2) -> None:
        self.requests_per_frame = requests_per_frame

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        features = observation["uav_features"]
        battery = features[:, 2]
        dead = features[:, 3 + 3] > 0.5  # status one-hot order ends in DEAD
        candidates = np.flatnonzero(~dead)
        order = candidates[np.lexsort((candidates, battery[candidates]))]
        action = np.zeros(len(features), dtype=np.int8)
        action[order[: self.requests_per_frame]] = 1
        return action
