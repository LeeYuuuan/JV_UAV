from __future__ import annotations

from typing import Any, Mapping

from .joint_runner import JointEpisodeRunner, LowestBatteryRequestPolicy
from .low_env import LowEnv, ZeroLowPolicy
from .low_runner import LowFrameRunner
from .metrics import EpisodeMetrics
from .models import LowestArrivalBatteryScheduler, SimpleEnergyModel
from .scene import Scene
from .upper_env import UpperEnv


def build_smoke_test_runner(cfg: Mapping[str, Any]) -> JointEpisodeRunner:
    scene = Scene(cfg)
    energy = SimpleEnergyModel(cfg)
    metrics = EpisodeMetrics(cfg, scene.sensor_pos)
    low_env = LowEnv(scene, energy, metrics, cfg)
    low_runner = LowFrameRunner(low_env, ZeroLowPolicy(), energy)
    scheduler = LowestArrivalBatteryScheduler(cfg, energy)
    upper_env = UpperEnv(cfg, scene, scheduler, energy, low_runner, metrics)
    return JointEpisodeRunner(upper_env, LowestBatteryRequestPolicy())
