from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import section
from .types import EpisodeTrace, LowStepResult


class EpisodeMetrics:
    """Evaluation-only accumulators; none of these values enters an observation."""

    def __init__(self, cfg: Mapping[str, Any], sensor_pos: np.ndarray) -> None:
        self.world_size = float(section(cfg, "world")["size_m"])
        self.radius = float(section(cfg, "uavs")["coverage_radius_m"])
        self.grid_size = int(section(cfg, "metrics")["coverage_grid_size"])
        axis = (np.arange(self.grid_size, dtype=np.float32) + 0.5) * (
            self.world_size / self.grid_size
        )
        gx, gy = np.meshgrid(axis, axis, indexing="xy")
        self.grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1)
        self.sensor_pos = sensor_pos
        self.trace = EpisodeTrace()
        self.reset(np.zeros(len(sensor_pos), dtype=np.float32))

    def reset(self, initial_packets: np.ndarray) -> EpisodeTrace:
        self.trace = EpisodeTrace(
            backlog_max_post_timeline=[float(initial_packets.max(initial=0.0))],
            backlog_mean_post_timeline=[float(initial_packets.mean())],
            sensor_visit_count=np.zeros(len(self.sensor_pos), dtype=np.int64),
            sensor_collected_packets=np.zeros(len(self.sensor_pos), dtype=np.float64),
            coverage_grid_step_count=np.zeros((self.grid_size, self.grid_size), dtype=np.int64),
            uav_presence_grid_count=np.zeros((self.grid_size, self.grid_size), dtype=np.int64),
        )
        return self.trace

    def record_low_step(self, result: LowStepResult) -> None:
        self.trace.backlog_max_post_timeline.append(result.system_max_post_service)
        self.trace.backlog_mean_post_timeline.append(result.system_mean_post_service)
        covered = result.sensor_owner >= 0
        self.trace.sensor_visit_count[covered] += 1
        self.trace.sensor_collected_packets += result.collected_per_sensor

        active_positions = result.position_after[result.serving_ids]
        if active_positions.size:
            distances = np.linalg.norm(
                self.grid_points[:, None, :] - active_positions[None, :, :], axis=2
            )
            covered_grid = np.any(distances <= self.radius, axis=1)
            self.trace.coverage_grid_step_count += covered_grid.reshape(
                self.grid_size, self.grid_size
            )
            cell = np.floor(active_positions / self.world_size * self.grid_size).astype(int)
            cell = np.clip(cell, 0, self.grid_size - 1)
            for x_index, y_index in cell:
                self.trace.uav_presence_grid_count[y_index, x_index] += 1
