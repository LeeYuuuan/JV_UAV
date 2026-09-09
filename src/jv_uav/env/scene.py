from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping

import numpy as np


class UAVStatus(IntEnum):
    SERVING = 0
    WAITING = 1
    CHARGING = 2
    DEAD = 3


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing config section: {name}")
    return value


class Scene:
    """Owns all static and dynamic variables of the simulated scene."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        seeds = _section(cfg, "seeds")
        world_cfg = _section(cfg, "world")
        users_cfg = _section(cfg, "users")
        uav_cfg = _section(cfg, "uav")

        self.world_size = float(world_cfg["size"])
        self.airship_pos = np.asarray(world_cfg["airship_pos"], np.float32)
        self.max_uavs = int(uav_cfg["max_uavs"])
        self.num_users = int(users_cfg["num_users"])

        self._map_seed = int(seeds["map_seed"])
        self._dyn_seed = int(seeds["world_dyn_seed"])
        self._packet_lam = float(users_cfg["lam"])
        self._cold_start_dt = float(users_cfg.get("cold_start_dt", 60.0))
        self._init_battery = float(uav_cfg.get("init_battery", 1.0))

        self.map_rng = np.random.default_rng(self._map_seed)
        self.dyn_rng = np.random.default_rng(self._dyn_seed)

        # Static scene variables.
        self.user_pos = self._generate_user_positions(users_cfg)
        self.user_lam = np.full(
            self.num_users, self._packet_lam, dtype=np.float32
        )

        # Dynamic scene variables.
        self.user_pkts = np.zeros(self.num_users, dtype=np.float32)
        self.user_last_visit = np.zeros(self.num_users, dtype=np.float32)
        self.uav_pos = np.zeros((self.max_uavs, 2), dtype=np.float32)
        self.uav_prev_pos = np.zeros((self.max_uavs, 2), dtype=np.float32)
        self.uav_status = np.zeros(self.max_uavs, dtype=np.int8)
        self.uav_battery = np.zeros(self.max_uavs, dtype=np.float32)
        self.now = 0.0
        self.low_step = 0

        self.reset()

    def reset(
        self,
        *,
        reset_dyn_rng: bool = False,
        cold_start: bool = True,
    ) -> None:
        if reset_dyn_rng:
            self.dyn_rng = np.random.default_rng(self._dyn_seed)

        self.user_pkts.fill(0.0)
        self.user_last_visit.fill(0.0)
        self.uav_pos[:] = self.airship_pos
        self.uav_prev_pos[:] = self.airship_pos
        self.uav_status.fill(int(UAVStatus.WAITING))
        self.uav_battery.fill(self._init_battery)
        self.now = 0.0
        self.low_step = 0

        if cold_start and self._cold_start_dt > 0.0:
            self.update_packets(self._cold_start_dt)

    def update_packets(self, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        increments = self.dyn_rng.poisson(self.user_lam * dt).astype(np.float32)
        self.user_pkts += increments
        self.user_last_visit += dt
        self.now += dt

    def set_uav_status(self, status: np.ndarray) -> None:
        status = np.asarray(status, dtype=np.int8)
        if status.shape != (self.max_uavs,):
            raise ValueError(f"status must have shape ({self.max_uavs},)")
        if not np.isin(status, [int(x) for x in UAVStatus]).all():
            raise ValueError("unknown UAV status")
        self.uav_status[:] = status

    def serving_ids(self) -> np.ndarray:
        return np.flatnonzero(
            self.uav_status == int(UAVStatus.SERVING)
        ).astype(np.int64)

    def waiting_ids(self) -> np.ndarray:
        return np.flatnonzero(
            self.uav_status == int(UAVStatus.WAITING)
        ).astype(np.int64)

    def charging_ids(self) -> np.ndarray:
        return np.flatnonzero(
            self.uav_status == int(UAVStatus.CHARGING)
        ).astype(np.int64)

    def dead_ids(self) -> np.ndarray:
        return np.flatnonzero(
            self.uav_status == int(UAVStatus.DEAD)
        ).astype(np.int64)

    def _generate_user_positions(
        self,
        users_cfg: Mapping[str, Any],
    ) -> np.ndarray:
        if self.num_users <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        cluster_ratio = float(users_cfg.get("cluster_ratio", 0.8))
        n_clustered = int(round(cluster_ratio * self.num_users))
        n_clustered = int(np.clip(n_clustered, 0, self.num_users))
        n_uniform = self.num_users - n_clustered
        n_clusters = max(1, int(users_cfg.get("n_clusters", 3)))
        cluster_std = float(users_cfg.get("cluster_std", 180.0))
        parts: list[np.ndarray] = []

        if n_clustered:
            margin = min(50.0, self.world_size / 4.0)
            centers = self.map_rng.uniform(
                margin,
                self.world_size - margin,
                size=(n_clusters, 2),
            )
            counts = np.full(n_clusters, n_clustered // n_clusters, dtype=int)
            counts[: n_clustered % n_clusters] += 1
            for center, count in zip(centers, counts):
                if count:
                    parts.append(
                        self.map_rng.normal(center, cluster_std, size=(count, 2))
                    )

        if n_uniform:
            parts.append(
                self.map_rng.uniform(
                    0.0, self.world_size, size=(n_uniform, 2)
                )
            )

        positions = np.vstack(parts).astype(np.float32)
        return np.clip(positions, 0.0, self.world_size).astype(np.float32)
