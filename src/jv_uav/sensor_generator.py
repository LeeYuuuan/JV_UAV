from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import section


def generate_sensor_positions(cfg: Mapping[str, Any]) -> np.ndarray:
    sensors = section(cfg, "sensors")
    world = section(cfg, "world")
    seeds = section(cfg, "seeds")
    rng = np.random.default_rng(int(seeds["map_seed"]))

    count = int(sensors["count"])
    cluster_count = int(sensors["cluster_count"])
    clustered_count = int(round(count * float(sensors["cluster_ratio"])))
    map_size = float(world["size_m"])
    margin = float(sensors["cluster_center_margin_m"])
    min_distance = float(sensors["min_cluster_center_distance_m"])

    centers: list[np.ndarray] = []
    for _ in range(100_000):
        candidate = rng.uniform(margin, map_size - margin, size=2)
        if all(np.linalg.norm(candidate - c) >= min_distance for c in centers):
            centers.append(candidate)
            if len(centers) == cluster_count:
                break
    if len(centers) != cluster_count:
        raise RuntimeError("could not place sensor cluster centers")

    per_cluster = np.full(cluster_count, clustered_count // cluster_count, int)
    per_cluster[: clustered_count % cluster_count] += 1
    parts: list[np.ndarray] = []
    std = float(sensors["cluster_std_m"])
    for center, n in zip(centers, per_cluster):
        samples = rng.normal(center, std, size=(int(n), 2))
        parts.append(np.clip(samples, 0.0, map_size))
    uniform_count = count - clustered_count
    if uniform_count:
        parts.append(rng.uniform(0.0, map_size, size=(uniform_count, 2)))
    positions = np.vstack(parts).astype(np.float32)
    return positions[rng.permutation(count)]
