from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import section


def generate_sensor_positions(cfg: Mapping[str, Any]) -> np.ndarray:
    mode = section(cfg, "sensors").get("distribution", "cluster_uniform")
    if mode == "radial_clusters":
        return generate_radial_sensor_distribution(cfg)["positions"]
    if mode != "cluster_uniform":
        raise ValueError(f"unknown sensor distribution: {mode}")
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


def generate_radial_sensor_distribution(cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Exact radial quotas with clustered cores and local or independent scatter.

    All points are rejection sampled in their requested radial region and map.
    Labels follow the final shuffled sensor order. Legacy maps use the separate
    cluster_uniform mode so saved configurations preserve their original map.
    """
    sensors, world = section(cfg, "sensors"), section(cfg, "world")
    rng = np.random.default_rng(int(section(cfg, "seeds")["map_seed"]))
    count, clusters = int(sensors["count"]), int(sensors["cluster_count"])
    inner_count, inner_clusters = int(sensors["inner_count"]), int(sensors["inner_cluster_count"])
    radius = float(sensors["partition_radius_m"])
    outer_min, outer_max = map(float, sensors["outer_radius_range_m"])
    size = float(world["size_m"])
    airship = np.asarray(world["airship_xy_m"], dtype=float)
    std = float(sensors["cluster_std_m"])
    ratio = float(sensors["cluster_ratio"])
    spread = float(sensors["local_scatter_radius_m"])
    margin = float(sensors["cluster_center_margin_m"])
    separation = float(sensors["min_cluster_center_distance_m"])
    outer_margin = float(sensors.get("outer_sensor_margin_m", 0.0))
    if not np.isfinite(outer_margin) or not 0 <= outer_margin < size / 2:
        raise ValueError("outer_sensor_margin_m must be finite and inside half the map size")
    if not (0 < inner_count < count and 0 < inner_clusters < clusters):
        raise ValueError("radial maps require nonempty inner/outer sensor and cluster groups")
    if not (0 < radius < outer_min < outer_max and 0 < std <= spread and 0 <= ratio <= 1):
        raise ValueError("invalid radial bounds, cluster spread or ratio")
    if not (0 <= margin < size / 2 and separation >= 0):
        raise ValueError("invalid cluster center margin or separation")

    def balanced(total, n):
        values = np.full(n, total // n, dtype=int)
        values[:total % n] += 1
        return values

    independent = sensors.get("scatter_mode", "local") == "independent"
    if sensors.get("scatter_mode", "local") not in ("local", "independent"):
        raise ValueError("unknown scatter mode")
    free_counts = [int(sensors.get("inner_scatter_count", 0)), int(sensors.get("outer_scatter_count", 0))]
    centers, points, labels, scattered = [], [], [], []
    regions = [(inner_count, inner_clusters, (0.0, radius), sensors["inner_center_radius_range_m"]),
               (count - inner_count, clusters - inner_clusters, (outer_min, outer_max), sensors["outer_center_radius_range_m"])]
    for region_index, (total, n, (lo, hi), center_bounds) in enumerate(regions):
        point_margin = outer_margin if region_index == 1 else 0.0
        cmin, cmax = map(float, center_bounds)
        if total < n or not (lo <= cmin < cmax <= hi):
            raise ValueError("cluster center bounds must lie inside their radial region")
        free_count = free_counts[region_index] if independent else 0
        if free_count < 0 or total - free_count < n:
            raise ValueError("scatter quota must leave at least one sensor per cluster")
        group_counts = balanced(total - free_count, n)
        core_counts = group_counts.copy() if independent else balanced(int(round(total * ratio)), n)
        phase = rng.uniform(0, 2 * np.pi)
        for group, (group_count, core_count) in enumerate(zip(group_counts, core_counts)):
            for _ in range(20000):
                angle = phase + group * 2 * np.pi / n + rng.uniform(-0.9 * np.pi / n, 0.9 * np.pi / n)
                candidate = airship + rng.uniform(cmin, cmax) * np.array([np.cos(angle), np.sin(angle)])
                if np.all((candidate >= margin) & (candidate <= size - margin)) and all(np.linalg.norm(candidate - c) >= separation for c in centers):
                    break
            else:
                raise ValueError("cannot place radial cluster centers; relax bounds, margin or separation")
            cluster_id = len(centers)
            centers.append(candidate)
            for is_scattered, amount in [(False, core_count), (True, group_count - core_count)]:
                accepted = []
                for _ in range(20000):
                    if len(accepted) == amount:
                        break
                    if is_scattered:
                        theta = rng.uniform(0, 2 * np.pi)
                        distance = spread * np.sqrt(rng.uniform(0.25, 1.0))
                        p = candidate + distance * np.array([np.cos(theta), np.sin(theta)])
                    else:
                        p = rng.normal(candidate, std, 2)
                    # Validate the actual float32 values that Scene will receive.
                    p = p.astype(np.float32)
                    distance = np.linalg.norm(p.astype(float) - airship)
                    if (np.all((p >= point_margin) & (p <= size - point_margin)) and lo <= distance <= hi
                            and np.linalg.norm(p - candidate) <= spread):
                        accepted.append(p)
                if len(accepted) != amount:
                    raise ValueError("cannot sample requested radial sensor quota")
                points.extend(accepted)
                labels.extend([cluster_id] * amount)
                scattered.extend([is_scattered] * amount)
        # Uniform over the map intersection with this radial region; no cluster
        # proximity or exclusion condition. Independent points have cluster ID -1.
        accepted = []
        for _ in range(100000):
            if len(accepted) == free_count:
                break
            p = rng.uniform(0, size, 2).astype(np.float32)
            distance = np.linalg.norm(p.astype(float) - airship)
            if lo <= distance <= hi and np.all((p >= point_margin) & (p <= size - point_margin)):
                accepted.append(p)
        if len(accepted) != free_count:
            raise ValueError("cannot sample independent radial scatter")
        points.extend(accepted)
        labels.extend([-1] * free_count)
        scattered.extend([True] * free_count)
    order = rng.permutation(count)
    positions = np.asarray(points, dtype=np.float32)[order]
    seed = int(section(cfg, "seeds")["map_seed"])
    overrides = sensors.get("position_overrides_by_seed", {}).get(seed, {})
    original_labels = np.asarray(labels, dtype=np.int64)[order]
    for sensor_id, xy in overrides.items():
        index = int(sensor_id)
        point = np.asarray(xy, dtype=np.float32)
        if not 0 <= index < count or original_labels[index] != -1:
            raise ValueError("position overrides must target independent scatter IDs")
        if point.shape != (2,) or not np.isfinite(point).all() or not np.all((point >= 0) & (point <= size)):
            raise ValueError("position override must be a finite point inside the map")
        was_inner = np.linalg.norm(positions[index].astype(float) - airship) <= radius
        if not was_inner and not np.all((point >= outer_margin) & (point <= size - outer_margin)):
            raise ValueError("outer position override violates outer_sensor_margin_m")
        distance = np.linalg.norm(point.astype(float) - airship)
        if not (distance <= radius if was_inner else outer_min <= distance <= outer_max):
            raise ValueError("position override must preserve the sensor's radial region")
        positions[index] = point
    return {
        "positions": positions,
        "centers": np.asarray(centers, dtype=np.float32),
        "cluster_ids": np.asarray(labels, dtype=np.int64)[order],
        "is_scattered": np.asarray(scattered, dtype=bool)[order],
        "inside": np.linalg.norm(positions.astype(float) - airship, axis=1) <= radius,
    }
