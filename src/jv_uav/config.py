from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing configuration section: {name}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("The YAML root must be a mapping")
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    world = section(cfg, "world")
    sensors = section(cfg, "sensors")
    uavs = section(cfg, "uavs")
    time = section(cfg, "time")
    energy = section(cfg, "energy")
    zones = section(cfg, "zones")
    service = section(cfg, "service")

    size = float(world["size_m"])
    airship = world["airship_xy_m"]
    if size <= 0 or len(airship) != 2:
        raise ValueError("world geometry is invalid")
    if not all(0 <= float(v) <= size for v in airship):
        raise ValueError("airship must be inside the map")

    if int(sensors["count"]) <= 0:
        raise ValueError("sensors.count must be positive")
    if float(sensors["arrival_rate_per_sec"]) < 0:
        raise ValueError("arrival rate cannot be negative")

    n_uavs = int(uavs["count"])
    if n_uavs <= 0:
        raise ValueError("uavs.count must be positive")
    if not 0 <= float(uavs["initial_battery_frac"]) <= 1:
        raise ValueError("initial battery must be in [0, 1]")
    if float(uavs["speed_mps"]) <= 0:
        raise ValueError("uav speed must be positive")
    if not 0 < float(uavs["max_move_time_sec"]) <= float(time["low_step_sec"]):
        raise ValueError("max move time must fit inside a low step")
    initial_ids = [int(x) for x in uavs.get("initial_serving_ids", [])]
    if len(initial_ids) != len(set(initial_ids)):
        raise ValueError("initial_serving_ids contains duplicates")
    if any(i < 0 or i >= n_uavs for i in initial_ids):
        raise ValueError("initial_serving_ids contains an invalid ID")

    if int(time["low_steps_per_frame"]) <= 0:
        raise ValueError("low_steps_per_frame must be positive")
    if int(time["episode_upper_frames"]) <= 0:
        raise ValueError("episode_upper_frames must be positive")
    if float(energy["battery_capacity_wh"]) <= 0:
        raise ValueError("battery capacity must be positive")
    if int(zones["charging_slots"]) < 0 or int(zones["waiting_slots"]) < 0:
        raise ValueError("slot counts cannot be negative")
    if zones["allocation_rule"] != "lowest_arrival_battery":
        raise ValueError("only lowest_arrival_battery is implemented")
    if service["mode"] != "full_clear_nearest":
        raise ValueError("only full_clear_nearest is implemented")
