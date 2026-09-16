"""Policy input transforms; physical state and replay stay in raw units."""
from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """A single cumulative distribution shared by all sensor last-visit values."""

    def __init__(self, clip=10.0, min_std=1.0):
        self.count = 0
        self.mean = 0.0
        self.var = 1.0
        self.clip = float(clip)
        self.min_std = float(min_std)
        self.training = True

    def update(self, values):
        if not self.training:
            return
        x = np.asarray(values, dtype=np.float64).ravel()
        if not x.size:
            return
        if not np.isfinite(x).all():
            raise ValueError("non-finite observation")
        n, mean, var = x.size, float(x.mean()), float(x.var())
        if self.count == 0:
            self.count, self.mean, self.var = n, mean, var
            return
        total = self.count + n
        delta = mean - self.mean
        m2 = self.var * self.count + var * n + delta * delta * self.count * n / total
        self.mean += delta * n / total
        self.var = max(m2 / total, 0.0)
        self.count = total

    def normalize(self, values):
        scale = max(np.sqrt(self.var), self.min_std)
        return np.clip((np.asarray(values) - self.mean) / scale, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return dict(vars(self))

    def load_state_dict(self, state):
        self.__dict__.update(state)


def copy_lower(obs):
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def lower_arrays(obs, rms, world_size, fleet_size=None):
    features = obs["active_uav_features"].copy()
    features[:, :2] /= world_size
    if fleet_size is not None:
        ids = np.asarray(obs["active_uav_ids"])
        if fleet_size <= 0 or np.any(ids < 0) or np.any(ids >= fleet_size):
            raise ValueError("active UAV IDs must belong to the configured fleet")
        identity = ids.astype(np.float32)[:, None] / max(fleet_size - 1, 1)
        features = np.concatenate([features, identity], axis=1)
    return rms.normalize(obs["sensor_last_visit_sec"]), features


def upper_observation(scene, energy):
    """Shared actor: all SOCs, own one-hot ID/status, occupancy and return time."""
    n = scene.num_uavs
    status = np.eye(4, dtype=np.float32)[scene.uav_status]
    occupancy = np.asarray([
        len(scene.charging_ids()) / max(int(scene.cfg["zones"]["charging_slots"]), 1),
        len(scene.waiting_ids()) / max(int(scene.cfg["zones"]["waiting_slots"]), 1),
    ], dtype=np.float32)
    times = energy.return_trip(scene.uav_pos, scene.airship_pos).time_sec.copy()
    times[np.isin(scene.uav_status, [1, 2])] = 0.0
    times /= scene.low_step_sec * scene.low_steps_per_frame
    actor = np.concatenate([
        np.broadcast_to(scene.uav_battery, (n, n)), np.eye(n), status,
        np.broadcast_to(occupancy, (n, 2)), times[:, None],
    ], axis=1).astype(np.float32)
    critic = np.concatenate([
        np.column_stack([scene.uav_battery, status, times]).ravel(), occupancy,
    ]).astype(np.float32)
    return {"actor": actor, "critic": critic}
