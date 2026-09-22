"""Persistent spatial warm-up without sensor positions or privileged backlog."""
from __future__ import annotations

import numpy as np


class WaypointExploration:
    def __init__(self, scene, rng, random_fraction=0.2):
        self.size = scene.world_size
        self.center = scene.airship_pos.astype(float).copy()
        self.fleet = scene.num_uavs
        self.step_m = scene.max_move_distance_m
        self.rng = rng
        self.random_fraction = float(random_fraction)
        if not 0 <= self.random_fraction <= 1:
            raise ValueError("warmup_random_fraction must be in [0, 1]")
        self.reset()

    def reset(self):
        self.targets = np.zeros((self.fleet, 2), dtype=np.float32)
        self.valid = np.zeros(self.fleet, dtype=bool)
        self.phase = None

    def _target(self, uid, stratified):
        for _ in range(128):
            angle = (self.phase + 2 * np.pi * uid / self.fleet
                     + self.rng.uniform(-np.pi / self.fleet, np.pi / self.fleet)
                     if stratified else self.rng.uniform(0, 2 * np.pi))
            radius = np.sqrt(self.rng.uniform((0.25*self.size)**2, (0.65*self.size)**2))
            point = self.center + radius * np.array([np.cos(angle), np.sin(angle)])
            if np.all((point >= 0) & (point <= self.size)):
                return point
        return self.rng.uniform(0, self.size, 2)

    def __call__(self, obs):
        ids = obs['active_uav_ids']
        positions = obs['active_uav_features'][:, :2]
        if self.phase is None:
            self.phase = float(self.rng.uniform(0, 2*np.pi))
        active = np.zeros(self.fleet, dtype=bool)
        active[ids] = True
        self.valid[~active] = False
        actions = np.zeros((len(ids), 2), dtype=np.float32)
        for row, uid in enumerate(ids):
            if not self.valid[uid] or np.linalg.norm(self.targets[uid]-positions[row]) < self.step_m/2:
                self.targets[uid] = self._target(uid, stratified=not self.valid[uid])
                self.valid[uid] = True
            delta = self.targets[uid].astype(float) - positions[row]
            action = delta / max(np.linalg.norm(delta), self.step_m)
            action *= self.rng.uniform(0.8, 1.0)
            if self.rng.random() < self.random_fraction:
                angle = self.rng.uniform(0, 2*np.pi)
                action = np.sqrt(self.rng.random()) * np.array([np.cos(angle), np.sin(angle)])
            # Keep warm-up endpoints in the map; the learned policy retains the
            # unchanged environment boundary penalty and physical action rules.
            endpoint = np.clip(positions[row] + action*self.step_m, 0, self.size)
            actions[row] = (endpoint-positions[row]) / self.step_m
        return actions

    def state_dict(self):
        return dict(targets=self.targets.copy(), valid=self.valid.copy(), phase=self.phase)

    def load_state_dict(self, state):
        self.targets[:] = state['targets']
        self.valid[:] = state['valid']
        self.phase = state['phase']
