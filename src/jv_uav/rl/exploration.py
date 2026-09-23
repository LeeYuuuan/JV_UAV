"""Persistent spatial warm-up without sensor positions or privileged backlog."""
from __future__ import annotations

import numpy as np


class WaypointExploration:
    def __init__(self, scene, rng, random_fraction=0.2, *, energy_guard=False, reserve_frac=0.02, energy_model=None):
        from ..models import SimpleEnergyModel
        self.scene = scene
        self.energy_guard = bool(energy_guard)
        self.reserve_frac = float(reserve_frac)
        if not np.isfinite(self.reserve_frac) or not 0 <= self.reserve_frac < 1:
            raise ValueError("warmup reserve must be finite and in [0, 1)")
        self.energy_model = energy_model if energy_model is not None else SimpleEnergyModel(scene.cfg)
        self.counts = dict(random_actions=0, guided_actions=0, turnbacks=0,
                           random_infeasible_forecasts=0, guided_infeasible_forecasts=0)
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
        self.returning = np.zeros(self.fleet, dtype=bool)
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

    def _home_action(self, position):
        delta = self.center - position
        return delta / max(np.linalg.norm(delta), self.step_m)

    def _frame_end_margin(self, position, soc, first_action):
        """Forecast this action then direct inward steps until this frame ends.

        This is a feasible fallback check, not a guarantee under future random
        overrides or a future upper decision to keep serving another frame.
        """
        position = np.asarray(position, dtype=float).copy()
        remaining = self.scene.low_steps_per_frame - self.scene.step_in_frame
        for step in range(remaining):
            action = first_action if step == 0 else self._home_action(position)
            endpoint = np.clip(position + action * self.step_m, 0, self.size)
            move_time = np.linalg.norm(endpoint-position) / self.scene.speed_mps
            soc -= float(self.energy_model.serving_energy_fraction(
                move_time, self.scene.low_step_sec-move_time))
            position = endpoint
            if soc <= 0:
                return -float('inf')
        trip = self.energy_model.return_trip(position[None], self.center)
        if trip.time_sec[0] > self.scene.low_step_sec * self.scene.low_steps_per_frame:
            return -float('inf')
        reserve = max(self.reserve_frac, self.scene.return_reserve_frac)
        return soc - float(trip.energy_frac[0]) - reserve

    def __call__(self, obs):
        ids = obs['active_uav_ids']
        positions = obs['active_uav_features'][:, :2]
        if self.phase is None:
            self.phase = float(self.rng.uniform(0, 2*np.pi))
        active = np.zeros(self.fleet, dtype=bool)
        active[ids] = True
        self.valid[~active] = False
        self.returning[~active] = False
        actions = np.zeros((len(ids), 2), dtype=np.float32)
        for row, uid in enumerate(ids):
            if not self.returning[uid] and (not self.valid[uid] or np.linalg.norm(self.targets[uid]-positions[row]) < self.step_m/2):
                self.targets[uid] = self._target(uid, stratified=not self.valid[uid])
                self.valid[uid] = True
            delta = self.targets[uid].astype(float) - positions[row]
            action = delta / max(np.linalg.norm(delta), self.step_m)
            action *= self.rng.uniform(0.8, 1.0)
            is_random = self.rng.random() < self.random_fraction
            if is_random:
                angle = self.rng.uniform(0, 2*np.pi)
                action = np.sqrt(self.rng.random()) * np.array([np.cos(angle), np.sin(angle)])
            if self.energy_guard:
                soc = float(obs['active_uav_features'][row, 2])
                if is_random:
                    # Deliberately retain energy-unsafe random actions.
                    self.counts['random_actions'] += 1
                    self.counts['random_infeasible_forecasts'] += int(self._frame_end_margin(positions[row], soc, action) < 0)
                else:
                    self.counts['guided_actions'] += 1
                    if not self.returning[uid] and self._frame_end_margin(positions[row], soc, action) < 0:
                        self.returning[uid] = True
                        self.counts['turnbacks'] += 1
                    if self.returning[uid]:
                        action = self._home_action(positions[row])
                        self.counts['guided_infeasible_forecasts'] += int(self._frame_end_margin(positions[row], soc, action) < 0)
            # Keep warm-up endpoints in the map; the learned policy retains the
            # unchanged environment boundary penalty and physical action rules.
            endpoint = np.clip(positions[row] + action*self.step_m, 0, self.size)
            actions[row] = (endpoint-positions[row]) / self.step_m
        return actions

    def state_dict(self):
        return dict(targets=self.targets.copy(), valid=self.valid.copy(), phase=self.phase,
                    returning=self.returning.copy(), counts=dict(self.counts))

    def load_state_dict(self, state):
        self.targets[:] = state['targets']
        self.valid[:] = state['valid']
        self.phase = state['phase']
        self.returning[:] = state.get('returning', False)
        self.counts.update(state.get('counts', {}))
