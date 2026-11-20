"""Epsilon selects a strategy, never averages actor and greedy coordinates."""
import copy
import numpy as np


class WaypointSampler:
    def __init__(self, cfg, env_cfg, seed):
        self.cfg = copy.deepcopy(cfg)
        self.size = float(env_cfg["world"]["size_m"])
        self.radius = float(env_cfg["uavs"]["coverage_radius_m"])
        self.rng = np.random.default_rng(seed)
        self.counts = dict(random=0, greedy=0, actor=0)

    def epsilon(self, physical_steps):
        ratio = np.clip(physical_steps / self.cfg["decay_physical_steps"], 0, 1)
        return float(self.cfg["epsilon_start"] + ratio * (self.cfg["epsilon_end"] - self.cfg["epsilon_start"]))

    def propose(self, obs, actor_action, physical_steps):
        actor_action = np.asarray(actor_action, np.float32)
        if actor_action.shape != (len(obs["ids"]), 2):
            raise ValueError("actor action must match active UAVs")
        eps = self.epsilon(physical_steps)
        labels = []
        probabilities = [self.cfg["random_probability"], self.cfg["greedy_probability"], self.cfg["actor_probability"]]
        for _ in obs["ids"]:
            labels.append(self.rng.choice(["random", "greedy", "actor"], p=probabilities) if self.rng.random() < eps else "actor")
        targets = (actor_action.copy() + 1) * (self.size / 2)
        for i, label in enumerate(labels):
            if label == "random":
                targets[i] = (self.rng.uniform(-1, 1, 2) + 1) * (self.size / 2)
        sensor_pos = obs["sensor_pos"]
        reserved = np.zeros(len(sensor_pos), bool)
        for i, label in enumerate(labels):
            if label != "greedy":
                reserved |= np.linalg.norm(sensor_pos - targets[i], axis=1) <= self.radius
        chosen = [-1] * len(labels)
        for i, label in enumerate(labels):
            if label != "greedy":
                continue
            available = np.flatnonzero(~reserved)
            if not len(available):
                available = np.arange(len(sensor_pos))
            order = np.argsort(-obs["backlog"][available], kind="stable")
            pool = available[order[:self.cfg["candidate_count"]]]
            best = int(pool[np.linalg.norm(sensor_pos[pool] - obs["positions"][i], axis=1).argmin()])
            point = sensor_pos[best] + self.rng.uniform(-self.cfg["greedy_jitter_m"], self.cfg["greedy_jitter_m"], 2)
            targets[i] = np.clip(point, 0, self.size)
            chosen[i] = best
            reserved |= np.linalg.norm(sensor_pos - targets[i], axis=1) <= self.radius
        for label in labels:
            self.counts[label] += 1
        action = np.clip(targets / (self.size / 2) - 1, -1, 1).astype(np.float32)
        return action, dict(epsilon=eps, sources=labels, greedy_sensor_ids=chosen)

    def state_dict(self):
        return copy.deepcopy(dict(rng=self.rng.bit_generator.state, counts=self.counts))

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.counts = copy.deepcopy(state["counts"])
