"""Raw variable-length replay with independent current/next masks and ID maps."""
from __future__ import annotations

import numpy as np
import torch

from .observations import copy_lower, lower_arrays


def pack_observations(observations, rms, world_size, device, fleet_size=None):
    size = max(1, max(len(obs["active_uav_ids"]) for obs in observations))
    b = len(observations)
    features = np.zeros((b, size, 3 if fleet_size is None else 4), np.float32)
    masks = np.zeros((b, size), bool)
    ids = np.full((b, size), -1, np.int64)
    globals_ = []
    for index, obs in enumerate(observations):
        global_, tokens = lower_arrays(obs, rms, world_size, fleet_size)
        n = len(tokens)
        features[index, :n] = tokens
        masks[index, :n] = True
        ids[index, :n] = obs["active_uav_ids"]
        globals_.append(global_)
    return {
        "global": torch.as_tensor(np.stack(globals_), device=device),
        "tokens": torch.as_tensor(features, device=device),
        "mask": torch.as_tensor(masks, device=device),
        "ids": torch.as_tensor(ids, device=device),
    }


class Replay:
    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.items = []
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def add(self, obs, action, reward, next_obs, done):
        action = np.asarray(action, np.float32)
        if action.shape != (len(obs["active_uav_ids"]), 2):
            raise ValueError("action rows must match current active IDs")
        if not done and not len(next_obs["active_uav_ids"]):
            raise ValueError("nonterminal next observation requires serving UAVs")
        item = (copy_lower(obs), action.copy(), float(reward), copy_lower(next_obs), bool(done))
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, rms, world_size, device, fleet_size=None):
        indices = self.rng.choice(len(self.items), batch_size, replace=False)
        items = [self.items[i] for i in indices]
        current = pack_observations([x[0] for x in items], rms, world_size, device, fleet_size)
        next_ = pack_observations([x[3] for x in items], rms, world_size, device, fleet_size)
        actions = np.zeros((*current["mask"].shape, 2), np.float32)
        for i, item in enumerate(items):
            actions[i, :len(item[1])] = item[1]
        return current, torch.as_tensor(actions, device=device), torch.tensor([x[2] for x in items], dtype=torch.float32, device=device), next_, torch.tensor([x[4] for x in items], dtype=torch.bool, device=device)

    def state_dict(self):
        return {"items": self.items, "pos": self.pos, "capacity": self.capacity, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.items, self.pos, self.capacity = state["items"], state["pos"], state["capacity"]
        self.rng.bit_generator.state = state["rng"]
