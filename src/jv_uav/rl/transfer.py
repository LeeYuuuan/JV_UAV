"""Explicit Paper1 notebook actor transfer; no critic/optimizer/replay import."""
from pathlib import Path
import hashlib
import torch
from torch import nn


class Paper1Actor(nn.Module):
    """Paper1 feature order, optionally extended with separate return SOC costs."""
    def __init__(self, fleet, hidden=512, include_return_soc=False, include_own_return_soc=False):
        super().__init__()
        self.fleet = fleet
        self.include_return_soc = include_return_soc
        self.include_own_return_soc = include_own_return_soc
        if include_own_return_soc and not include_return_soc:
            raise ValueError("own return SOC requires the separate return SOC feature block")
        layers = []
        for i in range(4):
            layers.extend([nn.Linear(fleet + 8 + (fleet if include_return_soc else 0) + int(include_own_return_soc) if i == 0 else hidden, hidden), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.logits = nn.Linear(hidden, 2)

    def old_features(self, obs):
        n = self.fleet
        soc, identity = obs[..., :n], obs[..., n:2*n]
        own_soc = (soc * identity).sum(dim=-1, keepdim=True)
        ids = (identity * torch.arange(n, device=obs.device, dtype=obs.dtype)).sum(dim=-1, keepdim=True) / max(n-1, 1)
        features = torch.cat([own_soc, obs[..., 2*n:2*n+4], obs[..., 2*n+4:2*n+6], ids, soc], dim=-1)
        if self.include_return_soc:
            costs = obs[..., 2*n+6:3*n+6]
            features = torch.cat([features, costs], dim=-1)
            if self.include_own_return_soc:
                own_cost = (costs * identity).sum(dim=-1, keepdim=True)
                # Append a dedicated self feature without moving legacy columns.
                features = torch.cat([features, own_cost], dim=-1)
        return features

    def categorical_logits(self, obs):
        return self.logits(self.net(self.old_features(obs)))

    def forward(self, obs):
        logits = self.categorical_logits(obs)
        # Old categorical action 0 requests charge; new Bernoulli action 1 does.
        return logits[..., :1] - logits[..., 1:2]


def read_paper1_actor(path, fleet, observation_mode="arrival_soc", *, include_own_return_soc=False):
    if observation_mode not in ("arrival_soc", "current_soc_return_soc"):
        raise ValueError("unsupported Paper1 target observation mode")
    if include_own_return_soc and observation_mode != "current_soc_return_soc":
        raise ValueError("own return SOC requires current_soc_return_soc observations")
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "actor" not in payload:
        raise ValueError("Expected a Paper1 notebook checkpoint with an actor state dict")
    state = payload["actor"]
    required = {f"net.{i}.{kind}" for i in (0, 2, 4, 6) for kind in ("weight", "bias")}
    required |= {"logits.weight", "logits.bias"}
    if set(state) != required or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Unsupported Paper1 actor architecture; expected four ReLU layers and two logits")
    if state["net.0.weight"].ndim != 2:
        raise ValueError("Source actor first layer must be a matrix")
    source_fleet = state["net.0.weight"].shape[1] - 8
    if source_fleet != fleet:
        raise ValueError(f"Paper1 actor input implies {source_fleet} UAVs ({source_fleet+8} features), current environment has {fleet}. Supply a matching checkpoint; no padding/truncation is performed.")
    hidden = state["net.0.weight"].shape[0]
    # Validate all parameter shapes without consuming the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = Paper1Actor(fleet, hidden)
        model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Paper1 actor contains non-finite parameters")
    metadata = dict(source_path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_fleet=source_fleet, hidden=hidden, source_charge_action=0,
                    target_charge_action=1, soc_mapping="current_minus_return_cost",
                    transferred="actor_only", critic="fresh", optimization="joint_finetune",
                    actor_input_order=["own_arrival_soc", "own_status_onehot_4", "charging_ratio",
                                       "waiting_ratio", "id_div_N_minus_1", "all_arrival_soc"],
                    actor_input_dim=fleet+8, source_soc="current_soc_fraction",
                    upper_running_normalization=False, negative_arrival_soc="preserved",
                    probability_mapping="sigmoid(old_charge_logit_0 - old_serve_logit_1)")
    if observation_mode == "current_soc_return_soc":
        # The original 14 features now retain their original current-SOC meaning.
        # Zero extra columns preserve the old policy until return-cost weights learn.
        state = dict(state)
        weight = state["net.0.weight"]
        state["net.0.weight"] = torch.cat([weight, weight.new_zeros(hidden, fleet + int(include_own_return_soc))], dim=1)
        metadata.update(soc_mapping="current_soc_plus_separate_return_cost",
                        actor_input_dim=2*fleet+8+int(include_own_return_soc), added_return_cost_columns="zero_initialized",
                        actor_input_order=["own_current_soc", "own_status_onehot_4", "charging_ratio",
                                           "waiting_ratio", "id_div_N_minus_1", "all_current_soc",
                                           "all_return_energy_soc"])
        if include_own_return_soc:
            metadata["actor_input_order"].append("own_return_energy_soc")
        metadata["include_own_return_soc"] = bool(include_own_return_soc)
        metadata.pop("negative_arrival_soc", None)
    return state, metadata
