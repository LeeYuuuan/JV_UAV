"""Optional training rewards, shared by joint training and evaluation."""
from __future__ import annotations
import copy
import numpy as np
from ..low_env import LowReward
from ..upper_env import UpperEnv
from ..factory import build_smoke_test_runner


def build_training_env(env_cfg, train_cfg):
    env_cfg = copy.deepcopy(env_cfg)
    if "failure_responsibility" in train_cfg:
        env_cfg["failure_responsibility"] = copy.deepcopy(train_cfg["failure_responsibility"])
    env = build_smoke_test_runner(env_cfg).upper_env
    reward = train_cfg.get("lower_reward")
    if reward is None:
        return env
    if reward.get("mode") != "topk":
        raise ValueError("lower_reward.mode must be topk")
    k, rho = reward.get("k", 5), reward.get("rho", 0.5)
    scale, offset = reward.get("scale", 300.0), reward.get("offset", 0.0)
    env.low_runner.low_env.reward_model = TopKReward(env_cfg, k, rho, scale, offset)
    if reward.get("terminal_tail", True):
        env = FailureTailEnv(env, train_cfg["sac"]["gamma"], scale, offset)
    return env


class TopKReward:
    def __init__(self, cfg, k=5, rho=0.5, scale=300.0, offset=0.0):
        if k < 1 or not 0 <= rho < 1 or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Require K >= 1, 0 <= rho < 1 and a positive reward scale.")
        if not np.isfinite(offset):
            raise ValueError("Reward offset must be finite.")
        self.k, self.rho, self.scale, self.offset = k, rho, scale, offset
        self.constraints = LowReward(cfg)

    def __call__(self, result, **kwargs):
        _, old = self.constraints(result, **kwargs)
        pre = np.asarray(result.packets_pre_service, dtype=np.float64)
        # Stable sensor-index tie break. Actual full-clear collection is counted once.
        ids = np.argsort(-pre, kind="stable")[:self.k]
        denominator = pre[ids].sum()
        h = float(result.collected_per_sensor[ids].sum() / denominator) if denominator else 0.0
        b = float(result.system_max_post_service)
        terms = {
            "offset": self.offset / self.scale,
            "system_max_post_service": -b / self.scale,
            "topk_clear": self.rho * b * h / self.scale,
            "oob": old["oob"],
            "return_failure": old["return_failure"],
            "return_distance": old["return_distance"],
        }
        return float(sum(terms.values())), terms


class FailureTailEnv(UpperEnv):
    """Settle the last lower transition before exposing it to replay."""

    def __init__(self, env, gamma, scale, offset):
        super().__init__(env.cfg, env.scene, env.scheduler, env.energy_model,
                         env.low_runner, env.metrics)
        self.failure_gamma = float(gamma)
        self.failure_scale = float(scale)
        self.failure_offset = float(offset)

    def step(self, action):
        callback = self.low_runner.on_step
        deferred = []

        def on_step(obs, low_action, result):
            failed = result.dead_during_step.size or result.return_unsafe_ids.size
            last = result.step_in_frame == self.scene.low_steps_per_frame - 1
            if failed or last:
                deferred.append((obs, low_action, result))
            elif callback is not None:
                callback(obs, low_action, result)

        self.low_runner.on_step = on_step
        try:
            transition = super().step(action)
        finally:
            self.low_runner.on_step = callback
        _, _, terminated, _, info = transition
        for obs, low_action, result in deferred:
            if terminated:
                if "terminal_future_backlog" in result.reward_terms:
                    raise RuntimeError("Failure tail already applied.")
                horizon = self.max_frames * self.scene.low_steps_per_frame
                remaining = max(0, horizon - self.scene.low_step_total)
                k = np.arange(1, remaining + 1, dtype=np.float64)
                # All collection stops after failure. Use mean arrivals for the
                # virtual remaining horizon, without consuming dynamics RNG.
                future = (self.scene.sensor_backlog[:, None].astype(np.float64) +
                          self.scene.arrival_rate[:, None] * self.scene.low_step_sec * k)
                maxima = future.max(axis=0)
                tail = np.sum(self.failure_gamma ** k *
                              (self.failure_offset - maxima) / self.failure_scale)
                weight = float(info.get("failure_tail_lower_weight", 1.))
                info["failure_tail"] = dict(unweighted_cost=float(tail), lower_weight=weight,
                                            weighted_lower_cost=float(tail*weight))
                result.reward_terms["terminal_future_backlog"] = float(tail*weight)
                result.reward = float(sum(result.reward_terms.values()))
            if callback is not None:
                callback(obs, low_action, result)
        return transition
