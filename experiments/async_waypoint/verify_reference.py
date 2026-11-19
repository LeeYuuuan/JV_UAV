"""Handwritten conservative reference, solely to audit long physical rollouts."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from engine import Engine, Role, Phase
from jv_uav.config import load_config


def reference_run(env_cfg, cfg, seed):
    env = Engine(env_cfg, cfg, seed=seed)
    coverage = np.linalg.norm(env.xy[:, None] - env.xy[None, :], axis=-1) <= env_cfg["uavs"]["coverage_radius_m"]
    while not env.done:
        previous_time = env.now
        if env.upper_due:
            # Rotate2 serving,2 charging,2 waiting through the actual allocator.
            env.apply_upper((env.role != Role.CHARGE).astype(int))
        elif env.current_uid is not None:
            uid = env.current_uid
            candidates = np.vstack((env.xy, env.home))
            move = np.linalg.norm(candidates - env.pos[uid], axis=1) / env.energy.horizontal_speed_mps
            cost = env.energy.serving_energy_fraction(move, np.full(len(move), cfg["service_sec"]))
            cost += env.energy.return_trip(candidates, env.home).energy_frac
            safe = cost + .02 <= env.soc[uid]
            weights = (env.backlog / max(float(env.backlog.max()), 1.0)) ** 4
            for other in range(env.n):
                if other != uid and env.phase[other] in (Phase.MOVE, Phase.SERVICE):
                    dest = env.target[other] if env.phase[other] == Phase.MOVE else env.pos[other]
                    weights[np.linalg.norm(env.xy - dest, axis=1) <= env_cfg["uavs"]["coverage_radius_m"]] = 0
            scores = np.concatenate((coverage @ weights, [0.0]))
            scores[~safe] = -np.inf
            target = candidates[int(scores.argmax())] if safe.any() else env.home
            env.apply_lower(target / env.size * 2 - 1)
        else:
            env.advance()
        assert env.now >= previous_time
        assert np.sum(env.role == Role.CHARGE) <= env_cfg["zones"]["charging_slots"]
        assert np.sum(env.role == Role.WAIT) <= env_cfg["zones"]["waiting_slots"]
        assert np.all(env.soc >= 0) and np.all(env.soc <= 1 + 1e-7)
        assert np.isclose(env.initial_packets + env.arrived_packets - env.collected, env.backlog.sum())
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ec = load_config(ROOT / "configs/default.yaml")
    cfg = yaml.safe_load((ROOT / "configs/training_async_waypoint.yaml").read_text())
    results = [reference_run(ec, cfg, seed).summary() | dict(seed=seed, policy="handwritten_reference_not_RL") for seed in (1001, 1002, 1003)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
