#!/usr/bin/env python3
"""Inspect a trusted JV checkpoint on fresh rollouts; never save training weights."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from jv_uav.rl.mappo import MAPPO, gae
from jv_uav.rl.sac import SAC
from jv_uav.rl.observations import RunningMeanStd, upper_observation
from jv_uav.rl.training_env import build_training_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        parser.error("output already has diagnostic data; choose a new directory")
    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    initial_stat = checkpoint.stat()
    # Load only checkpoints produced by this trusted project.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env_cfg, cfg = state["env_cfg"], state["train_cfg"]
    original_counters = state["counters"]
    upper_state, lower_actor, rms_state = state["mappo"], state["sac"]["actor"], state["rms"]
    del state  # Replay is not used by this independent probe.
    upper = MAPPO(env_cfg["uavs"]["count"], cfg["mappo"], "cpu")
    upper.load_state_dict(upper_state)
    lower = SAC(env_cfg, cfg["sac"], "cpu")
    lower.actor.load_state_dict(lower_actor)
    rms = RunningMeanStd()
    rms.load_state_dict(rms_state)
    rms.training = False
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg["seeds"]["dynamics_seed"] = args.seed
    env = build_training_env(env_cfg, cfg)
    env.reset()
    env.low_runner.low_policy = lambda obs: lower.act(obs, rms)
    torch.manual_seed(args.seed)
    # A lower bound: maximal allowed movement time minimizes energy for this model.
    dt = env_cfg["time"]["low_step_sec"]
    max_move = env_cfg["uavs"]["max_move_time_sec"]
    energy = env_cfg["energy"]
    power = energy["horizontal_power_w"]*max_move + energy["hover_power_w"]*(dt-max_move)
    if energy["horizontal_power_w"] > energy["hover_power_w"]:
        power = energy["hover_power_w"] * dt
    min_frame_soc = power * env_cfg["time"]["low_steps_per_frame"] / (3600*energy["battery_capacity_wh"])
    records, rollout = [], []
    for frame in range(args.frames):
        obs = upper_observation(env.scene, env.energy_model, upper.observation_mode)
        action, logp, value, probabilities = upper.act(obs, include_probabilities=True)
        battery = env.scene.uav_battery.copy()
        _, reward, terminated, truncated, info = env.step(action)
        roles = info["frame_plan"].assigned_status
        unsafe = np.flatnonzero((roles == 0) & (battery < min_frame_soc))
        done = terminated or truncated
        records.append(dict(frame=frame+1, value=value, reward=reward, done=bool(done),
            battery_start=battery.tolist(), actions=action.tolist(), assigned_status=roles.tolist(),
            p_charge_before=probabilities.tolist(), unavoidable_service_ids=unsafe.tolist(),
            dead_ids=info["dead_ids_end"].tolist(), reward_terms=info["reward_terms"],
            termination_reason=env.trace.frames[-1].termination_reason))
        rollout.append(dict(obs=obs, action=action, logp=logp, value=value, reward=reward, done=done))
        if done and frame+1 < args.frames:
            env.reset(reset_rng=False)
    bootstrap = 0. if rollout[-1]["done"] else upper.value(upper_observation(env.scene, env.energy_model, upper.observation_mode))
    rewards = torch.tensor([r["reward"] * cfg["mappo"]["reward_scale"] for r in rollout])
    values = torch.tensor([r["value"] for r in rollout])
    dones = torch.tensor([r["done"] for r in rollout], dtype=torch.float32)
    advantage, targets = gae(rewards, values, dones, bootstrap, cfg["mappo"]["gamma"], cfg["mappo"]["gae_lambda"])
    normalized = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8) if advantage.numel()>1 else advantage
    # One PPO update on this copied policy/optimizer in RAM. No checkpoint is written.
    update = upper.update(rollout, bootstrap, upper_steps=original_counters["upper_steps"]+args.frames)
    unsafe_credit = []
    for index, (record, sample) in enumerate(zip(records, rollout)):
        after = upper.request_probabilities(sample["obs"])
        record.update(advantage_raw=float(advantage[index]), advantage_normalized=float(normalized[index]),
                      value_target=float(targets[index]), p_charge_after=after.tolist())
        for uid in record["unavoidable_service_ids"]:
            unsafe_credit.append(dict(frame=record["frame"],uav=uid,
                advantage=record["advantage_normalized"],
                p_charge_before=record["p_charge_before"][uid],p_charge_after=float(after[uid]),
                initial_score_gradient_abs=abs(record["advantage_normalized"])*record["p_charge_before"][uid]))
    summary = dict(probe_version="upper_credit_v1", checkpoint=str(checkpoint),
        checkpoint_bytes_at_open=initial_stat.st_size, checkpoint_mtime_ns_at_open=initial_stat.st_mtime_ns,
        original_counters=original_counters, probe_frames=args.frames, probe_seed=args.seed,
        upper_config=cfg["mappo"], reward_config=env_cfg["reward"],
        bootstrap_value=bootstrap, minimum_service_soc=min_frame_soc,
        policy_mode="stochastic upper and lower; frozen lower weights and RMS; fresh environment episodes",
        checkpoint_modified_by_probe=False, update_performed_on_in_memory_copy_only=True,
        ppo_update=update, unavoidable_service_decisions=len(unsafe_credit),
        upper_source_sha256=hashlib.sha256((ROOT/"src/jv_uav/rl/mappo.py").read_bytes()).hexdigest())
    if unsafe_credit:
        summary.update(unsafe_positive_advantage_fraction=float(np.mean([r["advantage"]>0 for r in unsafe_credit])),
            unsafe_charge_probability_increased_fraction=float(np.mean([r["p_charge_after"]>r["p_charge_before"] for r in unsafe_credit])),
            unsafe_initial_score_gradient_median=float(np.median([r["initial_score_gradient_abs"] for r in unsafe_credit])),
            unsafe_charge_probability_mean_before=float(np.mean([r["p_charge_before"] for r in unsafe_credit])),
            unsafe_charge_probability_mean_after=float(np.mean([r["p_charge_after"] for r in unsafe_credit])))
    with gzip.open(args.output/"credit_frames.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    package = args.output.with_name(args.output.name + ".zip")
    with ZipFile(package, "x", compression=ZIP_DEFLATED) as archive:
        for name in ("summary.json", "credit_frames.jsonl.gz"):
            archive.write(args.output/name, name)
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Diagnostic ZIP: {package.resolve()}", flush=True)


if __name__ == "__main__":
    main()
