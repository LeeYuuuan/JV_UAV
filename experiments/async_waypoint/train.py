"""Independent asynchronous hierarchical waypoint experiment."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import numpy as np
import torch
import yaml
from jv_uav.config import load_config
from trainer import Trainer
from plots import curves, write_json


def validate(env_cfg, cfg):
    if cfg.get("lower_update_mode", "per_transition") not in ("per_transition", "matched_upper"):
        raise ValueError("unknown lower update mode")
    energy = env_cfg["energy"]
    for key in ("horizontal_power_w", "hover_power_w", "climb_power_w", "charging_power_w", "horizontal_speed_mps", "climb_speed_mps"):
        if not np.isfinite(energy[key]) or energy[key] <= 0:
            raise ValueError(f"energy.{key} must be finite and positive")
    if energy["service_to_airship_height_m"] < 0 or env_cfg["uavs"]["return_reserve_frac"] < 0:
        raise ValueError("height and reserve cannot be negative")
    if not np.isclose(env_cfg["uavs"]["speed_mps"], energy["horizontal_speed_mps"]):
        raise ValueError("this experiment uses one horizontal speed for outbound and return")
    for key in ("episode_sec", "upper_period_sec", "service_sec", "integration_sec", "backlog_input_scale"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("total_upper_steps", "upper_rollout_steps", "eval_every_upper", "full_eval_every_upper", "save_every_upper"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if sorted(env_cfg["uavs"]["initial_serving_ids"]) != list(range(env_cfg["uavs"]["count"])):
        raise ValueError("this experiment starts all UAVs available for the initial upper decision")
    for name in ("upper", "lower"):
        block = cfg[name]
        if not 0 < block["gamma"] <= 1 or block["reference_sec"] <= 0:
            raise ValueError("invalid physical-time discount")
        if type(block["hidden"]) is not int or block["hidden"] <= 0:
            raise ValueError("hidden size must be positive")
    up, low, rw = cfg["upper"], cfg["lower"], cfg["lower_reward"]
    if not 0 <= up["gae_lambda"] <= 1 or min(up["actor_lr"], up["critic_lr"], up["clip"], up["actor_grad_norm"], up["critic_grad_norm"]) <= 0:
        raise ValueError("invalid upper optimization settings")
    if up["epochs"] <= 0 or up["minibatch_frames"] <= 0 or up["entropy_coef"] < 0 or up["value_coef"] <= 0:
        raise ValueError("invalid upper epochs/minibatch/loss settings")
    if low["heads"] <= 0 or low["hidden"] % low["heads"] or min(low["lr"], low["alpha"], low["max_grad_norm"]) <= 0 or not 0 < low["tau"] <= 1:
        raise ValueError("invalid lower optimization settings")
    if low["batch_size"] <= 0 or low["replay_capacity"] < low["batch_size"] or min(low["random_steps"], low["learning_starts"]) < 0:
        raise ValueError("invalid lower replay/warmup settings")
    if min(rw["backlog_scale"], rw["reference_sec"], rw["distance_scale_m"]) <= 0 or not 0 <= rw["rho"] < 1 or rw["top_k"] <= 0:
        raise ValueError("invalid lower reward settings")
    if min(rw["death_weight"], rw["distance_weight"]) < 0 or min(cfg["upper_reward"]["divisor"], cfg["upper_reward"]["reference_sec"]) <= 0:
        raise ValueError("invalid reward cost/scale")
    r = cfg["responsibility"]
    if not 0 < r["inward_fraction"] < 1 or not 0 <= r["shared_upper_weight"] <= 1 or r["margin_soc"] < 0:
        raise ValueError("invalid reference responsibility settings")


def provenance(output, trainer, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = output / "provenance" / stamp; folder.mkdir(parents=True)
    paths = sorted(set(ROOT.glob("*.py")) | set((ROOT / "src").rglob("*.py")) |
                   set((ROOT / "configs").glob("*.yaml")) | set(Path(__file__).resolve().parent.glob("*.py")))
    hashes = {}
    with ZipFile(folder / "source.zip", "w", ZIP_DEFLATED) as archive:
        for path in paths:
            name = path.relative_to(ROOT).as_posix(); content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest(); archive.writestr(name, content)
    write_json(folder / "manifest.json", dict(revision=trainer.cfg["revision"], source_sha256=hashes,
               arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               starting_upper_steps=trainer.upper_steps, starting_lower_steps=trainer.lower_steps,
               device=str(trainer.device), torch=torch.__version__, numpy=np.__version__))
    write_json(folder / "resolved_config.json", dict(environment=trainer.env_cfg, training=trainer.cfg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs/training_async_waypoint.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upper-steps", type=int, help="Additional upper intervals; every1024 intervals triggers one PPO update")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lower-update-mode", choices=["per_transition", "matched_upper"],
                        help="Explicit cadence override; also supported for checkpoint continuation")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.upper_steps is not None and args.upper_steps <= 0:
        parser.error("upper steps must be positive")
    if args.resume and args.smoke:
        parser.error("cannot change resumed config with --smoke")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((output / "train.jsonl").exists() or (output / "checkpoint.pt").exists()):
        parser.error("output contains a prior run; choose a new output or --resume")
    if args.resume:
        trainer = Trainer.load(args.resume, device)
        if args.seed is not None and args.seed != trainer.cfg["seed"]:
            parser.error("resume seed must match")
        if args.lower_update_mode is not None and args.lower_update_mode != trainer.cfg["lower_update_mode"]:
            trainer.cfg["lower_update_mode"] = args.lower_update_mode
            trainer.env.cfg["lower_update_mode"] = args.lower_update_mode
            trainer.cfg["revision"] = "async_waypoint_600s_matched_v2" if args.lower_update_mode == "matched_upper" else "async_waypoint_600s_per_transition_v2"
            trainer.env.cfg["revision"] = trainer.cfg["revision"]
        if (output / "live_status.json").exists():
            previous = json.loads((output / "live_status.json").read_text())
            if previous["upper_steps"] > trainer.upper_steps:
                parser.error("logs newer than checkpoint; resume into a new --output directory")
    else:
        cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8")); ec = load_config(args.env_config)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.lower_update_mode is not None:
            cfg["lower_update_mode"] = args.lower_update_mode
            cfg["revision"] = "async_waypoint_600s_matched_v2" if args.lower_update_mode == "matched_upper" else "async_waypoint_600s_per_transition_v2"
        if args.smoke:
            cfg.update(episode_sec=180.0, upper_period_sec=60.0, total_upper_steps=8, upper_rollout_steps=4,
                       eval_every_upper=4, full_eval_every_upper=4, save_every_upper=4, evaluation_seeds=[1001])
            cfg["upper"].update(hidden=16, epochs=2, minibatch_frames=2)
            cfg["lower"].update(hidden=16, heads=2, batch_size=2, replay_capacity=128, random_steps=2, learning_starts=2)
        validate(ec, cfg); trainer = Trainer(ec, cfg, device)
    cfg = trainer.cfg; validate(trainer.env_cfg, cfg)
    write_json(output / "resolved_config.json", dict(environment=trainer.env_cfg, training=cfg, device=device))
    provenance(output, trainer, args)
    evaluations = []
    if (output / "eval.jsonl").exists():
        for line in (output / "eval.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row["upper_steps"] <= trainer.upper_steps:
                evaluations.append(row)

    def evaluate(full):
        rows = []
        for seed in cfg["evaluation_seeds"]:
            folder = output / f"eval_{trainer.upper_steps:08d}" / f"seed_{seed}" if full else None
            rows.append(trainer.evaluate(seed, folder))
        with (output / "eval.jsonl").open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        evaluations.extend(rows); write_json(output / "latest_eval.json", rows)
        print("  eval | " + " | ".join(f"seed{r['seed']} alive={r['simulation_sec']/60:.1f}min "
              f"S={r['mean_serving']:.2f} cover={r['sensor_coverage_fraction']:.1%} B={r['time_mean_max_backlog']:.1f} "
              f"dead={len(r['failures'])}" for r in rows), flush=True)

    steps = args.upper_steps or cfg["total_upper_steps"]
    print(f"Async hierarchical waypoint | {cfg['revision']} | {device} | upper period {cfg['upper_period_sec']}s | "
          f"additional intervals {steps} | rollout {cfg['upper_rollout_steps']} | lower updates {cfg['lower_update_mode']}", flush=True)
    print(f"Output: {output.resolve()}", flush=True)
    if trainer.upper_steps == 0:
        evaluate(True)
    began = time.monotonic()
    try:
        with (output / "train.jsonl").open("a", encoding="utf-8") as log, gzip.open(output / "episodes.jsonl.gz", "at", encoding="utf-8", compresslevel=5) as episodes:
            for index in range(steps):
                row = trainer.step(); summary = row.pop("episode_summary")
                if row["upper_steps"] % 10 == 0 or row["terminal"] or index == steps - 1:
                    log.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"); log.flush()
                write_json(output / "live_status.json", row)
                if summary is not None:
                    episodes.write(json.dumps(summary, allow_nan=False, separators=(",", ":")) + "\n"); episodes.flush()
                    if trainer.episodes % 100 == 0:
                        folder = output / "episodes"; folder.mkdir(exist_ok=True)
                        with gzip.open(folder / f"episode_{trainer.episodes:06d}.json.gz", "wt", encoding="utf-8") as stream:
                            json.dump(dict(upper=trainer.env.allocations, lower=trainer.env.paths, collections=trainer.env.collections,
                                           segments=trainer.env.segments, failures=trainer.env.failures), stream, allow_nan=False)
                if row["upper_updated"]:
                    recent = trainer.history[-100:]
                    mean_s = np.mean([r["mean_serving"] for r in recent]) if recent else row["mean_serving_so_far"]
                    print(f"[update {trainer.upper_updates:05d}] rollout_T={trainer.cfg['upper_rollout_steps']} "
                          f"upper={trainer.upper_steps} lower={trainer.lower_steps} SAC={trainer.lower_updates} "
                          f"S_episode_mean={mean_s:.2f} pi={trainer.last_losses['upper_actor_loss']:.4f} "
                          f"vf={trainer.last_losses['upper_critic_loss']:.3f} elapsed={(time.monotonic()-began)/60:.1f}m", flush=True)
                if trainer.upper_steps % cfg["save_every_upper"] == 0:
                    trainer.save(output / "checkpoint.pt")
                if index < steps - 1 and trainer.upper_steps % cfg["eval_every_upper"] == 0:
                    evaluate(trainer.upper_steps % cfg["full_eval_every_upper"] == 0)
                    curves(trainer, output, evaluations)
    except KeyboardInterrupt:
        print("Interrupted; resume the last completed atomic checkpoint.", flush=True)
        return
    trainer.save(output / "checkpoint.pt"); evaluate(True); curves(trainer, output, evaluations)
    print(f"Saved {output/'checkpoint.pt'} (partial PPO rollout and in-flight lower transition retained)", flush=True)


if __name__ == "__main__":
    main()
