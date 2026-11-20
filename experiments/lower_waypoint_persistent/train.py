"""Train persistent real-speed waypoints with epsilon/greedy/actor exploration."""
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
from persistent_trainer import Trainer
from persistent_plots import curves, write_json


def provenance(output, trainer, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = output / "provenance" / stamp
    folder.mkdir(parents=True)
    files = sorted(set(ROOT.glob("*.py")) | set((ROOT / "src").rglob("*.py")) |
                   set((ROOT / "configs").glob("*.yaml")) | set(Path(__file__).resolve().parent.glob("*.py")))
    hashes = {}
    with ZipFile(folder / "source.zip", "w", ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(ROOT).as_posix(); content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest(); archive.writestr(name, content)
    write_json(folder / "manifest.json", dict(revision=trainer.cfg["revision"], source_sha256=hashes,
               starting_steps=trainer.steps, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               device=str(trainer.device), torch=torch.__version__, numpy=np.__version__))
    write_json(folder / "resolved_config.json", dict(environment=trainer.env_cfg, training=trainer.cfg))


def validate(cfg):
    for key in ("total_steps", "eval_every", "full_eval_every", "save_every", "plot_every", "log_every_episodes", "trace_every_steps"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 1 <= cfg["long_count_min"] <= cfg["long_count_max"] <= 6:
        raise ValueError("long group count invalid")
    if any(type(cfg[k]) is not int for k in ("episode_steps", "block_steps", "short_steps", "long_count_min", "long_count_max")):
        raise ValueError("group and episode horizons/counts must be integers")
    if not 0 < cfg["short_steps"] < cfg["block_steps"] or cfg["episode_steps"] <= 0:
        raise ValueError("invalid episode/group horizons")
    if cfg["step_sec"] <= 0 or cfg["observation_backlog_scale"] <= 0:
        raise ValueError("time/input scales must be positive")
    rw, sc = cfg["reward"], cfg["sac"]
    if rw["backlog_scale"] <= 0 or rw["reference_sec"] <= 0 or not 0 <= rw["rho"] < 1:
        raise ValueError("invalid reward settings")
    if not np.isfinite(rw.get("movement_weight", 0.0)) or rw.get("movement_weight", 0.0) < 0:
        raise ValueError("movement_weight must be finite and nonnegative")
    if not 0 < sc["gamma_per_reference"] <= 1 or sc["hidden"] % sc["heads"] or sc["alpha"] <= 0:
        raise ValueError("invalid SAC discount/architecture/alpha")
    if sc["batch_size"] <= 0 or sc["replay_capacity"] < sc["batch_size"]:
        raise ValueError("invalid replay batch/capacity")
    if type(cfg["updates_per_physical_step"]) is not int or cfg["updates_per_physical_step"] <= 0 or cfg["arrival_tolerance_m"] <= 0:
        raise ValueError("invalid update ratio or arrival tolerance")
    ex = cfg["exploration"]
    probabilities = [ex["random_probability"], ex["greedy_probability"], ex["actor_probability"]]
    if min(probabilities) < 0 or not np.isclose(sum(probabilities), 1.0):
        raise ValueError("exploration probabilities must sum to1")
    if not 0 <= ex["epsilon_end"] <= ex["epsilon_start"] <= 1 or ex["decay_physical_steps"] <= 0:
        raise ValueError("invalid epsilon decay")
    if type(ex["candidate_count"]) is not int or ex["candidate_count"] <= 0 or ex["greedy_jitter_m"] < 0:
        raise ValueError("invalid greedy candidate count/jitter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs/training_lower_waypoint_persistent.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="Additional physical60s steps; ends at next natural waypoint decision boundary")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.steps is not None and args.steps <= 0:
        parser.error("steps must be positive")
    if args.resume and args.smoke:
        parser.error("smoke cannot change a resumed run")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((output / "train.jsonl").exists() or (output / "checkpoint.pt").exists()):
        parser.error("Output contains a prior run; use --resume or a new directory")
    if args.resume:
        trainer = Trainer.load(args.resume, device)
        if args.seed is not None and args.seed != trainer.cfg["seed"]:
            parser.error("resume seed must match saved config")
        # Avoid mixing a replayed suffix with log rows written after this checkpoint.
        if (output / "train.jsonl").exists():
            with (output / "train.jsonl").open("rb") as stream:
                stream.seek(max(0, (output / "train.jsonl").stat().st_size - 65536))
                lines = stream.read().splitlines()
            for line in reversed(lines):
                try:
                    last = json.loads(line)
                except ValueError:
                    continue
                if last["training_steps"] > trainer.steps:
                    parser.error("checkpoint is older than output logs; resume into a new --output directory")
                break
    else:
        cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.smoke:
            cfg["exploration"].update(epsilon_start=1.0, epsilon_end=1.0)
            cfg.update(total_steps=12, episode_steps=8, block_steps=4, short_steps=2, eval_every=4, full_eval_every=8,
                       save_every=4, plot_every=4, log_every_episodes=1, evaluation_seeds=[1001])
            cfg["sac"].update(hidden=16, heads=2, batch_size=2, replay_capacity=64, learning_starts=2)
        validate(cfg)
        trainer = Trainer(load_config(args.env_config), cfg, device)
    cfg = trainer.cfg
    validate(cfg)
    write_json(output / "resolved_config.json", dict(environment=trainer.env_cfg, training=cfg, device=device))
    provenance(output, trainer, args)
    evaluations = []
    if (output / "eval.jsonl").exists():
        for line in (output / "eval.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row["training_steps"] <= trainer.steps:
                evaluations.append(row)

    def evaluate(full=False, baselines=False):
        summaries = []
        for seed in cfg["evaluation_seeds"]:
            folder = output / f"eval_{trainer.steps:08d}" / f"seed_{seed}"
            summaries.append(trainer.evaluate(seed, folder, detailed=full))
            if baselines:
                summaries.append(trainer.evaluate(seed, output / "eval_00000000" / f"random_seed_{seed}", mode="random"))
                for mode in ("center", "greedy_reference"):
                    summaries.append(trainer.evaluate(seed, mode=mode))
        with (output / "eval.jsonl").open("a", encoding="utf-8") as stream:
            for summary in summaries:
                stream.write(json.dumps(summary, allow_nan=False) + "\n")
            stream.flush()
        evaluations.extend(summaries); write_json(output / "latest_eval.json", summaries)
        print("  eval | " + " | ".join(f"{r['policy_mode']} seed{r['seed']} cover={r['sensor_coverage_fraction']:.1%} "
              f"B_max={r['mean_max_backlog']:.1f} pkt/s={r['collection_rate_packets_sec']:.2f}"
              for r in summaries), flush=True)
        print(f"  trajectory plots: {output / f'eval_{trainer.steps:08d}'} / seed_* / trajectory.png", flush=True)

    target_steps = args.steps or cfg["total_steps"]
    print(f"Persistent-waypoint SAC | revision {cfg['revision']} | device {device} | additional physical steps {target_steps} | "
          f"{cfg['episode_steps']} steps/episode, {cfg['step_sec']:g}s/step | seed {cfg['seed']}", flush=True)
    print(f"Output: {output.resolve()}", flush=True)
    print(f"Movement penalty weight: {cfg['reward'].get('movement_weight', 0.0):g} (mean active distance / map diagonal)", flush=True)
    if trainer.steps == 0:
        evaluate(True, baselines=True)
    started = time.monotonic()
    ending = trainer.steps + target_steps
    thresholds = {key: (trainer.steps // cfg[key] + 1) * cfg[key] for key in
                  ["save_every", "eval_every", "full_eval_every", "plot_every", "trace_every_steps"]}
    def crossed(key):
        if trainer.steps < thresholds[key]:
            return False
        while thresholds[key] <= trainer.steps:
            thresholds[key] += cfg[key]
        return True
    try:
        with (output / "train.jsonl").open("a", encoding="utf-8") as log, gzip.open(output / "episodes.jsonl.gz", "at", encoding="utf-8", compresslevel=5) as episodes:
            while trainer.steps < ending:
                row = trainer.step(); summary = row.pop("episode_summary")
                # Commitments can cross logging thresholds by several physical slots.
                if crossed("trace_every_steps") or row["done"] or trainer.steps >= ending:
                    log.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"); log.flush()
                write_json(output / "live_status.json", row)
                if summary is not None:
                    episodes.write(json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n"); episodes.flush()
                    if trainer.episodes % 100 == 0:
                        folder = output / "episodes"; folder.mkdir(exist_ok=True)
                        with gzip.open(folder / f"episode_{trainer.episodes:06d}.json.gz", "wt", encoding="utf-8") as stream:
                            json.dump(dict(rows=trainer.env.records, decisions=trainer.env.decision_records), stream, allow_nan=False)
                    if trainer.episodes % cfg["log_every_episodes"] == 0:
                        recent = trainer.history[-cfg["log_every_episodes"]:]
                        print(f"[SAC update {trainer.updates:07d}] physical_steps={trainer.steps:08d} decisions={trainer.decisions:07d} ep={trainer.episodes:06d} "
                              f"R={np.mean([r['lower_return'] for r in recent]):.2f} "
                              f"cover={np.mean([r['sensor_coverage_fraction'] for r in recent]):.1%} "
                              f"B_max={np.mean([r['mean_max_backlog'] for r in recent]):.1f} "
                              f"elapsed={(time.monotonic() - started)/60:.1f}m", flush=True)
                if crossed("save_every"):
                    trainer.save(output / "checkpoint.pt")
                full = crossed("full_eval_every")
                if (crossed("eval_every") or full) and trainer.steps < ending:
                    evaluate(full)
                if crossed("plot_every"):
                    curves(trainer, output, evaluations)
    except KeyboardInterrupt:
        # A KeyboardInterrupt can occur during a partial optimizer/environment step.
        # Keep the last atomic, completed checkpoint rather than saving that state.
        print("Interrupted; resume from the last completed checkpoint.", flush=True)
        curves(trainer, output, evaluations)
        return
    trainer.save(output / "checkpoint.pt"); evaluate(True); curves(trainer, output, evaluations)
    print(f"Saved {output / 'checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
