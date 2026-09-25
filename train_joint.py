"""Train or evaluate the confirmed attention SAC + MAPPO hierarchy."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import gzip
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from jv_uav import load_config
from jv_uav.rl.trainer import JointTrainer
from jv_uav.rl.curves import TrainingCurves
from jv_uav.rl.provenance import record_run


def format_progress(rows, elapsed):
    """Summarize the frames since the previous console line."""
    last = rows[-1]
    count = len(rows)
    reward = sum(row["upper_reward"] for row in rows) / count
    backlog = sum(row["system_max_backlog"] for row in rows) / count
    deaths = sum(len(row["dead_ids"]) for row in rows)
    return (
        f"frame {last['upper_steps']:07d} | ep {last['episodes']:05d} | "
        f"low_steps {last['low_steps']:08d} | avg{count}_reward {reward:9.2f} | "
        f"avg{count}_max_backlog {backlog:9.2f} | dead {deaths:3d} | "
        f"SAC {last['sac_updates']:07d} | MAPPO {last['mappo_updates']:05d} | "
        f"{last['phase']:5s} | elapsed {elapsed / 60:7.1f}m"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs/training.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-note", default="", help="Short experiment description saved with source and effective configuration.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--upper-steps", type=int, help="Additional upper environment steps in this invocation.")
    parser.add_argument("--log-every", type=int, help="Console summary interval in upper frames (default: config or 50); also overrides resume settings.")
    parser.add_argument("--plot-every", type=int, help="Overwrite training_curves.png every N upper frames (default: config or 100; 0 disables).")
    parser.add_argument("--resume", type=Path, help="Resume this project's trusted checkpoint, including replay and pending transitions.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Small networks, six upper steps with constant SAC updates; not convergence training.")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; one is efficient for the small smoke networks.")
    args = parser.parse_args()
    if args.log_every is not None and args.log_every <= 0:
        parser.error("log-every must be positive")
    if args.plot_every is not None and args.plot_every < 0:
        parser.error("plot-every cannot be negative")
    if args.threads <= 0 or (args.upper_steps is not None and args.upper_steps <= 0):
        parser.error("threads and upper-steps must be positive")
    if args.smoke and args.resume:
        parser.error("--smoke cannot override a resumed checkpoint")
    if args.eval_only and not args.resume:
        parser.error("--eval-only requires --resume")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    output = args.output or ROOT / "runs" / datetime.now().strftime("joint_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((output / "train.jsonl").exists() or (output / "checkpoint.pt").exists()):
        parser.error("output already contains training data; use --resume or a new output directory")
    if args.resume:
        trainer = JointTrainer.load(args.resume, device)
    else:
        env_cfg = load_config(args.env_config)
        cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
        if args.smoke:
            env_cfg["time"]["episode_upper_frames"] = 3
            cfg.update(total_upper_steps=6, upper_rollout_steps=2, checkpoint_every_mappo=1, evaluate_every_mappo=0, evaluation_seeds=[1001])
            cfg["sac"].update(hidden=32, heads=2, layers=1, batch_size=8, replay_capacity=256, random_steps=0, learning_starts=0)
            cfg["mappo"].update(hidden=32, epochs=2, minibatch_frames=2)
        trainer = JointTrainer(env_cfg, cfg, device)
    cfg = trainer.cfg
    log_every = args.log_every if args.log_every is not None else cfg.get("log_every_frames", 50)
    if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every <= 0:
        parser.error("log_every_frames must be a positive integer")
    cfg["log_every_frames"] = log_every
    disk_log_every = cfg.get("train_log_every_frames", 10)
    if isinstance(disk_log_every, bool) or not isinstance(disk_log_every, int) or disk_log_every <= 0:
        parser.error("train_log_every_frames must be a positive integer")
    cfg["train_log_every_frames"] = disk_log_every
    plot_every = args.plot_every if args.plot_every is not None else cfg.get("plot_every_frames", 100)
    if isinstance(plot_every, bool) or not isinstance(plot_every, int) or plot_every < 0:
        parser.error("plot_every_frames must be a non-negative integer")
    cfg["plot_every_frames"] = plot_every
    if cfg["checkpoint_every_mappo"] < 0 or cfg["evaluate_every_mappo"] < 0:
        parser.error("checkpoint/evaluation intervals cannot be negative")
    (output / "resolved_config.json").write_text(json.dumps({"environment": trainer.env_cfg, "training": cfg, "device": device}, indent=2), encoding="utf-8")
    record_run(ROOT, output, trainer, args)

    def evaluate(label):
        summaries = []
        for seed in cfg["evaluation_seeds"]:
            summaries.append(trainer.evaluate(seed, output / label / f"seed_{seed}"))
        for summary in summaries:
            print(
                f"  eval {label} | seed {summary['seed']} | "
                f"return {summary['upper_return']:10.2f} | "
                f"avg_max_backlog {summary['mean_max_backlog']:9.2f} | "
                f"frames {summary['frames']:4d}/{summary['configured_frames']} | "
                f"low_steps {summary['low_steps']:4d} | dead {len(summary['dead_ids']):2d} | "
                f"charge requests/assigned {summary['charge_requests']}/{summary['charging_assignments']} | "
                f"stop {summary['termination_reason']}",
                flush=True,
            )

    if args.eval_only:
        evaluate(f"eval_{trainer.upper_steps:08d}")
        return
    steps = args.upper_steps or cfg["total_upper_steps"]
    print(f"Joint SAC + MAPPO | device {device} | additional frames {steps} | log every {log_every} frames", flush=True)
    print(f"Output: {output.resolve()}", flush=True)
    started = monotonic()
    console_rows = []
    curves = TrainingCurves(output, resume=args.resume, upper_steps=trainer.upper_steps) if plot_every else None
    try:
        with (output / "train.jsonl").open("a", encoding="utf-8") as log, gzip.open(
                output / 'charging_frames.jsonl.gz', 'at', encoding='utf-8', compresslevel=5) as charge_log:
            for step_index in range(steps):
                row = trainer.step()
                charge_log.write(json.dumps(dict(upper_steps=row['upper_steps'],
                    episode=trainer.episodes if trainer.episode_over else trainer.episodes + 1,
                    **row['charging_diagnostics']), separators=(',', ':')) + '\n')
                charge_log.flush()  # Persist every completed frame, independently of console output.
                if (row['upper_steps'] % disk_log_every == 0 or row['terminated']
                        or row['truncated'] or row['mappo_updated'] or step_index == steps - 1):
                    serialized = json.dumps(row, allow_nan=False, separators=(',', ':'))
                    log.write(serialized + "\n")
                    log.flush()  # Preserve completed records on process termination.
                    live_tmp = output / 'live_status.tmp.json'
                    live_tmp.write_text(serialized, encoding='utf-8')
                    live_tmp.replace(output / 'live_status.json')
                if row['episode_metrics'] is not None:
                    episode_dir = output / 'episodes'
                    episode_dir.mkdir(exist_ok=True)
                    path = episode_dir / f"episode_{row['episodes']:06d}.json.gz"
                    temporary = path.with_suffix('.tmp')
                    summary = dict(episode=row['episodes'], upper_steps=row['upper_steps'],
                                   upper_return=row['episode_return'], termination_reason=row['termination_reason'],
                                   **row['episode_metrics'])
                    with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                        json.dump(summary, stream, allow_nan=False)
                    temporary.replace(path)
                if curves is not None:
                    curves.add(row)
                    if (step_index + 1) % plot_every == 0:
                        curves.save()
                console_rows.append(row)
                if len(console_rows) >= log_every:
                    print(format_progress(console_rows, monotonic() - started), flush=True)
                    console_rows.clear()
                if row["mappo_updated"]:
                    interval = cfg["checkpoint_every_mappo"]
                    if interval and trainer.mappo_updates % interval == 0:
                        trainer.save(output / "checkpoint.pt")
                    interval = cfg["evaluate_every_mappo"]
                    if interval and trainer.mappo_updates % interval == 0:
                        evaluate(f"eval_{trainer.upper_steps:08d}")
    except KeyboardInterrupt:
        # A physical frame may be incomplete. Keep the previous safe checkpoint
        # rather than saving inconsistent policy/rollout state mid-step.
        print("Interrupted. Resume the last completed checkpoint; the active frame was not saved.", flush=True)
        if curves is not None:
            curves.save()
        return
    if console_rows:
        print(format_progress(console_rows, monotonic() - started), flush=True)
    trainer.save(output / "checkpoint.pt")
    if curves is not None:
        curves.save()
    evaluate(f"eval_{trainer.upper_steps:08d}")
    print(f"Saved checkpoint: {(output / 'checkpoint.pt').resolve()}", flush=True)


if __name__ == "__main__":
    main()
