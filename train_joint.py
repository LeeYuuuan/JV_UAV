"""Train or evaluate the confirmed attention SAC + MAPPO hierarchy."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from jv_uav import load_config
from jv_uav.rl.trainer import JointTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs/training.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--upper-steps", type=int, help="Additional upper environment steps in this invocation.")
    parser.add_argument("--resume", type=Path, help="Resume this project's trusted checkpoint, including replay and pending transitions.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Small networks, six upper steps and a low switch threshold; not convergence training.")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; one is efficient for the small smoke networks.")
    args = parser.parse_args()
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
            cfg.update(total_upper_steps=6, upper_rollout_steps=2, switch_after_sac_updates=12, checkpoint_every_mappo=1, evaluate_every_mappo=0, evaluation_seeds=[1001])
            cfg["sac"].update(hidden=32, heads=2, layers=1, batch_size=8, replay_capacity=256, random_steps=0, learning_starts=0)
            cfg["mappo"].update(hidden=32, epochs=2, minibatch_frames=2)
        trainer = JointTrainer(env_cfg, cfg, device)
    cfg = trainer.cfg
    if cfg["checkpoint_every_mappo"] < 0 or cfg["evaluate_every_mappo"] < 0:
        parser.error("checkpoint/evaluation intervals cannot be negative")
    (output / "resolved_config.json").write_text(json.dumps({"environment": trainer.env_cfg, "training": cfg, "device": device}, indent=2), encoding="utf-8")

    def evaluate(label):
        summaries = []
        for seed in cfg["evaluation_seeds"]:
            summaries.append(trainer.evaluate(seed, output / label / f"seed_{seed}"))
        print(json.dumps({"evaluation": label, "results": summaries}), flush=True)

    if args.eval_only:
        evaluate(f"eval_{trainer.upper_steps:08d}")
        return
    steps = args.upper_steps or cfg["total_upper_steps"]
    print(json.dumps({"device": device, "additional_upper_steps": steps, "switch_after_sac_updates": cfg["switch_after_sac_updates"], "upper_rollout_steps": cfg["upper_rollout_steps"], "output": str(output.resolve())}), flush=True)
    try:
        with (output / "train.jsonl").open("a", encoding="utf-8") as log:
            for _ in range(steps):
                row = trainer.step()
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if row["mappo_updated"] or row["terminated"] or row["truncated"]:
                    print(json.dumps(row, allow_nan=False), flush=True)
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
        return
    trainer.save(output / "checkpoint.pt")
    evaluate(f"eval_{trainer.upper_steps:08d}")
    print(f"Saved checkpoint: {(output / 'checkpoint.pt').resolve()}", flush=True)


if __name__ == "__main__":
    main()
