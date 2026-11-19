"""Render a saved waypoint checkpoint without changing or stopping training."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import torch
from waypoint_trainer import Trainer
from waypoint_plots import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    trainer = Trainer.load(args.checkpoint, device)
    output = args.output or args.checkpoint.parent / f"eval_{trainer.steps:08d}"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds or trainer.cfg["evaluation_seeds"]:
        folder = output / f"seed_{seed}"
        summaries.append(trainer.evaluate(seed, folder, detailed=True))
        print(f"seed {seed} | trajectory: {folder / 'trajectory.png'}", flush=True)
    write_json(output / "evaluation_manifest.json", dict(checkpoint=str(args.checkpoint.resolve()),
               checkpoint_steps=trainer.steps, revision=trainer.cfg["revision"], summaries=summaries))


if __name__ == "__main__":
    main()
