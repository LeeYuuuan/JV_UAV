"""Combine saved full evaluations without loading weights or interrupting training."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from persistent_plots import plt, fleet_map, fleet_overlay, save_figure, write_json


def completed_evals(run, seed):
    records = []
    for folder in sorted(run.glob("eval_*")):
        if not folder.is_dir() or not folder.name[5:].isdigit():
            continue
        target = folder / f"seed_{seed}"
        try:
            trace = json.loads((target / "trace.json").read_text())
            summary = json.loads((target / "summary.json").read_text())
        except (OSError, ValueError):
            continue
        rows = trace.get("rows", [])
        if not rows or not rows[-1].get("done") or len(rows) != summary["steps"]:
            continue
        if summary["training_steps"] != int(folder.name[5:]):
            continue
        records.append(dict(step=summary["training_steps"], trace=trace, summary=summary, folder=target))
    return records


def make_env(record, env_cfg):
    scene = SimpleNamespace(sensor_pos=np.asarray(record["trace"]["sensor_pos"]),
                            airship_pos=np.asarray(env_cfg["world"]["airship_xy_m"]),
                            world_size=env_cfg["world"]["size_m"], num_uavs=env_cfg["uavs"]["count"])
    return SimpleNamespace(scene=scene, records=record["trace"]["rows"], env_cfg=env_cfg, summary=lambda: record["summary"])


def render(run, output, seed=1001, selected_steps=None):
    config = json.loads((run / "resolved_config.json").read_text())
    env_cfg = config["environment"]
    records = completed_evals(run, seed)
    if not records:
        raise ValueError("No completed full evaluation trace.json found for this seed yet")
    if selected_steps:
        chosen = [r for r in records if r["step"] in selected_steps]
        missing = set(selected_steps) - {r["step"] for r in chosen}
        if missing:
            raise ValueError(f"No full trace for requested steps: {sorted(missing)}")
    else:
        indices = np.unique(np.linspace(0, len(records) - 1, min(4, len(records)), dtype=int))
        chosen = [records[i] for i in indices]
    output.mkdir(parents=True, exist_ok=True)
    latest = records[-1]
    env = make_env(latest, env_cfg)
    fleet_overlay(env, output / "latest_fleet.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for ax, ids in zip(axes, ([0, 1, 2], [3, 4, 5])):
        handles = fleet_map(ax, env.scene, env.records, env_cfg, ids=ids,
                            title=f"UAVs {','.join(map(str, ids))} | training step {latest['step']:,}")
        ax.legend(handles=handles, loc="upper center", ncol=3)
    fig.suptitle("Latest complete evaluation | groups of 3 UAVs | all selections and service footprints\n"
                 "Dashed lines denote actual executed movement", fontsize=11)
    save_figure(fig, output / "latest_groups.png")
    ncols = min(2, len(chosen)); nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.7 * nrows), squeeze=False, layout="constrained")
    for ax, record in zip(axes.flat, chosen):
        env = make_env(record, env_cfg); s = record["summary"]
        handles = fleet_map(ax, env.scene, env.records, env_cfg,
                            title=f"Training {record['step']:,} | visited {s['sensor_coverage_fraction']:.1%} | mean max {s['mean_max_backlog']:.0f}")
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=10)
    fig.suptitle(f"Training-stage comparison | evaluation seed {seed} | fixed map and UAV colors\n"
                 "All selected positions and service footprints within each complete evaluation", fontsize=12)
    save_figure(fig, output / "stage_comparison.png")
    write_json(output / "overview.json", dict(run=str(run.resolve()), seed=seed, latest_full_eval_step=latest["step"],
               stages=[dict(training_step=r["step"], source=str(r["folder"].resolve()), summary=r["summary"]) for r in chosen],
               note="Reads actual saved full traces; no new trajectories sampled. Existing process saves full traces every5000 steps."))
    return latest["step"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--watch", type=float, help="Poll interval in seconds; overwrites plots only when new full traces arrive")
    args = parser.parse_args()
    if args.watch is not None and args.watch < 10:
        parser.error("watch interval must be at least10 seconds")
    output = args.output or args.run / "overview"
    last = None
    try:
        while True:
            try:
                records = completed_evals(args.run, args.seed)
                newest = records[-1]["step"] if records else None
                if newest != last or last is None:
                    last = render(args.run, output, args.seed, args.steps)
                    print(f"Updated full eval {last:,}: {output / 'latest_fleet.png'}; {output / 'latest_groups.png'}; {output / 'stage_comparison.png'}", flush=True)
            except (OSError, ValueError) as exc:
                if args.watch is None:
                    raise
                print(f"Waiting for complete evaluation files: {exc}", flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("Overview watcher stopped; training was not modified.", flush=True)


if __name__ == "__main__":
    main()
