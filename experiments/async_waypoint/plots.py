"""Plots use actual event segments/collection sites, never planned full paths."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def write_json(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"); temp.replace(path)


def save(fig, path):
    path = Path(path); temp = path.with_suffix(".tmp")
    fig.savefig(temp, format="png", dpi=140, bbox_inches="tight"); plt.close(fig); temp.replace(path)


def dashboard(env, path, label="Deterministic policy"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax = axes[0, 0]; colors = plt.get_cmap("tab10")
    dots = ax.scatter(*env.xy.T, c=env.backlog, s=14, cmap="YlOrRd", zorder=5)
    fig.colorbar(dots, ax=ax, label="Final sensor backlog / packets")
    for segment in env.segments:
        start, end = np.asarray(segment["positions_start"]), np.asarray(segment["positions_end"])
        for uid in range(env.n):
            if np.linalg.norm(end[uid] - start[uid]) > 1e-5:
                style = "--" if segment["phase_start"][uid] in (3, 4) else "-"
                ax.plot([start[uid, 0], end[uid, 0]], [start[uid, 1], end[uid, 1]], c=colors(uid), alpha=.35, lw=.7, ls=style)
    for collection in env.collections:
        for uid, xy in zip(collection["uids"], collection["positions"]):
            ax.add_patch(Circle(xy, env.env_cfg["uavs"]["coverage_radius_m"], color=colors(uid), alpha=.04))
    ax.add_patch(Circle(env.home, env.env_cfg["sensors"]["partition_radius_m"], fill=False, ls="--", color="gray"))
    ax.scatter(*env.home, marker="*", c="navy", s=120, zorder=6)
    ax.set(xlim=(0, env.size), ylim=(0, env.size), aspect="equal", title="Actual paths and completed-service footprints", xlabel="East / m", ylabel="North / m")
    if env.segments:
        times = [0] + [r["end"] / 60 for r in env.segments]
        batteries = np.asarray([env.segments[0]["soc_start"]] + [r["soc_end"] for r in env.segments]) * 100
        for uid in range(env.n):
            axes[0, 1].plot(times, batteries[:, uid], c=colors(uid), label=f"U{uid}")
        axes[0, 1].legend(ncol=3, fontsize=8)
        axes[1, 0].plot(times[1:], [r["max_backlog"] for r in env.segments])
        axes[1, 1].plot(times[1:], [r["cumulative_covered"] / len(env.xy) for r in env.segments])
    axes[0, 1].set(title="Actual SOC (no sortie resets)", ylabel="SOC / %")
    axes[1, 0].set(title="Maximum backlog after event processing", ylabel="Packets")
    axes[1, 1].set(title="Unique sensors visited", ylim=(0, 1.02))
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("Simulation time / min"); ax.grid(alpha=.15)
    summary = env.summary()
    fig.suptitle(f"{label} | {env.now / 60:.1f} min | mean serving {summary['mean_serving']:.2f} | "
                 f"coverage {summary['sensor_coverage_fraction']:.1%} | deaths {len(env.failures)}")
    fig.tight_layout(); save(fig, path)


def curves(trainer, output, evaluations):
    output = Path(output)
    write_json(output / "training_curves.json", dict(revision=trainer.cfg["revision"], episodes=trainer.history,
               losses=trainer.loss_history, evaluations=evaluations))
    if trainer.history:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, key in zip(axes.flat, ("upper_return", "lower_return", "time_mean_max_backlog", "mean_serving", "sensor_coverage_fraction", "simulation_sec")):
            y = np.array([r[key] for r in trainer.history]); x = np.arange(1, len(y) + 1)
            ax.plot(x, y, lw=.7, alpha=.35)
            if len(y) >= 10:
                ax.plot(x[9:], np.convolve(y, np.ones(10) / 10, mode="valid"))
            ax.set(title=key, xlabel="Completed episode"); ax.grid(alpha=.15)
        fig.tight_layout(); save(fig, output / "training_curves.png")
    if trainer.loss_history:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, prefix, suffixes in ((axes[0, 0], "upper_", ("actor_loss", "policy_loss", "entropy_loss")),
                                     (axes[0, 1], "upper_", ("critic_loss",)),
                                     (axes[1, 0], "lower_", ("actor_loss", "policy_loss", "entropy_loss")),
                                     (axes[1, 1], "lower_", ("critic_loss",))):
            for suffix in suffixes:
                key = prefix + suffix
                rows = [r for r in trainer.loss_history if key in r]
                if rows:
                    ax.plot([r["upper_steps"] for r in rows], [r[key] for r in rows], label=key)
            ax.set_xlabel("Upper intervals"); ax.set_title(prefix + "loss"); ax.grid(alpha=.15)
            if ax.lines:
                ax.legend(fontsize=8)
        fig.tight_layout(); save(fig, output / "loss_curves.png")
    if evaluations:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, key in zip(axes.flat, ("mean_serving", "simulation_sec", "time_mean_max_backlog", "sensor_coverage_fraction")):
            for seed in sorted({r["seed"] for r in evaluations}):
                rows = [r for r in evaluations if r["seed"] == seed]
                ax.plot([r["upper_steps"] for r in rows], [r[key] for r in rows], label=str(seed))
            ax.set(title=key, xlabel="Upper intervals"); ax.grid(alpha=.15)
        axes[0, 0].legend(); fig.tight_layout(); save(fig, output / "evaluation_curves.png")
