"""Compact overwrite-in-place plots for training and full waypoint evaluations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_figure(fig, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    fig.savefig(temp, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig); temp.replace(path)


def dashboard(env, path):
    rows, s = env.records, env.scene
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    ax = axes[0, 0]
    points = ax.scatter(*s.sensor_pos.T, c=s.sensor_backlog, s=13, cmap="YlOrRd", zorder=4)
    fig.colorbar(points, ax=ax, label="Backlog after final return / packets")
    palette = plt.get_cmap("tab10")
    for row in rows:
        if not row["service_completed"]:
            continue
        for event in row["return_events"]:
            if not event["service_survived"]:
                continue
            xy = event["position"]
            ax.add_patch(Circle(xy, env.env_cfg["uavs"]["coverage_radius_m"], color=palette(event["uid"]), alpha=.035))
    for row in rows[-16:]:
        for event in row["return_events"]:
            xy = event["position"]
            ax.plot([s.airship_pos[0], xy[0]], [s.airship_pos[1], xy[1]], c=palette(event["uid"]), alpha=.3, lw=.7)
    ax.add_patch(Circle(s.airship_pos, env.env_cfg["sensors"]["partition_radius_m"], fill=False, ls="--", color="gray"))
    ax.scatter(*s.airship_pos, marker="*", s=120, c="navy", zorder=5)
    ax.set(xlim=(0, s.world_size), ylim=(0, s.world_size), aspect="equal", xlabel="East / m", ylabel="North / m",
           title="All service footprints; last 16 sorties' outbound paths")
    time = np.array([r["end_sec"] for r in rows]) / 60
    axes[0, 1].plot(time, [r["max_backlog"] for r in rows])
    axes[0, 1].set(title="Maximum backlog after each return", xlabel="Simulation time / min", ylabel="Packets")
    axes[1, 0].plot(time, [r["sensor_coverage_fraction"] for r in rows], label="All sensors")
    outer_n = int((np.linalg.norm(s.sensor_pos - s.airship_pos, axis=1) > env.env_cfg["sensors"]["partition_radius_m"]).sum())
    axes[1, 0].plot(time, [r["outer_visited"] / max(outer_n, 1) for r in rows], label="Outside r=1800")
    axes[1, 0].set(title="Unique sensor coverage within episode", ylim=(0, 1.02), xlabel="Simulation time / min")
    axes[1, 0].legend()
    frames = np.arange(1, len(rows) + 1)
    for key in rows[0]["reward_terms"]:
        axes[1, 1].plot(frames, [r["reward_terms"][key] for r in rows], label=key)
    axes[1, 1].set(title="Reward terms per synchronous decision", xlabel="Sortie / frame")
    axes[1, 1].legend(fontsize=8)
    axes[2, 0].plot(frames, [r["service_sec"] / 60 for r in rows], label="Outbound + service/wait")
    axes[2, 0].plot(frames, [r["return_sec"] / 60 for r in rows], label="Automatic return")
    axes[2, 0].set(title="Actual simulated duration", xlabel="Sortie / frame", ylabel="Minutes")
    axes[2, 0].legend(fontsize=8)
    used = np.array([np.mean([e["service_soc"] + e["return_paid_soc"] for e in r["return_events"]]) for r in rows])
    axes[2, 1].plot(frames, used * 100, label="Mean sortie SOC use")
    axes[2, 1].set(title="Full battery each sortie; no charging policy", xlabel="Sortie / frame", ylabel="SOC / %")
    summary = env.summary()
    fig.suptitle(f"Synchronous waypoint | {len(rows)} sorties | {summary['sensor_coverage_fraction']:.1%} coverage | "
                 f"{summary['failures']} failures | time-mean max backlog {summary['time_mean_max_backlog']:.1f}")
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.tight_layout(); save_figure(fig, path)


def curves(trainer, output, evaluations):
    output = Path(output)
    history = trainer.history
    write_json(output / "training_curves.json", dict(revision=trainer.cfg["revision"], episodes=history, evaluations=evaluations,
               comparison_note="Different task/horizon/observations from 20/10 baseline; raw returns are not comparable."))
    if history:
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        x = [r["episode"] for r in history]
        keys = [("lower_return", "Episode reward"), ("time_mean_max_backlog", "Time-mean max backlog / packets"),
                ("sensor_coverage_fraction", "Unique sensor coverage"), ("collection_rate_packets_sec", "Collection rate / packets per sec"),
                ("simulation_sec", "Episode simulation duration / sec"), ("failures", "Failed sorties' UAV count")]
        for ax, (key, title) in zip(axes.flat, keys):
            y = np.array([r[key] for r in history])
            ax.plot(x, y, alpha=.25, lw=.7)
            if len(y) >= 10:
                ax.plot(x[9:], np.convolve(y, np.ones(10) / 10, mode="valid"))
            ax.set(title=title, xlabel="Training episode"); ax.grid(alpha=.15)
        fig.tight_layout(); save_figure(fig, output / "lower_training_curves.png")
    losses = trainer.loss_history
    if losses:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        x = [r["training_steps"] for r in losses]
        for key in ("actor_loss", "policy_loss", "entropy_loss"):
            axes[0, 0].plot(x, [r[key] for r in losses], label=key)
        axes[0, 0].set_title("Actor and its two terms"); axes[0, 0].legend()
        axes[0, 1].plot(x, [r["critic_loss"] for r in losses]); axes[0, 1].set_title("Critic Huber loss")
        for key in ("q_mean", "target_mean"):
            axes[1, 0].plot(x, [r[key] for r in losses], label=key)
        axes[1, 0].legend(); axes[1, 0].set_title("Q and bootstrap target")
        axes[1, 1].plot(x, [r["alpha"] for r in losses]); axes[1, 1].set_title("Automatic entropy coefficient")
        for ax in axes.flat:
            ax.set_xlabel("Waypoint decisions"); ax.grid(alpha=.15)
        fig.tight_layout(); save_figure(fig, output / "loss_curves.png")
    policy = [r for r in evaluations if r["policy_mode"] == "policy"]
    if policy:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, key in zip(axes.flat, ("time_mean_max_backlog", "sensor_coverage_fraction", "collection_rate_packets_sec", "failures")):
            for seed in sorted({r["seed"] for r in policy}):
                values = [r for r in policy if r["seed"] == seed]
                ax.plot([r["training_steps"] for r in values], [r[key] for r in values], label=f"seed {seed}")
            ax.set(title=key, xlabel="Waypoint decisions"); ax.grid(alpha=.15)
        axes[0, 0].legend(); fig.tight_layout(); save_figure(fig, output / "evaluation_curves.png")
