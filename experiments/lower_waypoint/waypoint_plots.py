"""Compact overwrite-in-place plots for training and full waypoint evaluations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


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


def trajectories(env, path):
    """All sorties, with true executed outbound and return geometry per UAV."""
    s, rows = env.scene, env.records
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), layout="constrained")
    norm = Normalize(1, max(len(rows), 2))
    cmap = plt.get_cmap("viridis")
    climb = env.energy.height_m / env.energy.climb_speed_mps
    for uid, ax in enumerate(axes.flat):
        outbound, returns, circles, values, return_values = [], [], [], [], []
        targets, target_values = [], []
        for row in rows:
            for event in row["return_events"]:
                if event["uid"] != uid:
                    continue
                position = np.asarray(event["position"])
                outbound.append([s.airship_pos, position]); values.append(row["frame"])
                targets.append(position); target_values.append(row["frame"])
                if row["service_completed"] and event["service_survived"]:
                    circles.append(Circle(position, env.env_cfg["uavs"]["coverage_radius_m"]))
                distance = float(np.linalg.norm(position - s.airship_pos))
                horizontal = max(0, event["return_executed_sec"] - climb)
                fraction = min(1, horizontal * env.energy.horizontal_speed_mps / max(distance, 1e-12))
                if fraction > 0:
                    end = position + (s.airship_pos - position) * fraction
                    returns.append([position, end]); return_values.append(row["frame"])
        ax.add_collection(PatchCollection(circles, facecolor="#278c88", edgecolor="none", alpha=.07))
        ax.add_collection(LineCollection(outbound, colors=cmap(norm(values)), linewidths=.85, alpha=.55))
        ax.add_collection(LineCollection(returns, colors=cmap(norm(return_values)), linewidths=.65, linestyles="dashed", alpha=.65))
        if targets:
            ax.scatter(*np.asarray(targets).T, c=target_values, norm=norm, cmap=cmap, s=14, zorder=4)
        ax.scatter(*s.sensor_pos.T, color="#4c5966", s=9, zorder=3)
        ax.scatter(*s.airship_pos, c="#b3314b", marker="*", s=110, zorder=5)
        ax.add_patch(Circle(s.airship_pos, env.env_cfg["sensors"]["partition_radius_m"], fill=False, ls="--", color="gray", lw=.8))
        ax.set(xlim=(0, s.world_size), ylim=(0, s.world_size), aspect="equal",
               title=f"U{uid} | {len(outbound)} sorties", xlabel="East / m", ylabel="North / m")
        ax.grid(alpha=.15)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat), shrink=.6, label="Decision / sortie number")
    fig.legend(handles=[Line2D([], [], color="#4c5966", marker="o", ls="", label="Sensor"),
                        Line2D([], [], color="#278c88", lw=7, alpha=.3, label="Actual service footprint"),
                        Line2D([], [], color="#405c85", label="Outbound"),
                        Line2D([], [], color="#405c85", ls="--", label="Executed return (same route)")],
               loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(f"All UAV trajectories | {s.now_sec/3600:g} simulated hours | {len(rows)} decisions\n"
                 "Waypoints colored by time; partial terminal travel shown only as executed", fontsize=13)
    save_figure(fig, path)


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
