"""Persistent waypoint diagnostics: lines are actual executed movement."""
from pathlib import Path
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def write_json(path, data):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers/scanners can briefly deny atomic replacement. Retain the
    # atomic write, but retry transient sharing/access failures before failing.
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def save_figure(fig, path):
    path = Path(path); temporary = path.with_suffix(".tmp")
    fig.savefig(temporary, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig); temporary.replace(path)


def fleet_map(ax, scene, rows, env_cfg, ids=None, title=None):
    """All selected UAVs on a common map; shared colors across panels."""
    ids = list(range(scene.num_uavs)) if ids is None else list(ids)
    palette = plt.get_cmap("tab10")
    radius = env_cfg["uavs"]["coverage_radius_m"]
    for uid in ids:
        segments, positions = [], []
        for row in rows:
            if uid in row["active_ids"]:
                segments.append([row["positions_start"][uid], row["positions_end"][uid]])
                positions.append(row["positions_end"][uid])
        color = palette(uid)
        ax.add_collection(PatchCollection([Circle(xy, radius) for xy in positions], facecolor=color, edgecolor="none", alpha=.025))
        ax.add_collection(LineCollection(segments, colors=[color], linewidths=.5, linestyles="solid", alpha=.2))
        if positions:
            ax.scatter(*np.asarray(positions).T, s=7, color=color, alpha=.55, zorder=3)
            ax.scatter(*rows[-1]["positions_end"][uid], s=45, color=color, marker="^", edgecolor="white", lw=.5, zorder=6)
    ax.scatter(*scene.sensor_pos.T, c="#303c47", s=11, zorder=4)
    ax.scatter(*scene.airship_pos, marker="*", c="#b3314b", s=110, zorder=6)
    ax.add_patch(Circle(scene.airship_pos, env_cfg["sensors"]["partition_radius_m"], fill=False, ls="--", color="gray", lw=.8))
    ax.set(xlim=(0, scene.world_size), ylim=(0, scene.world_size), aspect="equal", title=title,
           xlabel="East / m", ylabel="North / m")
    ax.grid(alpha=.12)
    return [Line2D([], [], marker="o", color=palette(uid), lw=.7, label=f"U{uid}") for uid in ids]


def fleet_overlay(env, path):
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    summary = env.summary()
    handles = fleet_map(ax, env.scene, env.records, env.env_cfg,
                        title=f"All {env.scene.num_uavs} UAVs | coverage {summary['sensor_coverage_fraction']:.1%} | mean max backlog {summary['mean_max_backlog']:.0f}")
    ax.legend(handles=handles, ncol=6, loc="upper center", fontsize=9)
    fig.suptitle("Complete evaluation episode | colored dots=selected positions, triangles=final positions\n"
                 "Solid segments are real executed movements; shaded circles show all service footprints", fontsize=10)
    save_figure(fig, path)


def trajectories(env, path):
    s, rows = env.scene, env.records
    columns = min(2, s.num_uavs)
    nrows = (s.num_uavs + columns - 1) // columns
    fig, axes = plt.subplots(nrows, columns, figsize=(6 * columns, 4.7 * nrows), layout="constrained", squeeze=False)
    norm, cmap = Normalize(1, max(len(rows), 2)), plt.get_cmap("viridis")
    radius = env.env_cfg["uavs"]["coverage_radius_m"]
    for uid, ax in enumerate(axes.flat):
        if uid >= s.num_uavs:
            ax.set_visible(False)
            continue
        segments, positions, times, circles = [], [], [], []
        for row in rows:
            if uid not in row["active_ids"]:
                continue
            start, end = row["positions_start"][uid], row["positions_end"][uid]
            segments.append([start, end]); positions.append(end); times.append(row["step"])
            circles.append(Circle(end, radius))
        ax.add_collection(PatchCollection(circles, facecolor="#278c88", edgecolor="none", alpha=.035))
        ax.add_collection(LineCollection(segments, colors=cmap(norm(times)), linewidths=.65, alpha=.5, linestyles="solid"))
        if positions:
            ax.scatter(*np.asarray(positions).T, c=times, norm=norm, cmap=cmap, s=10, zorder=4)
        ax.scatter(*s.sensor_pos.T, c="#4c5966", s=9, zorder=3)
        ax.scatter(*s.airship_pos, marker="*", c="#b3314b", s=100, zorder=5)
        ax.add_patch(Circle(s.airship_pos, env.env_cfg["sensors"]["partition_radius_m"], fill=False, ls="--", color="gray", lw=.8))
        ax.set(xlim=(0, s.world_size), ylim=(0, s.world_size), aspect="equal", title=f"U{uid} | {len(times)} active steps",
               xlabel="East / m", ylabel="North / m")
        ax.grid(alpha=.15)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat), shrink=.6, label="Step number")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", c="#4c5966", label="Sensor"),
                        Line2D([], [], c="#278c88", lw=7, alpha=.3, label="Service footprint"),
                        Line2D([], [], c="#405c85", ls="-", label="Executed flight path")],
               loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(f"Persistent waypoints | {len(rows)} steps | {s.now_sec/60:g} simulated min\n"
                 "All selections shown; real speed and time; energy disabled", fontsize=13)
    save_figure(fig, path)


def dashboard(env, path):
    s, rows = env.scene, env.records
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    points = axes[0, 0].scatter(*s.sensor_pos.T, c=s.sensor_backlog, s=16, cmap="YlOrRd")
    fig.colorbar(points, ax=axes[0, 0], label="Final backlog / packets")
    axes[0, 0].scatter(*s.airship_pos, marker="*", c="navy", s=100)
    axes[0, 0].add_patch(Circle(s.airship_pos, env.env_cfg["sensors"]["partition_radius_m"], fill=False, ls="--", color="gray"))
    axes[0, 0].set(xlim=(0, s.world_size), ylim=(0, s.world_size), aspect="equal", title="Final sensor backlog", xlabel="East / m", ylabel="North / m")
    x = np.arange(1, len(rows) + 1)
    axes[0, 1].plot(x, [r["max_backlog"] for r in rows], label="Maximum")
    axes[0, 1].plot(x, [r["mean_backlog"] for r in rows], label="Mean")
    axes[0, 1].set(title="Post-collection backlog", xlabel="Step", ylabel="Packets"); axes[0, 1].legend()
    axes[1, 0].plot(x, [r["sensor_coverage_fraction"] for r in rows], label="All")
    outer_n = int((np.linalg.norm(s.sensor_pos - s.airship_pos, axis=1) > env.env_cfg["sensors"]["partition_radius_m"]).sum())
    axes[1, 0].plot(x, [r["outer_visited"] / max(outer_n, 1) for r in rows], label="Outside circle")
    axes[1, 0].set(title="Unique sensor coverage", xlabel="Step", ylim=(0, 1.02)); axes[1, 0].legend()
    for key in rows[0]["reward_terms"]:
        axes[1, 1].plot(x, [r["reward_terms"][key] for r in rows], label=key)
    axes[1, 1].set(title="Reward terms", xlabel="Step"); axes[1, 1].legend()
    active = np.zeros((s.num_uavs, len(rows)))
    for i, row in enumerate(rows):
        active[row["active_ids"], i] = 1
    axes[2, 0].imshow(active, aspect="auto", origin="upper", cmap="Blues", vmin=0, vmax=1,
                      extent=[.5, len(rows) + .5, s.num_uavs - .5, -.5], interpolation="nearest")
    axes[2, 0].set(title="Active UAVs (blue); no rotation" if env.cfg.get("disable_rotation", False) else f"Active UAVs (blue); groups every {env.cfg['block_steps']} steps", xlabel="Step", ylabel="UAV ID")
    axes[2, 1].plot(x, [r["collected_packets"] for r in rows])
    axes[2, 1].set(title="Actual collected packets per step", xlabel="Step", ylabel="Packets")
    summary = env.summary()
    fig.suptitle(f"Persistent waypoint coverage | R={summary['lower_return']:.1f} | coverage={summary['sensor_coverage_fraction']:.1%} | "
                 f"mean max backlog={summary['mean_max_backlog']:.1f}")
    for ax in axes.flat:
        ax.grid(alpha=.12)
    fig.tight_layout(); save_figure(fig, path)


def curves(trainer, output, evaluations):
    output = Path(output)
    history = trainer.history
    write_json(output / "training_curves.json", dict(revision=trainer.cfg["revision"], episodes=history, evaluations=evaluations,
               comparison_note="Persistent real-speed waypoint experiment; energy disabled; macro decisions use discounted physical rewards."))
    if history:
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        x = [r["episode"] for r in history]
        keys = [("lower_return", "Episode reward"), ("mean_max_backlog", "Mean post-collection max backlog / packets"),
                ("sensor_coverage_fraction", "Unique sensor coverage"), ("collection_rate_packets_sec", "Collection rate / packets per sec"),
                ("mean_sensor_backlog", "Mean sensor backlog / packets"), ("mean_clear_fraction", "Mean cleared fraction of all backlog")]
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
        for ax, key in zip(axes.flat, ("mean_max_backlog", "sensor_coverage_fraction", "collection_rate_packets_sec", "mean_clear_fraction")):
            for seed in sorted({r["seed"] for r in policy}):
                values = [r for r in policy if r["seed"] == seed]
                ax.plot([r["training_steps"] for r in values], [r[key] for r in values], label=f"seed {seed}")
            ax.set(title=key, xlabel="Waypoint decisions"); ax.grid(alpha=.15)
        axes[0, 0].legend(); fig.tight_layout(); save_figure(fig, output / "evaluation_curves.png")
