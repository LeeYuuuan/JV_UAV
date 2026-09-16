"""Read-only Matplotlib views of the current scene and its recorded episode."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .upper_env import UpperEnv

UAV_COLORS = ("#167d9a", "#e19b26", "#9063bd", "#dc677b", "#3b9672", "#6574c4")
ROLE_COLORS = {"SERVING": "#167d9a", "CHARGING": "#e4ad42", "WAITING": "#a9b6c7", "DEAD": "#d45e6b"}
INK = "#20354c"
MUTED = "#6d7f90"


def render_env(
    env: UpperEnv,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    view: str = "dashboard",
    sensor_value: str = "backlog",
    trail_frames: int = 6,
    draw_coverage: bool = True,
    dpi: int = 150,
):
    """Render without advancing time, consuming RNG, or modifying the scene.

    ``view`` is ``dashboard`` or ``heatmaps``. Map sensor colors show raw
    ``backlog`` (packets) or ``last_visit`` (minutes). Trajectories include only
    recorded frames; dashed segments are settled returns, not sampled paths.
    Coverage circles include every serving endpoint in the displayed frames;
    these are discrete geometric footprints, not continuous in-flight service.
    Heatmap grids retain the recorder's geometric footprint semantics.
    Returns the Figure; with show=False it is closed after optional saving.
    """
    if view not in {"dashboard", "heatmaps"}:
        raise ValueError("view must be 'dashboard' or 'heatmaps'")
    if sensor_value not in {"backlog", "last_visit"}:
        raise ValueError("sensor_value must be 'backlog' or 'last_visit'")
    if not isinstance(trail_frames, int) or trail_frames < 0:
        raise ValueError("trail_frames must be a non-negative integer")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#dce4ec", "axes.facecolor": "white",
        "figure.facecolor": "#f3f6fa", "savefig.facecolor": "#f3f6fa",
        "axes.titleweight": "bold", "axes.titlesize": 11,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(16, 11), dpi=dpi)
        try:
            _header(fig, env, view)
            if view == "dashboard":
                _dashboard(fig, env, sensor_value, trail_frames, draw_coverage)
            else:
                _heatmaps(fig, env)
            _footer(fig, env)
            if save_path is not None:
                target = Path(save_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(target, dpi=dpi)
            if show:
                plt.show()
            return fig
        finally:
            if not show:
                plt.close(fig)


def _header(fig, env, view):
    scene, trace = env.scene, env.trace
    fig.text(0.055, 0.955, "UAV  /  FIELD OPERATIONS", fontsize=21, weight="bold")
    fig.text(0.055, 0.929, "Episode dashboard" if view == "dashboard" else "Spatial collection & coverage", color=MUTED, fontsize=11)
    fig.text(0.95, 0.948, f"{scene.now_sec / 60:,.0f} min   |   {scene.low_step_total:,} low steps", ha="right", fontsize=11)
    fig.text(0.95, 0.925, f"{len(trace.frames)} recorded frames   /   {scene.num_uavs} UAVs   /   {scene.num_sensors} sensors", ha="right", color=MUTED)
    packets = float(trace.sensor_collected_packets.sum()) if trace.sensor_collected_packets is not None else 0
    cards = [
        ("CURRENT MAX BACKLOG", f"{scene.sensor_backlog.max():,.0f}", "packets"),
        ("PACKETS COLLECTED", f"{packets:,.0f}", "episode total"),
        ("SERVING / CHARGING / WAITING", f"{len(scene.serving_ids())} / {len(scene.charging_ids())} / {len(scene.waiting_ids())}", "current status"),
        ("FLEET BATTERY", f"{scene.uav_battery.mean():.0%}", "mean state of charge"),
    ]
    from matplotlib.patches import FancyBboxPatch
    for index, (label, value, unit) in enumerate(cards):
        x = 0.055 + index * 0.227
        fig.add_artist(FancyBboxPatch((x, 0.818), 0.215, 0.088, boxstyle="round,pad=0.006,rounding_size=0.008", transform=fig.transFigure, facecolor="white", edgecolor="#e3eaf1", zorder=0))
        fig.text(x + 0.012, 0.884, label, fontsize=8, color=MUTED)
        fig.text(x + 0.012, 0.846, value, fontsize=23, weight="bold")
        fig.text(x + 0.012, 0.825, unit, fontsize=8, color=MUTED)


def _style_axes(ax, title, grid=True):
    ax.set_title(title, loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis="y", color="#e8eef4", linewidth=0.7)
        ax.set_axisbelow(True)


def _map_axes(ax, scene, title):
    _style_axes(ax, title, grid=False)
    ax.set(xlim=(0, scene.world_size), ylim=(0, scene.world_size), xlabel="East / m", ylabel="North / m")
    ax.set_aspect("equal")
    ax.grid(color="#edf1f5", linewidth=0.7)


def _airship(ax, scene):
    ax.scatter(*scene.airship_pos, marker="*", s=230, c=INK, edgecolors="white", linewidths=1.2, zorder=9)
    ax.annotate("Airship", scene.airship_pos, xytext=(9, -16), textcoords="offset points", color=INK, fontsize=8, zorder=10, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})


def _dashboard(fig, env, sensor_value, trail_frames, draw_coverage):
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch
    scene, trace = env.scene, env.trace
    gs = fig.add_gridspec(3, 2, left=0.06, right=0.95, bottom=0.13, top=0.765, width_ratios=[1.04, 1], height_ratios=[1, 1, 0.95], hspace=0.64, wspace=0.25)
    ax_map = fig.add_subplot(gs[:2, 0])
    frames = trace.frames[-trail_frames:] if trail_frames else []
    shown_steps = sum(round(frame.duration_sec / scene.low_step_sec) for frame in frames)
    _map_axes(ax_map, scene, f"01  Flight map  /  {len(frames)} frames, {shown_steps} low steps")
    values = scene.sensor_backlog if sensor_value == "backlog" else scene.sensor_last_visit_sec / 60
    label = "Current backlog / packets" if sensor_value == "backlog" else "Time since service / min"
    sc = ax_map.scatter(scene.sensor_pos[:, 0], scene.sensor_pos[:, 1], c=values, s=15, cmap="YlOrBr", vmin=0, vmax=max(float(values.max()), 1.0), edgecolors="none", alpha=0.9, zorder=2)
    cb = fig.colorbar(sc, ax=ax_map, fraction=0.035, pad=0.025)
    cb.set_label(label, fontsize=8)
    cb.outline.set_visible(False)
    footprints, footprint_colors = [], []
    for frame in frames:
        for uav in frame.uavs:
            path = np.asarray(uav.trajectory_xy_m)
            if draw_coverage and uav.role_during_frame == "SERVING":
                # The first point is the frame start, not an extra service step.
                for endpoint in path[1:]:
                    footprints.append(Circle(endpoint, float(scene.cfg["uavs"]["coverage_radius_m"])))
                    footprint_colors.append(UAV_COLORS[uav.uav_id % len(UAV_COLORS)])
            if len(path) > 1:
                returning = uav.role_during_frame in {"WAITING", "CHARGING"}
                ax_map.plot(path[:, 0], path[:, 1], "--" if returning else "-", color=UAV_COLORS[uav.uav_id % len(UAV_COLORS)], linewidth=1.6, alpha=0.65, zorder=3)
    if footprints:
        historical = PatchCollection(footprints, facecolors=footprint_colors,
                                     edgecolors=footprint_colors, linewidths=0.35,
                                     alpha=0.055, zorder=1)
        historical.set_gid("historical_coverage")
        ax_map.add_collection(historical)
    for uid in range(scene.num_uavs):
        color = UAV_COLORS[uid % len(UAV_COLORS)]
        status = int(scene.uav_status[uid])
        if status in (1, 2):
            continue  # Docked UAVs are individually listed in the fleet panel.
        pos = scene.uav_pos[uid]
        at_airship = np.linalg.norm(pos - scene.airship_pos) < 1e-4
        if draw_coverage and status == 0:
            ax_map.add_patch(Circle(pos, float(scene.cfg["uavs"]["coverage_radius_m"]), facecolor=color, edgecolor=color, alpha=0.1, linewidth=1, zorder=1))
        ax_map.scatter(*pos, marker="X" if status == 3 else "^", s=85, color=color, edgecolors="white", linewidths=1, zorder=8)
        if not at_airship:
            ax_map.annotate(f"U{uid}", pos, xytext=(8, 8), textcoords="offset points", fontsize=8, weight="bold", color=color)
    _airship(ax_map, scene)
    at_base = np.flatnonzero(np.linalg.norm(scene.uav_pos - scene.airship_pos, axis=1) < 1e-4)
    if at_base.size:
        ax_map.text(0.98, 0.98, "At airship: " + ", ".join(f"U{i}" for i in at_base), transform=ax_map.transAxes, ha="right", va="top", fontsize=7, color=MUTED, bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 3})
    legend = [Line2D([], [], color=INK, lw=1.5, label="Service path"), Line2D([], [], color=INK, ls="--", label="Settled return")]
    if draw_coverage:
        legend.append(Patch(facecolor=UAV_COLORS[0], alpha=0.2, label="Each step's footprint"))
    ax_map.legend(handles=legend, loc="upper left", fontsize=7, framealpha=0.9)

    ax = fig.add_subplot(gs[0, 1])
    _style_axes(ax, "02  Sensor backlog")
    minutes = np.arange(len(trace.backlog_max_post_timeline)) * scene.low_step_sec / 60
    ax.fill_between(minutes, trace.backlog_mean_post_timeline, color="#cce6ed", alpha=0.7)
    ax.plot(minutes, trace.backlog_max_post_timeline, color="#d59226", lw=1.8, label="Maximum")
    ax.plot(minutes, trace.backlog_mean_post_timeline, color="#167d9a", lw=1.8, label="Mean")
    ax.set(xlabel="Simulation time / min", ylabel="Packets", ylim=(0, None))
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncols=2)

    ax = fig.add_subplot(gs[1, 1])
    _style_axes(ax, "03  Fleet now", grid=False)
    ids = np.arange(scene.num_uavs)
    names = ("SERVING", "WAITING", "CHARGING", "DEAD")
    ax.barh(ids, np.ones(scene.num_uavs) * 100, color="#edf1f6", height=0.5)
    ax.barh(ids, scene.uav_battery * 100, color=[UAV_COLORS[i % len(UAV_COLORS)] for i in ids], height=0.5)
    for uid in ids:
        ax.text(103, uid, f"{scene.uav_battery[uid]:.0%}", va="center", fontsize=8, weight="bold")
        ax.text(121, uid, names[int(scene.uav_status[uid])], va="center", fontsize=7, color=ROLE_COLORS[names[int(scene.uav_status[uid])]])
    ax.set(yticks=ids, yticklabels=[f"U{i}" for i in ids], xlim=(0, 159), xticks=[0, 25, 50, 75, 100], xlabel="Battery / %")
    ax.invert_yaxis()

    ax = fig.add_subplot(gs[2, 0])
    _style_axes(ax, "04  Battery at frame boundaries")
    if trace.frames:
        times = np.r_[0, np.cumsum([f.duration_sec for f in trace.frames])] / 60
        for uid in range(scene.num_uavs):
            series = [trace.frames[0].uavs[uid].battery_start_frac] + [f.uavs[uid].battery_end_frac for f in trace.frames]
            ax.plot(times, np.array(series) * 100, lw=1.3, color=UAV_COLORS[uid % len(UAV_COLORS)], label=f"U{uid}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncols=min(scene.num_uavs, 8), frameon=False, fontsize=7)
    else:
        ax.text(0.5, 0.5, "No completed transitions yet", transform=ax.transAxes, ha="center", color=MUTED)
    ax.set(xlabel="Simulation time / min", ylabel="Battery / %", ylim=(-3, 103))

    ax = fig.add_subplot(gs[2, 1])
    _style_axes(ax, "05  Assigned role during each frame", grid=False)
    start = 0.0
    for frame in trace.frames:
        width = frame.duration_sec / 60
        for uav in frame.uavs:
            ax.broken_barh([(start, width)], (uav.uav_id - 0.35, 0.7), facecolors=ROLE_COLORS[uav.role_during_frame], edgecolors="white", linewidth=0.35, hatch=None if frame.settled else "///")
        start += width
    ax.set(yticks=ids, yticklabels=[f"U{i}" for i in ids], xlabel="Simulation time / min", xlim=(0, max(start, 1)), ylim=(scene.num_uavs - 0.4, -0.6))
    ax.legend(handles=[Patch(color=color, label=role.title()) for role, color in ROLE_COLORS.items()], loc="upper center", bbox_to_anchor=(0.5, -0.30), ncols=4, frameon=False, fontsize=7)


def _heatmaps(fig, env):
    scene, trace = env.scene, env.trace
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.95, bottom=0.095, top=0.765, hspace=0.38, wspace=0.20)
    specs = [
        ("01  Sensor service frequency", trace.sensor_visit_count, "Service events", "YlGnBu"),
        ("02  Collected packets by sensor", trace.sensor_collected_packets, "Packets", "YlOrBr"),
        ("03  Geometric coverage footprint", trace.coverage_grid_step_count, "Low steps covered", "YlGnBu"),
        ("04  UAV endpoint presence", trace.uav_presence_grid_count, "UAV endpoint samples", "PuBu"),
    ]
    for index, (title, data, label, cmap) in enumerate(specs):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        _map_axes(ax, scene, title)
        values = np.asarray(data)
        if index < 2:
            artist = ax.scatter(scene.sensor_pos[:, 0], scene.sensor_pos[:, 1], c=values, s=19, cmap=cmap, vmin=0, vmax=max(float(values.max()), 1), edgecolors="none")
        else:
            artist = ax.imshow(values, origin="lower", extent=(0, scene.world_size, 0, scene.world_size), interpolation="nearest", cmap=cmap, vmin=0, vmax=max(float(values.max()), 1))
        _airship(ax, scene)
        cb = fig.colorbar(artist, ax=ax, fraction=0.035, pad=0.025)
        cb.set_label(label, fontsize=8)
        cb.outline.set_visible(False)


def _footer(fig, env):
    trace = env.trace
    last = trace.frames[-1] if trace.frames else None
    if last is not None and not last.settled:
        note = "UNSETTLED TERMINAL FRAME  |  Hatched roles are assignments. Non-serving battery/position is not settled; planned durations are not actual."
        color = "#b45b38"
    elif last is not None and last.termination_reason:
        note = "TERMINATED  |  " + last.termination_reason.replace("_", " ")
        color = "#b45b38"
    else:
        note = "Raw physical units  |  Backlog includes the cold-start sample  |  Dashed returns show endpoints, not flight-time samples"
        color = MUTED
    fig.text(0.055, 0.027, note, fontsize=8, color=color)
    fig.text(0.055, 0.010, "Coverage grids use step-start serving UAV endpoints, including failures in that step; they are geometric footprints, not successful collection.", fontsize=7, color=MUTED)
