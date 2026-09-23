"""Run only the initial joint warm-up and aggregate coverage across resets."""
from pathlib import Path
import argparse
import json
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from jv_uav import load_config
from jv_uav.rl.trainer import JointTrainer


class WarmupComplete(Exception):
    """Stop after the requested lower callback without inventing a terminal."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--legacy-waypoints', action='store_true', help='Disable energy guard for matched baseline')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/warmup_coverage')
    args = parser.parse_args()
    torch.set_num_threads(1)
    env_cfg = load_config(ROOT / 'configs/default.yaml')
    cfg = yaml.safe_load((ROOT / 'configs/training.yaml').read_text(encoding='utf-8'))
    cfg['seed'] = args.seed
    if args.legacy_waypoints:
        cfg['sac']['warmup_energy_guard'] = False
    if not 0 < args.steps <= cfg['sac']['random_steps']:
        parser.error('--steps must be positive and within configured waypoint warm-up')
    if cfg['sac']['warmup_mode'] != 'waypoint':
        parser.error('This diagnostic requires waypoint warm-up')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    trainer = JointTrainer(env_cfg, cfg, args.device)
    scene = trainer.env.scene
    sensors = scene.sensor_pos.copy()
    center = scene.airship_pos.copy()
    radius = trainer.env.metrics.radius
    grid_n = 100
    axis = (np.arange(grid_n) + .5) * scene.world_size / grid_n
    xx, yy = np.meshgrid(axis, axis)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    footprint = np.zeros(grid_n * grid_n, np.int64)
    visits = np.zeros(len(sensors), np.int64)
    packets = np.zeros(len(sensors))
    by_uav = np.zeros((scene.num_uavs, len(sensors)), np.int64)
    first_seen = np.full(len(sensors), -1, np.int64)
    positions, segments, segment_ids, rows, progression = [], [], [], [], []
    generated, reached = [], []
    service_runs = []
    run_lengths = np.zeros(scene.num_uavs, np.int64)
    last_episode = -1
    termination_counts = {'mid_step_death_steps': 0, 'final_return_failure_steps': 0}
    start = monotonic()
    original_target = trainer.exploration._target

    def target(uid, stratified):
        point = original_target(uid, stratified)
        generated.append([trainer.low_steps, int(uid), float(point[0]), float(point[1])])
        if not stratified:
            reached.append([trainer.low_steps, int(uid)])
        return point

    trainer.exploration._target = target
    original_callback = trainer.env.low_runner.on_step

    def collect(obs, action, result):
        nonlocal last_episode
        original_callback(obs, action, result)
        step = trainer.low_steps
        if last_episode != trainer.episodes:
            service_runs.extend(run_lengths[run_lengths > 0].tolist())
            run_lengths[:] = 0
            last_episode = trainer.episodes
        active = np.zeros(scene.num_uavs, bool)
        active[result.serving_ids] = True
        ended = (~active) & (run_lengths > 0)
        service_runs.extend(run_lengths[ended].tolist())
        run_lengths[~active] = 0
        run_lengths[active] += 1
        owned = result.sensor_owner >= 0
        first_seen[owned & (first_seen < 0)] = step
        visits[:] += owned
        packets[:] += result.collected_per_sensor
        for uid in result.serving_ids:
            by_uav[uid] += result.sensor_owner == uid
            positions.append([step, trainer.episodes, int(uid), *result.position_after[uid]])
            segments.append([result.position_before[uid].copy(), result.position_after[uid].copy()])
            segment_ids.append(int(uid))
        current = result.position_after[result.serving_ids]
        if len(current):
            delta = grid_points[:, None] - current[None, :]
            footprint[:] += np.any(np.sum(delta * delta, axis=2) <= radius ** 2, axis=1)
        termination_counts['mid_step_death_steps'] += bool(result.dead_during_step.size)
        termination_counts['final_return_failure_steps'] += bool(result.return_unsafe_ids.size)
        progression.append([step, int((visits > 0).sum()), int((visits[outer] > 0).sum())])
        if step % 1000 == 0:
            print(f'low {step:05d} | sensors {(visits>0).sum()}/{len(visits)} | outer {(visits[outer]>0).sum()}/{outer.sum()} | MAPPO {trainer.mappo_updates} | elapsed {monotonic()-start:.1f}s', flush=True)
        if step == args.steps:
            raise WarmupComplete()

    outer = np.linalg.norm(sensors - center, axis=1) > 1800
    trainer.env.low_runner.on_step = collect
    try:
        while True:
            rows.append(trainer.step())
    except WarmupComplete:
        pass
    # The last run for each active UAV is censored at the observation cutoff.
    seen = visits > 0
    pos = np.asarray(positions)
    max_radius = float(np.linalg.norm(pos[:, 3:5] - center, axis=1).max()) if len(pos) else 0.
    summary = dict(seed=args.seed, low_steps=trainer.low_steps,
        completed_upper_frames=trainer.upper_steps, final_frame_observed_lower_steps=scene.step_in_frame,
        completed_episodes=trainer.episodes, mappo_updates=trainer.mappo_updates,
        sac_updates=trainer.sac_updates, visited_sensors=int(seen.sum()), total_sensors=len(seen),
        first_full_coverage_step=int(first_seen.max()) if seen.all() else None,
        sensor_visit_min_median_max=[int(visits.min()), float(np.median(visits)), int(visits.max())],
        outer_visit_min_median_max=[int(visits[outer].min()), float(np.median(visits[outer])), int(visits[outer].max())] if outer.any() else [],
        inner_visited=int(seen[~outer].sum()), inner_total=int((~outer).sum()),
        outer_visited=int(seen[outer].sum()), outer_total=int(outer.sum()),
        unvisited_sensor_ids=np.flatnonzero(~seen).tolist(),
        geometric_grid_coverage_fraction=float((footprint>0).mean()), max_serving_radius_m=max_radius,
        generated_waypoints=len(generated), reached_waypoints=len(reached),
        exploration_counts=dict(trainer.exploration.counts), energy_guard=trainer.exploration.energy_guard,
        completed_service_run_lengths_low_steps=service_runs,
        censored_service_run_lengths_low_steps=run_lengths[run_lengths>0].tolist(),
        elapsed_seconds=monotonic()-start, **termination_counts,
        notes=['All coverage accumulates across episode resets; it is not coverage in one episode.',
               'Sensor visits use actual service ownership; heatmap uses geometric post-move footprints, including steps that fail.',
               'Stops immediately after the final lower callback; final upper settlement/update is intentionally not executed.',
               'Waypoint reaches count replacements after coming within 90m; a last-step arrival may not yet be counted.',
               'Normal MAPPO sampling/updates and SAC warm-up threshold are preserved.'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'resolved_config.json').write_text(json.dumps(dict(env=env_cfg, training=cfg), indent=2), encoding='utf-8')
    (out/'upper_frames.json').write_text(json.dumps(rows), encoding='utf-8')
    np.savez_compressed(out/'coverage_data.npz', sensor_positions=sensors, sensor_visits=visits,
        sensor_packets=packets, first_seen_step=first_seen, visits_by_uav=by_uav,
        footprint=footprint.reshape(grid_n, grid_n), serving_positions=pos,
        generated_waypoints=np.asarray(generated), reached_waypoints=np.asarray(reached),
        coverage_progress=np.asarray(progression))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), facecolor='#f3f6fa', layout='constrained')
    extent = [0, scene.world_size, 0, scene.world_size]
    for ax in axes.flat[:3]:
        ax.set(xlim=extent[:2], ylim=extent[2:], aspect='equal', xlabel='East / m', ylabel='North / m')
        ax.add_patch(Circle(center, 1800, fill=False, ls='--', lw=1.5, color='#596779'))
        ax.scatter(*center, marker='*', s=160, c='#172c45', zorder=5)
    heat = footprint.reshape(grid_n, grid_n)
    im = axes[0,0].imshow(np.ma.masked_equal(heat,0), origin='lower', extent=extent,
        cmap='YlOrRd', norm=LogNorm(vmin=1, vmax=max(2,int(heat.max()))))
    axes[0,0].scatter(sensors[:,0], sensors[:,1], s=9, c='#16354c', alpha=.65)
    fig.colorbar(im, ax=axes[0,0], label='Lower steps with geometric coverage (log scale)')
    axes[0,0].set_title('01 | All serving footprints accumulated across resets', loc='left')
    im = axes[0,1].scatter(sensors[seen,0], sensors[seen,1], c=visits[seen], cmap='viridis',
        norm=LogNorm(vmin=1,vmax=max(2,int(visits.max()))),s=32,edgecolors='white',linewidths=.3)
    axes[0,1].scatter(sensors[~seen,0], sensors[~seen,1], facecolors='none',edgecolors='#d62839',s=50, label='Never visited')
    for uid in np.flatnonzero(~seen):
        axes[0,1].annotate(str(uid), sensors[uid], xytext=(3,3),textcoords='offset points',fontsize=7,color='#ae2434')
    axes[0,1].legend(loc='lower left')
    fig.colorbar(im, ax=axes[0,1], label='Actual sensor service steps (log scale)')
    axes[0,1].set_title(f'02 | Sensors visited: {seen.sum()}/{len(seen)}; outer {seen[outer].sum()}/{outer.sum()}',loc='left')
    lc = LineCollection(segments, colors=plt.cm.tab10(np.asarray(segment_ids)%10),linewidths=.4,alpha=.13)
    axes[1,0].add_collection(lc)
    axes[1,0].scatter(sensors[:,0], sensors[:,1],s=12,c='#18344a')
    axes[1,0].set_title('03 | Actual service paths (no artificial links across resets)',loc='left')
    prog=np.asarray(progression)
    axes[1,1].plot(prog[:,0],prog[:,1],label='All sensors',color='#187c94')
    axes[1,1].plot(prog[:,0],prog[:,1]-prog[:,2],label='Inside 1800m',color='#438455')
    axes[1,1].plot(prog[:,0],prog[:,2],label='Outside 1800m',color='#d58c24')
    axes[1,1].axhline(len(seen),color='#187c94',ls=':',alpha=.5)
    axes[1,1].set(xlabel='Cumulative lower environment steps',ylabel='Distinct sensors visited',ylim=(0,len(seen)+5))
    axes[1,1].legend();axes[1,1].grid(alpha=.15)
    axes[1,1].set_title('04 | How quickly exploration discovers sensors',loc='left')
    fig.suptitle(f'JOINT WARM-UP COVERAGE | seed {args.seed} | {args.steps:,} lower steps\n'
                 f'MAPPO updates: {trainer.mappo_updates}   |   Waypoints generated/reached: {len(generated)}/{len(reached)}   |   Max radius: {max_radius:.0f} m',fontsize=17,fontweight='bold')
    fig.savefig(out/'warmup_coverage.png',dpi=160)
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('visited_sensors','inner_visited','outer_visited','generated_waypoints','reached_waypoints','max_serving_radius_m','completed_episodes','mappo_updates','sac_updates','elapsed_seconds')},indent=2))
    print(str((out/'warmup_coverage.png').resolve()))


if __name__ == '__main__':
    main()
