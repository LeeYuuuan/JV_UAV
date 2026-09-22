"""Plot the configured radial sensor map without running training."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from jv_uav import load_config
from jv_uav.sensor_generator import generate_radial_sensor_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/sensor_distribution')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg['sensors'].get('distribution') != 'radial_clusters':
        parser.error('this plot requires distribution: radial_clusters')
    data = generate_radial_sensor_distribution(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    positions, centers = data['positions'], data['centers']
    inside, scattered = data['inside'], data['is_scattered']
    base = np.asarray(cfg['world']['airship_xy_m'])
    radius = cfg['sensors']['partition_radius_m']
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle
    fig = Figure(figsize=(10, 10), dpi=160, facecolor='#f5f7fb')
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    ax.add_patch(Circle(base, radius, facecolor='#eaf4f7', edgecolor='#42576b',
                        linestyle=(0, (6, 4)), linewidth=1.6, zorder=1))
    for mask, color, name in [(inside, '#147f9a', 'Inside'), (~inside, '#dc7b29', 'Outside')]:
        ax.scatter(*positions[mask].T, s=30, marker='o', color=color,
                   edgecolors='white', linewidths=0.5,
                   label=f'{name}: {int(mask.sum())} sensors', zorder=3)
    ax.scatter(*centers.T, marker='x', s=70, color='#344054', linewidths=1.5, label='Cluster centers', zorder=5)
    for index, center in enumerate(centers):
        ax.annotate(f'C{index+1}', center, xytext=(7, 10), textcoords='offset points',
                    fontsize=9, color='#344054', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
    ax.scatter(*base, marker='*', s=260, color='#20354c', edgecolors='white', linewidths=1,
               label='Airship (2000, 2000)' if np.array_equal(base, [2000,2000]) else 'Airship', zorder=6)
    ax.annotate(f'r = {radius:,.0f} m', base + [0, radius], xytext=(0, -20),
                textcoords='offset points', ha='center', fontsize=11, color='#42576b')
    size = cfg['world']['size_m']
    ax.set(xlim=(0, size), ylim=(0, size), xlabel='East / m', ylabel='North / m', aspect='equal')
    ax.grid(alpha=0.15)
    ax.set_title(f'Sensor distribution | {len(positions)} sensors, {len(centers)} clusters\n'
                 f'{int(inside.sum())} inside r={radius:.0f} m / {int((~inside).sum())} outside',
                 fontsize=16, color='#20354c', pad=18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.025, f"Outer sensors: {cfg['sensors']['outer_radius_range_m'][0]:.0f}-{cfg['sensors']['outer_radius_range_m'][1]:.0f} m from airship | Map seed {cfg['seeds']['map_seed']}", ha='center', fontsize=9, color='#667085')
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.17, top=0.89)
    fig.savefig(args.output / 'sensor_distribution.png')
    table = np.column_stack([np.arange(len(positions)), positions, data['cluster_ids'], scattered, inside])
    np.savetxt(args.output / 'sensor_positions.csv', table, delimiter=',',
               header='sensor_id,x_m,y_m,cluster_id,is_scattered,inside_1800m', comments='',
               fmt=['%d','%.6f','%.6f','%d','%d','%d'])
    distances = np.linalg.norm(positions.astype(float) - base, axis=1)
    summary = dict(count=len(positions), inner_count=int(inside.sum()), outer_count=int((~inside).sum()),
                   cluster_count=len(centers), core_count=int((~scattered).sum()), scattered_count=int(scattered.sum()),
                   min_outer_radius_m=float(distances[~inside].min()), max_outer_radius_m=float(distances[~inside].max()),
                   beyond_one_frame_service_reach=int((distances > cfg['time']['low_steps_per_frame'] * cfg['uavs']['speed_mps'] * cfg['uavs']['max_move_time_sec'] + cfg['uavs']['coverage_radius_m']).sum()))
    (args.output / 'distribution_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.output / 'environment_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f"Saved: {(args.output / 'sensor_distribution.png').resolve()}")


if __name__ == '__main__':
    main()
