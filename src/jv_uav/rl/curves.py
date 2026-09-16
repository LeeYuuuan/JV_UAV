"""Completed-episode training curves, independent of policy and optimizer state."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


class TrainingCurves:
    def __init__(self, output, *, resume=None, upper_steps=0):
        self.output = Path(output)
        self.episodes = {}
        # Import history when resuming into a new output directory as well.
        sources = [Path(resume).parent] if resume is not None else []
        if self.output not in sources:
            sources.append(self.output)
        for source in sources:
            history = source / 'training_curves.json'
            if history.exists():
                for item in json.loads(history.read_text(encoding='utf-8')):
                    if item['upper_steps'] <= upper_steps:
                        self.episodes[item['episode']] = item
            log = source / 'train.jsonl'
            if log.exists():
                with log.open(encoding='utf-8') as stream:
                    for line in stream:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # An interrupted write may leave a partial line.
                        if row['upper_steps'] <= upper_steps:
                            self.add(row)

    def add(self, row):
        if not (row['terminated'] or row['truncated']):
            return
        item = dict(row.get('episode_metrics') or {})
        item.update(episode=int(row['episodes']), upper_steps=int(row['upper_steps']),
                    upper_return=float(row['episode_return']), dead_count=len(row['dead_ids']))
        # Old logs contain episode return but not full-episode backlog statistics.
        # Never substitute the final frame's backlog for an episode mean.
        self.episodes[item['episode']] = item

    def save(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.ticker import MaxNLocator

        rows = [self.episodes[key] for key in sorted(self.episodes)]
        fig = Figure(figsize=(12, 8), dpi=130, facecolor='#f5f7fb')
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        panels = [
            ('upper_return', 'Upper episode return', 'Raw reward sum', '#167d9a'),
            ('mean_max_backlog', 'Average max buffer after collection', 'Packets', '#9063bd'),
            ('peak_max_backlog', 'Worst max buffer in episode', 'Packets', '#e19b26'),
            ('low_steps', 'Episode length', 'Lower steps', '#3b9672'),
        ]
        for ax, (key, title, ylabel, color) in zip(axes.flat, panels):
            valid = [row for row in rows if key in row and row[key] is not None]
            ax.set(title=title, xlabel='Completed episode', ylabel=ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.18)
            ax.spines[['top', 'right']].set_visible(False)
            if not valid:
                ax.text(0.5, 0.5, 'Waiting for completed-episode metrics',
                        transform=ax.transAxes, ha='center', va='center', color='#667085')
                continue
            x = np.array([row['episode'] for row in valid])
            y = np.array([row[key] for row in valid], dtype=float)
            ax.plot(x, y, color=color, alpha=0.35, lw=0.8, marker='.' if len(x) < 3 else None, label='Raw')
            if len(y) >= 10:
                means = np.convolve(y, np.ones(10) / 10, mode='valid')
                # Do not draw a rolling average across missing episodes.
                means[(x[9:] - x[:-9]) != 9] = np.nan
                ax.plot(x[9:], means, color=color, lw=1.7, label='10-episode mean')
            if key == 'low_steps':
                dead = np.array([row['dead_count'] > 0 for row in valid])
                ax.scatter(x[dead], y[dead], color='#d45e6b', s=10, label='Death termination', zorder=3)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle('Training progress | completed episodes', fontsize=16, color='#20354c')
        fig.text(0.5, 0.02, 'Backlog uses every post-service lower step; cold start excluded. Short episodes can hide poor performance.',
                 ha='center', fontsize=9, color='#667085')
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        self.output.mkdir(parents=True, exist_ok=True)
        # Readers see either the previous complete image or the new complete image.
        image_tmp = self.output / 'training_curves.tmp.png'
        fig.savefig(image_tmp)
        image_tmp.replace(self.output / 'training_curves.png')
        history_tmp = self.output / 'training_curves.tmp.json'
        history_tmp.write_text(json.dumps(rows, allow_nan=False), encoding='utf-8')
        history_tmp.replace(self.output / 'training_curves.json')
        fig.clear()
