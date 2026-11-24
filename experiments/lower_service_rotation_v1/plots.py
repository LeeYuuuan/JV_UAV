"""Headless, bounded-output scientific figures for server runs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np


COLORS = plt.get_cmap('tab10').colors


def atomic_figure(fig, path):
    path = Path(path)
    tmp = path.with_suffix('.tmp.png')
    fig.savefig(tmp, dpi=140, bbox_inches='tight')
    plt.close(fig)
    tmp.replace(path)


def evaluation(env, output, trajectory_steps=30, page_steps=15):
    if trajectory_steps <= 0 or page_steps <= 0:
        raise ValueError('trajectory_steps and page_steps must be positive')
    output = Path(output)
    horizon = min(trajectory_steps, len(env.rows))
    for start in range(0, horizon, page_steps):
        end = min(start+page_steps, horizon)
        rows = env.rows[start:end]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(9, 9))
        # Backlog values correspond to this panel's endpoint, not the end of evaluation.
        points = ax.scatter(*env.sensor_xy.T, c=rows[-1]['backlog'], cmap='YlOrBr',
                            s=22, edgecolors='#374151', linewidths=.3, zorder=4)
        fig.colorbar(points, ax=ax, fraction=.045, label=f'Backlog after step {end} / packets')
        ax.add_patch(Circle(env.home, 1800, fill=False, ls='--', color='gray', alpha=.6))
        ax.scatter(*env.home, marker='*', s=180, color='#b83354', zorder=6)
        # Physical position_after is BEFORE a reset; next_positions is AFTER it.
        # Never connect these two positions as a flight segment.
        reset_rows = [row for row in rows if row.get('cycle_reset', False)]
        reset_steps = [row['step'] for row in reset_rows]
        latest_segment_start = max(reset_steps)+1 if reset_steps and max(reset_steps)<end else start+1
        for i in range(env.n):
            for row in rows:
                before = np.array(row['position_before'])[i]
                after = np.array(row['position_after'])[i]
                status = row['status_before'][i]
                old_segment = row['step'] < latest_segment_start
                alpha = .35 if old_segment else .9
                if status in (0, 1):
                    ax.plot([before[0], after[0]], [before[1], after[1]],
                            color=COLORS[i], ls='-' if status == 0 else '--', lw=1.3, alpha=alpha)
                if i in row['action_ids'] and row['soc_after'][i] > 0:
                    ax.add_patch(Circle(after, env.radius, color=COLORS[i], alpha=.025 if old_segment else .045))
                    ax.scatter(*after, s=10, color=COLORS[i], alpha=alpha)
            for reset in reset_rows:
                position = np.array(reset['position_after'])[i]
                if np.linalg.norm(position-env.home)>1:
                    ax.scatter(*position, marker='s', s=40, facecolors='none', edgecolors=COLORS[i], zorder=5)
                    separation=np.linalg.norm(np.array(reset['position_after'])-position,axis=1)
                    separation[i]=np.inf
                    if separation.min()>100:
                        ax.annotate(f'U{i}@{reset["step"]}', position, xytext=(4,-10),
                                    textcoords='offset points', fontsize=7, color=COLORS[i])
            final = np.array(rows[-1]['position_after'])[i]
            ax.scatter(*final, marker='^', s=65, color=COLORS[i], edgecolor='white', zorder=5)
            # At-home IDs remain identifiable by color without overlapping labels.
            separation=np.linalg.norm(np.array(rows[-1]['position_after'])-final,axis=1)
            separation[i]=np.inf
            if np.linalg.norm(final-env.home)>1 and separation.min()>100:
                ax.annotate(f'U{i}@{rows[-1]["step"]}', final, xytext=(4,4), textcoords='offset points', fontsize=8)
        title=f'All UAVs | steps {start+1}-{end}\nSolid: serving; dashed: actual return; shading: service footprints'
        if reset_steps:
            title+='\nFull/home reset after step '+', '.join(map(str,reset_steps))+' (not a return flight)'
            if latest_segment_start>start+1:
                title+=f'; faded: before reset'
        ax.set(xlim=(0, env.size), ylim=(0, env.size), aspect='equal', xlabel='East / m', ylabel='North / m',
               title=title)
        ax.grid(alpha=.15)
        handles = [Line2D([0],[0],color=COLORS[i], label=f'U{i}') for i in range(env.n)]
        if reset_steps:
            handles.extend([Line2D([],[],color='gray',marker='s',markerfacecolor='none',ls='none',label='Before reset'),
                            Line2D([],[],color='gray',marker='^',ls='none',label='Panel end')])
        ax.legend(handles=handles, ncol=4 if reset_steps else env.n, loc='upper center', bbox_to_anchor=(.5,-.07))
        atomic_figure(fig, output/f'trajectory_{start+1:02d}_{end:02d}.png')
    rows = env.rows
    x = np.arange(1, len(rows)+1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0,0].plot(x, [r['max_backlog'] for r in rows], label='Maximum')
    axes[0,0].plot(x, [r['mean_backlog'] for r in rows], label='Mean')
    axes[0,0].set(title='Backlog throughout complete evaluation', ylabel='Packets')
    axes[0,0].legend()
    soc = np.array([rows[0]['soc_before']] + [r['next_soc'] for r in rows])
    for i in range(env.n):
        axes[0,1].plot(np.arange(len(rows)+1), soc[:,i]*100, color=COLORS[i], label=f'U{i}')
    axes[0,1].set(title='SOC at physical step boundaries', ylabel='SOC / %', ylim=(-2,102))
    for row in rows:
        if row['cycle_reset']:
            axes[0,1].axvline(row['step'], color='gray', lw=.5, alpha=.25)
    axes[0,1].legend(ncol=3, fontsize=8)
    for key in rows[0]['terms']:
        axes[1,0].plot(x, np.cumsum([r['terms'][key] for r in rows]), label=key)
    axes[1,0].plot(x, np.cumsum([r['reward'] for r in rows]), color='black', label='total', lw=2)
    axes[1,0].set(title='Cumulative reward components')
    axes[1,0].legend(fontsize=8)
    status = np.array([r['status_before'] for r in rows]).T
    image = axes[1,1].imshow(status, aspect='auto', interpolation='nearest', origin='lower',
                             extent=(.5,len(rows)+.5,-.5,env.n-.5), vmin=-.5, vmax=5.5,
                             cmap=plt.get_cmap('tab10',6))
    axes[1,1].set(title='Role at step start', yticks=np.arange(env.n),
                  yticklabels=[f'U{i}' for i in range(env.n)])
    cb = fig.colorbar(image, ax=axes[1,1], ticks=range(6))
    cb.ax.set_yticklabels(['Serving','Returning','Charging','Waiting','Idle','Stopped'], fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel('Physical step'); ax.grid(alpha=.15)
    fig.tight_layout()
    atomic_figure(fig, output/'evaluation_metrics.png')


def smooth(values, window=10):
    a = np.array(values, dtype=float)
    return np.array([a[max(0,i-window+1):i+1].mean() for i in range(len(a))])


def training(history, losses, evals, output):
    output = Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(13,8))
    if history:
        x = [r['training_steps'] for r in history]
        for ax, key, title in ((axes[0,0], 'episode_reward', 'Training episode reward / 300 steps'),
                               (axes[0,1], 'mean_max_backlog', 'Training mean maximum backlog'),
                               (axes[1,0], 'worst_max_backlog', 'Training worst maximum backlog')):
            y = [r[key] for r in history]
            ax.plot(x,y,alpha=.25,lw=.7)
            ax.plot(x,smooth(y),lw=1.7,label='Trailing 10 episodes')
            ax.set_title(title); ax.legend(fontsize=8)
        for key in ('target','auxiliary','boundary','return_failure'):
            axes[1,1].plot(x,smooth([r[f'reward_{key}'] for r in history]),label=key)
        axes[1,1].set_title('Training reward components / episode'); axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Training physical steps');ax.grid(alpha=.2)
    fig.tight_layout();atomic_figure(fig,output/'training_curves.png')
    fig, axes = plt.subplots(2,2,figsize=(13,8))
    if losses:
        x = [r['training_steps'] for r in losses]
        for key in ('actor_loss','policy_loss','entropy_loss'):
            axes[0,0].plot(x,[r[key] for r in losses],label=key)
        for key in ('q1_loss','q2_loss'):
            axes[0,1].plot(x,[r[key] for r in losses],label=key)
        for key in ('q_mean','target_mean'):
            axes[1,0].plot(x,[r[key] for r in losses],label=key)
        axes[1,1].plot(x,[r['alpha'] for r in losses],label='alpha')
        axes[1,1].plot(x,[r['sigma_mean'] for r in losses],label='sigma mean')
    for ax,title in zip(axes.flat,('Actor objective components','Critic losses','Q and target scale','Exploration scale')):
        ax.set(title=title,xlabel='Training physical steps');ax.grid(alpha=.2)
        if losses:ax.legend(fontsize=8)
    fig.tight_layout();atomic_figure(fig,output/'losses.png')
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for seed in sorted({r['seed'] for r in evals}):
        subset = [r for r in evals if r['seed']==seed]
        x = [r['training_steps'] for r in subset]
        for ax,key in zip(axes,('episode_reward','mean_max_backlog','worst_max_backlog')):
            ax.plot(x,[r[key] for r in subset],label=f'seed {seed}',marker='.',ms=3)
    for ax,title in zip(axes,('Eval episode reward / 300 steps','Eval mean maximum backlog','Eval worst maximum backlog')):
        ax.set(title=title,xlabel='Training physical steps');ax.grid(alpha=.2)
        if evals:ax.legend(fontsize=8)
    fig.tight_layout();atomic_figure(fig,output/'evaluation_curves.png')
