"""Two consecutive evaluation pages; physical step labels remain global."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from persistent_plots import save_figure, write_json


def evaluation_pages(env, output):
    for offset in (0, 15):
        rows = env.records[offset:offset+15]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
        ax = axes[0, 0]; xy = env.scene.sensor_pos
        ax.scatter(*xy.T, c='black', s=35, zorder=4)
        for i, point in enumerate(xy):
            ax.annotate(f'S{i}', point, xytext=(4, 5), textcoords='offset points', fontsize=9)
        path = np.array([rows[0]['positions_start'][0]] + [r['positions_end'][0] for r in rows])
        ax.plot(*path.T, 'o-', color='#1d8096', lw=1.4, ms=4)
        ax.scatter(*path[0], c='red', marker='*', s=110, label='Segment start', zorder=5)
        for row in rows:
            p = row['positions_end'][0]
            ax.annotate(str(row['step']), p, fontsize=7)
            ax.add_patch(Circle(p, env.env_cfg['uavs']['coverage_radius_m'], fc='#1d8096', alpha=.05, ec='none'))
        ax.set(xlim=(0, env.scene.world_size), ylim=(0, env.scene.world_size), aspect='equal',
               xlabel='East / m', ylabel='North / m', title='Executed trajectory and service footprints')
        ax.legend(fontsize=8); ax.grid(alpha=.15)
        x = [r['step'] for r in rows]
        axes[0, 1].plot(x, [r['max_backlog'] for r in rows], label='Max backlog')
        axes[0, 1].plot(x, [r['mean_backlog'] for r in rows], label='Mean backlog')
        axes[0, 1].set(title='Post-collection backlog', xlabel='Physical step', ylabel='Packets')
        axes[0, 1].legend()
        axes[1, 0].plot(x, [r['reward'] for r in rows], 'k-', label='Total reward')
        for key in rows[0]['reward_terms']:
            axes[1, 0].plot(x, [r['reward_terms'][key] for r in rows], label=key, alpha=.75)
        axes[1, 0].set(title='Physical-step reward terms', xlabel='Physical step')
        axes[1, 0].legend(fontsize=8)
        collected = np.zeros((env.scene.num_sensors, len(rows)))
        for k, r in enumerate(rows):collected[r['covered_ids'], k] = 1
        axes[1, 1].imshow(collected, aspect='auto', cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        axes[1, 1].set(xticks=range(len(rows)), xticklabels=x, yticks=range(env.scene.num_sensors),
                       yticklabels=[f'S{i}' for i in range(env.scene.num_sensors)],
                       title='Actually cleared sensors (blue)', xlabel='Physical step')
        for ax in (axes[0, 1], axes[1, 0]):ax.grid(alpha=.15)
        fig.suptitle(f'Actor-only evaluation | steps {offset+1}-{offset+15}\n'
                     'One continuous rollout; no reset at step 15; energy disabled', fontsize=12)
        save_figure(fig, output / f'eval_steps_{offset+1:02d}_{offset+15:02d}.png')


def curves(trainer, output, evaluations):
    write_json(output/'training_curves.json', dict(revision=trainer.cfg['revision'], windows=trainer.history,
               evaluations=evaluations, note='Windows are statistics only; the environment and bootstrap continue.'))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    if trainer.history:
        x = [r['training_steps'] for r in trainer.history]
        axes[0, 0].plot(x, [r['mean_step_reward'] for r in trainer.history], lw=.7)
        axes[0, 1].plot(x, [r['mean_max_backlog'] for r in trainer.history], lw=.7)
    axes[0, 0].set(title='Training: reward per step (15-step windows)')
    axes[0, 1].set(title='Training: mean max backlog (15-step windows)')
    policy = [r for r in evaluations if r['policy_mode']=='policy']
    for seed in sorted({r['seed'] for r in policy}):
        rows = [r for r in policy if r['seed']==seed]
        x = [r['training_steps'] for r in rows]
        axes[1, 0].plot(x, [r['mean_step_reward'] for r in rows], label=f'seed{seed}')
        axes[1, 1].plot(x, [r['mean_max_backlog'] for r in rows], label=f'seed{seed}')
    axes[1, 0].set(title='Pure actor evaluation: mean reward per step')
    axes[1, 1].set(title='Pure actor evaluation: mean maximum backlog')
    for ax in axes.flat:ax.set_xlabel('Physical training steps'); ax.grid(alpha=.15)
    if policy:axes[1, 0].legend();axes[1, 1].legend()
    save_figure(fig, output/'training_curves.png')
    if trainer.loss_history:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        x=[r['training_steps'] for r in trainer.loss_history]
        for key in ('actor_loss','policy_loss','entropy_loss'):
            axes[0,0].plot(x,[r[key] for r in trainer.loss_history],label=key)
        axes[0,0].legend();axes[0,0].set_title('Actor loss terms')
        axes[0,1].plot(x,[r['critic_loss'] for r in trainer.loss_history]);axes[0,1].set_title('Critic loss')
        for key in ('q_mean','target_mean'):
            axes[1,0].plot(x,[r[key] for r in trainer.loss_history],label=key)
        axes[1,0].legend();axes[1,0].set_title('Q and TD target')
        axes[1,1].plot(x,[r['alpha'] for r in trainer.loss_history]);axes[1,1].set_title('SAC alpha')
        for ax in axes.flat:ax.set_xlabel('Physical training steps');ax.grid(alpha=.15)
        save_figure(fig,output/'loss_curves.png')
