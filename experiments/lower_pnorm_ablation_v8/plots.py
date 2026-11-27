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


def evaluation(env, output, trajectory_steps=80, page_steps=20):
    if trajectory_steps <= 0 or page_steps != 20:
        raise ValueError('v2 uses four 20-step trajectory panels')
    output = Path(output)
    horizon = min(80, trajectory_steps, len(env.rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    vmax = max(1., max(max(r['backlog']) for r in env.rows[:horizon]))
    for panel, ax in enumerate(axes.flat):
        start=panel*20; end=min(start+20,horizon)
        rows=env.rows[start:end]
        if not rows:
            ax.set_axis_off(); continue
        points=ax.scatter(*env.sensor_xy.T,c=rows[-1]['backlog'],cmap='YlOrBr',
                          vmin=0,vmax=vmax,s=18,edgecolors='#374151',linewidths=.25,zorder=4)
        radius=env.world['sensors'].get('partition_radius_m')
        if radius is not None:ax.add_patch(Circle(env.home,radius,fill=False,ls='--',color='gray',alpha=.6))
        ax.scatter(*env.home,marker='*',s=140,color='#b83354',zorder=6)
        for i in range(env.n):
            for row in rows:
                before=np.asarray(row['position_before'])[i];after=np.asarray(row['position_after'])[i]
                status=row['status_before'][i]
                if status in (0,1):
                    ax.plot([before[0],after[0]],[before[1],after[1]],color=COLORS[i],
                            ls='-' if status==0 else '--',lw=1.2,alpha=.85)
                if i in row['action_ids'] and row['soc_after'][i]>0:
                    ax.add_patch(Circle(after,env.radius,color=COLORS[i],alpha=.04))
                    ax.scatter(*after,s=9,color=COLORS[i])
            endpoints=np.asarray(rows[-1]['position_after']);endpoint=endpoints[i]
            ax.scatter(*endpoint,marker='^',s=48,color=COLORS[i],edgecolors='white',zorder=6)
            separation=np.linalg.norm(endpoints-endpoint,axis=1);separation[i]=np.inf
            if separation.min()<240:
                angle=2*np.pi*i/max(env.n,1)
                offset=np.array([35*np.cos(angle),27*np.sin(angle)])
            else:offset=np.array([9.,8.])
            if endpoint[0]<250:offset[0]=abs(offset[0])+8
            if endpoint[0]>env.size-250:offset[0]=-abs(offset[0])-8
            if endpoint[1]<250:offset[1]=abs(offset[1])+10
            if endpoint[1]>env.size-250:offset[1]=-abs(offset[1])-10
            ax.annotate(f'U{i}',endpoint,xytext=tuple(offset),textcoords='offset points',
                        fontsize=8,color=COLORS[i],ha='center',va='center',
                        arrowprops=dict(arrowstyle='-',color=COLORS[i],lw=.65),zorder=7)
        ax.set(xlim=(0,env.size),ylim=(0,env.size),aspect='equal',xlabel='East / m',ylabel='North / m',
               title=f'Steps {start+1}-{end} | backlog at step {end}')
        ax.grid(alpha=.15)
    fig.suptitle('All UAVs | solid: service; dashed: actual return; shading: service footprints',fontsize=13,y=.98)
    handles=[Line2D([0],[0],color=COLORS[i],label=f'U{i}') for i in range(env.n)]
    fig.legend(handles=handles,ncol=env.n,loc='lower center',bbox_to_anchor=(.45,.025))
    fig.text(.45,.009,'Endpoints precede the 20-step full/home reset; panels are successive UAV cycles.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.06,right=.88,bottom=.09,top=.93,wspace=.22,hspace=.25)
    fig.colorbar(points,ax=list(axes.flat),fraction=.025,pad=.025,label='Sensor backlog at each panel end / packets')
    atomic_figure(fig,output/f'trajectory_01_{horizon:02d}.png')
    rows=env.rows;x=np.arange(1,len(rows)+1)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].plot(x,[r['max_backlog'] for r in rows],label='Maximum')
    axes[0].plot(x,[r['mean_backlog'] for r in rows],label='Mean')
    axes[0].set(title='Backlog throughout complete evaluation',ylabel='Packets');axes[0].legend()
    for key in rows[0]['terms']:
        axes[1].plot(x,np.cumsum([r['terms'][key] for r in rows]),label=key)
    axes[1].plot(x,np.cumsum([r['reward'] for r in rows]),color='black',label='total',lw=2)
    axes[1].set_title('Cumulative reward components');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Physical step');ax.grid(alpha=.15)
    fig.tight_layout();atomic_figure(fig,output/'evaluation_metrics.png')
    if output.name=='seed_1001' and len(rows)>=100:
        for offset in range(0,len(rows),300):
            full_evaluation(env,output,offset)


def full_evaluation(env, output, offset=0):
    """Complete rollout in100-step windows; never connect home resets."""
    count=min(3,(len(env.rows)-offset+99)//100)
    fig,axes=plt.subplots(2,count,figsize=(6*count,9),squeeze=False,
                          gridspec_kw={'height_ratios':[3,1]})
    vmax=max(1.,max(max(r['backlog']) for r in env.rows[offset:offset+300]))
    for panel in range(count):
        rows=env.rows[offset+panel*100:min(offset+(panel+1)*100,len(env.rows))];ax=axes[0,panel]
        pts=ax.scatter(*env.sensor_xy.T,c=rows[-1]['backlog'],cmap='YlOrBr',vmin=0,vmax=vmax,
                       s=18,edgecolors='#374151',linewidths=.25,zorder=5)
        ax.add_patch(Circle(env.home,env.world['sensors'].get('partition_radius_m',1800),fill=False,ls='--',color='gray',alpha=.5))
        ax.scatter(*env.home,s=140,marker='*',c='#b83354',zorder=7)
        for i in range(env.n):
            for r in rows:
                a=np.asarray(r['position_before'])[i];b=np.asarray(r['position_after'])[i]
                if r['status_before'][i] in (0,1):
                    ax.plot([a[0],b[0]],[a[1],b[1]],c=COLORS[i],
                            ls='-' if r['status_before'][i]==0 else '--',lw=.9,alpha=.6)
                if i in r['action_ids'] and r['soc_after'][i]>0:
                    ax.add_patch(Circle(b,env.radius,color=COLORS[i],alpha=.018))
                    ax.scatter(*b,s=5,color=COLORS[i])
        ax.set(xlim=(0,env.size),ylim=(0,env.size),aspect='equal',xlabel='East / m',ylabel='North / m',
               title=f'Steps {rows[0]["step"]}-{rows[-1]["step"]} | collected {len({i for r in rows for i in r["covered_ids"]})}/{env.m}')
        ax.grid(alpha=.15)
        bx=axes[1,panel];x=[r['step'] for r in rows]
        bx.plot(x,[r['max_backlog'] for r in rows],label='Maximum')
        bx.plot(x,[r['mean_backlog'] for r in rows],label='Mean')
        bx.set(xlabel='Physical step',ylabel='Packets',ylim=(0,vmax*1.05));bx.grid(alpha=.15);bx.legend(fontsize=8)
    fig.suptitle('Complete deterministic evaluation | every step shown\nSolid: service; dashed: actual return; shading: footprints; reset jumps omitted',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],c=COLORS[i],label=f'U{i}') for i in range(env.n)],
               ncol=env.n,loc='lower center')
    fig.subplots_adjust(left=.055,right=.91,bottom=.08,top=.9,hspace=.22,wspace=.25)
    fig.colorbar(pts,ax=list(axes[0]),fraction=.02,pad=.025,label='Panel-end backlog / packets')
    atomic_figure(fig,Path(output)/f'trajectory_full_{offset+1:03d}_{min(offset+300,len(env.rows)):03d}.png')


def smooth(values, window=10):
    a = np.array(values, dtype=float)
    return np.array([a[max(0,i-window+1):i+1].mean() for i in range(len(a))])


def training(history, losses, evals, output):
    output = Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(13,8))
    if history:
        x = [r['training_steps'] for r in history]
        for ax, key, title in ((axes[0,0], 'episode_reward', 'Training return / 300-step report'),
                               (axes[0,1], 'mean_max_backlog', 'Training mean maximum backlog'),
                               (axes[1,0], 'worst_max_backlog', 'Training worst maximum backlog')):
            y = [r[key] for r in history]
            ax.plot(x,y,alpha=.25,lw=.7)
            ax.plot(x,smooth(y),lw=1.7,label='Trailing 10 episodes')
            ax.set_title(title); ax.legend(fontsize=8)
        for key in ('task','boundary','return_failure'):
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
    for ax,title in zip(axes,(f"Eval return / {evals[0]['steps'] if evals else 600} steps",'Eval mean maximum backlog','Eval worst maximum backlog')):
        ax.set(title=title,xlabel='Training physical steps');ax.grid(alpha=.2)
        if evals:ax.legend(fontsize=8)
    fig.tight_layout();atomic_figure(fig,output/'evaluation_curves.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    if history:
        x=[r['training_steps'] for r in history]
        axes[0,0].plot(x,[r['completed_rounds'] for r in history],alpha=.3)
        axes[0,0].plot(x,smooth([r['completed_rounds'] for r in history]),label='Trailing10 reports')
        axes[0,1].plot(x,[100*r['round_progress'] for r in history])
    for seed in sorted({r['seed'] for r in evals}):
        rr=[r for r in evals if r['seed']==seed];x=[r['training_steps'] for r in rr]
        axes[1,0].plot(x,[r['completed_rounds'] for r in rr],label=f'seed {seed}')
        axes[1,1].plot(x,[100*r['round_progress'] for r in rr],label=f'seed {seed}')
    for ax,title in zip(axes.flat,['Training rounds / report','Training unfinished-round progress %',
                                  'Eval completed full-map rounds','Eval unfinished-round progress %']):
        ax.set(title=title,xlabel='Training physical steps');ax.grid(alpha=.2)
    if history:axes[0,0].legend(fontsize=8)
    if evals:axes[1,0].legend(fontsize=8);axes[1,1].legend(fontsize=8)
    fig.tight_layout();atomic_figure(fig,output/'renewal_curves.png')
