"""Read-only actor-mean / sampled-actor / greedy diagnostic comparison."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from environment import Environment
from agent import SAC
from plots import evaluation, atomic_figure, plt
from train import FORMAT, write_json


def greedy_action(env, obs):
    actions=[]
    for i in obs['ids']:
        if not obs['assigned_valid'][i]:raise RuntimeError('missing target assignment')
        delta=obs['assigned_xy'][i]-obs['xy'][i]
        actions.append(delta/max(env.max_distance,float(np.linalg.norm(delta)),1e-8))
    return np.array(actions,dtype=np.float32).reshape(-1,2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seeds',type=int,nargs='+',default=[1001,1002,1003])
    p.add_argument('--sample-repeats',type=int,default=3)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=p.parse_args();torch.set_num_threads(1)
    if args.steps<=0 or args.sample_repeats<=0:p.error('steps/repeats must be positive')
    if len(set(args.seeds))!=len(args.seeds):p.error('duplicate seeds')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):p.error('use a fresh output directory')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    with args.checkpoint.open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        stream.seek(0);saved=torch.load(stream,map_location='cpu',weights_only=False)
    if saved.get('format')!=FORMAT:p.error('wrong experiment checkpoint')
    cfg,world=copy.deepcopy(saved['cfg']),copy.deepcopy(saved['world'])
    weights=saved['sac']['actor'];trained_steps=saved['steps'];del saved
    physical_cfg=copy.deepcopy(cfg);physical_cfg['episode_steps']=max(args.steps,cfg['episode_steps'])
    env=Environment(world,physical_cfg,args.seeds[0])
    sac=SAC(env,cfg['sac'],device);sac.actor.load_state_dict(weights);sac.actor.eval();del weights,env
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'resolved_config.json',dict(environment=world,training=cfg,
        evaluation=dict(steps=args.steps,seeds=args.seeds,sample_repeats=args.sample_repeats,
                        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest.hexdigest(),
                        checkpoint_training_steps=trained_steps,training_updated=False,
                        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')},
                        comparison_note='Same map/config/initial seed. Availability affects random dispatch and shared RNG consumption, so different policies are not strict matched counterfactuals.')))
    results=[]
    for policy in ('actor_mean','actor_sample','greedy'):
        for seed in args.seeds:
            for repeat in range(args.sample_repeats if policy=='actor_sample' else 1):
                action_seed=seed*10000+repeat+600
                torch.manual_seed(action_seed)
                if torch.cuda.is_available():torch.cuda.manual_seed_all(action_seed)
                env=Environment(world,physical_cfg,seed);visits=np.zeros(env.m,dtype=int)
                for _ in range(args.steps):
                    obs=env.observation()
                    action=greedy_action(env,obs) if policy=='greedy' else sac.act(obs,deterministic=policy=='actor_mean')
                    _,_,_,_,row=env.step(action);visits[row['covered_ids']]+=1
                summary=env.summary() | dict(policy=policy,seed=seed,repeat=repeat,action_seed=action_seed,
                    training_steps=trained_steps,never_visited=np.flatnonzero(visits==0).tolist(),
                    visit_counts=visits.tolist(),training_updated=False)
                folder=output/f'policy_{policy}'/f'seed_{seed}_repeat_{repeat}';folder.mkdir(parents=True)
                write_json(folder/'summary.json',summary)
                with gzip.open(folder/'trace.json.gz','wt',encoding='utf-8',compresslevel=5) as stream:
                    json.dump(dict(sensor_xy=env.sensor_xy.tolist(),rows=env.rows),stream,separators=(',',':'),allow_nan=False)
                evaluation(env,folder)
                results.append(summary)
                print(f"{policy} seed={seed} repeat={repeat} R={summary['episode_reward']:.2f} mean_Bmax={summary['mean_max_backlog']:.1f} coverage={summary['coverage_fraction']:.1%} missed={summary['never_visited']}",flush=True)
    aggregate=[]
    for policy in ('actor_mean','actor_sample','greedy'):
        rows=[r for r in results if r['policy']==policy]
        aggregate.append(dict(policy=policy,rollouts=len(rows),**{key:float(np.mean([r[key] for r in rows]))
            for key in ('episode_reward','mean_max_backlog','worst_max_backlog','coverage_fraction','penalty_count')}))
    write_json(output/'comparison_summary.json',dict(aggregate=aggregate,rollouts=results))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,title in zip(axes,('episode_reward','mean_max_backlog','coverage_fraction'),
                            ('Return / evaluation','Time-average max backlog','Ever-visited fraction')):
        ax.bar(range(3),[r[key] for r in aggregate],color=['#267d9b','#bc7a42','#509c72'],alpha=.8)
        for i,row in enumerate(aggregate):
            values=[r[key] for r in results if r['policy']==row['policy']]
            ax.scatter(np.full(len(values),i),values,color='black',s=15,zorder=3)
        ax.set(xticks=range(3),xticklabels=['Actor mean','Actor sampled','Greedy'],title=title)
        ax.grid(axis='y',alpha=.2)
    fig.tight_layout();atomic_figure(fig,output/'comparison.png')
    print(f'Output: {output}',flush=True)


if __name__=='__main__':main()
