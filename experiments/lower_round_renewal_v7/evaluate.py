"""Evaluate a saved actor over a chosen horizon; never resume or update training."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from environment import Environment
from agent import SAC
from plots import evaluation
from train import FORMAT, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--page-steps',type=int,choices=[20],default=20)
    p.add_argument('--trajectory-steps',type=int,default=80,help='At most 80 steps, in one 2x2 figure')
    p.add_argument('--seeds',type=int,nargs='+',default=[1001,1002,1003])
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=p.parse_args();torch.set_num_threads(1)
    if args.steps<=0 or args.page_steps<=0 or (args.trajectory_steps is not None and args.trajectory_steps<=0):
        p.error('step counts must be positive')
    if len(set(args.seeds))!=len(args.seeds):p.error('evaluation seeds must be unique')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):p.error('use a new empty evaluation output directory')
    # Hash and load the SAME opened file, even if training atomically replaces
    # checkpoint.pt concurrently. The already-open handle pins this snapshot.
    with args.checkpoint.open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        stream.seek(0)
        saved=torch.load(stream,map_location='cpu',weights_only=False)
    if saved.get('format')!=FORMAT:p.error('checkpoint belongs to another experiment')
    cfg,world=copy.deepcopy(saved['cfg']),copy.deepcopy(saved['world'])
    actor_state=saved['sac']['actor']
    trained_steps,updates=saved['steps'],saved['updates']
    del saved  # Evaluation does not need the potentially large training replay.
    env_cfg=copy.deepcopy(cfg)
    # The300-step reporting period stays fixed; the task itself is continuous.
    probe=Environment(world,env_cfg,args.seeds[0])
    sac=SAC(probe,cfg['sac'],device)
    sac.actor.load_state_dict(actor_state);sac.actor.eval()
    del actor_state,probe
    output.mkdir(parents=True,exist_ok=True)
    settings=dict(steps=args.steps,seeds=args.seeds,page_steps=args.page_steps,
                  trajectory_steps=min(80,args.trajectory_steps),
                  checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest.hexdigest(),
                  checkpoint_training_steps=trained_steps,checkpoint_updates=updates,
                  policy='actor_mean_only',training_updated=False,
                  source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in HERE.glob('*.py')})
    write_json(output/'resolved_config.json',dict(environment=world,training=cfg,evaluation=settings))
    results=[]
    print(f"Evaluation only | trained steps={trained_steps} | horizon={args.steps} | round reward={cfg['reward']['round_reward']} | {device}",flush=True)
    for seed in args.seeds:
        env=Environment(world,env_cfg,seed)
        probes=[]
        for _ in range(args.steps):
            obs=env.observation()
            probes.append(dict(step=env.step_count+1,**sac.diagnostics(obs)))
            env.step(sac.act(obs,deterministic=True))
        summary=env.summary() | dict(seed=seed,training_steps=trained_steps,updates=updates,
                                    revision=cfg['revision'],policy='actor_mean_only',
                                    evaluation_steps=args.steps,evaluation_cutoff=True,training_updated=False)
        folder=output/f'seed_{seed}';folder.mkdir()
        write_json(folder/'summary.json',summary)
        with gzip.open(folder/'trace.json.gz','wt',encoding='utf-8',compresslevel=5) as stream:
            json.dump(dict(sensor_xy=env.sensor_xy.tolist(),rows=env.rows,action_distribution=probes),
                      stream,separators=(',',':'),allow_nan=False)
        evaluation(env,folder,trajectory_steps=settings['trajectory_steps'],page_steps=args.page_steps)
        results.append(summary)
        print(f"seed={seed} steps={args.steps} R={summary['episode_reward']:.2f} mean/worst Bmax={summary['mean_max_backlog']:.1f}/{summary['worst_max_backlog']:.1f} coverage={summary['coverage_fraction']:.1%}",flush=True)
    write_json(output/'summary.json',results)
    print(f'Output: {output}',flush=True)


if __name__=='__main__':main()
