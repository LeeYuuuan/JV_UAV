"""Server entry point for the independent displacement-service-rotation experiment."""
from pathlib import Path
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import random
import sys
import time
import zipfile
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jv_uav.config import load_config
from environment import Environment
from agent import SAC, Replay
from plots import training, evaluation

FORMAT = 'displacement_age_distance_v6'


def configure_reward(cfg, *, resume=False, **overrides):
    """Fresh replay for a reward change; omitted resume options restore saved values."""
    cfg = copy.deepcopy(cfg)
    for key,value in overrides.items():
        if value is None:
            continue
        if key not in cfg['reward']:
            raise ValueError(f'unknown reward option {key}')
        saved=cfg['reward'][key]
        equal=(saved==value if isinstance(value,str) else math.isclose(float(saved),value,rel_tol=0,abs_tol=1e-12))
        if resume and not equal:
            raise ValueError(f'resume reward mismatch: {key}={saved}, requested {value}; start a fresh run')
        cfg['reward'][key]=value
    cfg['revision']=FORMAT
    return cfg


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def append(path, data):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
        stream.flush()


def validate(cfg, world):
    for name in ('episode_steps','dispatch_every','check_every','total_steps','eval_every','save_every',
                 'loss_log_every','trace_every_episodes','target_count'):
        if type(cfg[name]) is not int or cfg[name]<=0:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['check_every']%cfg['dispatch_every'] or cfg['episode_steps']%cfg['check_every']:
        raise ValueError('dispatch/check/episode periods must align')
    n = world['uavs']['count']
    if cfg['target_count']<n:
        raise ValueError('target_count must be at least fleet size for assignment padding')
    if not 1<=cfg['service_count_min']<=cfg['service_count_max']<=n:
        raise ValueError('invalid service counts')
    if cfg['charging_slots'] is not None and not 1<=cfg['charging_slots']<=n:
        raise ValueError('invalid charging slots')
    rw=cfg['reward']
    for name in ('backlog_reference_packets','distance_reference_m','pnorm_p'):
        if not math.isfinite(rw[name]) or rw[name]<=0:
            raise ValueError(f'invalid {name}')
    if not math.isfinite(rw['navigation_weight']) or rw['navigation_weight']<0:
        raise ValueError('invalid navigation weight')
    if any(key in rw for key in ('target_bonus','auxiliary_max','cost_mode','potential_beta')):
        raise ValueError('old task/shaping/collection rewards are not allowed in v6')
    if cfg['age_scale_sec']<=0 or cfg['reward']['return_distance_scale_m']<=0:
        raise ValueError('invalid normalization scale')
    if cfg['reward']['energy_deficit_scale_soc']<=0 or cfg['reward']['energy_deficit_weight']<0:
        raise ValueError('invalid linear energy-deficit settings')
    sac = cfg['sac']
    if not 0<sac['gamma']<1 or sac['hidden']%sac['heads']:
        raise ValueError('invalid gamma or attention dimensions')
    if not 0<sac['batch_size']<=sac['replay_capacity']:
        raise ValueError('invalid replay batch size')
    ex = cfg['exploration']
    if not 0<=ex['epsilon_end']<=ex['epsilon_start']<=1 or ex['decay_steps']<=0:
        raise ValueError('invalid epsilon schedule')
    if min(ex['random_probability'],ex['greedy_probability'])<0 or ex['random_probability']+ex['greedy_probability']>1:
        raise ValueError('invalid behavior mixture')


class Trainer:
    def __init__(self, world, cfg, device):
        self.world, self.cfg = copy.deepcopy(world), copy.deepcopy(cfg)
        validate(cfg,world)
        seed = cfg['seed']
        random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        self.env = Environment(world,cfg,seed)
        self.sac = SAC(self.env,cfg['sac'],device)
        self.replay = Replay(cfg['sac']['replay_capacity'],seed+100)
        self.behavior_rng = np.random.default_rng(seed+200)
        self.steps = self.updates = self.episodes = 0
        self.history, self.losses, self.evals = [], [], []
        self.pending_losses = []
        self.source_counts = dict(actor=0,greedy=0,random=0)
        self.last_loss = {}
        self.best_eval_score=float('inf')
        self.best_eval_step=0

    def behavior(self, obs):
        action = self.sac.act(obs)
        ex = self.cfg['exploration']
        epsilon = ex['epsilon_start']+(ex['epsilon_end']-ex['epsilon_start'])*min(1.,self.steps/ex['decay_steps'])
        sources = []
        for k,i in enumerate(obs['ids']):
            source = 'actor'
            if self.behavior_rng.random()<epsilon:
                roll = self.behavior_rng.random()
                if roll<ex['random_probability']:
                    action[k] = self.behavior_rng.uniform(-1,1,2);source='random'
                elif roll<ex['random_probability']+ex['greedy_probability']:
                    if not obs['assigned_valid'][i]:
                        raise RuntimeError('serving UAV has no assigned target')
                    delta=obs['assigned_xy'][i]-obs['xy'][i]
                    action[k]=delta/max(self.env.max_distance,float(np.linalg.norm(delta)),1e-8)
                    source='greedy'
            sources.append(source);self.source_counts[source]+=1
        return action,sources,epsilon

    def save(self,path):
        state = dict(format=FORMAT,world=self.world,cfg=self.cfg,env=self.env.state_dict(),
                     sac=self.sac.state_dict(),replay=self.replay,behavior_rng=self.behavior_rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if self.sac.device.type=='cuda' else None)
        for key in ('steps','updates','episodes','history','losses','evals','source_counts','last_loss','pending_losses',
                    'best_eval_score','best_eval_step'):
            state[key]=getattr(self,key)
        path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

    @classmethod
    def load(cls,path,device):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('format')!=FORMAT:raise ValueError('checkpoint belongs to another experiment')
        if 'replay' not in state:raise ValueError('policy snapshots are for evaluation; resume from checkpoint.pt')
        obj=cls(state['world'],state['cfg'],device)
        obj.env.load_state_dict(state['env']);obj.sac.load_state_dict(state['sac'])
        obj.replay=state['replay'];obj.behavior_rng.bit_generator.state=state['behavior_rng']
        for key in ('steps','updates','episodes','history','losses','evals','source_counts','last_loss','pending_losses',
                    'best_eval_score','best_eval_step'):
            setattr(obj,key,state[key])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if obj.sac.device.type=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        return obj

    def evaluate(self,output):
        results=[]
        for seed in self.cfg['eval_seeds']:
            env=Environment(self.world,self.cfg,seed)
            probes=[]
            for _ in range(self.cfg['episode_steps']):
                obs=env.observation()
                if env.step_count<80:probes.append(dict(step=env.step_count+1,**self.sac.diagnostics(obs)))
                env.step(self.sac.act(obs,deterministic=True))
            summary=env.summary() | dict(seed=seed,training_steps=self.steps,updates=self.updates,
                                         revision=self.cfg['revision'],policy='actor_mean_only')
            self.evals.append(summary)
            results.append(summary)
            folder=output/f'eval_{self.steps:08d}'/f'seed_{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'summary.json',summary)
            with gzip.open(folder/'trace.json.gz','wt',encoding='utf-8',compresslevel=5) as stream:
                json.dump(dict(sensor_xy=env.sensor_xy.tolist(),rows=env.rows,action_distribution=probes),stream,separators=(',',':'),allow_nan=False)
            evaluation(env,folder)
            append(output/'eval.jsonl',summary)
            print(f"  eval seed={seed} R={summary['episode_reward']:.2f} Bmax(mean/worst)={summary['mean_max_backlog']:.1f}/{summary['worst_max_backlog']:.1f} coverage={summary['coverage_fraction']:.1%} serving={summary['mean_serving']:.2f}",flush=True)
        score=float(np.mean([r['mean_max_backlog'] for r in results]))
        def policy_snapshot(path):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            state=dict(format=FORMAT,world=self.world,cfg=self.cfg,steps=self.steps,updates=self.updates,
                       sac={'actor':self.sac.actor.state_dict()},evaluation_score=score,
                       selection_metric='mean of evaluation seeds time-average maximum backlog')
            temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)
        policy_snapshot(output/'policy_latest.pt')
        if score<self.best_eval_score:
            self.best_eval_score,self.best_eval_step=score,self.steps
            policy_snapshot(output/'policy_best.pt')
            write_json(output/'best_eval.json',dict(training_steps=self.steps,mean_max_backlog=score,
                       revision=self.cfg['revision'],metric='mean of evaluation seeds time-average maximum backlog'))
        if self.steps and self.steps%30000==0:
            policy_snapshot(output/'policy_snapshots'/f'policy_{self.steps:08d}.pt')


def provenance(output,trainer,args):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=output/'provenance'/stamp;folder.mkdir(parents=True)
    sources=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.yaml'))+sorted(HERE.glob('*.md'))+sorted((ROOT/'src/jv_uav').rglob('*.py'))
    manifest={}
    with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            name=source.relative_to(ROOT).as_posix();data=source.read_bytes()
            manifest[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
    write_json(folder/'manifest.json',dict(revision=trainer.cfg['revision'],checkpoint_format=FORMAT,files_sha256=manifest,
               arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               torch=torch.__version__,device=str(trainer.sac.device),starting_steps=trainer.steps,
               description='V6: direct mean estimated-target-backlog times remaining coverage distance; no global backlog cost/potential; v4 safety/physics.'))
    write_json(folder/'resolved_config.json',dict(environment=trainer.world,training=trainer.cfg))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'config.yaml')
    parser.add_argument('--env-config',type=Path,default=HERE/'environment.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--steps',type=int,help='Additional physical steps, including when resuming')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--backlog-ref',type=float)
    parser.add_argument('--distance-ref',type=float)
    parser.add_argument('--navigation-weight',type=float)
    args=parser.parse_args();torch.set_num_threads(1)
    reward_options=dict(backlog_reference_packets=args.backlog_ref,
                        distance_reference_m=args.distance_ref,navigation_weight=args.navigation_weight)
    if args.steps is not None and args.steps<=0:parser.error('--steps must be positive')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        trainer=Trainer.load(args.resume,device)
        try:configure_reward(trainer.cfg,resume=True,**reward_options)
        except ValueError as exc:parser.error(str(exc))
        if trainer.env.cfg['reward'] != trainer.cfg['reward']:
            parser.error('checkpoint environment reward differs from trainer configuration')
        if args.seed is not None and args.seed!=trainer.cfg['seed']:parser.error('resume seed mismatch')
        for name in ('episodes.jsonl','losses.jsonl','progress.jsonl','eval.jsonl'):
            path=output/name
            if path.exists():
                for line in reversed(path.read_text(encoding='utf-8').splitlines()):
                    try:last=json.loads(line)
                    except ValueError:continue
                    if last['training_steps']>trainer.steps:parser.error('logs newer than checkpoint; resume into a new output directory')
                    break
    else:
        if any(output.iterdir()):parser.error('output directory is not empty; use --resume or a new directory')
        cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
        try:cfg=configure_reward(cfg,**reward_options)
        except ValueError as exc:parser.error(str(exc))
        if args.seed is not None:cfg['seed']=args.seed
        trainer=Trainer(load_config(args.env_config),cfg,device)
    cfg=trainer.cfg
    write_json(output/'resolved_config.json',dict(environment=trainer.world,training=cfg))
    provenance(output,trainer,args)
    end=trainer.steps+(args.steps or cfg['total_steps']);started=time.monotonic()
    print(f"{cfg['revision']} | device={device} | additional steps={end-trainer.steps} | output={output}",flush=True)
    print(f"gamma={cfg['sac']['gamma']} | direct age-distance cost | weight={cfg['reward']['navigation_weight']} | B_ref={cfg['reward']['backlog_reference_packets']} | distance_ref={cfg['reward']['distance_reference_m']}m | cycle_reset={cfg['cycle_reset']}",flush=True)
    print(f"Linear energy cost: weight={cfg['reward']['energy_deficit_weight']} per deficit scale={cfg['reward']['energy_deficit_scale_soc']} SOC | one-time failure={cfg['reward']['return_failure_per_uav']} + {cfg['reward']['return_distance_weight']}*distance/{cfg['reward']['return_distance_scale_m']}",flush=True)
    if trainer.steps==0:
        trainer.evaluate(output)
        training(trainer.history,trainer.losses,trainer.evals,output)
    try:
        while trainer.steps<end:
            if trainer.env.step_count>=cfg['episode_steps']:trainer.env.reset()
            obs=trainer.env.observation()
            action,sources,epsilon=trainer.behavior(obs)
            nxt,reward,terminated,truncated,row=trainer.env.step(action)
            row.update(sources=sources,epsilon=epsilon)
            # Store the physical successor BEFORE reset. Truncation is not termination.
            trainer.replay.add(obs,action,reward,nxt,terminated)
            trainer.steps+=1
            if trainer.steps>=cfg['sac']['learning_starts'] and len(trainer.replay.items)>=cfg['sac']['batch_size']:
                trainer.last_loss=trainer.sac.update(trainer.replay.sample(cfg['sac']['batch_size']))
                trainer.updates+=1;trainer.pending_losses.append(trainer.last_loss)
            if trainer.steps%cfg['loss_log_every']==0:
                status=dict(revision=cfg['revision'],training_steps=trainer.steps,updates=trainer.updates,
                            episode=trainer.episodes+1,episode_step=trainer.env.step_count,
                            reward=reward,max_backlog=row['max_backlog'],serving=len(obs['ids']),
                            episode_reward_so_far=float(sum(r['reward'] for r in trainer.env.rows)),
                            soc=trainer.env.soc.tolist(),status=trainer.env.status.tolist(),
                            last_losses=trainer.last_loss.copy(),
                            source_counts=trainer.source_counts.copy(),epsilon=epsilon)
                status.update(reward_terms=row['terms'],navigation_cost=row['navigation_cost'])
                append(output/'progress.jsonl',status);write_json(output/'live_status.json',status)
                if trainer.pending_losses:
                    loss={k:float(np.mean([r[k] for r in trainer.pending_losses])) for k in trainer.pending_losses[0]}
                    loss.update(training_steps=trainer.steps,updates=trainer.updates)
                    trainer.losses.append(loss);append(output/'losses.jsonl',loss);trainer.pending_losses.clear()
            if truncated:
                trainer.episodes+=1
                summary=trainer.env.summary() | dict(revision=cfg['revision'],episode=trainer.episodes,
                         training_steps=trainer.steps,updates=trainer.updates,terminated=False,truncated=True,
                         source_counts=trainer.source_counts.copy(),epsilon=epsilon)
                trainer.history.append(summary);append(output/'episodes.jsonl',summary)
                if trainer.episodes%cfg['trace_every_episodes']==0:
                    trace=output/'episode_traces';trace.mkdir(exist_ok=True)
                    with gzip.open(trace/f'episode_{trainer.episodes:06d}.json.gz','wt',encoding='utf-8') as stream:
                        json.dump(trainer.env.rows,stream,separators=(',',':'),allow_nan=False)
                trainer.source_counts=dict(actor=0,greedy=0,random=0)
                print(f"[update {trainer.updates:07d}] steps={trainer.steps:08d} ep={trainer.episodes:05d} R={summary['episode_reward']:.2f} Bmax={summary['mean_max_backlog']:.1f} serving={summary['mean_serving']:.2f} actor={trainer.last_loss.get('actor_loss',0):.3f} critic={trainer.last_loss.get('critic_loss',0):.3f} elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)
                training(trainer.history,trainer.losses,trainer.evals,output)
            if trainer.steps%cfg['eval_every']==0:
                trainer.evaluate(output)
                training(trainer.history,trainer.losses,trainer.evals,output)
            if trainer.steps%cfg['save_every']==0:trainer.save(output/'checkpoint.pt')
    except KeyboardInterrupt:
        print('Interrupted; saving current state for resume.',flush=True)
        trainer.save(output/'checkpoint.pt')
        training(trainer.history,trainer.losses,trainer.evals,output)
        return
    if trainer.steps%cfg['eval_every']:
        trainer.evaluate(output)
    trainer.save(output/'checkpoint.pt')
    training(trainer.history,trainer.losses,trainer.evals,output)
    print(f"Saved {output/'checkpoint.pt'}",flush=True)


if __name__=='__main__':main()
