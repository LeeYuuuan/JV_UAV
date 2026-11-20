"""Train an actual continuing small-map SAC task, with bounded diagnostic windows."""
from pathlib import Path
import argparse,json,gzip,time
import numpy as np
import torch
import yaml
from train import ROOT, validate, provenance
from jv_uav.config import load_config
from continuing_trainer import ContinuingTrainer
from continuing_plots import curves
from persistent_plots import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-config',type=Path,default=ROOT/'configs/env_lower_continuing6.yaml')
    p.add_argument('--train-config',type=Path,default=ROOT/'configs/training_lower_continuing6_greedy_rho08.yaml')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--steps',type=int)
    p.add_argument('--resume',type=Path)
    p.add_argument('--seed',type=int)
    args=p.parse_args();torch.set_num_threads(1)
    if args.steps is not None and args.steps<=0:p.error('steps must be positive')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    if not args.resume and any((output/n).exists() for n in ('checkpoint.pt','train.jsonl')):
        p.error('prior run exists; resume explicitly or use a new directory')
    if args.resume:
        t=ContinuingTrainer.load(args.resume,device)
        if args.seed is not None and args.seed!=t.cfg['seed']:p.error('resume seed mismatch')
        if (output/'train.jsonl').exists():
            with (output/'train.jsonl').open('rb') as f:
                f.seek(max(0,(output/'train.jsonl').stat().st_size-65536));lines=f.read().splitlines()
            for line in reversed(lines):
                try:last=json.loads(line)
                except ValueError:continue
                if last['training_steps']>t.steps:p.error('checkpoint older than logs; use a new output directory')
                break
    else:
        cfg=yaml.safe_load(args.train_config.read_text(encoding='utf-8'))
        if args.seed is not None:cfg['seed']=args.seed
        validate(cfg);t=ContinuingTrainer(load_config(args.env_config),cfg,device)
    cfg=t.cfg;validate(cfg)
    write_json(output/'resolved_config.json',dict(environment=t.env_cfg,training=cfg,device=device))
    provenance(output,t,args)
    evals=[]
    if (output/'eval.jsonl').exists():
        for s in (output/'eval.jsonl').read_text().splitlines():
            try:r=json.loads(s)
            except ValueError:continue
            if r['training_steps']<=t.steps:evals.append(r)
    def evaluate(initial=False):
        results=[]
        for seed in cfg['evaluation_seeds']:
            folder=output/f'eval_{t.steps:08d}'/f'seed_{seed}'
            results.append(t.evaluate(seed,folder))
            if initial:results.append(t.evaluate(seed,mode='greedy_reference'))
        with (output/'eval.jsonl').open('a',encoding='utf-8') as f:
            for r in results:f.write(json.dumps(r,allow_nan=False)+'\n')
            f.flush()
        evals.extend(results);write_json(output/'latest_eval.json',results)
        print('  eval | '+' | '.join(f"{r['policy_mode']} seed{r['seed']} R/step={r['mean_step_reward']:.4f} Bmax={r['mean_max_backlog']:.1f} cover={r['sensor_coverage_fraction']:.0%}" for r in results),flush=True)
        page=cfg.get('evaluation_page_steps',15)
        print(f"  plots: {output}/eval_{t.steps:08d}/seed_*/eval_steps_01_{page:02d}.png and eval_steps_{page+1:02d}_{2*page:02d}.png",flush=True)
    budget=args.steps or cfg['total_steps'];end=t.steps+budget
    print(f"Continuing SAC | {cfg['revision']} | {device} | additional steps {budget} | no environment reset or terminal at window/budget boundaries",flush=True)
    print(f"Reward: -B/300 * (1-{cfg['reward']['rho']}*h_top{cfg['reward']['top_k']}) - movement | gamma={cfg['sac']['gamma_per_reference']}",flush=True)
    print(f'Output: {output.resolve()}',flush=True)
    if t.steps==0:evaluate(initial=True)
    started=time.monotonic()
    thresholds={k:(t.steps//cfg[k]+1)*cfg[k] for k in ('eval_every','save_every','plot_every','trace_every_steps')}
    def crossed(key):
        if t.steps<thresholds[key]:return False
        while thresholds[key]<=t.steps:thresholds[key]+=cfg[key]
        return True
    try:
        with (output/'train.jsonl').open('a',encoding='utf-8') as log,gzip.open(output/'windows.jsonl.gz','at',encoding='utf-8',compresslevel=5) as windows:
            while t.steps<end:
                row=t.step();summary=row.pop('window_summary')
                if crossed('trace_every_steps') or row['statistics_window_end'] or t.steps==end:
                    log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');log.flush()
                write_json(output/'live_status.json',row)
                if summary:
                    windows.write(json.dumps(summary,separators=(',',':'),allow_nan=False)+'\n');windows.flush()
                    if t.windows%cfg['log_every_windows']==0:
                        recent=t.history[-cfg['log_every_windows']:]
                        print(f"[SAC update {t.updates:07d}] steps={t.steps:08d} window={t.windows:06d} R/step={np.mean([r['mean_step_reward'] for r in recent]):.4f} Bmax={np.mean([r['mean_max_backlog'] for r in recent]):.1f} elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)
                if crossed('save_every'):t.save(output/'checkpoint.pt')
                if crossed('eval_every') and t.steps<end:evaluate()
                if crossed('plot_every'):curves(t,output,evals)
    except KeyboardInterrupt:
        print('Interrupted; last completed atomic checkpoint retained.',flush=True)
        return
    t.save(output/'checkpoint.pt');evaluate();curves(t,output,evals)
    print(f"Saved {output/'checkpoint.pt'} | environment remains nonterminal",flush=True)


if __name__=='__main__':main()
