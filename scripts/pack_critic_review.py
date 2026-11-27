"""Export compact critic diagnostics from a trusted local training checkpoint.

Read-only with respect to training: no optimizer/replay is modified on disk.
Run from the JV_UAV repository root; full checkpoint/replay is not included.
"""
from pathlib import Path
from datetime import datetime,timezone
from io import BytesIO
from zipfile import ZipFile,ZIP_DEFLATED
import argparse,hashlib,json,subprocess,sys

def worker(run,root):
    import torch
    run=run.resolve()
    resolved=json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))
    revision=resolved['training']['revision']
    if revision.startswith('displacement_backlog_potential_v5'):
        folder='lower_backlog_potential_v5'
    elif revision.startswith('displacement_round_renewal_v7'):
        folder='lower_round_renewal_v7'
    elif revision.startswith('displacement_pnorm_ablation_v8'):
        folder='lower_pnorm_ablation_v8'
    else:raise ValueError(f'Unsupported experiment revision: {revision}')
    sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'experiments'/folder))
    # The checkpoint's Replay object is pickled under its original agent module.
    with (run/'checkpoint.pt').open('rb') as stream:
        state=torch.load(stream,map_location='cpu',weights_only=False)
    sac=state['sac'];required=['actor','critic','target','log_alpha']
    if any(k not in sac for k in required):raise ValueError('Need full checkpoint.pt, not an actor-only policy snapshot')
    replay=state.get('replay')
    samples=replay.sample(min(256,len(replay.items))) if replay is not None and replay.items else []
    payload=dict(format='lower_critic_review_v1',training_format=state['format'],world=state['world'],
        cfg=state['cfg'],steps=state['steps'],updates=state['updates'],
        sac={k:sac[k] for k in required},env={k:v for k,v in state['env'].items() if k!='rows'},
        replay_samples=samples,torch_rng=state.get('torch_rng'),description='Actor, twin critics, target critics, alpha, current simulator state, and <=256 replay transitions. No optimizer/full replay.')
    buffer=BytesIO();torch.save(payload,buffer);data=buffer.getvalue()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=run.with_name(run.name+'_critic_'+stamp+'.zip')
    meta=dict(run=str(run),revision=revision,training_steps=state['steps'],updates=state['updates'],
              replay_sample_count=len(samples),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    with ZipFile(out,'x',compression=ZIP_DEFLATED,compresslevel=3) as z:
        z.writestr('critic_review.pt',data)
        z.writestr('critic_manifest.json',json.dumps(meta,indent=2))
        for f in run.glob('provenance/*/manifest.json'):
            z.write(f,'provenance/'+f.parent.name+'/manifest.json')
    print(f'{out}\nsteps={state["steps"]}, replay samples={len(samples)}, ZIP={out.stat().st_size/1024**2:.2f} MiB',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();root=args.root.resolve()
    if not (root/'src'/'jv_uav').is_dir():p.error('Run from JV_UAV repository root or pass --root')
    if args.worker:
        worker(args.runs[0],root)
    else:
        for run in args.runs:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),str(run),'--root',str(root),'--worker'],check=True)

if __name__=='__main__':main()
