"""Export small lower-policy diagnostics, not the full training replay."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import io
import json
import sys
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS={
    'displacement_service_assignment_v2':'lower_service_assignment_v2',
    'displacement_service_energy_v4':'lower_service_energy_v4',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+')
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.samples<1:p.error('--samples must be positive')
    paths=[r.resolve() for r in args.runs]
    if len({r.name for r in paths})!=len(paths):p.error('run directory names must be unique')
    for run in paths:
        if not (run/'resolved_config.json').is_file() or not (run/'checkpoint.pt').is_file():
            p.error(f'missing resolved config or full checkpoint: {run}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.output.resolve() if args.output else ROOT/'runs'/f'lower_diagnostics_{stamp}.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(training_changed=False,full_replay_included=False,runs=[])
    sys.path.insert(0,str(ROOT/'src'))
    with ZipFile(output,'x',compression=ZIP_DEFLATED,compresslevel=5) as archive:
        for run in paths:
            resolved=json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))
            revision=resolved['training']['revision']
            family=next((k for k in EXPERIMENTS if revision.startswith(k)),None)
            if family is None:raise ValueError(f'unsupported experiment revision: {revision}')
            # Replay is pickled as agent.Replay. Both supported versions share
            # this class/schema; reimport from the correct independent folder.
            module_path=str(ROOT/'experiments'/EXPERIMENTS[family])
            sys.path.insert(0,module_path);sys.modules.pop('agent',None);importlib.import_module('agent')
            with (run/'checkpoint.pt').open('rb') as stream:
                digest=hashlib.sha256()
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
                stream.seek(0);saved=torch.load(stream,map_location='cpu',weights_only=False)
            if saved['format']!=family:raise ValueError('checkpoint/config version mismatch')
            replay=saved['replay']
            count=min(args.samples,len(replay.items))
            indices=np.linspace(0,len(replay.items)-1,count,dtype=int).tolist() if count else []
            example=[replay.items[i] for i in indices]
            environment={k:v for k,v in saved['env'].items() if k!='rows'}
            diagnostic=dict(format=saved['format'],diagnostic_schema='lower_diagnostics_v1',
                cfg=saved['cfg'],world=saved['world'],steps=saved['steps'],updates=saved['updates'],
                sac={k:saved['sac'][k] for k in ('actor','critic','target','log_alpha')},
                env=environment,replay_examples=example,replay_indices=indices,
                source_checkpoint_sha256=digest.hexdigest(),training_updated=False)
            buffer=io.BytesIO();torch.save(diagnostic,buffer)
            archive.writestr(f'{run.name}/latest_diagnostics.pt',buffer.getvalue())
            for name in ('resolved_config.json','best_eval.json','policy_best.pt','policy_latest.pt'):
                path=run/name
                if path.is_file():archive.writestr(f'{run.name}/{name}',path.read_bytes())
            item=dict(run=str(run),revision=revision,training_steps=saved['steps'],
                      updates=saved['updates'],replay_size=len(replay.items),exported_examples=count,
                      source_checkpoint_sha256=digest.hexdigest())
            manifest['runs'].append(item)
            print(f"Packed {run.name}: step {saved['steps']}, {count} replay examples; full replay omitted",flush=True)
            del diagnostic,environment,example,replay,saved,buffer
            sys.path.remove(module_path)
        archive.writestr('diagnostics_manifest.json',json.dumps(manifest,indent=2))
    print(output)
    print(f'{output.stat().st_size/1024/1024:.1f} MiB; existing training and checkpoints unchanged')


if __name__=='__main__':main()
