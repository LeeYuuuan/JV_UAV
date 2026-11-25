"""Compact read-only snapshots: first/latest/best and intermediate evals, actors, logs."""
from pathlib import Path
from datetime import datetime,timezone
from zipfile import ZipFile,ZIP_DEFLATED
import argparse,gzip,hashlib,json,os

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--recent-evals',type=int,default=3);args=p.parse_args()
    run=args.run.resolve()
    if not (run/'resolved_config.json').is_file():p.error('expected a run with resolved_config.json')
    if args.recent_evals<1:p.error('recent-evals must be positive')
    evals=sorted(f for f in run.glob('eval_*') if f.is_dir())
    selected=set(evals[:1]+evals[-args.recent_evals:])
    if evals:
        for fraction in (.25,.5,.75):selected.add(evals[round((len(evals)-1)*fraction)])
    for name in ['best_eval.json','best_round_eval.json']:
        try:
            step=json.loads((run/name).read_text())['training_steps'];folder=run/f'eval_{step:08d}'
            if folder.is_dir():selected.add(folder)
        except (OSError,ValueError,KeyError):pass
    paths=[f for f in run.iterdir() if f.is_file() and f.suffix in ('.json','.jsonl','.png')]
    for name in ['policy_best.pt','policy_latest.pt','policy_best_rounds.pt']:
        if (run/name).is_file():paths.append(run/name)
    for folder in selected:
        paths.extend(f for f in folder.rglob('*') if f.is_file() and f.suffix in ('.json','.png','.gz'))
    for folder in (run/'provenance').glob('*'):
        paths.extend(f for f in folder.iterdir() if f.suffix in ('.json','.zip'))
    paths.extend(sorted((run/'episode_traces').glob('*.json.gz'))[-3:])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=run.with_name(run.name+'_'+stamp+'.zip')
    manifest={'run':str(run),'full_checkpoint_included':False,'actors_included':True,
              'selected_evals':sorted(f.name for f in selected),'files':{},'skipped':[]}
    with ZipFile(output,'x',compression=ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(set(paths)):
            name=f.relative_to(run).as_posix()
            try:
                with f.open('rb') as stream:data=stream.read(os.fstat(stream.fileno()).st_size)
                if f.suffix=='.jsonl':
                    good=[]
                    for line in data.splitlines(keepends=True):
                        if not line.endswith(b'\n'):continue
                        try:json.loads(line)
                        except (ValueError,UnicodeDecodeError):continue
                        good.append(line)
                    data=b''.join(good)
                elif f.suffix=='.json':json.loads(data)
                elif f.suffix=='.gz':json.loads(gzip.decompress(data))
                z.writestr(name,data)
                manifest['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            except (OSError,EOFError,ValueError,UnicodeDecodeError,gzip.BadGzipFile):manifest['skipped'].append(name)
        z.writestr('review_manifest.json',json.dumps(manifest,indent=2))
    print(output);print(f'{len(paths)} candidate files; {len(selected)} eval snapshots; checkpoint/replay excluded')

if __name__=='__main__':main()
