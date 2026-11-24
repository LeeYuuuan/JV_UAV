"""Package this experiment's diagnostics, including snapshots of running jobs."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--recent-evals',type=int,default=3)
    args=p.parse_args()
    run=args.run.resolve()
    if not (run/'resolved_config.json').is_file():p.error('not an experiment output directory')
    if args.recent_evals<1:p.error('--recent-evals must be positive')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=run.with_name(f'{run.name}_{stamp}.zip')
    paths=[p for p in run.iterdir() if p.suffix in ('.json','.jsonl','.png')]
    # Standalone evaluation outputs contain seed folders directly under the root.
    for directory in list(run.glob('seed_*'))+list(run.glob('policy_*')):
        if directory.is_dir():
            paths.extend(p for p in directory.rglob('*') if p.suffix in ('.json','.png','.gz'))
    evals=sorted(p for p in run.glob('eval_*') if p.is_dir())
    selected=sorted(set(evals[:1]+evals[-args.recent_evals:]))
    for directory in selected:
        paths.extend(p for p in directory.rglob('*') if p.suffix in ('.json','.png','.gz'))
    for directory in sorted((run/'provenance').glob('*')):
        paths.extend(p for p in directory.iterdir() if p.suffix in ('.json','.zip'))
    paths.extend(sorted((run/'episode_traces').glob('*.json.gz'))[-3:])
    manifest=dict(run=str(run),checkpoints_included=False,files={},skipped=[])
    with ZipFile(output,'x',compression=ZIP_DEFLATED,compresslevel=5) as archive:
        for path in sorted(set(paths)):
            name=path.relative_to(run).as_posix()
            try:
                # Freeze the size observed when opening; ignore newly appended bytes.
                with path.open('rb') as stream:data=stream.read(path.stat().st_size)
                if path.suffix=='.jsonl':
                    valid=[]
                    for line in data.splitlines(keepends=True):
                        if not line.endswith(b'\n'):continue
                        try:json.loads(line)
                        except (ValueError,UnicodeDecodeError):continue
                        valid.append(line)
                    data=b''.join(valid)
                elif path.suffix=='.json':json.loads(data)
                elif path.suffix=='.gz':json.loads(gzip.decompress(data))
                archive.writestr(name,data);manifest['files'][name]=len(data)
            except (OSError,EOFError,ValueError,UnicodeDecodeError):
                manifest['skipped'].append(name)
        archive.writestr('snapshot_manifest.json',json.dumps(manifest,indent=2))
    print(output)
    print(f'{len(manifest["files"])} files; checkpoints excluded; in-progress incomplete files skipped: {len(manifest["skipped"])}')


if __name__=='__main__':main()
