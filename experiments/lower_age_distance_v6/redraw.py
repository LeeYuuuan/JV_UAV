"""Redraw saved evaluation traces without loading models or changing training."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from environment import Environment
from plots import evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_directory',type=Path,help='A completed eval_XXXXXXXX directory')
    args=parser.parse_args()
    directory=args.evaluation_directory.resolve()
    resolved=directory.parent/'resolved_config.json'
    if not resolved.is_file():parser.error('expected RUN/eval_XXXXXXXX with RUN/resolved_config.json')
    config=json.loads(resolved.read_text(encoding='utf-8'))
    traces=sorted(directory.glob('seed_*/trace.json.gz'))
    if not traces:parser.error('no completed seed_*/trace.json.gz files found')
    for trace in traces:
        # Read a frozen byte snapshot. Incomplete in-progress gzip files fail clearly.
        data=trace.read_bytes()
        payload=json.loads(gzip.decompress(data))
        seed=int(trace.parent.name.removeprefix('seed_'))
        env=Environment(config['environment'],config['training'],seed)
        env.sensor_xy=np.array(payload['sensor_xy'],dtype=np.float32)
        env.rows=payload['rows']
        output=trace.parent/'plots_v2';output.mkdir(exist_ok=True)
        evaluation(env,output)
        metadata=dict(source_trace=str(trace),trace_sha256=hashlib.sha256(data).hexdigest(),
                      plot_source_sha256=hashlib.sha256((HERE/'plots.py').read_bytes()).hexdigest(),
                      original_figures_preserved=True,model_rerun=False,
                      cycle_reset_steps=[r['step'] for r in env.rows if r.get('cycle_reset',False)])
        (output/'redraw_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(output)


if __name__=='__main__':main()
