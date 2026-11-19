"""Train only SAC on random20/10step service horizons, using real energy."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
import sys
import time
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(Path(__file__).resolve().parent)]
import torch
import yaml
from jv_uav.config import load_config
from isolated_trainer import LowerTrainer,write_json
from isolated_plots import curves


def provenance(output,trainer,args):
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder=output/"provenance"/stamp;folder.mkdir(parents=True)
    files=sorted(set(ROOT.glob("*.py"))|set((ROOT/"src").rglob("*.py"))|set((ROOT/"configs").glob("*.yaml"))|set(Path(__file__).resolve().parent.glob("*.py")))
    hashes={}
    with ZipFile(folder/"source.zip","w",ZIP_DEFLATED) as z:
        for path in files:
            name=path.relative_to(ROOT).as_posix();data=path.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
    write_json(folder/"manifest.json",dict(revision=trainer.cfg["revision"],source_sha256=hashes,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        starting_low_steps=trainer.low_steps,seed=trainer.cfg["seed"],device=str(trainer.device)))
    write_json(folder/"resolved_config.json",dict(environment=trainer.env.env_cfg,training=trainer.cfg))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config",type=Path,default=ROOT/"configs/default.yaml")
    parser.add_argument("--train-config",type=Path,default=ROOT/"configs/training_lower_isolated.yaml")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,help="Additional lower environment steps")
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--smoke",action="store_true")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--return-penalty",dest="return_penalty",action="store_true")
    group.add_argument("--no-return-penalty",dest="return_penalty",action="store_false")
    parser.set_defaults(return_penalty=None)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.steps is not None and args.steps<=0:parser.error("steps must be positive")
    if args.resume and args.smoke:parser.error("smoke cannot alter a resumed checkpoint")
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device
    if device=="cuda" and not torch.cuda.is_available():parser.error("CUDA requested but unavailable")
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    if not args.resume and ((output/"train.jsonl").exists() or (output/"checkpoint.pt").exists()):parser.error("use a fresh output or --resume")
    if args.resume:
        trainer=LowerTrainer.load(args.resume,device)
        if args.seed is not None and args.seed!=trainer.cfg["seed"]:parser.error("resume seed must match")
        if args.return_penalty is not None and args.return_penalty!=trainer.cfg["isolated"]["return_penalty"]:parser.error("resume return penalty must match; use a fresh run for a control")
    else:
        env_cfg=load_config(args.env_config);cfg=yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
        if args.seed is not None:cfg["seed"]=args.seed;cfg["isolated"]["selection_seed"]=args.seed+2
        if args.return_penalty is not None:cfg["isolated"]["return_penalty"]=args.return_penalty
        if args.smoke:
            cfg.update(total_low_steps=12,eval_every=4,full_eval_every=8,save_every=4,plot_every=4,log_every_episodes=1,evaluation_seeds=[1001])
            cfg["isolated"].update(long_steps=4,short_steps=2)
            cfg["sac"].update(hidden=16,heads=2,layers=1,batch_size=2,replay_capacity=64,random_steps=0,learning_starts=0)
        trainer=LowerTrainer(env_cfg,cfg,device)
    cfg=trainer.cfg
    for key in ["eval_every","full_eval_every","save_every","plot_every","log_every_episodes"]:
        if type(cfg[key]) is not int or cfg[key]<=0:parser.error(f"{key} must be a positive integer")
    write_json(output/"resolved_config.json",dict(environment=trainer.env.env_cfg,training=cfg,device=device))
    provenance(output,trainer,args)
    evaluations=[]
    if (output/"eval.jsonl").exists():
        for line in (output/"eval.jsonl").read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if row["low_steps"]<=trainer.low_steps:evaluations.append(row)
    def evaluate(full):
        summaries=[]
        for seed in cfg["evaluation_seeds"]:
            path=output/f"eval_{trainer.low_steps:08d}"/f"seed_{seed}" if full else None
            summaries.append(trainer.evaluate(seed,path))
        with (output/"eval.jsonl").open("a",encoding="utf-8") as stream:
            for summary in summaries:stream.write(json.dumps(summary,allow_nan=False)+"\n")
        evaluations.extend(summaries);write_json(output/"latest_eval.json",summaries)
        print(f"  eval | SAC={trainer.sac_updates} | "+" | ".join(f"seed{r['seed']} cover={r['sensor_coverage_fraction']:.1%} outer={r['outer_sensor_visited']} return={r['long_return_success']}/{r['long_return_count']} R={r['lower_return']:.1f}" for r in summaries),flush=True)
    steps=args.steps or cfg["total_low_steps"]
    print(f"Lower-only SAC | {cfg['isolated']['long_steps']}/{cfg['isolated']['short_steps']} steps | device {device} | additional steps {steps} | return penalty {trainer.env.return_penalty}",flush=True)
    print(f"Output: {output.resolve()}",flush=True)
    completed=[];start=time.monotonic()
    try:
        with (output/"train.jsonl").open("a",encoding="utf-8") as log,gzip.open(output/"episodes.jsonl.gz","at",encoding="utf-8",compresslevel=5) as episodes:
            for index in range(steps):
                row=trainer.step();summary=row.pop("episode_summary")
                if row["low_steps"]%10==0 or row["done"] or index==steps-1:
                    line=json.dumps(row,allow_nan=False,separators=(',',':'));log.write(line+"\n");log.flush();write_json(output/"live_status.json",row)
                if summary is not None:
                    episodes.write(json.dumps(summary,allow_nan=False,separators=(',',':'))+"\n");episodes.flush();completed.append(summary)
                    if trainer.episodes%250==0:
                        folder=output/"episodes";folder.mkdir(exist_ok=True)
                        with gzip.open(folder/f"episode_{trainer.episodes:06d}.json.gz","wt",encoding="utf-8") as stream:json.dump(trainer.env.trace(),stream,allow_nan=False)
                    if trainer.episodes%cfg["log_every_episodes"]==0:
                        n=max(len(completed),1);r=sum(s["lower_return"] for s in completed)/n;coverage=sum(s["sensor_coverage_fraction"] for s in completed)/n
                        print(f"episode={trainer.episodes:06d} steps={trainer.low_steps:08d} SAC={trainer.sac_updates:07d} avg_R={r:.2f} cover={coverage:.1%} elapsed={(time.monotonic()-start)/60:.1f}m",flush=True);completed=[]
                if trainer.low_steps%cfg["save_every"]==0:trainer.save(output/"checkpoint.pt")
                if index<steps-1 and trainer.low_steps%cfg["eval_every"]==0:evaluate(trainer.low_steps%cfg["full_eval_every"]==0)
                if trainer.low_steps%cfg["plot_every"]==0:curves(trainer,output,evaluations)
    except KeyboardInterrupt:
        print("Interrupted; the last completed checkpoint remains available.",flush=True);curves(trainer,output,evaluations);return
    trainer.save(output/"checkpoint.pt");evaluate(True);curves(trainer,output,evaluations)
    print(f"Saved {output/'checkpoint.pt'}",flush=True)


if __name__=="__main__":main()
