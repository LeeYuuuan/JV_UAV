"""Run failure-allocation scenarios using actual JV physics, without pytest."""
from pathlib import Path
import argparse
import inspect
import json
import sys
import traceback
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"tests"),str(ROOT/"src")]
import test_failure_responsibility as scenarios


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/failure_responsibility_checks")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    results=[]
    for name,function in vars(scenarios).items():
        if name.startswith("test_") and inspect.isfunction(function):
            try:
                function();results.append(dict(test=name,passed=True));print("PASS",name)
            except Exception:
                error=traceback.format_exc();results.append(dict(test=name,passed=False,error=error));print(error)
    table=[]
    for battery in [.395,.42,.5]:
        _,(_,_,_,_,info),_=scenarios.run_case([battery])
        record=info["failure_records"][0];terms=info["low_frame"].steps[-1].reward_terms
        table.append(dict(start_soc=battery,assessment=record,upper_dead_reward=info["reward_terms"]["dead"],
                          lower_final_reward_terms=terms,failure_tail=info["failure_tail"]))
    report=dict(scenarios=results,reward_table=table,passed=sum(r["passed"] for r in results),total=len(results))
    path=args.output/"scenario_results.json";path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(f"{report['passed']}/{report['total']} passed; results: {path}")
    if report["passed"]!=report["total"]:raise SystemExit(1)


if __name__=="__main__":
    main()
