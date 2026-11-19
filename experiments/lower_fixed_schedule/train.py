"""Train SAC against an exact, frozen 100-frame role schedule."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("JV_UAV_ROOT", HERE.parents[1])).resolve()
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import torch
import yaml

from jv_uav.config import load_config
from jv_uav.factory import build_smoke_test_runner
from jv_uav.rl.trainer import JointTrainer
from jv_uav.rl.sac import SAC
from jv_uav.rl.replay import Replay
from jv_uav.rl.observations import RunningMeanStd
from jv_uav.rl.exploration import WaypointExploration
from jv_uav.rl.diagnostics import EpisodeDiagnostics
from fixed_env import FixedScheduler, TopKReward, FailureTailEnv


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class LowerTrainer(JointTrainer):
    """Reuse joint replay transitions and SAC updates without creating MAPPO."""

    def __init__(self, env_cfg, cfg, schedule, reward_cfg, device):
        self.env_cfg, self.cfg = copy.deepcopy(env_cfg), copy.deepcopy(cfg)
        self.schedule, self.reward_cfg = copy.deepcopy(schedule), copy.deepcopy(reward_cfg)
        self.device = torch.device(device)
        torch.manual_seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.env = self.make_env(self.env_cfg)
        self.env.reset()
        self.exploration = WaypointExploration(
            self.env.scene, self.rng, cfg["sac"].get("warmup_random_fraction", .2),
            energy_guard=cfg["sac"].get("warmup_energy_guard", False),
            reserve_frac=cfg["sac"].get("warmup_return_reserve_frac", .02),
            energy_model=self.env.energy_model,
        )
        self.rms = RunningMeanStd(**dict(
            clip=cfg["normalization"]["clip"], min_std=cfg["normalization"]["min_std_sec"]))
        self.sac = SAC(env_cfg, cfg["sac"], device)
        self.replay = Replay(cfg["sac"]["replay_capacity"], cfg["seed"] + 1)
        self.env.low_runner.low_policy = self._low_policy
        self.env.low_runner.on_step = self._low_step
        self.low_steps = self.upper_steps = self.sac_updates = self.episodes = 0
        self.pending = None
        self.last_losses = {}
        self.episode_over = False
        self.diagnostics = EpisodeDiagnostics(cfg["sac"]["gamma"])

    def make_env(self, env_cfg):
        env = build_smoke_test_runner(env_cfg).upper_env
        env.scheduler = FixedScheduler(env_cfg, env.energy_model, self.schedule["assigned_status"])
        if self.reward_cfg["mode"] == "topk":
            values = {k: v for k, v in self.reward_cfg.items() if k not in ("mode", "terminal_tail")}
            env.low_runner.low_env.reward_model = TopKReward(env_cfg, **values)
            if self.reward_cfg.get("terminal_tail", False):
                env = FailureTailEnv(env, self.cfg["sac"]["gamma"],
                                     self.reward_cfg["scale"], self.reward_cfg["offset"])
        return env

    def step(self):
        if self.episode_over:
            self.env.reset(reset_rng=False)
            self.exploration.reset()
            self.diagnostics = EpisodeDiagnostics(self.cfg["sac"]["gamma"])
            self.episode_over = False
        _, _, terminated, truncated, info = self.env.step(
            np.zeros(self.env.scene.num_uavs, dtype=np.int8))
        self.upper_steps += 1
        done = terminated or truncated
        if done:
            self._finish_pending(self.env.scene.observe_lower(), True)
            self.episodes += 1
            self.episode_over = True
        summary = self.diagnostics.summary(self.env.scene.low_step_total)
        backlog = self.env.trace.backlog_max_post_timeline[1:]
        return dict(
            upper_steps=self.upper_steps, low_steps=self.low_steps,
            sac_updates=self.sac_updates, episode=self.episodes if done else self.episodes + 1,
            frame=len(self.env.trace.frames), terminated=terminated, truncated=truncated,
            termination_reason=self.env.trace.frames[-1].termination_reason,
            episode_return=summary["lower_return"], mean_max_backlog=float(np.mean(backlog)),
            peak_max_backlog=float(np.max(backlog)),
            sensor_coverage_fraction=float(np.mean(self.env.trace.sensor_visit_count > 0)),
            assigned_status=info["frame_plan"].assigned_status.tolist(),
            battery=self.env.scene.uav_battery.tolist(), lower=summary, **self.last_losses,
        )

    def evaluate(self, output, seed=1001):
        cfg = copy.deepcopy(self.env_cfg)
        cfg["seeds"]["dynamics_seed"] = seed
        env = self.make_env(cfg)
        env.reset()
        rms = copy.deepcopy(self.rms)
        rms.training = False
        env.low_runner.low_policy = lambda obs: self.sac.act(obs, rms, deterministic=True)
        terms = {}
        rewards = []
        def record(obs, action, result):
            rewards.append(result.reward)
            for key, value in result.reward_terms.items():
                terms[key] = terms.get(key, 0.0) + value
        env.low_runner.on_step = record
        roles = []
        while True:
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.scene.num_uavs, dtype=np.int8))
            roles.append(info["frame_plan"].assigned_status.tolist())
            if terminated or truncated:
                break
        output.mkdir(parents=True, exist_ok=True)
        summary = dict(
            upper_steps=self.upper_steps, frames=len(roles), low_steps=len(rewards),
            lower_return=float(sum(rewards)), lower_reward_terms_sum=terms,
            mean_max_backlog=float(np.mean(env.trace.backlog_max_post_timeline[1:])),
            peak_max_backlog=float(np.max(env.trace.backlog_max_post_timeline[1:])),
            sensor_coverage_fraction=float(np.mean(env.trace.sensor_visit_count > 0)),
            termination_reason=env.trace.frames[-1].termination_reason,
            completed_horizon=bool(truncated), assigned_status=roles,
        )
        write_json(output / "summary.json", summary)
        write_json(output / "trace.json", env.trace.to_serializable())
        env.render(show=False, save_path=output / "dashboard.png", trail_frames=len(roles))
        env.render(show=False, view="heatmaps", save_path=output / "heatmaps.png")
        return summary

    def save(self, path):
        scene = self.env.scene
        arrays = ("sensor_pos", "sensor_backlog", "sensor_last_visit_sec",
                  "uav_pos", "uav_battery", "uav_status")
        clocks = ("now_sec", "upper_step", "low_step_total", "step_in_frame")
        payload = dict(
            version="lower_fixed_schedule_v1", env_cfg=self.env_cfg, cfg=self.cfg,
            schedule=self.schedule, reward_cfg=self.reward_cfg,
            sac=self.sac.state_dict(), rms=self.rms.state_dict(), replay=self.replay.state_dict(),
            exploration=self.exploration.state_dict(), diagnostics=self.diagnostics.state_dict(),
            pending=self.pending, last_losses=self.last_losses,
            counters={k: getattr(self, k) for k in
                      ("upper_steps", "low_steps", "sac_updates", "episodes", "episode_over")},
            scene={k: getattr(scene, k).copy() for k in arrays} |
                  {k: getattr(scene, k) for k in clocks},
            trace=self.env.trace, rng=self.rng.bit_generator.state,
            scene_rng=scene.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        )
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def restore(cls, path, device):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["version"] != "lower_fixed_schedule_v1":
            raise ValueError("Unsupported lower-only checkpoint.")
        trainer = cls(state["env_cfg"], state["cfg"], state["schedule"], state["reward_cfg"], device)
        trainer.sac.load_state_dict(state["sac"])
        trainer.rms.load_state_dict(state["rms"])
        trainer.replay.load_state_dict(state["replay"])
        trainer.exploration.load_state_dict(state["exploration"])
        trainer.diagnostics.load_state_dict(state["diagnostics"])
        trainer.pending, trainer.last_losses = state["pending"], state["last_losses"]
        for k,v in state["counters"].items():
            setattr(trainer, k, v)
        for k,v in state["scene"].items():
            if isinstance(v, np.ndarray):
                getattr(trainer.env.scene, k)[:] = v
            else:
                setattr(trainer.env.scene, k, v)
        trainer.env.metrics.trace = state["trace"]
        trainer.rng.bit_generator.state = state["rng"]
        trainer.env.scene.rng.bit_generator.state = state["scene_rng"]
        torch.set_rng_state(state["torch_rng"])
        if trainer.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return trainer


def plot(output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    if not rows:
        return
    fig = Figure(figsize=(10,7))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2,2)
    for ax, key, label in zip(axes.flat,
            ("episode_return","mean_max_backlog","sensor_coverage_fraction","frame"),
            ("Lower episode return","Mean maximum backlog","Sensor coverage fraction","Episode upper frames")):
        ax.plot([r["episode"] for r in rows], [r[key] for r in rows], alpha=.7)
        ax.set(xlabel="Completed episode", title=label)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=140)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100000, help="Cumulative upper-frame budget")
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reward", choices=("topk","legacy"), default="topk")
    parser.add_argument("--no-terminal-tail", action="store_true",
                        help="Disable virtual remaining-horizon costs for a fresh top-K run")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rho", type=float, default=.5)
    parser.add_argument("--backlog-scale", type=float, default=300.)
    parser.add_argument("--offset", type=float, default=0.)
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluate every N upper frames (10 lower steps per complete frame)")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.frames,args.eval_every,args.save_every) <= 0:
        parser.error("Frame budgets and intervals must be positive.")
    torch.set_num_threads(1)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        if output != args.resume.resolve().parent:
            parser.error("Resume into the original run directory.")
        trainer=LowerTrainer.restore(args.resume,args.device)
    else:
        if (output/"train.jsonl").exists():
            parser.error("Use a new output directory or --resume.")
        env_cfg=load_config(PROJECT/"configs/default.yaml")
        env_cfg["reward"]["upper_reward_mode"]="serving_backlog"
        cfg=yaml.safe_load((PROJECT/"configs/training_transfer_n6.yaml").read_text(encoding="utf-8"))
        cfg["seed"]=args.seed
        schedule=json.loads((HERE/"schedule.json").read_text(encoding="utf-8"))
        reward=dict(mode=args.reward,k=args.top_k,rho=args.rho,scale=args.backlog_scale,
                    offset=args.offset,terminal_tail=not args.no_terminal_tail and args.reward=="topk")
        if args.smoke:
            cfg["sac"].update(hidden=32,heads=2,layers=1,batch_size=8,replay_capacity=128,
                              random_steps=0,learning_starts=0)
        trainer=LowerTrainer(env_cfg,cfg,schedule,reward,args.device)
    if args.frames<=trainer.upper_steps:
        parser.error("--frames must exceed completed frames.")
    hashes={str(f.relative_to(PROJECT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((PROJECT/"src").rglob("*.py"))}
    hashes.update({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in (HERE/"train.py",HERE/"fixed_env.py",HERE/"schedule.json")})
    write_json(output/"resolved_config.json",dict(version="lower_fixed_schedule_terminal_tail_v2",
        environment=trainer.env_cfg,training=trainer.cfg,schedule=trainer.schedule,
        lower_reward=trainer.reward_cfg,device=args.device,source_sha256=hashes))
    def evaluate():
        summary=trainer.evaluate(output/f"eval_{trainer.upper_steps:08d}")
        with (output/"eval.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(summary)+"\n")
        print(f"eval frame={trainer.upper_steps} | length={summary['frames']}/100 | "
              f"coverage={summary['sensor_coverage_fraction']:.1%} | "
              f"max_backlog={summary['mean_max_backlog']:.1f}",flush=True)
    print(f"Lower-only SAC | fixed 350-serving schedule | reward={trainer.reward_cfg}",flush=True)
    with (output/"train.jsonl").open("a",encoding="utf-8") as log, (output/"episodes.jsonl").open("a",encoding="utf-8") as episodes:
        try:
            while trainer.upper_steps<args.frames:
                row=trainer.step()
                if row["upper_steps"]%10==0 or row["terminated"] or row["truncated"] or trainer.upper_steps==args.frames:
                    log.write(json.dumps(row,allow_nan=False)+"\n")
                    log.flush()
                    write_json(output/"live_status.json",row)
                if row["terminated"] or row["truncated"]:
                    episodes.write(json.dumps(row,allow_nan=False)+"\n")
                    episodes.flush()
                if row["upper_steps"]%50==0:
                    print(f"frame={row['upper_steps']} | episode={row['episode']} | "
                          f"SAC={row['sac_updates']} | lower_R={row['episode_return']:.2f} | "
                          f"max_backlog={row['mean_max_backlog']:.1f}",flush=True)
                    plot(output)
                if trainer.upper_steps%args.save_every==0:
                    trainer.save(output/"checkpoint.pt")
                if trainer.upper_steps%args.eval_every==0:
                    evaluate()
        except KeyboardInterrupt:
            print("Interrupted. Use the last checkpoint to resume.",flush=True)
            plot(output)
            return
    trainer.save(output/"checkpoint.pt")
    plot(output)
    if trainer.upper_steps%args.eval_every:
        evaluate()


if __name__=="__main__":
    main()

