# Fixed-time, no-energy lower coverage experiment

Revision: `lower_coverage_fixed60_all_v1`, 2026-11-19.
Independent experiment: existing joint,20/10 physical and8h physical-waypoint code
and checkpoints are unchanged. This is a location-selection test, not a feasible
flight/charging schedule. No MAPPO, charging queue or energy model is executed.

## Environment

- Same configured map:4000m square,150 sensors,100 inside r1800 and50 outside,
  service radius173.2m. Mapseed42, trainingseed43.
- Every decision advances exactly60s. Active UAVs instantly relocate to selected
  absolute waypoints anywhere in the map. No speed limit, energy, travel time,
  return, or associated penalties. Packet arrivals are Poisson(rate*60s); then
  active footprint union is cleared once at the end of the slot.
- Every20 steps sample3or4 of6 UAV IDs as the long group. All6 serve for the first
  10 steps; only the long group serves for the next10. The short group stays at
  its position and does not collect while inactive.
- At the next group boundary, all6 reactivate at their last positions. No position
  reset or automatic return. Draw a new long/short split, not a new map or queue.
- Default200 steps per episode (12000s); sensor queues and ages reset only here.
  Initial positions reset to airship here. Cold-start60s packets are present before
  step1, with the episode clock starting at0.
- Group boundaries10/20 are nonterminal transitions. Only episode end has done=True.
  A partial final group is supported if episode_steps is not a multiple of20.

## Reward: all sensors, no Top-K gate

```
h_all = actual_collected_packets / sum(all_pre_collection_sensor_backlogs)
r = - Bmax_after / 300 + 0.5 * Bmax_after / 300 * h_all
```

Arrivals happen before computing h_all. Use the clear union, not the sum of
per-UAV overlapping footprints. Zero denominator means h_all=0. Every sensor
contributes to h_all, irrespective of backlog rank. No Top20 threshold remains.
If all queues are cleared, Bmax_after=0 and the whole reward is0, its best value;
there is no extra positive bonus in that case. Bonus size is still a fraction of
the backlog cost, not a direct packet-count reward or a guarantee of convergence.

No old integral, constant offset, movement, boundary or death costs are added.
Log both terms and collected-but-zero-bonus counts so scale/feedback can be audited.

## Input, network and learning

- Sensor tokens: `[x/4000,y/4000,backlog/3000]`; no CNN or future data.
- Active UAV tokens: `[x/4000,y/4000,ID/5,remaining_service/20,
  remaining_episode/200,active_count/6]`. SOC has been removed.
- Same attention SAC algorithm as the physical-waypoint experiment, copied locally
  to keep this experiment self-contained. UAV queries cross-attend to sensor
  tokens, followed by team self-attention. Gaussian mean/std -> tanh xy actions.
- Twin joint-action critics, hidden128, heads4, batch256, lr3e-4, gamma0.99 per60s,
  tau0.005, alpha initially0.05 with automatic tuning, gradient clip1.
- First5000 decisions use uniform whole-map waypoints, without a sensor oracle.
  Then1 SAC update per inserted transition. Replay capacity100000. Group changes
  are stored with their actual next active IDs, and bootstrap across them.
- Fresh training only; physical-waypoint checkpoints are intentionally rejected
  because both input semantics and environment/reward have changed.

## Run

PC PowerShell, from `D:\VScode\JV_UAV` with the RL environment activated:

```powershell
python -u experiments/lower_coverage_fixed/train.py --device cuda --steps 100000 --output ./runs/lower_coverage_fixed60_all_v1
```

Server, after committing locally and pulling in `~/Code/JV_UAV`:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_coverage_fixed/train.py --device cuda --steps 100000 --output ./runs/lower_coverage_fixed60_all_v1
```

100000 steps =500 complete200-step episodes, including5000 warmup steps.
Use `--device cpu` if CUDA is unavailable. `--seed` changes training/dynamics/group
RNG seeds, not map seed. Use a new output directory for a new seed/configuration.

Resume, adding100000 steps:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_coverage_fixed/train.py --device cuda --resume ./runs/lower_coverage_fixed60_all_v1/checkpoint.pt --steps 100000 --output ./runs/lower_coverage_fixed60_all_v1
```

Exact partial-episode state, replay, optimizers, target critics, alpha and RNG are
saved. If logs are newer than the checkpoint after interruption, resume into a
new output directory to avoid mixing/repeating logged steps. --steps is additional.

## Outputs and evaluation

- Every1000 steps (5 episodes): deterministic evaluation on seeds1001/1002/1003.
  EVERY eval saves per-seed trajectory.png and summary.json. Each UAV gets a panel;
  all targets and coverage circles are drawn. Dashed connecting lines denote
  instantaneous relocations, not physical flight paths.
- Every5000 steps and final exit: checkpoint, full evaluation trace, Gaussian
  parameters, dashboard with active-UAV timeline, queue/coverage/reward curves.
- At startup: policy, uniform-map random, no-collection center, and high-backlog
  greedy reference. References are evaluation only, never replay/warmup data.
- train.jsonl every10 steps plus episode ends; episodes.jsonl.gz every episode,
  flushed while running. Rewards, actions, groups, positions, actual packets,
  h_all, queue statistics, separate actor/policy/entropy/critic losses and alpha.
- training_curves.json and lower_training_curves.png/loss_curves.png/
  evaluation_curves.png overwrite every1000 steps. Console every5 episodes.
- Exact config and per-invocation source.zip/hashes in provenance.

Render a saved checkpoint while training continues in another process:

```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/lower_coverage_fixed/evaluate.py --device cuda --checkpoint ./runs/lower_coverage_fixed60_all_v1/checkpoint.pt --output ./runs/lower_coverage_fixed60_all_v1/eval_manual_latest
```

Package live results:

```bash
python3 pack_results.py ./runs/lower_coverage_fixed60_all_v1
```

## Verification and interpretation

```bash
python3 tests/test_lower_coverage_fixed.py
python3 experiments/lower_coverage_fixed/train.py --device cpu --smoke --output ./outputs/coverage_fixed_smoke
```

Tests cover group boundaries, inactive collection exclusion, retained positions,
queues, zero-energy immunity, all-sensor reward/overlap, actual feature scales,
fixed discounts, terminal targets, exact resume across10/20/episode boundaries,
evaluation isolation, and images at every evaluation.

Compare coverage (especially outside circle), repeated collection, mean/peak max
backlog and reward components with the two previous tasks. Different dynamics,
inputs and rewards mean raw returns are not a controlled single-variable ablation.
The no-energy environment is an intentionally permissive test. Success here must
later be revalidated after restoring motion and energy constraints.
