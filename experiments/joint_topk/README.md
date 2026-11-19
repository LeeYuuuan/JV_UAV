# Joint Top-K training, revision joint_topk_g095_staged_v1

This run learns both policies with actual JV energy and charging-slot allocation.
It does not replay the fixed Paper1 schedule. The default starts both policies
from scratch; matching Paper1 actor-only transfer remains optional via
`--upper-pretrained PATH`. No trained weights are present in the result ZIPs.

## Server

After committing and pushing locally, pull the JV_UAV repository, not P1_J:

```bash
cd ~/Code/JV_UAV
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=1 python3 -u train_joint.py --train-config configs/training_joint_topk.yaml --device cuda --output ./runs/joint_topk_g095_s43_v1
```

Default budget: 409600 upper frames (400 MAPPO updates, episodes at most 100
frames). Short episodes reset while the 1024-frame rollout keeps accumulating.
The training seed remains 43; map seed 42 and dynamics seed 43 are unchanged.
The current default map is 150 sensors (100 inside r=1800, 50 outside), 4000 m
square, airship at (2000,2000), UAV radius 173.2 m. This change does not alter it.

## Parameters and stages

Upper Paper1 architecture/PPO: gamma .95, GAE lambda .95, constant entropy .001,
rollout 1024, epochs 4, minibatch 128, actor/critic LR 1e-4. Reward remains the
existing Paper1-compatible serving/backlog/death reward, C/W weights zero,
death 10, divisor 10; lower return deaths also count for the upper reward.
Upper inputs remain arrival SOC (current minus actual return cost), identity,
status and C/W occupancy. No new privileged inputs are introduced.

Lower SAC parameters/observations remain unchanged (including gamma .99).
Its data reward is replaced, rather than augmented:
`(-B_after + .5*B_after*h_top5)/300`.
`h_top5` uses actual collection from the five highest pre-service sensor queues,
with deterministic sensor-index tie breaking and no overlap double counting.
Offset is zero. Existing OOB and final return count/distance costs remain.
Failure adds a discounted remaining-horizon backlog cost to the last lower
transition: no future collection, mean arrivals, no extra physical steps/RNG.
This is an explicit absorbing-failure surrogate, not measured future backlog;
it applies to any terminal failure to prevent benefiting from an early stop.
The upper reward does not receive that lower tail cost.

- Dense stage: 1 SAC update per inserted lower transition after 10000-step warmup.
  MAPPO updates every 1024 upper frames throughout.
- After 250000 actual SAC optimizer updates (the previously discussed 500*100*5),
  replay still records every step; SAC updates once per MAPPO update.
- The threshold is configurable under `update_schedule`; it is an experiment
  setting, not a claim that the lower policy has converged. Neither policy freezes.
- Old checkpoints retain their original configuration; a fresh output directory
  is necessary to use this new reward and stage schedule.

## Evaluation and logs

Every 200 upper frames: deterministic evaluation on seed 1001, up to 100 frames
or actual failure. Save scalar results, no full trace duplication.
Every 2000: all three seeds, full trajectories and dashboard/coverage heatmaps.
Final evaluation also stores complete trajectories. Training RNG and RMS are
unchanged by evaluation. Every 1000 frames and each upper update: overwrite
`checkpoint.pt`, including replay, pending transition, rollout, optimizers/RNG.

- `resolved_config.json`, `provenance/*`: revision, source snapshot/hash, effective
  lower override, exact arguments and seed settings.
- `train.jsonl`: every 10 frames, episode end and upper update; separate PPO
  policy/value/entropy/KL/clip logs and SAC Q/target/actor/critic/alpha terms.
  KL is logged only; it does not stop or modify updates.
- `charging_frames.jsonl.gz`: each frame, request probabilities, actual allocation,
  SOC start/end, request/rejection/charge counts and lower reward components.
- `episodes.jsonl.gz`: append episode summaries, returns and component sums/means,
  survival, coverage, collection and exploration diagnostics.
- `eval.jsonl`, `latest_eval.json`: ongoing evaluations including lower return,
  failure cause, coverage, sensor visits, component sums and charging information.
- Overwritten images: `training_curves.png`, `lower_training_curves.png`,
  `upper_request_probability.png`, `evaluation_curves.png`, `loss_curves.png`.

Interrupting retains completed flushed logs and the last safe checkpoint.
Resume with the same output path; `--upper-steps` is the number of *additional*
frames for this invocation:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u train_joint.py --resume ./runs/joint_topk_g095_s43_v1/checkpoint.pt --device cuda --upper-steps 100000 --output ./runs/joint_topk_g095_s43_v1
```

## Result audit

`batch_20261119_audit.json` records all eleven supplied ZIPs by their internal
configuration, not filenames. All contain 4000 completed updates. The misleading
old `fixed_l070...zip` filename actually contains seed45, entropy .01, gamma .95,
lambda .95. Seed45 entropy .005 duplicates an already complete run. Seed47 .005
ends at serving total 12, so small gamma does not guarantee every seed succeeds.
Entropy .001 was chosen as a conservative constant based on the available runs.
Standalone upper convergence is not a guarantee of joint convergence with new
energy dynamics and a changing lower policy.
