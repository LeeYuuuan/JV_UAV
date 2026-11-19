# Independent synchronous waypoint experiment

Revision: `lower_waypoint_sync_8h_v1`, 2026-11-19. No MAPPO or charging scheduler.
The existing joint experiment and random20/10 experiment are unchanged.

## Definition

- Map and energy come from `configs/default.yaml`: currently 4000m square,
  airship (2000,2000), 150 sensors (100 inside r1800,50 outside), coverage173.2m.
- One frame is ONE synchronous sortie. Randomly choose2-4 of6 UAV IDs, without
  replacement. Each chosen UAV starts at the airship with SOC1. This deliberately
  supplies a fresh-battery sortie, not a simulated charging queue or free midair
  recharge. Returning aircraft land before the next sortie starts.
- Each aircraft chooses an absolute waypoint: `xy=(tanh(z)+1)*world_size/2`.
  There is no180m movement limit and no boundary penalty. Flight speed remains6m/s.
- Aircraft fly directly; no en-route collection. Everyone serves at the destination
  until `max(flight_sec)+30s`. Early arrivals hover and consume326W for the whole wait;
  horizontal motion consumes257W. Collection is a simultaneous endpoint operation.
- Packet arrivals continue during flight, waiting, service, and return. At service
  completion clear the footprint union once. An overlapping sensor is credited
  only once. A UAV depleted before synchronous completion cannot clear data.
- Automatically return using the existing vertical-then-horizontal return model
  (400W climb50m at5m/s, then257W horizontal). As in the existing model, no separate
  outbound vertical/descent cost is added. Duration of the return phase is the
  slowest successful return; earlier landings cost no hover energy.
- Retain queues and collection ages throughout28800s (8h); reset only at episode start.
  Decision count varies with actual sortie duration. Stop at exactly28800s, including
  a partial final flight/service/return if necessary. An unfinished synchronous
  service phase does not clear future data. A partial return pays only executed
  climb/horizontal energy; feasible but unfinished returns are not battery failures.
  Cold-start60s packets are present initially; episode clock starts at0.
- A failed aircraft is penalized once in that sortie; it does not terminate the
  episode. The next sortie receives fresh ground-departing aircraft. Failure at
  reduced battery/capacity is tested, including partial travel before depletion.

### What this first version can and cannot test

Full battery + one target + return makes every default-map waypoint feasible:
even the farthest center-to-corner round trip uses about57.8% SOC. Return penalties
are implemented but normally inactive. This experiment tests location selection,
team coverage, and sustained queue control given replacement batteries. It does
not test multi-target sortie energy allocation or realistic charging scheduling.

## Reward and physical time

Let D include outbound/service/wait and actual successful return, and Bmax(t) be
the system maximum backlog. The reward is:

```
- integral(Bmax(t), over sortie) / (300 packets * 60 sec)
+ 0.5 * Bmax_after_service / 300 * h_top20
- 500 * failed_UAV_count
- 500 * sum(failed_return_distance_m) / 1800
```

`h_top20` is computed after arrivals and before clearing: the backlog of the
pre-clear Top20 sensors actually cleared by the union, divided by their total.
Ties use stable sensor-ID order. Zero denominator gives0. No old bonus, constant
offset, distance cost, or boundary cost is stacked on top.

Arrivals are independent Poisson counts in bins of at most10s. The integral uses
trapezoidal integration of per-bin maximum backlog. This is a documented numerical
approximation, not an exact integration over individual packet event times.

SAC's per-transition bootstrap discount is `0.99 ** (D/60)`, multiplied by0 at the
8-hour terminal state. Entropy is the joint normalized-waypoint log density,
one sample per sortie; alpha is automatically tuned. Current settings: hidden128,
4 attention heads, lr3e-4, batch256, gamma0.99 per60s, tau0.005, initialalpha0.05.
One SAC update per inserted decision after5000 uniformly random whole-map sorties.
No sensor-target oracle is used for warmup.

The original50-decision prototype failed an objective sanity check: staying home
shortened the simulated episode and beat a good collection heuristic. It was not
installed for user training. The user selected a fixed8-hour physical horizon,
which removes that specific escape. Regression tests compare the no-collection
policy with a high-backlog greedy collection reference under identical horizons.
Rewards are integrated within each transition without intra-transition discount;
duration discounts apply between decision epochs. This is the explicit SMDP
reward definition, not an exact continuously discounted integral objective.

## Observation and networks

- Sensor tokens: `[x/4000, y/4000, backlog/3000]`; no clipping/RMS or future arrivals.
- Active UAV tokens: `[x/4000, y/4000, SOC, ID/5, remaining_time/28800, active_count/6]`.
- UAV queries cross-attend to sensor tokens, followed by UAV-to-UAV self-attention.
  Actor outputs Gaussian mean/log_sigma per UAV xy, then reparameterized tanh sampling.
- Twin centralized critics see the same inputs and the complete joint action.
- No temporal sequence encoder or pretrained parameter import. This architecture
  intentionally differs from the old last-visit-only lower network.

## Run on server

From the JV_UAV repository after committing locally and pulling on the server:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_waypoint/train.py --device cuda --steps 100000 --output ./runs/lower_waypoint_sync_8h_v1
```

100000 means waypoint decisions, not60s steps. Episode count depends on learned
sortie durations; each completed episode always simulates exactly8 hours.
The default seed remains43 and the map remains42. `--seed` overrides dynamics,
selection, neural initialization, replay, and random-action seeds, not map seed.

Resume, adding another100000 decisions:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_waypoint/train.py --device cuda --resume ./runs/lower_waypoint_sync_8h_v1/checkpoint.pt --steps 100000 --output ./runs/lower_waypoint_sync_8h_v1
```

Checkpoint contains optimizers, target critics, alpha, replay, partial episode and
all RNG streams. Evaluation uses separate seeded environments and no sampling
from the training RNG. If interruption left logs newer than the checkpoint, use a
new output directory to avoid duplicated/mixed timelines; history stays in checkpoint.

Short functional check (small network; NOT a convergence experiment):

```bash
python3 experiments/lower_waypoint/train.py --device cpu --smoke --output ./outputs/waypoint_smoke
```

## Outputs while running

- `resolved_config.json`, per-invocation provenance with exact source.zip and hashes.
- `train.jsonl`: every10 decisions plus episode ends, actual sampled actions,
  targets, active IDs, durations, individual SOC/service/return budgets, Top-K,
  collection, coverage, reward terms, separate actor/policy/entropy/critic losses.
- `episodes.jsonl.gz`: every completed episode, flushed immediately. Full training
  trajectories saved every100 episodes under `episodes/`.
- `eval.jsonl`: every1000 decisions, all3 seeds1001/1002/1003. Deterministic policy.
  EVERY evaluation saves `eval_XXXXXXXX/seed_N/trajectory.png` and summary.json.
  The trajectory image separates6 UAVs and includes ALL sorties and actual service
  footprints in the episode, with waypoint color showing decision number. Partial
  terminal travel is drawn only as executed. Return paths retrace outbound lines.
  Initial evaluation also includes uniform-map random, stay-home, and high-backlog
  greedy-reference baselines. These do not contribute training/replay samples.
- Full8-hour evaluation paths/footprints and Gaussian parameters every5000
  decisions, at startup and at final exit. Full evaluations log every action.
- `checkpoint.pt`: atomic overwrite every5000 decisions and at successful exit.
- `lower_training_curves.png`, `loss_curves.png`, `evaluation_curves.png`,
  `training_curves.json`: overwrite every1000 decisions; no accumulating plot stack.
- Console summary every20 episodes, plus evaluations every1000 decisions.

Package results while running with the existing utility (no weights in ZIP):

```bash
python3 pack_results.py ./runs/lower_waypoint_sync_8h_v1
```

Render the latest saved checkpoint without stopping the current training process:

```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/lower_waypoint/evaluate.py --device cuda --checkpoint ./runs/lower_waypoint_sync_8h_v1/checkpoint.pt --output ./runs/lower_waypoint_sync_8h_v1/eval_manual_latest
```

This writes per-seed trajectory.png, dashboard.png, summary.json and trace.json.
It evaluates the saved weights, not unsaved in-memory updates. Existing running
training processes must be restarted/resumed to use a changed automatic renderer.

## Comparison to the20/10 experiment

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_random_horizon/train.py --device cuda --steps 200000 --output ./runs/lower_random20_10_real_v1
```

This is a comparison of complete experimental designs, not a one-variable ablation:
waypoint uses full-map actions, coordinates/backlog input, fresh-battery sorties,
8-hour persistent-queue episodes and duration-integrated reward;20/10 uses small relative
motions, last-visit input, persistent in-sortie energy and reset after20 steps.
Do not directly compare raw episode reward, episode number, or decision count.
First inspect actual coverage, outer coverage, failed-return rate, backlog in physical
units, simulated duration and collection rate. Old20/10 automatic return settles
energy without advancing the sensor clock, another reason time metrics differ.

## Verification

Run the standalone physical/learning checks:

```bash
python3 tests/test_lower_waypoint.py
```

Checks include synchronous waiting, return energy/time, union credit, exact constant
backlog integral, persistent queues, finite horizon, service depletion, physical
return failure, input scales, terminal/nonterminal SMDP targets, parameter updates,
evaluation isolation, and exact resume before/after the episode boundary.
