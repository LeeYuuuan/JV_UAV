# Lower service energy v4

Independent directory/checkpoint format: `displacement_service_energy_v4_aux05`.
V1/v2/v3 code/results are untouched. Changed rewards require fresh training/replay;
v1/v2/v3 checkpoints/replay are rejected.

## Server

Commit/push/pull, then run from **JV_UAV**:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_service_energy_v4/train.py --device cuda --auxiliary-max 0.5 --steps 900000 --output runs/service_energy_v4_aux05_seed43
```

900k physical steps =3000 episodes of300 steps, three times the preceding run.
Seed43, gamma0.95, batch256, LR3e-4, tau0.005, automatic alpha starting0.05,
one SAC update/step from step1024, epsilon0.8->0.1 over100k steps are preserved.
Relative to v2, the network, observations, assignments, physical model and
exploration remain unchanged. V4 calibrates cumulative linear shortfall cost to
roughly80% of the original v2 failure cost and retains75% of that original
one-time failure cost (a25% reduction, not a reduction to25%).

Resume an existing **v4** run for an additional budget:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_service_energy_v4/train.py --device cuda --auxiliary-max 0.5 --resume runs/service_energy_v4_aux05_seed43/checkpoint.pt --steps 300000 --output runs/service_energy_v4_aux05_seed43
```

Resume restores optimizer, target, replay, alpha, physical state, partial episode,
RNG and best-eval tracking. A changed auxiliary coefficient is rejected; energy
cost parameters are restored from the checkpoint. Ctrl-C saves
an atomic checkpoint. No long training job was started when creating this code.

## Per-step assignment

Only serving UAVs participate. Their processing order rotates every10-step
dispatch block. For each UAV, select the oldest still-unassigned age group,
then choose its nearest sensor; remove it before assigning the next UAV.
Age outranks distance. Recompute every step; observation calls are pure and
consume no RNG. Goals are references, not mandatory actions or permanent routes.
Distinct assigned sensors may have overlapping footprints; no region exclusion.

Each UAV token explicitly includes its own target relative x/y, age and validity,
alongside ID, position, SOC and one-hot status. Global context retains every
sensor's age, the assigned sensor coordinates/ages (at most6, padded/masked), and
20-step phase. Non-serving UAVs have no goal and reserve no sensor. No true
backlog or coordinates of non-target sensors enter actor/critic inputs.

Shared128-wide GELU/4-head attention and Gaussian/tanh displacement head; joint
twin Q. Null context tokens handle zero serving/target sets. Service movement
is radially limited to180m and then clipped to map bounds.

## Tie-aware reward

Freeze the6th-largest pre-action age as threshold tau. All sensors with age>=tau
are eligible, including ties; input assignments and reward eligibility differ.

```text
5 * covered_eligible_union_count
+ 0.5 * covered_backlog_P4 / (covered_backlog_P4 + 300)
- 0.1 * boundary_violation_count
- 70 * sum(max(required_return_SOC - current_SOC, 0)) / 0.1
- responsible_return_cost
```

Count overlapping collections once. Initially all ages are60s, so every sensor
is eligible and main reward can exceed30; no artificial cap is added. P4 uses
queues after arrivals/before collection. No negative-backlog term. Return cost
is now7.5+3.75*distance/1800 per responsible UAV without duplicate deployment costs.
CLI default auxiliary ceiling0.5 overrides stale fresh-run YAML.

Reward is settled after movement, energy consumption and collection, before
the20-step UAV reset. Covered P4 uses the saved after-arrivals/before-clear queue
snapshot; this does not mean reward is computed before the action. Age eligibility
is frozen before the action and intersected with the resulting collection union.

### Linear energy cost

The new cost is summed across deployed UAVs each physical step. A1 percentage
point SOC shortfall costs7;2 points costs14. Safe UAVs cost0. There is no squared
term or cap. The scale0.1 means10 percentage points, not0.1%.
Weight70 is calibrated at the **cumulative risky-deployment level**, not by
matching each individual step to a complete original failure charge.

Compute after physical evolution and BEFORE the20-step full/home reset. For a
return already in progress, include only remaining climb and horizontal flight.
Landing clears the deployment cost. An exhausted/stopped deployed UAV keeps
incurring its shortfall cost until cycle reset; setting SOC to0 cannot switch
off the cost. Docked charging/waiting/idle UAVs have no return-shortfall cost.
The existing reference-based responsibility rules still govern one-time service
depletion penalties; the continuous term is a team trajectory cost.

Calibration uses18 recorded late-v1 eval trajectories with59 failed checks and
the same original10+5*distance/1800 formula as v2. Mean original failure cost is
52.98 per300-step eval; reweighting those trajectories gives about42.12 linear
cost plus39.74 retained failure cost, total81.86 (about1.55x original).
Per failed deployment, the corresponding averages are approximately16.16 old,
12.85 linear and12.12 retained. These are offline scale estimates on unchanged
old trajectories, not a guarantee of ratios for newly learned v4 behavior.
Do not dynamically enforce a reward ratio: as shortfall vanishes, its cost is0.
Full calibration metadata is in config.yaml and every resolved run configuration.

Exploration preserves the previous per-UAV/per-step mixture: within epsilon,
25% random displacement,50% direction toward its assigned target,25% sampled
actor; outside epsilon sampled actor. No behavior cloning or frequency changes.

## Physical task and bootstrap

Pinned map:4000m square,150 sensors, map seed42, airship(2000,2000), radius173.2m,
arrival0.5packet/s/sensor.60s/step, at most30s service movement at6m/s;125Wh,
257W horizontal,326W hover,400W climb,500W charging,50m return climb.

Random2-6 service UAVs initially/every10 steps; availability may reduce actual
count. Deselected servers return physically, without collection. Arrival charges
only during remaining step time. Unlimited charging by default; an integer
charging_slots enables FIFO. Docked UAVs may leave before full charge if selected.

Every20 steps check return feasibility, settle penalties, reset UAVs only to
full/home, then dispatch. Backlog/age continue. Every300 steps reset everything,
but bootstrap from the final pre-reset next state. Both cycle changes and
artificial sampling truncation preserve value continuation. Stopped UAVs wait
for cycle reset. Same-step inward reference feasibility distinguishes avoidable
service depletion; unavoidable depletion remains schedule-caused.

## Figures, logs and model preservation

- `trajectory_01_80.png`: **one2x2 figure**, all UAVs together in each panel:
  steps1-20,21-40,41-60,61-80. All endpoint IDs retained with leader lines.
  Endpoints precede cycle reset; resets are never drawn as flight. Sensor colors
  use each panel's final backlog with a common scale.
- `evaluation_metrics.png`: full-eval backlog and cumulative reward components
  only; no SOC/role panels. Raw logs retain energy/status details.
- `training_curves.png` and `losses.png`: preserved training/reward/backlog and
  separate actor/critic/policy/entropy/Q/alpha/sigma views.
- `evaluation_curves.png`: eval return and average/worst maximum backlog versus
  training steps, for each seed. Retained from v1.

Eval every3000 steps, seeds1001/1002/1003, full300-step rollouts. Only trajectory
display is limited to80. Training plots overwrite after every episode. Flushed
loss/progress logs every100 steps, episode summaries every300, detailed compressed
training trace every10 episodes. Eval traces contain every step plus first80
distribution probes. Assignment IDs/order, eligible threshold/count, hits and
failures are logged. Resolved config and hashed source provenance identify runs.

`checkpoint.pt` is full/resumable. `policy_latest.pt` and `policy_best.pt` are
small evaluation-only actor snapshots. Best minimizes3-seed average time-average
maximum backlog; `best_eval.json` records the step. Additional actor snapshots
every30k steps preserve learning stages. Actor snapshots cannot resume training.

## Evaluate, compare, package

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_service_energy_v4/evaluate.py --checkpoint runs/service_energy_v4_aux05_seed43/policy_best.pt --device cuda --steps 300 --output runs/service_energy_v4_aux05_seed43/eval_best
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_service_energy_v4/compare_policies.py --checkpoint runs/service_energy_v4_aux05_seed43/policy_best.pt --device cuda --output runs/service_energy_v4_aux05_seed43/compare_best
python3 experiments/lower_service_energy_v4/pack_results.py runs/service_energy_v4_aux05_seed43 --recent-evals 6
```

Standalone eval defaults150 steps, changeable with --steps; trajectory remains
one four-panel figure. Compare reads one frozen model and tests actor mean,
sampled actor and assigned-target greedy. Availability can change random
dispatch/arrivals, so these are reference comparisons, not strict counterfactuals.
Pack comparison outputs by passing that directory directly. Result ZIPs omit
large checkpoints and policy weights; keep them on server.

```bash
python3 experiments/lower_service_energy_v4/verify.py
```

Checks cover age/distance priority, rotated order, rebinding, expanded/frozen
reward ties, own-target features, padding/empty sets, physical return/charging,
cycle/truncation boundaries and exact CPU resume.
