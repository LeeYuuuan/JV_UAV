# Independent lower age-distance experiment v6

V6 replaces BOTH v5 global backlog cost and potential shaping with a DIRECT
age-weighted remaining-distance cost. V1-v5 code and current runs are unchanged.
This is a new navigation surrogate objective, not a policy-invariant transformation
of maximum-backlog minimization. Actual mean/worst backlog remain reported and
best checkpoints are still selected by mean time-average maximum backlog.

## Exactly one goal per serving UAV

Every lower step: assign each serving UAV ONE unique sensor. Prefer oldest age;
within ties choose nearest; processing order rotates each10-step block. Non-serving
UAVs reserve no goals. There are2-6 serving UAVs, hence2-6 actual goals; width6
is only input padding. Goals can change at the next step; they are not locked for
the entire20-step service cycle. Inputs/network/physics/exploration/SAC are unchanged.

## Reward

```text
estimated_B_u = 0.5 * age_of_assigned_target_before_action_seconds
gap_u = max(distance(actual_post_move_position, pre_action_assigned_target)-173.2, 0)
navigation = -mean_serving( (estimated_B_u/2000) * (gap_u/1800) )
r = navigation - 0.1*boundary_count
    - 70*sum(return_SOC_shortfall)/0.1
    - sum_responsible_failures(7.5 + 3.75*distance_to_home/1800)
```

Navigation weight defaults1 and has no cap. Target and age are frozen for ONE
action; distance is evaluated after real movement and before any20-step home
reset or10-step dispatch. Goal rebinding after collection affects the NEXT step.
Within coverage, gap=0. There is no separate -B, potential difference, collection
bonus or target-hit bonus. Actual backlog does not enter the navigation term or
actor/critic inputs. Sensor arrivals/backlog/age and physical collection still run.

No serving UAV means zero navigation decision cost; deaths do not terminate early.
The inherited deficit cost persists for deployed stopped UAVs until cycle reset.
Direct navigation costs can still trade off against safety; inspect both separately.
Because completion reassigns a possibly distant goal, this direct cost can favor
waiting just outside a target footprint instead of clearing it and taking on the
next goal. It is an experimental surrogate, not a guarantee of sustained collection.
Inspect actual collection/revisit/backlog curves, not navigation return alone.

## Server (JV_UAV root, after commit/push/pull)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_age_distance_v6/train.py --device cuda --steps 900000 --output runs/age_distance_v6_seed43
```

Fresh training is required; v5 replay/checkpoints are rejected. Seed43, gamma0.95,
300-step episodes, random2-6 serving every10 steps, full/home every20 steps,
unchanged real energy, batch256, LR3e-4, one SAC update/step from1024;900k steps
is3000 episodes. Epsilon0.8->0.1 over100k, same random/greedy/actor mixture.

Continue for an additional budget:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_age_distance_v6/train.py --device cuda --resume runs/age_distance_v6_seed43/checkpoint.pt --steps 300000 --output runs/age_distance_v6_seed43
```

Explicit changed reward options on resume are rejected.300-step truncation still
bootstraps the actual successor.20-step reset does not reset sensor backlog/age.
Progress/loss/episode logs are incremental. Reward components, target IDs/coordinates,
estimated backlog, pre/post distances, gaps and per-UAV costs are logged. Eval every
3000 steps, plus initial/final; training/loss/eval curves overwrite existing files.
Seed1001 includes full1-100/101-200/201-300 trajectory/backlog panels, in addition
to existing1-80 trajectory views. No forced-reset flight lines are drawn.

```bash
python3 experiments/lower_age_distance_v6/verify.py
python3 experiments/lower_age_distance_v6/pack_results.py runs/age_distance_v6_seed43
```
