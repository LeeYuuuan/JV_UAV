# Reference failure responsibility, v4

Default config: configs/training_joint_responsibility_v4.yaml.
This adds event-based failure-cost allocation to the21feature actor experiment.
All other v3 learning, energy, map, seeds, upper reward coefficients, Top-K20,
staged update and evaluation settings remain unchanged.

## Definition

At the start of each actual serving allocation, record position/SOC before any
lower movement. In one frame maximum movement is1800m under current settings.
The reference endpoint advances half of the achievable inward distance:
`d_ref = d_start - .5*min(d_start,1800)`.
A reference trajectory distributes that movement equally over the ten lower
steps, using the real movement/hover model. Its SOC requirement is service cost
plus actual return-model cost, physical return reserve, and an extra .01SOC
reference margin. No actual failing trajectory is used to construct this budget.

An optimistic physical lower bound uses minimum per-step service energy and
return cost from the closest geometrically reachable endpoint, plus physical
reserve. Falling below that bound proves infeasibility; exceeding it alone does
not prove that a feasible trajectory exists. Boundary comparisons have1e-7SOC
tolerance. Equality with reference requirement is treated as reference-feasible.

For a serving UAV's final-step return failure:
- below optimistic lower bound: upper1/lower0;
- reference-feasible: upper0/lower1;
- intermediate gray region: upper.5/lower.5.

Mid-step battery exhaustion retains upper1/lower0, as previously agreed.
Failure of an actually allocated return at settlement also uses upper1/lower0.
Those categories cannot simultaneously create another final-return event for
the same UAV. Continuing charging/waiting and rejected requests follow existing
physical allocation rules; the reference is evaluated for actual serving roles.

## Single assessment, separate costs

`failure_responsibility.py` creates one assessment per newly dead UAV, with
frame-start budget/position, reference budget/endpoint, physical failure cause,
actual endpoint/SOC/return requirement and upper/lower responsibility weights.
The two weights sum to1. Missing, duplicated or inconsistent events raise errors.

Upper explicit death cost = existing unit death cost times sum(upper weights).
The previous unconditional full death cost is replaced, not added. Legacy extra
upper lower-return/episode failure penalties must be zero; invalid combinations
are rejected. At current settings the unit upper death cost is10/10=1.

Lower final-return count and distance costs use the same lower weight per UAV:
`-500*sum(wL)` and `-500*sum(wL*distance/1800)`.
Each episode's existing future-backlog tail is calculated only once. Its lower
weight is the arithmetic mean of lower weights across the actual death events.
Thus upper-only failure adds no lower failure tail; lower-only failure adds one
full tail; simultaneous upper/lower failures split that one tail. This averaging
is an explicit modeling convention, not a uniquely determined causal quantity.
The upper policy does not inherit the lower policy's500-scale or backlog tail.

Normal backlog/collection reward and distinct boundary violations remain. Real
episode termination and lost serving/future returns also remain: allocating
explicit failure costs does not make the two learning problems independent.

Final/failing lower callbacks are deferred until settlement and allocation are
finished, so replay sees the final responsibility-adjusted reward. Only then is
the terminal transition inserted. The episode tail's raw cost/weight/weighted
cost are metadata in info/logs, not three reward terms that could be summed.

## Audit logs

Each affected frame in charging_frames.jsonl.gz, train.jsonl and the actual PPO
upper_credit.jsonl.gz includes failure_records. Per-episode diagnostics and full
evaluation policy traces include the records as well. Failure tail metadata
records its raw cost, lower fraction and final contribution separately.
The training provenance manifest records failure_responsibility settings.

## Verified scenarios

106 CPU regression checks passed, including12scenario test functions. With
one UAV starting at the airship and actually flying1800m outward in ten steps:

| Start SOC | Classification | Upper death cost | Lower return count cost | Lower distance cost | Lower tail fraction |
|---|---|---:|---:|---:|---:|
| .395 | upper | -1 | 0 | 0 | 0 |
| .420 | shared | -.5 | -250 | -250 | .5 |
| .500 | lower | 0 | -500 | -500 | 1 |

Additional cases check two simultaneous failures, mid-step exhaustion, settlement
failure, normal completion, last-episode-frame failure (zero remaining tail),
exhaustion on the tenth step, operation with tail disabled, reference refresh
on a new frame, exact threshold handling, reserve counted once, reference versus
executed energy, unchanged dynamics RNG, exact resume and reward/replay equality.
The full local scenario evidence is in failure_scenarios_verified.json.
These are behavior checks, not a convergence test.

To repeat the12physical scenario tests on the server without pytest:

```bash
python3 run_failure_scenarios.py
```

The generated report is outputs/failure_responsibility_checks/scenario_results.json.

## Run

Commit/push locally, then on server:

```bash
cd ~/Code/JV_UAV
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=1 python3 -u train_joint.py --train-config configs/training_joint_responsibility_v4.yaml --device cuda --output ./runs/joint_reference_responsibility_v4
```

Start a fresh run: older replay rewards do not use this allocation. Resuming
older checkpoints preserves their saved legacy reward configuration. No network
dimensions change from v3: actor21/critic38. No persistent training was started.
