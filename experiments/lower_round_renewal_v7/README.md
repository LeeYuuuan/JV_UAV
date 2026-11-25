# Lower full-map renewal-round experiment v7

Independent experiment/checkpoint format: `displacement_round_renewal_v7`.
Fresh training required; v1-v6 snapshots/replay are rejected. No long job is
started by installation/tests. V5 and its P8 run are preserved unchanged.

## Global task and anti-repeat rule

At fresh task start all150 sensors are pending. A real collection marks each
covered sensor visited in the current round. A sensor can advance a round only
ONCE. Repeated collection still clears data/age physically, but earns no new
progress. All150 must be updated to finish a round. Completion pays5 once and
starts a new pending list; the same collection is NOT credited into that new list.

Each serving UAV receives one pending goal, oldest age first, nearest within
ties, rotating UAV priority. Goals are unique while at least that many pending
sensors exist. At the round tail, if fewer targets remain than serving UAVs,
UAVs may cooperate on duplicate targets; collection/progress still counts once.
Completed goals cannot be substituted for difficult pending goals.

## Reward

```text
Phi(s) = 4 * visited_fraction_this_round
         + (1/6) * sum_serving clip(1 - coverage_gap / Dmax, 0, 1)
Dmax = sqrt(2)*4000 - 173.2
r = 5 * round_completed + gamma*Phi(next_replay_state) - Phi(current_state)
    - 0.1 * boundary_violations - 5 * responsible_failure_count
gamma = 0.99
```

Phi is bounded0..5. At completion, the task reward5 covers the maximum possible
potential drop, so completion's round+shaping subtotal is nonnegative (safety is
separate). No raw backlog reward, per-sensor hit bonus or repeated SOC-deficit
penalty. Actual backlog is still evolved, logged and used for quality evaluation.
Deficit metrics remain logged, but deficit weight is0. Real energy is enabled.
Failure is deduplicated once per deployment as in the preceding physical model.

Shaping uses the same state function before/after collection, pending-list reset,
dispatch and home reset. Repeated partial progress cannot create unbounded
discounted shaping return. A partial round is not a completed task reward.
Potential shaping preserves the discounted ROUND objective; that objective is
not mathematically identical to time-average maximum backlog. Both are evaluated.

## Continuous task:300 is a report boundary

Every10 physical steps randomly choose2-6 serving UAVs; deselected deployed UAVs
physically return and charge. Every20 steps check return feasibility and reset
UAVs full/home, preserving sensor age/backlog AND round mask.

Every300 steps write one report (episode counter in console). It does NOT reset
sensor age/backlog/mask/round clock. This avoids farming partial progress by
restarting near airship. The physical step counter continues across reports;
the replay done flag isFalse and the actual successor is bootstrapped. Fresh
evals begin from a new task. Resume restores the continuous state exactly.

Actor/critic retain the14-feature fleet tokens and assigned-target context.
The global age MLP input adds the150-bit round mask and visited fraction, so
both networks know which tasks remain. Actual backlog is NOT an input. This
input change makes v5/v6 weights incompatible. Other SAC settings stay the same:
seed43, batch256, LR3e-4, tau0.005, auto alpha starting0.05, update each step from
1024, unchanged epsilon0.8->0.1 over100k and random/greedy/actor behavior mixture.

## Server commands from JV_UAV root

After local commit/push and server pull:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_round_renewal_v7/train.py --device cuda --steps 900000 --output runs/round_renewal_v7_seed43
```

900k steps =3000 report blocks of300. Eval every6000 training steps, on three
seeds1001/1002/1003 for600 physical steps each, with initial/final evaluation.
Progress/loss logs every100 steps and report JSONL every300 are incrementally
flushed. Config/source hashes and compressed traces are saved.

Resume for an ADDITIONAL budget, with unchanged saved settings:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_round_renewal_v7/train.py --device cuda --resume runs/round_renewal_v7_seed43/checkpoint.pt --steps 300000 --output runs/round_renewal_v7_seed43
```

## Outputs and checks

Logs include completed-round count, completion durations, newly updated IDs,
progress before/after collection and next state, round mask, goal coordinates,
Phi before/after, reward components, failure/deficit statistics and backlog.
`policy_best.pt` minimizes time-average max backlog; `policy_best_rounds.pt`
maximizes completed rounds then minimizes max backlog. Latest/periodic actor
snapshots and resumable checkpoint remain available. Traces include600 steps;
seed1001 full trajectory panels cover1-300 and301-600, plus first80-step detail.
Training/loss/eval figures overwrite the same files. No reset-flight lines.

```bash
python3 experiments/lower_round_renewal_v7/verify.py
python3 scripts/pack_lower_review.py runs/round_renewal_v7_seed43
python3 experiments/lower_round_renewal_v7/evaluate.py --checkpoint runs/round_renewal_v7_seed43/policy_best_rounds.pt --steps 600 --device cuda --output runs/round_renewal_v7_best_rounds_eval
```

The test suite checks repeat farming, union counting, completion and next-round
reset, pending-goal masks,20/300 boundaries, discounted telescoping, physical
return/energy, SAC finite updates and exact resume. A successful greedy reference
is a feasibility baseline, not a guarantee that untrained SAC will converge.
