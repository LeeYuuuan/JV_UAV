# Independent lower backlog-cost / potential experiment v5

Fresh training/replay is required. Checkpoint format: `displacement_backlog_potential_v5`.
V1-v4 weights/replay are not resumed here. Only reward, reward diagnostics and plots
change from v4. Actor/critic code, inputs, physics, assignment, dispatch, exploration
and SAC settings are preserved.

## Server (repository: JV_UAV)

After local commit/push and server pull:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_backlog_potential_v5/train.py --device cuda --cost-mode max --steps 900000 --output runs/backlog_potential_v5_max_seed43
```

Independent smooth-cost control, same seed and other settings:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_backlog_potential_v5/train.py --device cuda --cost-mode pnorm --cost-p 8 --steps 900000 --output runs/backlog_potential_v5_p8_seed43
```

900k steps =3000 episodes of300 steps. Seed43, gamma0.95, batch256, LR3e-4,
tau0.005, automatic alpha initialized0.05, one update/step from1024. Epsilon
0.8->0.1 over100k training steps; exploration branch25% random,50% assigned-goal
greedy,25% sampled actor. Eval uses only tanh(mu).

Continue for an ADDITIONAL budget, retaining saved settings:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_backlog_potential_v5/train.py --device cuda --resume runs/backlog_potential_v5_max_seed43/checkpoint.pt --steps 300000 --output runs/backlog_potential_v5_max_seed43
```

Omitted reward options restore saved values; explicit mismatches are rejected.
Changed rewards require fresh replay. Ctrl-C saves an atomic checkpoint.
Setup/verify do not start long training.

## Reward and ordering

```text
r = -cost(backlog_after_collection)
    + beta * (gamma * Phi(actual_replay_successor) - Phi(current_state))
    - 0.1 * boundary_violation_count
    - 70 * sum(return_SOC_shortfall) / 0.1
    - sum_responsible_failures(7.5 + 3.75 * distance_to_home / 1800)
```

Max cost: `max(B_after)/2000`, uncapped.
P8 control: `(mean((B_after/2000)^8))^(1/8)`, stably evaluated.
P8 is a smooth alternative objective, not exactly maximum backlog.
Old fixed target-hit and collected-backlog bonuses are REMOVED. Collection P4
and oldest-hit counts remain diagnostics only.

```text
gap_u(s) = max(distance(UAV_u, its assigned oldest target) - 173.2, 0)
Phi(s) = mean_over_serving_UAVs(exp(-gap_u(s)/1800))
Phi(s) = 0 if no UAV serves
beta = 5; gamma = SAME SAC gamma (0.95)
```

Phi is bounded in[0,1]. For fixed goals, closer movement has larger shaping feedback
than standing/retreating. Goal rebinding can yield negative shaping; dispatch/home
reset can cause jumps. These are the same state-potential difference, not task bonuses.
Compare TASK cost/base return/backlog separately from shaped episode return.

Successor Phi is computed AFTER10-step dispatch and20-step full/home reset,
from exactly the observation stored in replay. At300 the cutoff is truncation,
not death: bootstrap the physical successor, do NOT zero Phi or substitute an
external episode reset. Verification checks discounted telescoping across
target changes, resets and cutoff.

With bounded Phi and consistent continuation, shaping preserves the discounted
task objective. Approximate SAC convergence is not guaranteed. Gamma0.95 still
optimizes discounted cost, not exact infinite-horizon average cost. Best-model
selection continues to minimize three-seed time-average maximum backlog.

## Physics / observations

150 sensors, map42,4000m square, airship(2000,2000),173.2m coverage,180m maximum
displacement per60s. Dispatch2-6 UAVs every10 steps. Deselected deployed UAVs
physically return and charge with unlimited slots. Every20 steps check return
then reset UAVs full/home; sensor backlog/age continue. Episode length300.

Unchanged14-feature fleet tokens, all sensor ages, own oldest-target relative
coordinates/age, masked assigned-target context,20-step phase. Only serving
queries output actions. Actual backlog is simulator reward information ONLY;
it is not added to actor/critic input. Assignment remains oldest-first,
nearest-within-ties, unique, rotating priority. Radial action projection and
critic raw-action encoding remain as in v4 for this reward comparison.

## Logs, figures, checks

- Progress/loss JSONL flushed every100 steps; episode JSONL every300.
- Eval every3000 steps on1001/1002/1003, initial and final eval included.
- Rows log task/shaping/safety, cost/reference, Phi before/after, gamma/beta,
  own target coordinates/distances, actual/replay successors, dispatch/reset,
  collection/return/energy details. Summaries log base/shaped discounted return,
  reset shaping totals and telescoping residual. Config/source hashes saved.
- Training/loss/eval curves overwrite the same files. Eval metrics: backlog and
  reward components. Existing1-80 all-UAV2x2 trajectory figure retained.
- Seed1001 additionally gets `trajectory_full_001_300.png`: all steps in
  1-100/101-200/201-300 windows with corresponding backlog. No reset-flight lines.
- Best/latest actors, periodic snapshots and resumable checkpoint are saved.
  Result ZIPs exclude full training replay.

```bash
python3 experiments/lower_backlog_potential_v5/verify.py
python3 experiments/lower_backlog_potential_v5/pack_results.py runs/backlog_potential_v5_max_seed43
python3 experiments/lower_backlog_potential_v5/evaluate.py --checkpoint runs/backlog_potential_v5_max_seed43/policy_best.pt --steps 300 --output runs/backlog_potential_v5_max_best_eval --device cuda
```
