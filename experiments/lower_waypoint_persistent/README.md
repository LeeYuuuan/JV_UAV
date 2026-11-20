# Persistent waypoints with real speed and epsilon mixture

Revision `lower_persistent_speed_epsilon_v1`, 2026-11-19. New independent experiment.
The teleporting coverage experiment remains available for comparison; its replay
and weights must not be reused as real-motion transitions.

## Physical execution

- Same map, sensor process and coverage radius as configs/default.yaml. Default
  mapseed42, trainingseed43. Sensor inputs are actual coordinates and backlog.
- Physical step60s, speed6m/s, at most30s movement per step: maximum180m. Remaining
  time is service/hover. This matches the existing movement/service timing model.
  Collection uses actual service endpoints only; flying past other points is not
  treated as full continuous-path collection.
- Waypoints are absolute full-map targets, not instantaneous positions. Once
  selected, all active UAVs keep moving toward their targets. Early arrivals keep
  serving there. Replan synchronously when all active UAVs arrive, the service
  group changes, or the episode ends. No actor/random resampling in mid-interval.
- Groups remain20/10 physical steps: randomly3or4 long UAVs, all6 active initially,
  short group inactive for the last10. At group changes, interrupted targets are
  logged and new targets may be selected. Positions are retained, not reset.
- Episode200 physical steps. Queues, positions and groups reset only at episode
  reset; groups are resampled each20 physical steps. Queues persist across groups.
- Energy and automatic return are deliberately disabled, per user choice. This
  restores speed and time, not full battery-feasible missions.

## Exploration: no initial pure-random phase

Actor participates from decision1. At each NEW waypoint assignment, independently
for each active UAV, epsilon chooses whether to explore. Within exploration:

-25% full-map uniform xy: sample2 numbersU(-1,1), map to[0,4000].
-50% greedy: among unreserved sensors, take the current highest-backlog6; choose
  nearest to this UAV, then add independent uniform[-20,20]m in xy and clip to map.
-25% sampled actor action, with no extra perturbation.

Outside exploration: sampled actor action. Thus overall probabilities are
`random=.25*epsilon`, `greedy=.5*epsilon`, `actor=1-.75*epsilon`. Sampling sources
are not averaged in coordinate space. Before greedy assignments, reserve sensor
footprints of other actors/random targets; reserve each greedy target footprint
before assigning the next greedy UAV. Greedy uses current visible backlog, not
future arrivals. Candidate pools are refreshed after each reservation.

Epsilon linearly decays1.0->0.1 over100000 physical training steps, then stays0.1.
The exploration ratios count waypoint assignments, not physical time spent flying.
Jitter is in metres; worst-case28.3m is less than173.2m coverage radius. This keeps
the chosen sensor in range AT ARRIVAL; it grants no collection before actual reach.

Actor training always samples its Gaussian then applies tanh. No separate additive
noise is added to actor targets. Epsilon does not remove SAC's intrinsic sampling.
Evaluation uses actor-only `tanh(mu)`, no epsilon/random/greedy/jitter. The latter
is deterministic, not the exact mathematical mean of a tanh-normal distribution.

All6 fleet tokens remain visible as attention context, including inactive UAV
positions needed when they reactivate. A token is indexed by its physical UAV ID:
`[x/4000,y/4000,ID/5,remaining_service/20,remaining_episode/200,active_flag,active_count/6]`.
Only active tokens generate actions/log-probabilities. Replay actions are packed
by physical ID, not by the order of a changing active subset. No SOC is provided.

## Reward, replay and discount

Each physical slot uses the previous all-sensor reward, based on actual endpoints:

```
r_low = -Bmax_after/300 + 0.5*Bmax_after/300*h_all
        -0.02*mean_active_actual_move_m/(4000*sqrt(2))
h_all = actual_collected_packets / all_pre_collection_backlog
```

Overlap counted once, zero denominator gives0. No energy/death/return penalties.
For a committed interval lastingm physical slots, store ONE actual mixed waypoint
action, future observation, duration60*m, and `R=sum(gamma^j*r_low[j])` forj=0..m-1.
The SAC target uses `R + gamma^m*soft_V(next)`, with gamma0.95 per60s. Terminal
discount is0. Group boundaries are nonterminal, episode end is terminal. Never
store an imagined jump or a future target's unearned collection reward.

SAC entropy is per waypoint decision; the physical reward is discounted inside
the interval. Diagnostics report raw physical episode return separately from the
macro replay reward. Do not sum discounted macro rewards as raw episode reward.

Batch256 macro transitions must be available before learning starts; this is only
buffer filling, not a pure-random policy. Thereafter default1 update per elapsed
physical step, performed after each completed interval. Thus anm-slot interval
can trigger m updates on the completed replay buffer. Log physical steps,
waypoint decisions and SAC updates separately. Hidden128, heads4, lr3e-4, tau0.005,
initialalpha0.05/auto tuning, replay100000; other lower settings remain unchanged.

## Run

PC terminal atD:\VScode\JV_UAV, RL environment activated:

```powershell
python -u experiments/lower_waypoint_persistent/train.py --device cuda --steps 500000 --output ./runs/lower_persistent_speed_epsilon_v1
```

Server after commit/pull:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_waypoint_persistent/train.py --device cuda --steps 500000 --output ./runs/lower_persistent_speed_epsilon_v1
```

500000 physical steps =2500 complete200-step episodes. --steps is an additional
physical-step budget. A non-episode-aligned budget ends at the next natural
decision boundary (at most one group segment beyond it), not mid-flight with a
fabricated terminal transition. Full-episode budgets end exactly at the episode
boundary. Old teleport/physical-energy checkpoints are rejected by format.

Resume the same experiment, retaining mixed-policy RNG, replay, optimizers, target
networks, current episode/group/positions/queues and all neural RNG states:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_waypoint_persistent/train.py --device cuda --resume ./runs/lower_persistent_speed_epsilon_v1/checkpoint.pt --steps 100000 --output ./runs/lower_persistent_speed_epsilon_v1
```

If interrupted logs are newer than the checkpoint, use a new output folder.

## Plots and logs

- Eval at each1000-physical-step threshold, after the current committed interval.
  Actual eval step can therefore be a few slots later than the threshold. Seeds
  1001/1002/1003. Every eval saves per-UAV trajectory.png, fleet_overlay.png and
  summary.json; full trace/physical dashboard/Gaussian diagnostics every5000 steps
  and at final exit. Initial reference policies are separate from actor-only eval.
- Physical trace rows show true positions, <=180m movements, flight/service times,
  queues, coverage and each reward term. Decision records show targets, reached
  and interrupted IDs, source per UAV, epsilon, interval length and discounted reward.
- episodes.jsonl.gz flushed every episode; train.jsonl sampled by physical-step
  threshold, live_status.json each completed interval. Behavioral source counts
  are logged separately from actor-only evaluations.
- checkpoint.pt at each5000-step threshold and final exit, atomically overwritten.
- Training/loss/eval plots overwritten at1000-step thresholds. Source.zip, exact
  config and hashes recorded per invocation. Existing pack_results.py works.

```bash
python3 pack_results.py ./runs/lower_persistent_speed_epsilon_v1
python3 experiments/lower_waypoint_persistent/evaluate.py --device cuda --checkpoint ./runs/lower_persistent_speed_epsilon_v1/checkpoint.pt --output ./runs/lower_persistent_speed_epsilon_v1/eval_manual_latest
python3 experiments/lower_waypoint_persistent/live_overview.py ./runs/lower_persistent_speed_epsilon_v1 --watch 60
```

The overview keeps the user-approved common-map training-stage comparison format.
Lines in this experiment show real executed movements, not instantaneous jumps.

## Verification

```bash
python3 tests/test_lower_waypoint_persistent.py
python3 experiments/lower_waypoint_persistent/train.py --device cpu --smoke --output ./outputs/persistent_waypoint_smoke
```

Tests cover speed/time, held targets, early-arrival service, interrupted goals,
discounted physical rewards and gamma^m targets, branch frequencies, top6 nearest
assignment/jitter/reservation, actual-action replay, exact sampler/trainer resume,
actor-only evaluation isolation and images. Smoke uses a small network/short task;
reference metrics and functional checks are not learned-policy convergence evidence.
