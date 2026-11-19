# Independent asynchronous hierarchical waypoint experiment

Revision `async_waypoint_600s_v1`, 2026-11-19. Both policies train from scratch.
Existing joint,20/10 and synchronous-waypoint experiments are unchanged.

## Physical and scheduling contract

- Episode horizon28800s (8h). Any UAV death terminates early. No fresh-battery
  replacement within the episode; charge at the actual500W charging docks.
- Upper decision at0,600,1200,... seconds. At the8h terminal boundary no new decision.
-6 UAVs,2 charging slots,2 waiting places; the current default map is4000m square,
  airship(2000,2000),150 sensors,100 inside r1800 and50 outside. Map seed42 remains.
- Maintain separate roles SERVE/WAIT/CHARGE/DEAD and execution phases READY,
  MOVE,SERVICE,RETURN_CLIMB,RETURN_FLY,CHARGE,WAIT,DEAD. The two return phases are
  the public “returning” state; READY is an instantaneous waypoint-decision state.
- At an upper tick continuing SERVE retains the existing target and remaining
  flight/service time. Recall cancels the unfinished job at the interpolated
  current position; no premature collection. Rejected requests continue service.
- Continuing charging requests retain their dock. Existing waiters compete for
  vacancies before newcomers; within a group order is arrival SOC then UAV ID.
  This deliberately preserves the actual current controller, not strict FIFO.
- An inbound aircraft reserves its C/W destination immediately and remains locked
  to that return until arrival. Its sampled upper request is ignored and masked
  out of PPO actor/entropy loss. Its physical state remains visible to the critic.
- Return need is an energy check, NOT a “must arrive within600s” condition. Position
  and SOC advance continuously across upper ticks if the period is configured shorter.
- Actual charging starts on arrival; actual WAIT consumes no energy. Full charge
  stays docked until an upper decision releases it. Waiters do not auto-promote
  between upper ticks. A charger can leave before full if upper selects SERVE.
- Lower decisions occur initially for allocated serving UAVs, upon release from
  C/W, or after a service completes. Shuffle simultaneous ready IDs once; later
  selections see earlier targets. Selection itself consumes no physical time.
- Shared actor selects absolute full-map waypoint `(tanh(z)+1)*world_size/2`.
  Fly at6m/s, then hover/service30s. There is no180m step limit, boundary penalty,
  movement-distance penalty, or collection while traveling.
- At service completion clear the footprint union of simultaneous completers once.
  An aircraft whose service was interrupted or that depleted cannot collect.
- Between events sample Poisson arrivals for all sensors in bins<=10s. Packets
  added once globally. “Currently servicing” does not continuously hold buffers at0.
- Return is400W vertical climb50m at5m/s, then257W horizontal flight. Outbound
  vertical/descent energy is not modeled, consistent with the existing energy model.
  Early arrival means actual docking; no fictitious hover after landing.
- Inbound remaining-return SOC excludes already-paid climb/flight energy. Physical
  battery zero during flight/service fails; landing exactly at0 is allowed, matching
  the existing nonnegative arrival-SOC convention.

Event ordering: advance packet arrivals, positions and energy; settle return
landings and depletion (dead aircraft cannot collect); clear completed services;
process other arrivals; terminate if needed; run a due upper decision; then resolve
ready lower decisions. No fictitious waypoint choice immediately before a due recall.

`Engine.assert_invariants()` checks capacities, finite/bounded physical values,
packet conservation, queue validity and role/phase consistency during execution.

## Observations

Lower observes sensor `[x/4000,y/4000,backlog/3000]` tokens and all6 UAV tokens:

```
actual_xy/size, work_xy/size, SOC, remaining_return_SOC, ID/5,
phase_ETA/600, pending_order_rank/6, role_onehot4, phase_onehot8,
reference_start_SOC, reference_required_SOC, optimistic_required_SOC,
reference_elapsed/600
```

These are25 fields/UAV. Reference fields are zero for non-serving roles; they
expose the cached responsibility state to the centralized lower critic. Global
features are remaining episode fraction, time until upper decision/period, reserved
C ratio and reserved W ratio. The currently selected UAV is explicitly marked and
its contextual token is gathered; no need to infer self from an ID/SOC coincidence.
Working location is target when moving/returning, actual location otherwise. ETA
includes remaining climb plus horizontal return for a climbing UAV.

Upper shared actor has30 fields for6 UAVs: own SOC, own role onehot4, C/W reserved
ratios, normalized ID, all SOC6, all return SOC6, own return SOC, own phase onehot8,
own ETA/600. Central critic has154 fields: all25-field team tokens plus4 global
features. Input geometry/phase semantics differ from the earlier21-feature actor;
old checkpoints are rejected rather than silently imported.

## Rewards and responsibility

Upper interval reward:

```
(5 * integral(serving_role_count dt)/600 -20 * sum(upper_failure_weight)) /10
```

No C/W penalties or backlog term. Serving includes travel and actual collection
service; logs separate phases. Discrete death cost is charged once.

Lower reward accumulated on the one global decision stream:

```
- integral(system_max_backlog dt)/(300*60)
+ 0.5 * post_clear_max_backlog/300 * h_top20   # once per collection event
-500 * sum(lower_failure_weight)
-500 * sum(lower_failure_weight*return_distance_m)/1800
-responsibility_weighted_absorbing_failure_tail
```

Top20 is computed after arrivals, before clear; union-based weighted cleared
fraction, stable sensor-ID ties, zero denominator=>0. Trapezoidal integration on
<=10s bins approximates the backlog integral. Rewards within a transition are
undiscounted; physical-time discount applies between decision epochs.

Use the agreed reference-return responsibility idea, recalculated for continuous
movement rather than the old30s movement allowance: at each SERVE allocation cache
a600s reference that moves halfway toward the closest reachable inward endpoint,
then hovers, with0.01 extra SOC margin. The optimistic lower bound uses the smaller
of horizontal/hover power over the interval and the fully inward endpoint. If the
next interval would cross the8h horizon, use its actual remaining duration.

At an infeasible recall, compare cached starting SOC: below optimistic bound =>
upper1/lower0; enough for reference => upper0/lower1; intermediate =>0.5/0.5.
No preceding serving reference =>upper1. Preserve the previously agreed rule that
mid-service depletion and actual return depletion are upper responsibility.
The same failure record drives both explicit penalties; weights sum to1. This
does not remove indirect effects such as termination reducing future serving return.

When lower bears responsibility, terminal tail is the mean lower weight across
new deaths times `(Bmax*remaining_sec +0.5*max_arrival_rate*remaining_sec^2)/(300*60)`.
It is charged once, without advancing the physical clock. This is a shaping choice,
not a proof against every early-termination incentive. All components are logged.

## Learning: the two time axes are different

Upper: four512-ReLU-layer shared categorical actor, charge category0 mapped to
engine request1; three512-ReLU-layer centralized value network. Adam1e-4, PPO
clip0.2,4 epochs,128-frame minibatches, entropy0.01 constant, value coefficient0.5,
actor/critic gradient caps0.5/2. No KL early stopping or value clipping.

Upper rollout MUST accumulate1024 completed upper transitions across any episode
resets before updating. Do not update on every reset, checkpoint or CLI stop.
GAE uses actual interval duration d:

```
g = 0.95 ** (d/600)
l = 0.95 ** (d/600)
delta = reward + g*(1-done)*V(next) - V(current)
adv = delta + g*l*(1-done)*adv(next)
```

Normal intervals therefore retain old gamma=lambda=0.95. A death-shortened last
interval has the appropriate shorter discount, but terminal blocks bootstrap and
trace propagation. A rollout cut without termination bootstraps from the actual
next upper pre-decision state. Targets use raw advantages; only actor advantages
are population-standardized. Locked return actions do not contribute actor loss.

Lower: sensor cross-attention, UAV self-attention, shared2D tanh Gaussian actor,
two centralized Q networks and Polyak targets. Hidden128,4 heads, Adam3e-4,
Huber twin-Q losses, tau0.005, auto-alpha starting0.05, entropy target-2 (one UAV's
2D action, NOT all6 UAVs), batch256. Replay100000. Uniform whole-map warmup10000
lower choices; thereafter stochastic SAC. One update per replay insertion once
warmup/learning-start and batch requirements are met. No late frequency reduction.

One lower transition runs from a waypoint choice to the NEXT global lower choice,
not necessarily the same UAV's next choice. Its state contains decision UID and
pending-order ranks. Global backlog cost is NOT duplicated six times. Discount is
`0.99 ** (elapsed_sec/60)`; simultaneous consecutive choices have elapsed0 and
discount1. Their committed intentions change state even though time stays fixed.
The finite ready queue and positive30s service prevent an infinite zero-time cycle.
SAC has no GAE lambda. Target entropy is per decision; varying decision frequency
can influence entropy regularization and is logged for later evaluation.

At an upper boundary, a lower transition can remain open. Save its start observation,
action, clock and accumulated reward. Checkpoints also retain partial PPO rollouts,
optimizers, target critics, replay, alpha, all RNGs, phases/timers/positions, pending
order, reservations and responsibility references. Evaluation uses separate engines
and deterministic actions; it does not advance training RNG or update normalizers.

## Server commands

After local commit/push and server pull, from `~/Code/JV_UAV`:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/async_waypoint/train.py --device cuda --upper-steps 102400 --output ./runs/async_waypoint_600s_v1
```

102400 upper intervals =100 full PPO updates, potentially spanning many early
resets. Lower choice/update counts depend on the actual trajectories. This is a
fresh joint experiment, with no freezing or implicit transfer.

Continue another102400 intervals:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/async_waypoint/train.py --device cuda --resume ./runs/async_waypoint_600s_v1/checkpoint.pt --upper-steps 102400 --output ./runs/async_waypoint_600s_v1
```

If an interruption left live logs newer than the last checkpoint, choose a new
output directory for the continuation. Parameters come from the checkpoint.

Functional smoke check (shortened horizon and tiny networks, not training evidence):

```bash
python3 experiments/async_waypoint/train.py --device cpu --smoke --output ./outputs/async_smoke
```

## Logs, figures, packaging

- Effective config and per-invocation source archive/SHA hashes.
- `train.jsonl` every10 upper intervals plus terminals; live_status every interval.
  Requests/probabilities, effective action masks, allocated versus actual dock
  occupation, return counts, serving/backlog, separate actor/policy/entropy/critic losses.
- Every completed episode is flushed to `episodes.jsonl.gz`, including survival
  duration, reward decomposition, packet ledger, coverage and failure responsibility.
- Evaluation every200 upper intervals on seeds1001/1002/1003. Full trace/render
  every2000 intervals, initial and final evaluation. Traces include all upper
  probabilities, lower Gaussian parameters and actual physical segments/collection sites.
- Full training traces every100 episodes, gzip-compressed. Models/checkpoint atomic
  overwrite every1000 upper intervals and normal exit. No accumulating model copies.
- training_curves.png,loss_curves.png,evaluation_curves.png overwrite each evaluation.
  Upper/lower actor and critic scales are separated. Console summary each PPO update.

```bash
python3 pack_results.py ./runs/async_waypoint_600s_v1
```

Compare with previous experiments using survival, physical time, actual collection,
coverage and max backlog; raw rewards/decision counts are not equivalent objectives.
The fresh-policy evaluation may initially die or collect nothing; no convergence
claim follows from passing functional checks.

## Verification and review

```bash
python3 tests/test_async_waypoint.py
python3 experiments/async_waypoint/verify_reference.py --output ./outputs/async_reference.json
```

The conservative handwritten reference is NOT an RL model or warmup teacher. It
only verifies that real charging and event physics permit8-hour survival and full
sensor coverage. See `REVIEW.md` and `verification/` for scoped review/test evidence.
