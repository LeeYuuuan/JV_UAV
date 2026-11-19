# Implementation review record

Date2026-11-19. Scope: authored asynchronous experiment files, their configuration,
test cases, and actual invoked dependencies. Reviewed function bodies and their
call ordering, not just public signatures. A review/test pass is not a universal
proof of bug absence or evidence of RL convergence. No old environment runner,
frame settlement, observation adapter or learning loop is invoked by this experiment.

## Reviewed units and evidence

| Unit | Review focus | Evidence |
|---|---|---|
| engine.allocate | retained docks, existing waiter priority, low-arrival-SOC tie order, inbound locks, rejections |250 randomized parity cases against current legacy allocator; targeted locks/priority tests|
| Engine.reset | fresh episode only; persistent queues/energy within episode; independent dynamics/order RNG |seed and8h tests|
| Engine.return_cost | actual remaining climb/horizontal cost, no charging energy before docking |partial-climb and return tests|
| Engine._reference/_die/_finish_if_needed | continuous-time reference budgets, one failure/UAV, complementary weights, one terminal tail |upper/lower/shared scenarios and death ties|
| Engine.apply_upper | continue rather than restart; recall at actual XY; C/W pre-reservation; mask locked requests |flight/service continuation, recall, reservations, charging tests|
| Engine.apply_lower | single selected UAV, normalized absolute action, intent commit at zero time, positive task duration |same-time snapshot/intent test|
| Engine._queues_advance/_collect | one arrival update/global interval; union clear only for completed services, stable ties, packet ledger |analytic fluid case, duplicate coverage test, Poisson8h ledgers|
| Engine.advance/_process_zero_returns | minimum event time, real interpolation, powers/charge cap, depletion/service/upper ordering, exact horizon |timing conflicts, zero-SOC landing,8h segment energy checks|
| Engine.observation | true versus working location, explicit selected UID/order, ETA and remaining costs, cached reference inputs |shape/intent/cost tests and input-by-input review|
| Engine.state_dict/load_state_dict | include physical timers, queue, role reservations, RNG and reward buckets |exact trainer continuation tests|
| learning.temporal_factors/duration_gae |physical gamma/lambda exponents; death/reset blocks traces; nonterminal rollout bootstrap |hand-derived targets, reset and partial-duration cases|
| UpperPPO |charge action mapping, masked PPO loss/entropy, raw value targets, population advantage norm, optimizer/gradient bounds |rollout-splicing tests; all-masked actor remains unchanged|
| lower Encoder/LowerActor/Q |sensor geometry/backlog, full-team state, explicit selected-token gather, correct2D tanh Jacobian and joint-state Q |shape tests; real optimizer functional runs|
| LowerSAC.update |one-step duration discount; zero-time links; skip terminal Q; twin Huber; entropy target-2; Polyak targets |analytic constant-Q target test, terminal target test, finite losses|
| Replay |copy observations/action; duration/terminal retained; seeded sampling/cyclic overwrite |reward/time conservation and exact resume|
| Trainer._close_lower/step |one global reward stream, open lower action across upper ticks, no sixfold reward duplication |sum(replay rewards)=episode lower total; zero-time and timing checks|
| Trainer._close_upper |accumulate1024 across resets; no reset/checkpoint forced update; correct next pre-decision bootstrap |small-T mixed-episode test, exact resume before/after update|
| Trainer.evaluate/save/load |separate environment/no RNG side effects, complete pending state, atomic checkpoint |repeat eval and exact resume at partial rollout and terminal boundaries|
| train.validate/main/provenance |effective parameters/seeds, additional interval semantics, live flushing, config/source archive, explicit resume boundary |CLI smoke, full-width short run, continuation and packaging checks|
| plots.dashboard/curves |actual flown segments, completed-service circles only, initial SOC included, independent actor/critic scales |rendered-image inspection; data originate from saved event records|
| verify_reference |not used by learner; conservative rotating charge decisions, energy-feasible greedy sensor targeting |three8h reference runs with ledger and capacity assertions|

## Reused dependency scope

- `jv_uav.config.load_config/validate_config/section`: mapping and baseline geometry,
  energy/capacity checks; experiment adds its own physical/learning validation.
- `jv_uav.sensor_generator.generate_sensor_positions` and radial distribution helper:
  original map seed, quotas, rejection bounds and configured scatter overrides
  retained. No regeneration rule or coordinate override changed here.
- `jv_uav.models.SimpleEnergyModel`: capacity125Wh, serving and vertical/horizontal
  return units checked. Event engine does not call old Scene.evolve_low_step or
  LowestArrivalBatteryScheduler.make_plan; the latter is a test oracle only.
- NumPy/PyTorch/YAML/matplotlib standard operations used directly. CPU behavior is
  verified locally; server GPU runtime and stochastic training convergence remain
  to be verified by actual experiments.

## Explicit modeling limits

- No outbound descent energy, en-route collection or continuous streaming clear.
- Poisson endpoint counts are exact for each interval; the backlog time integral
  is a<=10s-bin trapezoidal approximation, not exact packet-event integration.
- Responsibility is an explicit counterfactual convention. Shared-weight penalties
  are not a causal proof, and terminal/future-return effects remain in value targets.
- Mid-service depletion retains upper responsibility as previously agreed; this
  does not guarantee the lower objective is free of all termination incentives.
- The upper policy changes while SAC replay retains earlier transitions, creating
  the usual joint-training nonstationarity. No convergence guarantee is claimed.
- SAC entropy is per decision; zero-time planning links do not multiply physical
  backlog costs but still carry the defined soft-policy entropy value.
