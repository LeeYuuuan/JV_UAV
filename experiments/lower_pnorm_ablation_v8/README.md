# V8: simple backlog p-mean / energy ablation

Independent from v5 and v7. New checkpoint format `displacement_pnorm_ablation_v8`;
fresh actor, critics and replay required. No training job is started by deployment.

## Reward

After arrivals and actual collection, let B contain ALL150 remaining sensor queues:

```text
task_cost = (mean((B / 2000)**8))**(1/8)
r = -task_cost - 0.1 * boundary_violation_count
    - 5 * responsible_return_failure_count
```

The calculation is a numerically stable generalized p-mean (not an unnormalized
sum norm). There is no clipping, collected-packet bonus, round bonus, potential,
distance shaping or repeated energy-deficit penalty. Failure rules/deduplication
match v7. In no-energy mode no energy failure can occur, so its failure term is0.

Round masks and goal assignment remain as v7: oldest pending goals first,
nearest within ties, distinct goals until the remaining list is shorter than
the serving fleet. The visited mask, round completion and durations are retained
as observations/diagnostics but pay no round-completion reward.

## Shared physical task

Fixed map seed42,150 sensors,4000m square; airship(2000,2000);6 UAVs; radius173.2m.
One physical step60s; at most30s of service movement at6m/s, so displacement
length<=180m. All serving UAVs move synchronously and actually clear covered data.

Every10 steps choose2-6 serving UAVs; deselected deployed UAVs fly back over real
time (climb first, then horizontal flight) and do not collect while returning.
Every20 steps both modes retain the full/home UAV reset. Sensor backlog/age and
round masks continue.300 steps is a reporting boundary, NOT a task reset; replay
done remains false and uses the actual successor. The phase/step is not observed.

## Two energy modes

- `--energy-mode real`: actual movement/hover/return power, SOC and charging;
  original v7 safety checks and failure penalty5. Unlimited charging slots as v7.
- `--energy-mode none`: SOC is always1 for ALL UAV tokens. No serving/return
  energy consumption, no depletion or energy failure. Return duration/speed are
  still physical; arrival makes the UAV idle/full without charging delay. This
  deliberately removes both battery shortage and recharge-time costs.

Map, neural architecture, initial weights, reward coefficients, service selection
rules, physical speeds,20-step resets and learning settings match across modes.
Actual later dispatch eligibility may differ because charging/depletion differs.
Compare both runs at matched training step counts and evaluation seeds.

## Observation and network

- Six14-feature UAV tokens: x/y divided by4000; own absolute SOC; ID/5;
  six-state one-hot (serving/returning/charging/waiting/idle/stopped);
  assigned-goal relative x/y divided by4000; bounded goal age; goal-valid bit.
- Up to six padded goal-context tokens: absolute x/y divided by4000, bounded age;
  valid-target mask. Serving IDs/masks determine actor output queries.
- Global301-vector:150 bounded sensor ages,150 current-round visited bits,
  one visited fraction. No actual backlog input or full sensor coordinate input.
- ALL age paths use `age/(age+6000 seconds)`. No rank, current step, phase,
  remaining service duration or remaining episode duration is fed to either net.

Shared set-based actor emits two normalized displacement components per serving
UAV. Twin centralized critics see the same state and joint6x2 masked actions,
and each return one team Q. Same128 hidden units,4 heads and GELU as v7.

SAC: gamma.99, batch256, LR.0003, tau.005, automatic alpha starting.05, one
update per step after1024. Exploration unchanged: epsilon.8->.1 over100k steps;
within exploration25% random,50% greedy,25% actor. Both modes default seed43.

## Server commands (JV_UAV root)

After local commit/push and server pull, run in separate tmux sessions:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_pnorm_ablation_v8/train.py --energy-mode real --device cuda --seed 43 --steps 900000 --output runs/pnorm_v8_real_seed43
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_pnorm_ablation_v8/train.py --energy-mode none --device cuda --seed 43 --steps 900000 --output runs/pnorm_v8_none_seed43
```

Resume an EXTRA budget (energy mode may be omitted: restored from checkpoint):

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_pnorm_ablation_v8/train.py --device cuda --resume runs/pnorm_v8_real_seed43/checkpoint.pt --steps 300000 --output runs/pnorm_v8_real_seed43
```

Changing energy mode on resume is rejected; do not reuse old v5/v7 output folders.

## Logs, figures and diagnostics

Incremental progress/loss logs every100 steps; report every300; checkpoint every
3000; pure-actor600-step evaluation on1001/1002/1003 every6000 and at the end.
Plots include all-UAV trajectory1-80 in four20-step panels, complete eval backlog
and task/boundary/failure components, training/eval curves, critic/actor losses,
and renewal-round diagnostics. Curves overwrite their current PNGs. Seed1001 also
gets complete1-300 and301-600 trajectory views. Raw ages stay in seconds in traces;
loss logs include both encoded maximum age and raw age maximum in seconds.

Mode, input schema, reward settings, seed, source hashes and code description are
recorded in resolved_config.json and per-launch provenance. Best/latest actor
snapshots and resumable full checkpoint are retained. To export results or critics:

The active reward is `training.reward` in resolved_config.json. The environment
section retains shared Scene configuration for provenance; its legacy reward
fields are not used by this independent environment.

```bash
python3 scripts/pack_lower_review.py runs/pnorm_v8_real_seed43
python3 scripts/pack_lower_review.py runs/pnorm_v8_none_seed43
python3 scripts/pack_critic_review.py runs/pnorm_v8_real_seed43 runs/pnorm_v8_none_seed43
```

Run `python3 experiments/lower_pnorm_ablation_v8/verify.py` for exact reward,
energy-isolation, no-phase/bounded-input, shared initial weights, SAC and resume
checks. These checks do not establish training convergence.
