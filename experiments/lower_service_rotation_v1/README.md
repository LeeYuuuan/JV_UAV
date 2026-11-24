# Displacement service rotation v1

Current revision: `displacement_service_rotation_v1_aux05`. The auxiliary reward
ceiling was reduced from 5 to 0.5; target bonuses and other settings are unchanged.
The CLI defaults explicitly to `--auxiliary-max 0.5`, overriding a stale YAML
value on fresh runs. A checkpoint with a different coefficient is rejected on
resume. To continue an old 5.0 run intentionally, pass `--auxiliary-max 5.0`.
Start a fresh run to change the reward; old replay rewards are never relabeled.

Independent lower-only experiment. This does not import or modify any previous
waypoint experiment, upper policy, or joint runner. It reuses only the existing
sensor generator, energy/service models, configuration loader, and gradient helper.

## Server commands

From the **JV_UAV** repository (not P1_J), after pulling the new files:

```bash
python3 -m pip install -r requirements-rl.txt
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_service_rotation_v1/train.py --device cuda --auxiliary-max 0.5 --steps 300000 --output runs/service_rotation_v1_aux05_seed43_corrected
```

The budget is physical 60-second steps: 300,000 steps = 1,000 training episodes.
The script does not start another process or choose a physical GPU for you.
Use tmux normally; CUDA sees the chosen physical GPU as local cuda:0.

To continue for another 300,000 steps with identical settings:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_service_rotation_v1/train.py --device cuda --auxiliary-max 0.5 --resume runs/service_rotation_v1_aux05_seed43_corrected/checkpoint.pt --steps 300000 --output runs/service_rotation_v1_aux05_seed43_corrected
```

Resume restores model/target/optimizers/alpha, replay, physical state, partial
episode, exploration schedule, and RNGs. `--steps` always means **additional** steps.
Ctrl-C saves an atomic checkpoint; SIGKILL/power loss can only retain the last
completed checkpoint. Never change experiment parameters while resuming.

## Explicit environment assumptions

- Map snapshot: `environment.yaml`, 4,000m square, 150 sensors, seed 42,
  airship (2000,2000), 173.2m coverage, 0.5 packet/s/sensor.
- Every 300 steps reset positions/SOC/backlog/age. This is a sampling truncation:
  bootstrap from the final pre-reset observation, never the reset observation.
- At episode start and after each 20-step safety check, fill every battery and
  place all UAVs at the airship (`cycle_reset: true`). Backlog and ages continue
  across these cycle boundaries. Physical return/charging operate within cycles.
- Randomly request 2-6 service UAVs initially and every 10 steps. Eligible UAVs
  are current servers and positive-SOC docked UAVs. Returning/stopped UAVs cannot
  be dispatched. Actual count can be below the requested count.
- Deselected servers return physically: climb first, then horizontal flight.
  They do not collect data. SOC and position update each physical step. On
  arrival, only time remaining inside that step is available for charging.
- Default unlimited charging (`charging_slots: null`); no waiting occurs in
  that mode. An integer enables FIFO charging with no limit on waiting space.
  Waiting/idle at the airship consume no energy. Charging ends at 100% SOC.
- Every 20 steps check remaining servers' return feasibility **before** new
  dispatch. A failed check is penalized once per service deployment; it does
  not magically return/refill the UAV or terminate the task. Subsequent physical
  return failure is not penalized again for that same deployment.
- Depleted UAVs stop until the next 20-step cycle reset. In-flight
  return failures are penalized once. Service depletion is penalized only if a
  same-step inward reference action could have completed service and retained
  return energy; unavoidable service depletion is logged as schedule-caused.
- `cycle_reset: false` is an explicit alternative: positions, SOC, return and
  charging continue across 20-step boundaries. Stopped UAVs then wait until the
  300-step episode reset. Random dispatch can exhaust the entire fleet, so this
  mode may contain long no-service tails. It requires a fresh output directory.

These assumptions resolve the two open design choices without changing the joint
environment. They are saved with every run and can be changed for a fresh run.

## Observation / architecture / actions

Shared actor, serving queries only. Inputs: every UAV's ID, position, SOC and
one-hot status; all sensor ages; current 6 oldest sensors' position/age; 20-step
cycle phase. No actual queues and no coordinates of non-target sensors enter
the networks. Fixed sensor index defines the full age vector order.

128-wide GELU embeddings, 4-head service-team attention, non-service-context
attention (with a null token for an empty set), target-sensor attention, global
age MLP, shared Gaussian mean/log-std action head. Tanh actions in [-1,1]^2 map
to displacement; radial clipping enforces a 180m Euclidean maximum, followed by
map-boundary clipping. Critics use joint observable information and valid actions.
Padding actions have no Q effect or policy/entropy loss. Zero-service states
still bootstrap their future value, but produce no actor update of their own.

## Reward and learning settings

`5 * target_hits + 0.5 * P4 / (P4 + 300) - 0.1 * oob_count - return_cost`.
Target IDs are frozen before the action; union coverage is counted once.
P4 uses covered sensors' queue sizes after arrivals and before collection.
There is **no negative backlog reward** and no inherited legacy bonus/penalty.
Return cost per responsible UAV is `10 + 5 * distance_to_airship / 1800`.

Gamma 0.95 per physical step, batch 256, LR 3e-4, tau .005, initial alpha .05
with automatic tuning over active action dimensions; 1 SAC update/step starting
at step 1024. No PPO/GAE. Epsilon .8 -> .1 over 100k steps; within epsilon,
25% uniform displacement, 50% nearest unreserved top-6 target direction, 25%
sampled actor; outside epsilon use sampled actor. There is no separate random
warmup. Eval always uses actor mean without exploration mixing.

## Outputs (written during training)

- `resolved_config.json`, `provenance/*`: revision, exact parameters, source hash
  manifest and compressed source snapshot for reproducing the run.
- `episodes.jsonl`: full 300-step returns, reward components, mean/worst/final
  maximum backlog, collection, coverage, active counts, failures, behavior counts.
- `progress.jsonl`, `live_status.json`: every 100 training steps.
- `losses.jsonl`: averaged over up to 100 updates; actor total, policy term,
  entropy term, separate Q losses, Q/target means, alpha, sigma, saturation, scales.
- `episode_traces/`: one compressed detailed episode per 10 episodes.
- `training_curves.png`: episode return, mean/worst maximum backlog, reward terms.
- `losses.png`: actor and critic on separate axes; Q/targets and exploration scale.
- `evaluation_curves.png`: evaluation return and mean/worst maximum backlog.
- `eval_STEP/seed_SEED/trajectory_01_15.png`, `trajectory_16_30.png`:
  **all UAVs on each map**, service footprints and real return segments. No
  per-UAV trajectory figures and no 100-step PDF.
  The 16-30 panel spans the reset after step20: faded paths and hollow squares
  mark the old cycle; stronger paths show the new cycle; triangles mark step30.
  The reset is explicitly labeled and is never drawn as a physical return flight.
  Sensor colors show backlog at panel end, so a previously visited sensor may
  already have accumulated new packets. Coverage circles mark endpoint service,
  not a continuously swept collection region during movement.
- `eval_STEP/seed_SEED/evaluation_metrics.png`: full 300-step backlog, SOC,
  cumulative reward and role histories. Trajectory panels show only first 30;
  scalar eval return/backlog summaries always use the complete 300 steps.
- `eval_STEP/seed_SEED/trace.json.gz`: all 300 steps and first-30 actor parameters.
- `checkpoint.pt`: latest atomic resumable checkpoint, overwritten every 3000 steps.

Eval runs initially, every 3000 steps (10 episodes), and at the final budget.
Three evaluation seeds share the map but have different dynamics/schedules.
Training figures overwrite the same files after every completed episode.

Check deterministic invariants without starting training:

```bash
python3 experiments/lower_service_rotation_v1/verify.py
```

Package completed or partially running results (no large checkpoint included):

```bash
python3 experiments/lower_service_rotation_v1/pack_results.py runs/service_rotation_v1_aux05_seed43_corrected
```

The packer includes all scalar logs/curves, source provenance, the initial eval,
the latest three evals and latest three detailed episode traces. It ignores
incomplete log lines/compressed traces and leaves the training process untouched.
Use this experiment's packer: the older repository packer does not know the new
loss/episode filenames.

Redraw a completed evaluation with explicit cycle-reset markers, without stopping
training or loading/running a model:

```bash
python3 experiments/lower_service_rotation_v1/redraw.py runs/service_rotation_v1_aux05_seed43/eval_00003000
```

New plots go into each seed's `plots_v2/`; original figures, logs and checkpoints
are preserved. The metadata records the exact source-trace and plotting hashes.
An already running Python training process keeps its imported plotting code;
pulling code alone does not update its future figures until restart/resume.
