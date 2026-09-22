# Joint attention SAC + MAPPO

The environment review and RL interfaces were confirmed before this training
implementation. The simulation remains usable without importing Torch. Training
code lives in `src/jv_uav/rl`; `Scene` contains no policy, optimizer or replay.

## Run

Use the existing `RLenv1` environment, which has CUDA-enabled PyTorch. To set up a
different environment, install `requirements-rl.txt` with its appropriate Torch
build. Commands below run from the project root.

```powershell
# A short engineering check; this does not train a converged policy.
python train_joint.py --smoke --device cuda --output runs/smoke

# Default full training: 400000 upper frames (4000 full-length episodes).
python train_joint.py --device cuda --output runs/joint_full

# Bounded run: 1000 additional upper decisions.
python train_joint.py --device cuda --upper-steps 1000 --output runs/joint_001

# Resume models, optimizers, replay, counters, RNG, scene and partial rollout.
python train_joint.py --resume runs/joint_001/checkpoint.pt --device cuda --upper-steps 1000 --output runs/joint_001

# Frozen deterministic evaluation only, with no training-state updates.
python train_joint.py --resume runs/joint_001/checkpoint.pt --eval-only --device cuda --output runs/joint_001_eval
```

`--upper-steps` means additional steps for this invocation, including on resume.
Without it, the configured `total_upper_steps` is used. Resume restores the saved
training/environment configuration; CLI config paths do not override it.
`--smoke` changes network sizes, horizon, warm-up only in
memory. It never rewrites the default YAML files.

## Training budget

The default new run collects 400000 upper frames. Each episode lasts at most
100 frames, with 10 lower steps per complete frame. This is equivalent to 4000
full-length episodes and at most 4000000 lower steps. UAV deaths end episodes
early, so the actual episode count can be higher. The frame budget stays fixed
across episode resets and allows 781 full MAPPO update calls plus a saved 128-frame partial rollout.

This is an initial training budget, not a convergence guarantee. Compare the
fixed-seed evaluation returns, backlog and survival across checkpoints before
deciding whether more training is useful. SAC updates continue at a constant per-transition rate. Evaluation/checkpoints occur every 1024 frames (two MAPPO updates).

An already running process keeps its loaded settings. Old checkpoints also
retain their saved budget; changing the YAML does not override resume settings.
To continue to a cumulative target of 400000 frames, pass `--upper-steps` equal
to 400000 minus the checkpoint's `upper_steps` counter.

## Confirmed schedule

Console summaries use aligned columns and appear every 50 upper frames by default,
counted across episode resets from the start of each invocation. Set
`log_every_frames` in `configs/training.yaml` or pass `--log-every 20` (also supported
with `--resume`). Older checkpoints without this setting default to 50.
The last partial interval is printed on normal completion. `avgN_reward` is the
mean upper reward over the N frames since the previous line; `avgN_max_backlog`
is the mean frame-end maximum backlog. `dead` counts UAV deaths in that interval;
`ep` counts completed episodes. SAC/MAPPO counters are cumulative update calls.
Full per-frame data still goes to `train.jsonl`; evaluation and checkpoint
intervals are unchanged. Evaluation summaries are printed separately.

All values are configurable in `configs/training.yaml`:

- Both levels collect and train concurrently; neither is alternately frozen.
- SAC always performs one minibatch update per finalized low transition once
  warm-up and batch availability permit. A final frame transition can be delayed
  until the next allocation; it still contributes at most one default
  update opportunity.
- Default warm-up: 10000 waypoint low actions and `learning_starts=10000`; minibatch
  size 256. Skipped updates do not advance `sac_updates`.
- MAPPO collects 512 upper decisions, across episode resets if needed, then
  performs one PPO update call. One call uses 3 epochs of minibatches containing
  128 frames each, i.e. 12 actor and 12 critic optimizer steps by default.
- `sac_updates_per_step: 1` stays fixed throughout training. MAPPO updates do
  not trigger additional SAC updates. The former frequency switch is removed.
- A run stopping before 512 upper decisions preserves its partial rollout in
  the checkpoint; it does not silently perform an extra partial PPO update.

Environment steps, SAC update calls, MAPPO update calls and PPO minibatches are
separate logged counters. Upper steps are frame decisions; an early terminal
frame can execute fewer than 10 low steps.

## Observation and normalization contract

New lower actor/critic tokens contain
`[x/world_size, y/world_size, SOC, id/max(total_uavs-1,1)]`. The user confirmed
restoring identity on 2026-09-16 to distinguish colocated, equal-battery UAVs.
Integer IDs remain the action-row mapping; the added scalar uses the fixed total
fleet size, never the current serving count. Global input is the
sensor last-visit vector; no sensor backlog or positions are added. A single
RunningMeanStd pools real sensor values across decision observations, once per
new low decision. It never updates from replay samples or duplicated UAV tokens.
A 60-second standard-deviation floor avoids amplifying initial constant values;
normalized values are clipped to +/-10. These are configurable starter settings.

Replay stores raw observations and normalized requested actions in [-1,1]. Both
current and next observations are transformed using the same current statistics
when sampled. Current and next variable lengths have independent padding masks.
Padding and IDs do not enter RunningMeanStd; IDs use the fixed scaling above. The physical environment
continues to perform the agreed component and radial action limiting.

Upper shared actor observation for UAV i, in order:

1. All UAV SOCs (N).
2. Own one-hot identity (N).
3. Own one-hot status (4).
4. Charging and waiting occupancy fractions (2).
5. Own return time divided by frame duration (1).

This is 2N+7 values (19 for six UAVs). The centralized critic receives per-UAV
`[SOC, status_one_hot(4), normalized_return_time]`, followed by the two occupancy
fractions: 6N+2 values (38 for six UAVs). Existing waiting/charging UAVs have zero
return time. The shared actor's independent Bernoulli action is a request, not
guaranteed admission; occupancy is enforced by the existing scheduler.

The upper features use fixed physical scaling, not adaptive RMS. SAC evaluation
uses a frozen copy of RMS. Checkpoints include mean, variance, count, clipping,
scale floor and training flag. Evaluation does not alter training RNG/replay.

## Frame boundary and termination

- Steps 1–9 connect to the next low observation in the same frame.
- Step 10 is retained with its final return-safety reward. After settlement,
  release/promotion and the next upper charging allocation, the first actual
  lower decision observation closes this transition.
- Changing serving identities/count does not terminate the transition.
- Death and the configured finite episode horizon both disable bootstrapping.
  A horizon-ended pending transition is closed terminal before episode reset.
- Empty terminal next sets are legal. They are not passed through attention or
  target networks; bootstrap is evaluated only for nonterminal samples.

The optional `LowFrameRunner.on_step` observer runs after the final-return reward
is applied, or after a mid-frame death is detected. Without a callback, simulation
behavior remains unchanged. It lets the trainer record transitions without
moving training code into Scene or duplicating the upper execution path.

## Algorithms and numerical choices

SAC uses an identity-aware UAV Transformer actor, two separate attention Q networks,
Polyak target critics, normalized tanh-Gaussian actions and automatic entropy
temperature. Log-probabilities sum over valid action dimensions only; the target
entropy is -2 times the current number of active UAVs. The critic pools valid
tokens and also receives active-count/fleet-size derived from the mask. Separate
global last-visit MLPs mean the sensor count is fixed for a checkpoint, even though
the active UAV count can change.

MAPPO uses a shared Bernoulli actor, scalar centralized team value, GAE, clipped
per-agent policy ratios, optional value clipping (disabled by default) and normalized advantages. Actor
samples share the team advantage; critic samples are counted once per frame.
There is no independent per-agent reward or occupancy action masking.

Raw evaluation backlog metrics remain unchanged. Training uses a
configurable overall reward multiplier (1.0 for each level) to set optimization
scale. This is not observation RMS and does not replace reward-weight selection.
The upper backlog cost is linear and uncapped; see the reward section below. Network widths, rates, entropy and optimizer
settings are starter values, not validated research hyperparameters.

Losses and gradients must be finite. Nonfinite values raise errors rather than
being silently replaced with zeros. Gradient norms are clipped. Replay retains
data across upper-policy changes; this implements the requested joint schedule
but does not eliminate its nonstationarity. Update counts alone are not evidence of convergence.

## Outputs and recovery

- `resolved_config.json`: exact configuration and runtime device.
- `train.jsonl`: one row per upper step, including raw reward, deaths, separate
  step/update counters, phase, pending status and latest losses.
- `checkpoint.pt`: both models and targets, all optimizers, temperature, RMS,
  replay and its sampler RNG, pending transition, partial PPO rollout, physical
  scene and RNG, current trace, counters, and Torch CPU/CUDA RNG state.
- `eval_*/seed_*/`: deterministic evaluation summary, trace, dashboard and heatmaps.

By default checkpoint/evaluation occur every 10 MAPPO updates; normal CLI exit
also saves and evaluates. A checkpoint writes to a temporary file then replaces
the target. Interruption mid-frame keeps the previous completed checkpoint
instead of saving an inconsistent frame. Only load trusted project-generated
checkpoints: this full-state format uses Python serialization. GPU recovery
restores state, but cross-device bit-for-bit numerical identity is not promised.

## Validation and references

Behavioral tests cover row-permutation behavior with attached identity, padding exclusion, shared
RMS, independently sized next observations, empty terminal sets, cross-frame
allocation, terminal/horizon targets, GAE reset boundaries, constant per-transition
update counts, CPU checkpoint continuation and read-only seeded evaluation.
Run `python -m pytest -q` when pytest is installed. In the current environment
the same test functions were invoked directly because pytest is absent.

Algorithm references:

- SAC: https://arxiv.org/abs/1812.05905
- MAPPO: https://arxiv.org/abs/2103.01955
- Running statistics reference: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_normalize.html

Short CPU/CUDA runs validate execution and recovery, not convergence, safety of
a learned policy, or performance relative to research baselines.

## Evaluation diagnostics (2026-09-16)

A normal evaluation targets `episode_upper_frames` (default 100), or 1000 lower
steps. Any UAV death still ends that episode immediately; evaluation does not
hide failure by resetting or forcing dead UAVs to continue. A smoke checkpoint
retains its shorter horizon and network configuration.

`summary.json` now includes configured and actual lengths, termination reason,
charge requests, charging/waiting assignments, actual charging energy by UAV,
and the fraction of sensors visited at least once. `policy_trace.json` records
per-frame request probabilities, selected actions, allocated roles, start/end
SOC, return cost, actual charging energy and upper reward components. Requests,
allocated charging roles and positive energy received are distinct quantities.
Evaluation dashboards plot the entire recorded episode (previously only the
last six frames). Interactive rendering still defaults to a six-frame trail.

The September audit found identical [x,y,SOC] inputs forced identical deterministic
ID-free actions. New runs now enable `sac.include_uav_id: true`. Old checkpoints
without that flag retain their three-feature architecture and emit a warning;
resuming them does not apply the identity fix. Start a fresh run for four-feature
models. This removes forced symmetry but does not itself prove learned coverage.

The confirmed September 22 reward and exploration settings are listed below.

## Live training curves

`training_curves.png` in the run output directory is overwritten every 100 upper
frames by default, and on normal completion or keyboard interruption. Set
`plot_every_frames` in the training YAML or pass `--plot-every 500` to change the
interval; `--plot-every 0` disables plotting. The CLI override works on resume.
Rendering uses the noninteractive Agg canvas and requires no server display.

The four panels show completed-episode upper reward sum, time-average maximum
buffer after collection, worst maximum buffer, and actual lower-step length
with death terminations marked. Backlog statistics use all post-service lower
steps in the episode and exclude cold start. Raw values and a 10-episode moving
average are drawn. Unfinished episodes are not plotted as finished returns.

`training_curves.json` stores the compact episode history; `train.jsonl` remains
the full frame log. Resume reloads history up to the checkpoint frame and imports
history from the checkpoint's directory when using a new output directory.
Older logs can recover episode returns, but missing full-episode backlog values
are left blank rather than estimated from frame-end values. These plots are
training curves; seed-specific evaluation dashboards remain separate.

## Radial sensor map (2026-09-18)

New runs use 150 sensors on the 4000m map: exactly 100 inside the airship-centered
1800m circle, and 50 outside in the 2000-2400m annulus. Seven clusters comprise
five inner groups and two outer groups. There are 141 Gaussian cluster points
(std 100m, within 300m of a center) and 9 independent scattered points. The
seed-42 point corrections are described below.
Rejection sampling preserves radial quotas and map boundaries without clipping.
The outer minimum exceeds 1800m travel plus 173.2m coverage, making a first-frame
visit from the airship impossible for every outer sensor.

Run `python run_sensor_distribution.py` to regenerate the actual configured map,
its dashed 1800m circle, CSV coordinates and summary under
`outputs/sensor_distribution`. Use `--output` or `--config` to override paths.
The map is reproducible from `map_seed`. Configurations lacking the new
`distribution` field retain the original `cluster_uniform` generator, so old
checkpoints preserve their map. Start a new training run for the 150-sensor input
dimension; resuming a 320-sensor checkpoint does not apply this new map.


## Reward and exploration revision (2026-09-21)

Upper reward uses the mean post-service maximum backlog over executed lower
steps, divided by 6000 without clipping. Lower-only return failures have no
upper death penalty. See the confirmed reward formulas below. Compare raw
backlog, coverage and survival alongside reward; convergence is not established.

Warm-up lasts 10000 low decisions; learning starts at 10000 with enough
replay samples. warmup_mode=waypoint keeps a random target across decisions,
with initial angular sectors assigned by UAV ID and a random episode rotation.
Targets are sampled 0.25-0.65 map widths from the airship, restricted to the map.
After arrival another random target is selected. 80% of actions head toward the
target at 80-100% step speed; 20% sample a random disk action. Map-boundary
clipping keeps warm-up endpoints inside the map. No sensor positions, backlog,
future arrivals, energy oracle or learned-policy action replacement is used.
Targets are reset after episode reset or departure from service, and saved in
checkpoints for exact resumption. Evaluation always uses the SAC actor.

New training should start in a new output directory WITHOUT --resume. Old
checkpoints retain their saved reward configuration, replay rewards, and model
architecture. They are not migrated by loading new YAML. Their retired frequency
switch is ignored and their former early rate becomes constant; a warning is
printed. Reusing old replay with a new reward is intentionally unsupported.

## Approved sensor map

The package includes the previously approved 150-sensor map: 100 inside 1800m
(94 clustered, 6 independent) and 50 outside (47 clustered, 3 independent).
There are 5 inner and 2 outer clusters. All sensors use the same marker shape.
For map seed 42 only, independent sensor IDs 42 and 106 are placed at (3300,3600)
and (3700,3200), preserving the other 148 positions and the outer radial region.


## Confirmed reward and exploration revision (2026-09-22)

New-run defaults:
- Upper: `S - mean_step_max_backlog/6000 - 0.05*C - 0.10*W - 20*upper_deaths`.
  Roles use actual frame assignments. Backlog is linear and uncapped. Lower-only
  return failures terminate the episode but add no upper death/failure penalty.
- Lower: `sum(owned_max_pre)/(N*3000) - 0.1*B/(B+3000) - 0.02*out_of_bounds
  - 30*sum_failed(1 + horizontal_distance/1800)`. N is the fixed fleet size;
  B is global post-service maximum backlog. Return costs apply on the final step.
- Both reward scales are 1. MAPPO value clipping is disabled; policy clip stays .2.
- MAPPO actor LR .0001, critic LR .0003; gamma .99 and GAE .95 unchanged.
  Entropy coefficient is .05 through cumulative upper frame 10000, linear to
  .005 at frame 100000, then constant. Logs include `mappo_entropy_coef` at updates.
  Resume uses the saved frame counter; no KL early stopping is introduced.
- SAC initial alpha .05, automatic alpha enabled, target entropy -2 per serving
  UAV. LR .0003, batch 256, gamma .99, tau .005, replay capacity 100000 unchanged.
- Warm-up uses persistent random waypoints with 20% random-action mixing for
  10000 lower decisions. SAC learning starts at 10000. No later frequency reduction.
- Episodes end at 100 frames or death. A 512-frame rollout spans resets; GAE stops
  at each terminal boundary. A rollout update itself does not reset the scene.
- New defaults require a fresh run. Resume preserves saved configuration, replay,
  optimizer state and partial rollout; editing YAML does not migrate checkpoints.
