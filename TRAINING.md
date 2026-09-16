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

# Default full training: 500000 upper frames (5000 full-length episodes).
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
`--smoke` changes network sizes, horizon, warm-up and update threshold only in
memory. It never rewrites the default YAML files.

## Training budget

The default new run collects 500000 upper frames. Each episode lasts at most
100 frames, with 10 lower steps per complete frame. This is equivalent to 5000
full-length episodes and at most 5000000 lower steps. UAV deaths end episodes
early, so the actual episode count can be higher. The frame budget stays fixed
across episode resets and allows 5000 MAPPO update calls from a fresh run.

This is an initial training budget, not a convergence guarantee. Compare the
fixed-seed evaluation returns, backlog and survival across checkpoints before
deciding whether more training is useful. The SAC frequency switch remains at
250000 successful SAC updates. Evaluation/checkpoint cadence is unchanged.

An already running process keeps its loaded settings. Old checkpoints also
retain their saved budget; changing the YAML does not override resume settings.
To continue to a cumulative target of 500000 frames, pass `--upper-steps` equal
to 500000 minus the checkpoint's `upper_steps` counter.

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
- SAC initially performs one minibatch update per finalized low transition once
  warm-up and batch availability permit. A final frame transition can be delayed
  until the next allocation; it still contributes at most one default early
  update opportunity.
- Default warm-up: 1000 random low actions and `learning_starts=1000`; minibatch
  size 256. Skipped updates do not advance `sac_updates`.
- MAPPO collects 100 upper decisions, across episode resets if needed, then
  performs one PPO update call. One call uses 5 epochs of minibatches containing
  25 frames each, i.e. 20 actor and 20 critic optimizer steps by default.
- After **250000 successful SAC update calls**, early per-transition SAC updates
  stop. Each MAPPO update call is then followed by **one SAC minibatch update**.
- If the threshold is reached inside a MAPPO rollout, the early updates stop
  immediately; the next completed MAPPO update is followed by the first late SAC
  update. The 250000th early update and that late update can occur in the same
  upper rollout.
- A run stopping before 100 upper decisions preserves its partial rollout in
  the checkpoint; it does not silently perform an extra partial PPO update.

Environment steps, SAC update calls, MAPPO update calls and PPO minibatches are
separate logged counters. Upper steps are frame decisions; an early terminal
frame can execute fewer than 10 low steps.

## Observation and normalization contract

Lower actor tokens contain only `[x/world_size, y/world_size, SOC]`. Integer IDs
stay in replay/action metadata and never enter the network. Global input is the
sensor last-visit vector; no sensor backlog or positions are added. A single
RunningMeanStd pools real sensor values across decision observations, once per
new low decision. It never updates from replay samples or duplicated UAV tokens.
A 60-second standard-deviation floor avoids amplifying initial constant values;
normalized values are clipped to +/-10. These are configurable starter settings.

Replay stores raw observations and normalized requested actions in [-1,1]. Both
current and next observations are transformed using the same current statistics
when sampled. Current and next variable lengths have independent padding masks.
Padding and integer IDs do not enter normalization. The physical environment
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

SAC uses an ID-free UAV Transformer actor, two separate attention Q networks,
Polyak target critics, normalized tanh-Gaussian actions and automatic entropy
temperature. Log-probabilities sum over valid action dimensions only; the target
entropy is -2 times the current number of active UAVs. The critic pools valid
tokens and also receives active-count/fleet-size derived from the mask. Separate
global last-visit MLPs mean the sensor count is fixed for a checkpoint, even though
the active UAV count can change.

MAPPO uses a shared Bernoulli actor, scalar centralized team value, GAE, clipped
per-agent policy ratios, clipped value loss and normalized advantages. Actor
samples share the team advantage; critic samples are counted once per frame.
There is no independent per-agent reward or occupancy action masking.

Environment rewards and raw evaluation metrics are unchanged. Training uses a
configurable overall reward multiplier (0.01 for each level) to set optimization
scale. This is not observation RMS and does not replace reward-weight selection.
The upper backlog weight in `configs/default.yaml` remains its provisional 1.0;
the user has not finalized it. Network widths, rates, entropy and optimizer
settings are starter values, not validated research hyperparameters.

Losses and gradients must be finite. Nonfinite values raise errors rather than
being silently replaced with zeros. Gradient norms are clipped. Replay retains
data across upper-policy changes; this implements the requested joint schedule
but does not eliminate its nonstationarity. Monitor learning before treating
250000 updates as evidence of convergence.

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

Behavioral tests cover ID-free permutation behavior, padding exclusion, shared
RMS, independently sized next observations, empty terminal sets, cross-frame
allocation, terminal/horizon targets, GAE reset boundaries, exact early/late
schedule counts, CPU checkpoint continuation and read-only seeded evaluation.
Run `python -m pytest -q` when pytest is installed. In the current environment
the same test functions were invoked directly because pytest is absent.

Algorithm references:

- SAC: https://arxiv.org/abs/1812.05905
- MAPPO: https://arxiv.org/abs/2103.01955
- Running statistics reference: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_normalize.html

Short CPU/CUDA runs validate execution and recovery, not convergence, safety of
a learned policy, or performance relative to research baselines.
