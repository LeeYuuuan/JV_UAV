# Archived V5 P8: gamma0.99 / energy comparison

This independent experiment is restored from the actual source archived with
`backlog_potential_v5_p8_seed43`. It is not based on V7 or V8. Original V5 and
all previous outputs remain untouched. Fresh runs use checkpoint format
`displacement_v5_p8_energy_ablation` and default gamma0.99 in both modes.

## Baseline fidelity

`config.yaml` comes from the P8 run's actual resolved TRAINING configuration,
not the old source template (which defaulted to max). Only gamma.95->.99 and
energy-mode/identification metadata are changed. Cost remains p-mean exponent8,
reference2000; potential beta5, exponential distance scale1800m; boundary.1;
real-mode shortfall penalty70 per.1SOC; failure7.5+3.75*distance/1800 unchanged.
Potential uses the same gamma.99 as SAC, preserving its discounted form.

`agent.py` is byte-identical to the archived V5 network/normalization source:
linear age/6000 on all age paths,150 global sensor ages,14-feature UAV tokens,
goal-context x/y/age, and20-step phase. Global-oldest distinct goal assignment,
nearest within ties, is preserved. No round/pending mask is added.

All150 sensor positions/map seed42, dynamics/training seed43, real speed/time,
movement cap180m, coverage173.2m,6-UAV fleet,10-step random2–6 service dispatch,
20-step full/home reset, original energy model and unlimited charging slots
match V5. Batch256, LR3e-4, replay100k, automatic alpha initial.05, Polyak.005,
exploration schedule and update-every-step after1024 are unchanged.

Every300 lower steps the sensor backlog, age and UAVs REALLY reset to a fresh
episode. Replay stores the physical successor before that reset, done=False
for truncation. Only the cumulative training/update counters continue. The
task is not a single never-restarted sensor world over the whole experiment.

`BASELINE_MANIFEST.json` retains the actual P8 configuration and archived source
hashes. It is included in source provenance ZIPs. Differential tests compare
the new real branch with the archived physical environment over600 fixed-action
steps across two300-step episodes: rewards, observations, SOC, queues and
positions match exactly when both use the same gamma.

## Modes

- `--energy-mode real`: complete V5 real consumption/charging, repeated linear
  shortfall cost and one-time failure/distance penalty.
- `--energy-mode none`: ALL SOC inputs fixed1; skip serving/return consumption,
  depletion, charging delay and ALL energy shortfall/failure rewards. Keep real
  service/return motion and climb/horizontal travel time. Return arrival becomes
  idle/full. All non-energy reward/physics/dispatch rules match the real group.

Both modes share identical initial actor/critic parameters for seed43. Future
actual dispatch availability may differ because charging/depletion are disabled.
Mode is saved in resolved config, traces, episode summaries and checkpoint;
changing it on resume is rejected. Old V5/V7/V8 checkpoints are rejected.

## Server launch from JV_UAV root

Commit/push locally and pull on the server. Use separate tmux sessions and idle
GPUs; examples below use0 and1.840000 physical steps is approximately the prior
12-hour V5 budget, not a wall-clock limit. The no-energy mode may run faster.

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_v5_p8_energy_ablation/train.py --energy-mode real --device cuda --seed 43 --steps 840000 --output runs/v5_p8_g099_real_seed43
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_v5_p8_energy_ablation/train.py --energy-mode none --device cuda --seed 43 --steps 840000 --output runs/v5_p8_g099_none_seed43
```

All saved curves/configs carry this new revision. Do not reuse old result folders.
Resume an ADDITIONAL budget with saved settings:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/lower_v5_p8_energy_ablation/train.py --device cuda --resume runs/v5_p8_g099_real_seed43/checkpoint.pt --steps 300000 --output runs/v5_p8_g099_real_seed43
```

## Outputs

V5 logging/evaluation cadence preserved: progress/loss every100 training steps,
episode report every300, checkpoint/eval every3000. Each eval300 physical steps
on1001/1002/1003, pure actor mean. All-UAV trajectories, full backlog/reward
components, training/eval curves and actor/critic losses are saved incrementally.
Shared Scene reward fields in the environment config are not used; the active
standalone reward is training.reward.

```bash
python3 scripts/pack_lower_review.py runs/v5_p8_g099_real_seed43
python3 scripts/pack_lower_review.py runs/v5_p8_g099_none_seed43
python3 scripts/pack_critic_review.py runs/v5_p8_g099_real_seed43 runs/v5_p8_g099_none_seed43
```

`python3 experiments/lower_v5_p8_energy_ablation/verify.py` checks archived V5
physical/reward/SAC invariants. Differential baseline/no-energy/config-equality
checks and short training/resume/packaging integration were run during deployment.
They establish implementation consistency, not convergence.
