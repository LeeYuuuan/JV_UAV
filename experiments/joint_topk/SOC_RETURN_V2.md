# Joint v2: separate current SOC and return cost

Revision: joint_soc_return_top20_entropy001_v2. New configuration:
configs/training_joint_topk_soc_v2.yaml. This is now the CLI default for fresh
runs. The previous configs/training_joint_topk.yaml remains unchanged.

## Changes

- Upper input mode current_soc_return_soc: all six actual current SOCs, all six
  actual return-energy fractions, own ID, own one-hot status and C/W occupancy.
  Return cost is zero for UAVs already waiting/charging at the airship. It is
  calculated by the same return-energy model used by the simulator.
- Raw actor observation24 values is adapted to20 network features: own SOC,
  status4, occupancy2, ID/5, all current SOC6, all return costs6. The previous
  network used14 features. Critic38 features: old32 current-SOC/status/occupancy
  values plus six return costs. Both energy inputs stay in SOC fractions, no
  upper RunningMeanStd, position input or additional clipping.
- Fixed upper entropy .01, increased from .001; gamma=.95, GAE lambda=.95.
- Lower Top-K20 instead of5, rho=.5, scale300, offset0. Neither the old p-norm
  bonus nor an extra collection bonus is added. Other rewards/penalties,
  physical parameters, map, seeds, PPO1024/128/fourepochs, SAC gamma=.99,
  phased update schedule and evaluation intervals remain unchanged.
- New upper_credit.jsonl.gz records each actual PPO rollout transition once:
  value, GAE target, raw/normalized advantage, terminal flag, current SOC,
  allocated roles, unsafe-serving threshold, actions, request probability before
  and after the complete PPO update, and its update/frame number. Logging does
  not consume RNG or change optimizer updates. pack_results.py includes it.
- Charging logs and full evaluation policy traces also retain the exact SOC
  values fed to the actor, with current and return-cost arrays separately named.

## Start on server

Commit/push locally, then:

```bash
cd ~/Code/JV_UAV
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=1 python3 -u train_joint.py --train-config configs/training_joint_topk_soc_v2.yaml --device cuda --output ./runs/joint_soc_return_top20_e001_s43_v2
```

This new input architecture requires a fresh run. `--resume` continues the
saved checkpoint configuration/architecture, so resuming v1 retains its old
arrival-margin input and K5. It does not silently convert to v2. Existing
arrival-SOC and legacy time-input checkpoints remain loadable.
Optional `--upper-pretrained` accepts the original six-UAV Paper1 actor and
appends six zero-weight return-cost columns. Original current-SOC inputs keep
their original meaning, so the transferred action probabilities are preserved
initially. The critic/lower policy still start fresh.

## Data checks supporting these choices

Previous gamma .95/lambda .95/fixed entropy .01 runs, seeds43..47, all4000updates:
last100 evaluation serving totals averaged344.77,352.00,350.66,351.99,348.25.
Survival fractions were .97,1,1,1,1. These support trying .01; they do not prove
that it will also solve JV's different physical scheduling problem.

Current joint snapshot lower return behavior is partially improved, not solved:
first100 seed1001 evaluations had92 return-failing UAVs across40 episodes,
mean failed distance2351m. Last100 had92 return-failing UAVs across92 episodes,
mean distance1141m. Mid-frame deaths changed from60 to8episodes. Therefore the
failure distance became smaller, but reliable return was not learned. Different
upper schedules confound attributing these changes solely to the lower policy.
Training first1000 versus last1000episodes: return failures .357 to.251/UAVs
per episode, mean failed distance2102m to1455m, boundary violations1.794 to.031
per episode. Survival remained short throughout.

Replayed the recorded186000/seed1001 full evaluation (30lowersteps), preserving
all UAV displacements and upper actions, with the original arrival seed.
Final backlog matched1004 exactly and K5 bonus matched. Number of steps with
positive Top-K feedback was1 forK5,7 forK20,8 forK30,9 forK100. Total bonuses
were .119,.368,.313,.378, respectively. Larger K increases eligible sensors but
also the denominator, so bonus magnitude does not grow monotonically. This is
one recorded trajectory, not a learning comparison; K20 is a conservative test.

## Validation

90 CPU regression tests passed, including separate SOC semantics and scales,
formerly aliased states,20/38network shapes, zero-column Paper1 migration,
exact checkpoint continuation and passive credit logging. Fresh CLI smoke,
optional transfer CLI smoke and diagnostic ZIP packing passed. No convergence
claim is made by these short tests. No persistent experiment was started.
