# Identify and package backlog_potential_v5_p8_seed43

Expected entry point: `experiments/lower_backlog_potential_v5/train.py` with
`--cost-mode pnorm --cost-p 8`; saved revision should end in`_pnorm`, not`_max`.
Folder names are labels, so resolved_config and source provenance are authoritative.

```text
cost = (mean((post_collection_backlog_i/2000)^8))^(1/8)
Phi = mean_serving exp(-max(distance_to_assigned_oldest_goal-173.2,0)/1800)
r = -cost + 5*(0.95*Phi(next_replay_state)-Phi(current_state))
    - 0.1*boundary_count
    - 70*sum(return_SOC_shortfall)/0.1
    - sum_responsible_failures(7.5 + 3.75*distance_to_home/1800)
```

P8 is a generalized mean of ALL remaining sensor queues, not a positive reward
for collected data. No fixed oldest-hit or collection-P4 bonus is active. There
is no backlog cap. Phi uses actual successor AFTER dispatch/home reset. Gamma0.95,
seed43, unchanged attention SAC, true energy. Random2-6 serving every10 steps;
UAV full/home reset every20 preserving sensor states; actual fresh sensor/task
reset after300 with the preceding physical successor bootstrapped as a cutoff.
This is DIFFERENT from v6 direct age-distance cost and v7 continuing round task.

Verify actual saved settings, from JV_UAV:

```bash
python3 -c 'import json; c=json.load(open("runs/backlog_potential_v5_p8_seed43/resolved_config.json"))["training"]; print(c["revision"]); print(c["reward"]); print(c["sac"])'
```

Package while training is running, after pulling the new helper:

```bash
python3 scripts/pack_lower_review.py runs/backlog_potential_v5_p8_seed43
```

The ZIP is placed BESIDE the run. It contains logs/config/source provenance,
first/latest/best/intermediate evals and best/latest actor weights, excluding the
large resumable checkpoint and replay. Incomplete append-log tails are ignored;
incomplete files are listed in review_manifest. Packaging does not stop training.

If you have not pulled the new helper yet, the existing fallback works:

```bash
python3 experiments/lower_backlog_potential_v5/pack_results.py runs/backlog_potential_v5_p8_seed43 --recent-evals 6
```

Fallback omits actor weights and may omit important middle/best eval traces.

After results review, resume the SAME run if worthwhile (additional300k steps):

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u experiments/lower_backlog_potential_v5/train.py --device cuda --resume runs/backlog_potential_v5_p8_seed43/checkpoint.pt --steps 300000 --output runs/backlog_potential_v5_p8_seed43
```

Omitted reward flags restore saved P8 settings; do NOT start a new process into an
actively running output folder. Read the archive before deciding to extend. A
drop in maximum backlog alone is not proof all sensors were updated: review will
reconstruct actual union visits and completed renewal rounds from raw traces.
