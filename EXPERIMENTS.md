# Experiment revisions

## 2026-10-01: lower covered-union p-norm

- Both map configs replace the lower packet-sum reward with the p=4 norm of uniquely collected sensors' pre-service backlogs, divided by 180 for six UAVs. Full-clear nearest-owner collection ensures overlap is counted once. Empty coverage gives zero.
- Keep `-(Bmax/2000)^2`, boundary and final-return penalties unchanged. Upper reward, observations and training settings are unchanged.
- Log the new component as `covered_pnorm`; actual packet throughput remains separately logged. Old total-packet and owned-max checkpoint modes remain supported.
- This norm is not additive across time: splitting a batch across steps can increase the sum of collection bonuses. The retained backlog cost and discount also affect this tradeoff; do not infer policy invariance or minimax optimality from this reward.
- Suggested run note: `lower-union-p4-div180-backlog2000quadratic`; start a fresh run to use the new reward without mixing old replay rewards.

## 2026-09-25: lower reward emphasizing delayed backlog

- Both default and spread-240 environments now use collected packets / 180 and `-(post-service max backlog / 2000)^2`, uncapped.
- Boundary cost stays 5 per UAV; final-return failure stays `500 * (1 + distance_m / 1800)` per UAV. Mid-service depletion has no added lower death penalty; upper responsibility remains 20 per UAV. Upper shared final-return failure remains 15 per UAV.
- The quadratic cost is driven by backlog, not episode time. Example costs at 300/3000/6000/12000 packets: 0.0225/2.25/9/36.
- Suggested run note: `packet180-backlog2000quadratic-return500-upper20shared15`.
- Fresh runs use these defaults; checkpoint resumes preserve saved reward/replay. Previous entries below describe earlier experiment revisions.

## 2026-09-25: upper arrival-SOC observation

- New runs use `mappo.observation_mode: arrival_soc`: actor inputs are all UAV arrival SOCs, own one-hot ID, own status and C/W occupancy (18 dimensions for six UAVs). Critic inputs are all arrival SOCs/statuses and occupancy (32 dimensions).
- Arrival SOC is current SOC minus horizontal plus climb return energy. Already waiting/charging UAVs have zero return cost. Negative values remain visible. No position or return-time feature is supplied.
- Old checkpoints without this flag retain their original 19/38-dimensional current-SOC/return-time inputs and warn on load. Start a new run to use the new observation.
- This change does not alter reward, scheduling, lower observations or entropy settings.

## 2026-09-25: dispersed 240-sensor map (separate environment config)

- Use `--env-config configs/sensors_spread_240.yaml` for a fresh training run.
- 240 sensors: 216 inside radius 1800 m, 24 outside in the 1850–2200 m annulus intersected with the square map.
- 135 independent scatter points (132 inside, 3 outside), uniformly sampled within each allowed region. 105 clustered points: three inner clusters of 28 and two outer clusters of 11 and 10. Cluster Gaussian standard deviation 220 m, rejected beyond 550 m from the center or outside the region/map. All outer sensors must be at least 300 m from the square boundary (`outer_sensor_margin_m`). Inner positions for seed 42 remain unchanged from the first 240-sensor map; final shuffled IDs may change.
- Seed 42 is reproducible. Legacy per-sensor position overrides are removed from this config. All sensors use the same circular plot marker.
- Per-sensor arrival rate stays 0.5 pkt/s; fleet-wide offered load increases from 75 to 120 pkt/s. This changes both geography and workload. A separate matched-load experiment could use 0.3125 pkt/s per sensor.
- New sensor count changes neural network input dimensions: start a fresh run, not a 150-sensor checkpoint resume. Existing default map remains available in `configs/default.yaml`.

## 2026-09-23: stronger lower return-failure costs

- Lower final-return failure: `500 * (1 + distance_m / 1800)` per failed UAV (previously `30 * (1 + distance_m / 1800)`).
- Upper-responsibility death: 20 per UAV, unchanged (the proposed increase to 100 was withdrawn).
- Collection, backlog and boundary terms are unchanged. Direct upper charging and waiting costs are now both zero (previously 1 and 0.5); role counts remain logged. Serving still earns 5 per assigned UAV.
- MAPPO entropy coefficient is now fixed at 0.05 throughout training; the previous decay to 0.005 is removed for new runs. Other learning settings are unchanged. Old checkpoint resumes preserve their saved entropy settings and warn when these differ.
- Shared responsibility: each lower final-return failure also costs the upper level 15, logged as `lower_return_failure`. Ordinary upper-responsibility deaths still cost 20; the same UAV is not charged both upper terms. Lower penalties remain unchanged. This is a failure-type rule, not a counterfactual causal attribution test.
- Start a fresh run to use the new defaults. Resuming a checkpoint preserves its saved reward configuration and replay; a warning identifies older reward coefficients.
- Suggested run note: `failure-cost-500-upper20-shared15-entropy0.05fixed; packet/6 backlog/10; C0 W0`.
- Compare horizon completion, return failures, outer sensor visits and backlog, not raw returns across reward revisions.

## 2026-09-24: charging diagnostics (no policy changes)

- `charging_frames.jsonl.gz` records every completed training frame and flushes each frame: request IDs, charging/waiting assignment IDs, rejected request IDs, actual positive-charge IDs, per-UAV SOC added, and whether settlement occurred.
- Counts represent UAV-frame occurrences, not unique UAVs or distinct request sessions. Continuing a request counts again on the next frame. Assigned charging is not proof of receiving energy.
- Episode diagnostics and evaluation summaries contain `charging_counts`; evaluation policy traces include the per-frame details. `charging_recorded_frames` shows how many frames were observed; old checkpoint resumes cannot reconstruct missing earlier counters.
- Scheduling remains lowest predicted arrival SOC within priority groups, not FIFO. Upper observations and sensor distribution are unchanged.


## 2026-10-01: upper reward aligned with lower data terms

- New configs use `upper_reward_mode: lower_data_mean`: mean over executed lower steps of covered-union p=4 norm /180 minus (post-service max backlog /2000)^2. This replaces serving-count and linear backlog rewards.
- Normal frames average 10 steps; death-shortened frames average actual executed K. No clipping or time gate. Lower OOB and distance penalties are excluded. Upper death costs remain 20 and shared final-return costs 15 per UAV, without double counting; C/W costs remain zero.
- Reward logs expose `frame_mean_lower_collection` and `frame_mean_lower_backlog`; existing resolved configs and provenance capture the mode. Changes to lower data weights intentionally also affect upper data rewards.
- Upper inputs remain arrival SOC, ID, own status and C/W occupancy. Old checkpoint configs without this reward mode retain serving/backlog reward and warn on resume. Start a fresh run to use this revision.


## 2026-10-02: Paper1 upper actor transfer interface

- Added `--upper-pretrained PATH` for a fresh joint run (mutually exclusive with `--resume`). The supported source is the Paper1 notebook categorical actor: four ReLU hidden layers, two output logits. This differs from the smaller actor in old src/agents/mappo.
- Current arrival-SOC observations are reordered into [own arrival SOC, own status one-hot, C/W ratios, normalized ID, all arrival SOCs]. Old action 0 means request charge; current action 1 means request charge. Bernoulli logit is old logit[0] minus old logit[1].
- Only actor parameters are imported. Critic, optimizers, SAC and replay start fresh. Joint fine-tuning starts normally; no automatic freezing. Architecture and source path/SHA256 persist in resolved configuration and checkpoints; resume requires no source file.
- Source audit: `Paper1_v1/code/checkpoints/charging/mappo_charging_n6/ckpt.pt`, SHA256 e6b8fbb081fbc0f45d2524f8a8aec88683ab884690dd685145e84c5fedf6a798, actually has actor input 15 and critic input 37, consistent with seven UAVs. Six-UAV notebooks report 14/32. Filename is not reliable evidence of fleet size. Loader rejects this file for the current six-UAV environment; no fabricated UAV or dropped column. Matching weights or an explicitly agreed approximate mapping are required before a six-UAV transfer run.
- Validation: 78 tests pass, including feature order, action probability conversion, gradient flow and exact checkpoint resume. For the actual source weights on seven-UAV test inputs, converted charge probabilities differ by at most 5.96e-8. This verifies numerical conversion, not performance on the new environment.


## 2026-11-12: six-UAV transfer configuration and wiring smoke

- Added configs/training_transfer_n6.yaml and TRANSFER_N6.md. Fresh runs require --upper-pretrained, six UAVs and source hidden width512. Actor warm start only; fresh critic/SAC/optimizers; joint fine-tuning with existing learning settings.
- Source architecture matches train_charging_mappo_n6_decay_4000.ipynb: 14 -> 512x4 ReLU -> 2. Input reorder and action reversal preserve charge probabilities for identical SOC values. Arrival SOC intentionally changes the old current-SOC semantics. Upper inputs use no RunningMeanStd.
- 80 tests pass. Additional CLI checks reject missing source and existing seven-UAV weights. Generated RANDOM UNTRAINED weights from actual old notebook classes and ran/resumed smoke: 4 frames,36 lower steps,2 MAPPO and28 SAC updates; actor parameters changed. This is wiring validation, not convergence evidence.
- All new temporary/test artifacts are under D:/VScode/UAV_Joint_Simulation/experiments/transfer_n6_smoke_20261112. Earlier source audit moved to experiments/source_audit_20261112. final_code is reserved; current development stays in JV_UAV.
