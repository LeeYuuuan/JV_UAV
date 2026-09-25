# Experiment revisions

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
