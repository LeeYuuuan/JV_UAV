# Experiment revisions

## 2026-09-23: stronger lower return-failure costs

- Lower final-return failure: `500 * (1 + distance_m / 1800)` per failed UAV (previously `30 * (1 + distance_m / 1800)`).
- Upper-responsibility death: 20 per UAV, unchanged (the proposed increase to 100 was withdrawn).
- Collection, backlog, boundary, charging and waiting terms are unchanged; learning settings are unchanged.
- Shared responsibility: each lower final-return failure also costs the upper level 15, logged as `lower_return_failure`. Ordinary upper-responsibility deaths still cost 20; the same UAV is not charged both upper terms. Lower penalties remain unchanged. This is a failure-type rule, not a counterfactual causal attribution test.
- Start a fresh run to use the new defaults. Resuming a checkpoint preserves its saved reward configuration and replay; a warning identifies older reward coefficients.
- Suggested run note: `failure-cost-500-upper20-shared15; packet/6 backlog/10; C1 W0.5`.
- Compare horizon completion, return failures, outer sensor visits and backlog, not raw returns across reward revisions.
