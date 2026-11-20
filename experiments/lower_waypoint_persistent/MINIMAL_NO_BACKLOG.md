# Minimal no-B reward ablation

Base: minimal15_greedy (1 UAV, 5 sensors, 15 steps, epsilon mixture, candidate_count=1).
Only learning-task change: reward.backlog_weight=0; revision updated for provenance.
Reward = 0.5 * (post-collection max backlog / 300) * cleared fraction
         - 0.02 * actual movement / (600 * sqrt(2)).
The B inside the positive term remains. Default backlog_weight=1 preserves old
configs and checkpoints. Reward terms still log backlog_post=0 for the ablation.
Start a fresh output directory without --resume; old replay rewards are different.
No training processes were stopped or started while preparing this experiment.
