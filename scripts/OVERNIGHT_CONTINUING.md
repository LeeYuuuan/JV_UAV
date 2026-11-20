# Overnight continuing-task runs

After local commit/push, server: cd ~/Code/JV_UAV; git pull.
In separate tmux sessions, choose free physical GPUs:

    bash ~/Code/JV_UAV/scripts/run_overnight_continuing.sh small 1
    bash ~/Code/JV_UAV/scripts/run_overnight_continuing.sh large 2

Small: same 1-UAV/6-sensor local continuing task, 180000 physical steps (3x60k),
15-step statistics windows, epsilon decay100k, candidate_count1, eval/save every5k.
Large: original 150-sensor map with6UAVs, 900000 physical steps (15x60k),
500-step statistics windows, epsilon decay300k, eval every20k, save every5k.
Large keeps real-speed persistent waypoints until all arrive or groups change;
service groups still use20/10 physical slots. Greedy processes active IDs in order,
top6 oldest unreserved sensors then nearest, reserving covered footprints.

Both: continuing task, no reset/terminal/bootstrap cutoff at statistics boundaries
or training budget; original top18 reward, rho.8, backlog_scale300, gamma.95,
active-state input and GELU. Inside epsilon branch25%random/50%greedy/25%actor.
Eval only deterministic actor, one40-step rollout split into01_20 and21_40 images.
Training budget and statistics windows are not physical task episode horizons.
Outputs: runs/continuing_small6_greedy_rho08_180k_seed43 and
runs/continuing_large150_uav6_top18_rho08_900k_seed43; console.log is captured.
Start fresh; script rejects existing outputs instead of overwriting them.
