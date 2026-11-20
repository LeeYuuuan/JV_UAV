# Server comparisons

Run `git pull` once in ~/Code/JV_UAV after committing/pushing local changes.
In three separate tmux windows/panes:

    bash scripts/run_server_comparisons.sh small_greedy_nob
    bash scripts/run_server_comparisons.sh small_actor_nob
    bash scripts/run_server_comparisons.sh large_greedy_long

Physical GPUs are fixed to 1, 2, 3 respectively through CUDA_VISIBLE_DEVICES.
Both small runs: 1 UAV / 5 sensors / 15 steps; independent -B term removed,
positive term still uses post-collection B; reward, gamma .95 and SAC settings match.
Greedy small run uses epsilon mixture (25% random, 50% greedy, 25% actor inside
exploration), oldest candidate_count=1; actor-only small run uses epsilon=0.
Large run: 150 sensors / 6 UAVs / 500-step episode / epsilon decay300k; ORIGINAL
reward including -B; candidate_count=6 with reservation. Budget500k physical steps.
Both small budgets60k physical steps. No automatic resume; output logs are preserved.
Eval is deterministic actor-only; initial reference baselines are separate.
This preparation does not start or stop local/server training.
