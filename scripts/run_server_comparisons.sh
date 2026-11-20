#!/usr/bin/env bash
# Three independent fresh runs. Invoke inside tmux; no existing process is stopped.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
experiment="${1:-}"
case "$experiment" in
  small_greedy_nob)
    gpu=1
    env_config=configs/env_lower_minimal5.yaml
    train_config=configs/training_lower_minimal15_greedy_noB.yaml
    run=runs/server_minimal15_greedy_noB_seed43
    steps=60000
    ;;
  small_actor_nob)
    gpu=2
    env_config=configs/env_lower_minimal5.yaml
    train_config=configs/training_lower_minimal15_actor_noB.yaml
    run=runs/server_minimal15_actor_noB_seed43
    steps=60000
    ;;
  large_greedy_long)
    gpu=3
    env_config=configs/default.yaml
    train_config=configs/training_lower_waypoint_gelu_ep500_slow.yaml
    run=runs/server_large150_ep500_decay300k_seed43
    steps=500000
    ;;
  *)
    echo "Usage: bash scripts/run_server_comparisons.sh {small_greedy_nob|small_actor_nob|large_greedy_long}" >&2
    exit 2
    ;;
esac
# Fail before touching logs if this directory already contains a prior training run.
if [[ -e "$run/checkpoint.pt" || -e "$run/train.jsonl" ]]; then
  echo "Prior run exists: $run. Use train.py --resume explicitly, or choose another output directory." >&2
  exit 2
fi
mkdir -p -- "$run"
echo "Experiment=$experiment | physical GPU=$gpu | training config=$train_config | steps=$steps"
CUDA_VISIBLE_DEVICES="$gpu" python3 -B -u experiments/lower_waypoint_persistent/train.py \
  --env-config "$env_config" --train-config "$train_config" \
  --device cuda --steps "$steps" --output "$run" 2>&1 | tee "$run/console.log"
