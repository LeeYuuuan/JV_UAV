#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
experiment="${1:-}"
gpu="${2:-}"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/run_overnight_continuing.sh {small|large} GPU_NUMBER" >&2
  exit 2
fi
case "$experiment" in
  small)
    env_config=configs/env_lower_continuing6.yaml
    train_config=configs/training_continuing_small_overnight.yaml
    steps=180000
    run=runs/continuing_small6_greedy_rho08_180k_seed43
    ;;
  large)
    env_config=configs/default.yaml
    train_config=configs/training_continuing_large_overnight.yaml
    steps=900000
    run=runs/continuing_large150_uav6_top18_rho08_900k_seed43
    ;;
  *) echo "Choose small or large" >&2; exit 2 ;;
esac
if [[ -e "$run/checkpoint.pt" || -e "$run/train.jsonl" ]]; then
  echo "Prior run exists: $run. Resume explicitly or choose a different output." >&2
  exit 2
fi
mkdir -p -- "$run"
echo "Experiment=$experiment | physical GPU=$gpu | total physical steps=$steps"
CUDA_VISIBLE_DEVICES="$gpu" python3 -B -u experiments/lower_waypoint_persistent/train_continuing.py \
  --env-config "$env_config" --train-config "$train_config" \
  --device cuda --steps "$steps" --output "$run" 2>&1 | tee "$run/console.log"
