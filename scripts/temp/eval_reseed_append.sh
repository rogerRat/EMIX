#!/usr/bin/env bash
# Append baseline (or emix) re-seed evaluations behind an already-running
# eval_reseed.sh queue. Same protocol as eval_reseed.sh (100 episodes, fresh
# seed), but parameterized by policy kind and chained on a predecessor PID so
# it never runs concurrently with the queue in front of it.
#
# Usage:
#   eval_reseed_append.sh <cuda_idx> <conda_env> <kind> <seed> [<wait_pid>]
#     <kind>      emix | baseline (selects the flow runtime config)
#     <wait_pid>  optional PID to wait for before starting; pass 0 to skip
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_IDX=$1
CONDA_ENV=$2
KIND=$3
SEED=$4
WAIT_PID=${5:-0}

TASKS=(
  blocks_ranking_rgb_left_central_wide_cam
  place_bread_basket_left_central_wide_cam
  place_burger_fries_left_central_wide_cam
)

# Mirror run.yaml environments.per_gpu.*.strip_ld_library_path: stale
# /usr/local/cuda entries break the cuBLASLt lookup on both cards.
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/usr/local/cuda$' | paste -sd: -)
export LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="$CUDA_IDX"
export PYTHONUNBUFFERED=1

echo "[reseed+$KIND] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES env=$CONDA_ENV seed=$SEED"
if [ "$WAIT_PID" != "0" ]; then
  echo "[reseed+$KIND] waiting for predecessor pid $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[reseed+$KIND] predecessor pid $WAIT_PID exited; waiting 30s to drain..."
  sleep 30
fi

for task in "${TASKS[@]}"; do
  config="configs/scripts/run/${task}/${KIND}_eval.yaml"
  out="outputs/eval_reseed/${task}_dp_${KIND}_seed${SEED}"
  echo "[reseed+$KIND] === $task seed=$SEED episodes=100 -> $out ==="
  /home/rogerspyke/miniforge3/bin/conda run --no-capture-output -n "$CONDA_ENV" \
    python -u evaluation/evaluate.py \
    --config "$config" \
    --episodes 100 \
    --seed "$SEED" \
    --output-dir "$out"
done
echo "[reseed+$KIND] all runs finished"
