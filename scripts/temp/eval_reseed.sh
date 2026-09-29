#!/usr/bin/env bash
# Re-evaluate EMIX checkpoints for tasks where emix < baseline, with fresh
# seeds and 200 episodes (see outputs/eval_reseed/).
#
# Reuses the run.py-generated configs/scripts/run/<task>/emix_eval.yaml and
# overrides episodes/seed/output-dir on the evaluate.py command line. The two
# seeds run 100 episodes each, for 200 new episodes per task in total.
#
# Usage:
#   eval_reseed.sh <cuda_idx> <conda_env> <seed> [<wait_pid>]
#     <cuda_idx>    CUDA_VISIBLE_DEVICES value (0 = RTX 4070, 1 = RTX 5060)
#     <conda_env>   emix-twin (4070) or emix-twin-5060 (5060)
#     <seed>        evaluation seed (1 or 2)
#     <wait_pid>    optional PID to wait for before starting (queue behind a
#                   running flow); pass 0 to skip waiting
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_IDX=$1
CONDA_ENV=$2
SEED=$3
WAIT_PID=${4:-0}

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

echo "[reseed] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES env=$CONDA_ENV seed=$SEED"
if [ "$WAIT_PID" != "0" ]; then
  echo "[reseed] waiting for pid $WAIT_PID (queued behind running program)..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[reseed] pid $WAIT_PID exited; waiting 30s for GPU to drain..."
  sleep 30
fi

for task in "${TASKS[@]}"; do
  config="configs/scripts/run/${task}/emix_eval.yaml"
  out="outputs/eval_reseed/${task}_dp_emix_seed${SEED}"
  echo "[reseed] === $task seed=$SEED episodes=100 -> $out ==="
  /home/rogerspyke/miniforge3/bin/conda run --no-capture-output -n "$CONDA_ENV" \
    python -u evaluation/evaluate.py \
    --config "$config" \
    --episodes 100 \
    --seed "$SEED" \
    --output-dir "$out"
done
echo "[reseed] all runs finished"
