#!/usr/bin/env bash
set -euo pipefail
# Collect 10 episodes for the remaining 9 central_wide_cam variants, running
# at most PARALLEL jobs concurrently with a per-job GPU assignment.
# Usage: collect_rest_wide.sh <parallel> <gpu0> <gpu1>
cd "$(dirname "$0")/.."
PARALLEL="${1:-4}"
GPU0="${2:-0}"
GPU1="${3:-1}"
LOG_DIR="$(pwd)/logs"
PYTHON=/home/rogerspyke/miniforge3/envs/emix-twin/bin/python
mkdir -p "${LOG_DIR}"

TASKS=(
  blocks_ranking_size_left_central_wide_cam
  place_bread_basket_left_central_wide_cam
  place_bread_skillet_left_central_wide_cam
  place_burger_fries_left_central_wide_cam
  place_can_basket_left_central_wide_cam
  place_cans_plasticbox_left_central_wide_cam
  place_dual_shoes_left_central_wide_cam
  stack_blocks_two_left_central_wide_cam
  stack_bowls_two_left_central_wide_cam
)

# Emit "task gpu" lines, then run each through a fully-inlined bash -c so no
# function/var export is needed in the xargs child.
idx=0
for task in "${TASKS[@]}"; do
  if [[ -d "data/robotwin/${task}/${task}_10ep" ]]; then
    echo "SKIP ${task}: collection already exists" >&2
    continue
  fi
  gpu=$(( idx % 2 == 0 ? GPU0 : GPU1 ))
  echo "${task} ${gpu}"
  idx=$((idx + 1))
done | xargs -P "${PARALLEL}" -n 2 bash -c '
  task="$0"; gpu="$1"
  echo "[collect] ${task} on cuda:${gpu}"
  '"${PYTHON}"' data_collection/robotwin.py "${task}" \
    --task-config "${task}_10ep" --cuda "${gpu}" \
    > "'"${LOG_DIR}"'/collect_${task}.log" 2>&1
'

echo "All remaining collections done."
