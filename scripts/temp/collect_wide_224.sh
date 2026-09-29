#!/usr/bin/env bash
set -euo pipefail
# Collect 10 episodes each for the 10 active central_wide_cam variants
# (hanging_mug is DEPRECATED and excluded), then convert to 224x224.
# Usage: collect_wide_224.sh <gpu0> <gpu1>   (defaults 0 1)
cd "$(dirname "$0")/.."
GPU0="${1:-0}"
GPU1="${2:-1}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

TASKS=(
  blocks_ranking_rgb_left_central_wide_cam
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

echo "Starting collection of ${#TASKS[@]} central_wide_cam variants (10ep each)."
pids=()
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  gpu=$(( i % 2 == 0 ? GPU0 : GPU1 ))
  echo "[$i] collecting ${task} on cuda:${gpu}"
  nohup conda run --no-capture-output -n emix-twin \
    python data_collection/robotwin.py "${task}" \
    --task-config "${task}_10ep" --cuda "${gpu}" \
    > "${LOG_DIR}/collect_${task}.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done
echo "Collection done. ${failed} background jobs failed."
exit $failed
