#!/usr/bin/env bash
set -euo pipefail
# Convert the 10 collected central_wide_cam 10ep collections to LeRobot 224x224.
cd "$(dirname "$0")/.."

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

for task in "${TASKS[@]}"; do
  src="data/robotwin/${task}/${task}_10ep"
  tgt="data/lerobot/${task}_10ep_224"
  if [[ ! -d "${src}" ]]; then
    echo "SKIP ${task}: no collection at ${src}"
    continue
  fi
  if [[ -f "${tgt}/meta/info.json" ]]; then
    echo "SKIP ${task}: already converted at ${tgt}"
    continue
  fi
  echo "Convert ${task} -> 224x224"
  conda run --no-capture-output -n emix python data_processing/convert_to_lerobot.py \
    --src "${src}" --tgt "${tgt}" --repo-id "${task}_10ep_224" \
    --instruction-type seen --image-width 224 --image-height 224 \
    --fps 17 --left-qpos-dim 7
done
echo "All conversions done."
