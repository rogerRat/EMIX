#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Pilot only: collect and convert ten episodes. It deliberately does not build
# an index, train, or evaluate a model.

TRAIN_ENV="${TRAIN_ENV:-emix}"
EVAL_ENV="${EVAL_ENV:-emix-twin}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

TASK="place_object_scale_left_oppo_cam"
COLLECTION_CONFIG="place_object_scale_left_oppo_cam_10ep"
DATASET_ID="place_object_scale_left_oppo_cam_10ep_84"
COLLECTION_ROOT="data/robotwin/${TASK}/${COLLECTION_CONFIG}"
LEROBOT_ROOT="data/lerobot/${DATASET_ID}"

rm -rf data outputs

conda run --no-capture-output -n "${EVAL_ENV}" python data_collection/robotwin.py \
  "${TASK}" --task-config "${COLLECTION_CONFIG}" --cuda "${CUDA_DEVICE}" --episodes 10

conda run --no-capture-output -n "${TRAIN_ENV}" python data_processing/convert_to_lerobot.py \
  --src "${COLLECTION_ROOT}" --tgt "${LEROBOT_ROOT}" --repo-id "${DATASET_ID}" \
  --instruction-type seen --image-width 84 --image-height 84 --fps 17 --left-qpos-dim 7
