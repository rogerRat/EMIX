#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Resumable opposite-camera pipeline. Existing completed stages are preserved;
# set FORCE_STAGE=collect,convert,index,train,eval (or all) to rerun one stage.
TRAIN_ENV="${TRAIN_ENV:-emix}"
EVAL_ENV="${EVAL_ENV:-emix-twin}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
FORCE_STAGE="${FORCE_STAGE:-}"
DRY_RUN="${DRY_RUN:-0}"

TASK="place_object_scale_left_oppo_cam"
DATASET_ID="${TASK}"
COLLECTION_CONFIG="${TASK}"
EXPECTED_DATASET_EPISODES=500
EXPECTED_EVAL_EPISODES=100
EXPECTED_EVAL_STEP_LIMIT=400

COLLECTION_ROOT="data/robotwin/${TASK}/${COLLECTION_CONFIG}"
LEROBOT_ROOT="data/lerobot/${DATASET_ID}"
EMIX_INDEX_ROOT="data/emix_index/${DATASET_ID}"
TRAIN_DIR="configs/train/oppo_cam"
EVAL_DIR="configs/eval/oppo_cam"
STEP_LIMIT_CONFIG="third_party/RoboTwin/task_config/_eval_step_limit.yml"

RUN_COLLECT=(conda run --no-capture-output -n "${EVAL_ENV}" python data_collection/robotwin.py)
RUN_CONVERT=(conda run --no-capture-output -n "${TRAIN_ENV}" python data_processing/convert_to_lerobot.py)
RUN_BUILD=(conda run --no-capture-output -n "${TRAIN_ENV}" python -m emix build)
RUN_TRAIN=(conda run --no-capture-output -n "${TRAIN_ENV}" python -m embodied_mixup train)
RUN_EVAL=(conda run --no-capture-output -n "${EVAL_ENV}" python -m embodied_mixup eval)

is_forced() {
  [[ ",${FORCE_STAGE}," == *,"$1",* || ",${FORCE_STAGE}," == *,all,* ]]
}

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    return
  fi
  "$@"
}

dataset_is_complete() {
  local info_path="${LEROBOT_ROOT}/meta/info.json"
  [[ -f "${info_path}" ]] && \
    grep -Eq "\"total_episodes\"[[:space:]]*:[[:space:]]*${EXPECTED_DATASET_EPISODES}" "${info_path}" && \
    [[ -f "${LEROBOT_ROOT}/data/chunk-000/file-000.parquet" ]] && \
    [[ -f "${LEROBOT_ROOT}/videos/observation.images.head_camera/chunk-000/file-000.mp4" ]]
}

collection_is_complete() {
  local trajectory_dir="${COLLECTION_ROOT}/_traj_data"
  [[ -d "${trajectory_dir}" ]] && \
    (( $(find "${trajectory_dir}" -maxdepth 1 -type f -name 'episode*.pkl' | wc -l) >= EXPECTED_DATASET_EPISODES ))
}

evaluation_is_complete() {
  local output_dir="$1"
  local metrics_path
  while IFS= read -r -d '' metrics_path; do
    if grep -Eq "\"episodes\"[[:space:]]*:[[:space:]]*${EXPECTED_EVAL_EPISODES}" "${metrics_path}"; then
      return 0
    fi
  done < <(find "${output_dir}" -mindepth 2 -maxdepth 2 -type f -name metrics.json -print0 2>/dev/null)
  return 1
}

require_eval_step_limit() {
  local configured_limit
  configured_limit="$(awk -F: -v task="${TASK}" '$1 == task { gsub(/[[:space:]]/, "", $2); print $2; exit }' "${STEP_LIMIT_CONFIG}")"
  if [[ "${configured_limit}" != "${EXPECTED_EVAL_STEP_LIMIT}" ]]; then
    printf 'Expected %s to set %s to %s steps; found %s. Aborting.\n' \
      "${STEP_LIMIT_CONFIG}" "${TASK}" "${EXPECTED_EVAL_STEP_LIMIT}" "${configured_limit:-missing}" >&2
    exit 1
  fi
}

require_eval_step_limit

if is_forced collect || ! collection_is_complete; then
  echo "[1/13] Collecting ${EXPECTED_DATASET_EPISODES} opposite-camera episodes"
  run "${RUN_COLLECT[@]}" "${TASK}" --task-config "${COLLECTION_CONFIG}" --cuda "${CUDA_DEVICE}" --episodes "${EXPECTED_DATASET_EPISODES}"
else
  echo "[1/13] Collection already complete; skipping"
fi

if is_forced convert || ! dataset_is_complete; then
  echo "[2/13] Converting to LeRobot 84x84"
  run "${RUN_CONVERT[@]}" --src "${COLLECTION_ROOT}" --tgt "${LEROBOT_ROOT}" \
    --repo-id "${DATASET_ID}" --instruction-type seen --image-width 84 --image-height 84 \
    --fps 17 --left-qpos-dim 7
else
  echo "[2/13] LeRobot dataset already complete; skipping"
fi

if is_forced index || [[ ! -f "${EMIX_INDEX_ROOT}/emix_index.json" ]]; then
  echo "[3/13] Building shared EmixIndex"
  run "${RUN_BUILD[@]}" --repo_id "${DATASET_ID}" --root "${LEROBOT_ROOT}" \
    --output_dir "${EMIX_INDEX_ROOT}"
else
  echo "[3/13] Shared EmixIndex already complete; skipping"
fi

run_model() {
  local number="$1"
  local label="$2"
  local train_config="$3"
  local eval_config="$4"
  local model_name="${train_config%.yaml}"
  local checkpoint="outputs/train/${model_name}/checkpoints/030000/pretrained_model/model.safetensors"
  local eval_output="outputs/eval/${model_name}"

  if is_forced train || [[ ! -f "${checkpoint}" ]]; then
    echo "[${number}/13] Training: ${label}"
    run "${RUN_TRAIN[@]}" --config "${TRAIN_DIR}/${train_config}"
  else
    echo "[${number}/13] Training already complete: ${label}; skipping"
  fi

  if is_forced eval || ! evaluation_is_complete "${eval_output}"; then
    echo "[$((number + 1))/13] Evaluation (${EXPECTED_EVAL_STEP_LIMIT}-step limit): ${label}"
    run "${RUN_EVAL[@]}" --config "${EVAL_DIR}/${eval_config}"
  else
    echo "[$((number + 1))/13] Evaluation already complete: ${label}; skipping"
  fi
}

run_model 4 "DP Baseline (84x84)" place_object_scale_left_oppo_cam_dp_baseline.yaml \
  place_object_scale_left_oppo_cam_dp_baseline.yaml
run_model 6 "EMIX Linear 100->0 (84x84)" place_object_scale_left_oppo_cam_dp_emix_linear.yaml \
  place_object_scale_left_oppo_cam_dp_emix_linear.yaml
run_model 8 "EMIX Log 100->0 (84x84)" place_object_scale_left_oppo_cam_dp_emix_log.yaml \
  place_object_scale_left_oppo_cam_dp_emix_log.yaml
run_model 10 "EMIX Step first half 50% (84x84)" place_object_scale_left_oppo_cam_dp_emix_step_first_half.yaml \
  place_object_scale_left_oppo_cam_dp_emix_step_first_half.yaml
run_model 12 "EMIX Step full training 50% (84x84)" place_object_scale_left_oppo_cam_dp_emix_step_constant_half.yaml \
  place_object_scale_left_oppo_cam_dp_emix_step_constant_half.yaml

echo "Opposite-camera pipeline completed."
