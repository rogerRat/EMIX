#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Reuse the completed 84x84 opposite-camera dataset. This script never
# collects, converts, or deletes data; it only builds a missing index, trains,
# and evaluates the three 30%-start EMIX schedules.
TRAIN_ENV="${TRAIN_ENV:-emix}"
EVAL_ENV="${EVAL_ENV:-emix-twin}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
FORCE_STAGE="${FORCE_STAGE:-}"
DRY_RUN="${DRY_RUN:-0}"

TASK="place_object_scale_left_oppo_cam"
DATASET_ROOT="data/lerobot/${TASK}"
INDEX_ROOT="data/emix_index/${TASK}"
TRAIN_DIR="configs/train/oppo_cam"
EVAL_DIR="configs/eval/oppo_cam"
STEP_LIMIT_CONFIG="third_party/RoboTwin/task_config/_eval_step_limit.yml"
EXPECTED_DATASET_EPISODES=500
EXPECTED_EVAL_EPISODES=100
EXPECTED_EVAL_STEP_LIMIT=400

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

require_dataset() {
  local info_path="${DATASET_ROOT}/meta/info.json"
  if [[ ! -f "${info_path}" ]] || \
    ! grep -Eq "\"total_episodes\"[[:space:]]*:[[:space:]]*${EXPECTED_DATASET_EPISODES}" "${info_path}" || \
    [[ ! -f "${DATASET_ROOT}/data/chunk-000/file-000.parquet" ]] || \
    [[ ! -f "${DATASET_ROOT}/videos/observation.images.head_camera/chunk-000/file-000.mp4" ]]; then
    printf 'Expected completed %s-episode dataset at %s. Run scripts/run_oppo_cam_84.sh first.\n' \
      "${EXPECTED_DATASET_EPISODES}" "${DATASET_ROOT}" >&2
    exit 1
  fi
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

require_dataset
require_eval_step_limit

if is_forced index || [[ ! -f "${INDEX_ROOT}/emix_index.json" ]]; then
  echo "[1/7] Building shared EmixIndex"
  run "${RUN_BUILD[@]}" --repo_id "${TASK}" --root "${DATASET_ROOT}" --output_dir "${INDEX_ROOT}"
else
  echo "[1/7] Shared EmixIndex already complete; skipping"
fi

run_model() {
  local number="$1"
  local label="$2"
  local config_name="$3"
  local checkpoint="outputs/train/${config_name}/checkpoints/030000/pretrained_model/model.safetensors"
  local eval_output="outputs/eval/${config_name}"

  if is_forced train || [[ ! -f "${checkpoint}" ]]; then
    echo "[${number}/7] Training: ${label}"
    run "${RUN_TRAIN[@]}" --config "${TRAIN_DIR}/${config_name}.yaml"
  else
    echo "[${number}/7] Training already complete: ${label}; skipping"
  fi

  if is_forced eval || ! evaluation_is_complete "${eval_output}"; then
    echo "[$((number + 1))/7] Evaluation (${EXPECTED_EVAL_STEP_LIMIT}-step limit): ${label}"
    run "${RUN_EVAL[@]}" --config "${EVAL_DIR}/${config_name}.yaml"
  else
    echo "[$((number + 1))/7] Evaluation already complete: ${label}; skipping"
  fi
}

run_model 2 "EMIX Constant 30% (84x84)" place_object_scale_left_oppo_cam_dp_emix_constant_30
run_model 4 "EMIX Linear 30%->0% (84x84)" place_object_scale_left_oppo_cam_dp_emix_linear_30
run_model 6 "EMIX Log 30%->0% (84x84)" place_object_scale_left_oppo_cam_dp_emix_log_30

echo "Opposite-camera 30% EMIX experiment completed."
