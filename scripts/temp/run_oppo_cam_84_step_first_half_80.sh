#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Final experiment: step scheduler, first 50% of training at 80% EMIX ratio,
# last 50% at 0% (no mixup). Reuses the completed 84x84 opposite-camera dataset;
# only builds a missing index, trains, and evaluates, pinned to GPU 0.
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
MODEL="place_object_scale_left_oppo_cam_dp_emix_step_first_half_80"

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

# Pin training and evaluation to the requested GPU (default GPU 0).
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if is_forced index || [[ ! -f "${INDEX_ROOT}/emix_index.json" ]]; then
  echo "[1/3] Building shared EmixIndex"
  run "${RUN_BUILD[@]}" --repo_id "${TASK}" --root "${DATASET_ROOT}" --output_dir "${INDEX_ROOT}"
else
  echo "[1/3] Shared EmixIndex already complete; skipping"
fi

if is_forced train || [[ ! -f "outputs/train/${MODEL}/checkpoints/030000/pretrained_model/model.safetensors" ]]; then
  echo "[2/3] Training: EMIX Step first half 80% (84x84)"
  run "${RUN_TRAIN[@]}" --config "${TRAIN_DIR}/${MODEL}.yaml"
else
  echo "[2/3] Training already complete: EMIX Step first half 80% (84x84); skipping"
fi

if is_forced eval || ! evaluation_is_complete "outputs/eval/${MODEL}"; then
  echo "[3/3] Evaluation (${EXPECTED_EVAL_STEP_LIMIT}-step limit): EMIX Step first half 80% (84x84)"
  run "${RUN_EVAL[@]}" --config "${EVAL_DIR}/${MODEL}.yaml"
else
  echo "[3/3] Evaluation already complete: EMIX Step first half 80% (84x84); skipping"
fi

echo "Opposite-camera first-half 80% EMIX experiment completed."
