#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-emix-twin}"

DEFAULT_CONFIGS=(
  "${PROJECT_ROOT}/configs/eval/default/place_object_scale_left_dp_baseline.yaml"
  "${PROJECT_ROOT}/configs/eval/default/place_object_scale_left_dp_emix.yaml"
  "${PROJECT_ROOT}/configs/eval/default/place_object_scale_left_dp_emix_beta16.yaml"
)

if [[ "$#" -eq 0 ]]; then
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
elif [[ "$#" -eq 3 ]]; then
  CONFIGS=("$1" "$2" "$3")
else
  echo "Usage: $0 [baseline-config emix-config emix-beta16-config]" >&2
  exit 2
fi

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Evaluation config not found: ${config}" >&2
    exit 1
  fi
  echo "[eval] Starting ${config}"
  (
    cd "${PROJECT_ROOT}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python -m emix eval --config "${config}"
  )
done
