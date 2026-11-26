#!/usr/bin/env bash
# Launch the training-only PI0.5 baseline in a BAAI scheduled GPU container.
set -uo pipefail

project_root=/home/zylong/EMIX
archive=${EMIX_ENV_ARCHIVE:-/home/zylong/emix-env.tar}
env_root=/tmp/emix-env
log_dir=${EMIX_JOB_LOG_DIR:-$project_root/job-logs}

mkdir -p "$log_dir" || exit 1
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
log_file="$log_dir/pi05-baai-baseline-$run_id.log"
status_file="$log_dir/pi05-baai-baseline-$run_id.exit"
ln -sfn "$(basename "$log_file")" "$log_dir/latest.log"

run_main() (
  set -Eeuo pipefail
  trap 'rc=$?; printf "[error] line=%s exit=%s command=%s\n" "$LINENO" "$rc" "$BASH_COMMAND" >&2' ERR
  printf "[start] run=%s utc=%s host=%s uid=%s\n" "$run_id" "$(date -u +%FT%TZ)" "$(hostname)" "$(id -u)"
  printf "[paths] project=%s archive=%s env=%s\n" "$project_root" "$archive" "$env_root"

  if [[ ! -f "$archive" ]]; then
    echo "Missing training environment archive: $archive" >&2
    exit 1
  fi
  if [[ ! -f "$project_root/configs/scripts/pi05_baai_baseline.yaml" ]]; then
    echo "Missing shared EMIX checkout: $project_root" >&2
    exit 1
  fi

  if [[ ! -f "$env_root/.emix-ready" ]]; then
    rm -rf "$env_root"
    mkdir -p "$env_root"
    echo "[setup] unpacking training environment"
    tar -xf "$archive" -C "$env_root"
    PATH="$env_root/bin:$PATH" "$env_root/bin/conda-unpack"
    touch "$env_root/.emix-ready"
  fi

  export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HOME="${HF_HOME:-/home/zylong/hf-cache}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-90}"
  cd "$project_root"
  echo "[train] starting PI0.5 baseline"
  "$env_root/bin/python" -u scripts/experiment.py pi05-baai-baseline --mode run
)

run_main 2>&1 | tee "$log_file"
pipeline_status=("${PIPESTATUS[@]}")
result=${pipeline_status[0]}
if [[ ${pipeline_status[1]} -ne 0 && $result -eq 0 ]]; then
  result=${pipeline_status[1]}
fi
printf "[finish] run=%s exit=%s utc=%s log=%s\n" "$run_id" "$result" "$(date -u +%FT%TZ)" "$log_file" | tee -a "$log_file"
printf "%s\n" "$result" > "$status_file"
exit "$result"
