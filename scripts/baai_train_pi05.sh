#!/usr/bin/env bash
# Launch the training-only PI0.5 baseline in a BAAI scheduled GPU container.
set -euo pipefail

project_root=/home/zylong/EMIX
archive=/home/zylong/emix-env.tar
env_root=/tmp/emix-env

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
  tar -xf "$archive" -C "$env_root"
  PATH="$env_root/bin:$PATH" "$env_root/bin/conda-unpack"
  touch "$env_root/.emix-ready"
fi

export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/zylong/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-90}"
cd "$project_root"
exec "$env_root/bin/python" -u scripts/experiment.py pi05-baai-baseline --mode run
