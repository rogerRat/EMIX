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
  trap 'rc=$?; printf "[error] line=%s exit=%s\n" "$LINENO" "$rc" >&2' ERR
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

  echo "[setup] linking bundled FFmpeg libraries"
  "$env_root/bin/python" - <<'PYFFMPEG'
from pathlib import Path
import sys

root = Path(sys.prefix)
lib_dir = root / "lib"
av_dir = root / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/av.libs"
if not av_dir.is_dir():
    raise RuntimeError(f"Missing bundled PyAV FFmpeg library directory: {av_dir}")
for name, major in (
    ("libavutil", 59), ("libavcodec", 61), ("libavformat", 61),
    ("libavdevice", 61), ("libavfilter", 10), ("libswscale", 8),
    ("libswresample", 5),
):
    matches = list(av_dir.glob(f"{name}-*.so.{major}.*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one bundled {name} library, found: {matches}")
    link = lib_dir / f"{name}.so.{major}"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Refusing to replace existing FFmpeg library: {link}")
    link.symlink_to(matches[0])
print("[setup] FFmpeg libraries linked", flush=True)
PYFFMPEG
  export LD_LIBRARY_PATH="$env_root/lib:$env_root/lib/python3.10/site-packages/av.libs:$env_root/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
  "$env_root/bin/python" - <<'PYGPU'
import os
import torch
from torchcodec.decoders import VideoDecoder

print("[setup] TorchCodec ready", flush=True)
print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible in this scheduled container. Assign one A800 GPU to the task and ensure the command runs on the GPU worker.")
properties = torch.cuda.get_device_properties(0)
print(f"[gpu] {properties.name}, {properties.total_memory / 1024**3:.1f} GiB", flush=True)
PYGPU

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
