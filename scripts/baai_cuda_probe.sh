#!/usr/bin/env bash
# Diagnose GPU visibility in a scheduled BAAI container without unpacking EMIX.
set -uo pipefail

log_dir=/home/zylong/EMIX/job-logs
mkdir -p "$log_dir" || exit 1
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
log_file="$log_dir/cuda-probe-$run_id.log"
ln -sfn "$(basename "$log_file")" "$log_dir/latest-cuda-probe.log"

probe() {
  printf '[start] run=%s host=%s utc=%s\n' "$run_id" "$(hostname)" "$(date -u +%FT%TZ)"
  printf '[devices]\n'
  ls -l /dev/nvidia* 2>&1 || true
  printf '[driver]\n'
  if [[ -r /proc/driver/nvidia/version ]]; then cat /proc/driver/nvidia/version; else echo 'No /proc/driver/nvidia/version'; fi
  for binary in "$(command -v nvidia-smi 2>/dev/null || true)" /usr/local/nvidia/bin/nvidia-smi /usr/bin/nvidia-smi; do
    if [[ -n "$binary" && -x "$binary" ]]; then "$binary" -L 2>&1; break; fi
  done
  base_python=$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)
  if [[ -z "$base_python" ]]; then echo 'No base Python executable'; return; fi
  printf '[python] %s\n' "$base_python"
  "$base_python" - <<'PY'
import ctypes
import os
import sys

print(f"python={sys.version.split()[0]}", flush=True)
for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "LD_LIBRARY_PATH"):
    print(f"{name}={os.environ.get(name, '<unset>')}", flush=True)

try:
    cuda = ctypes.CDLL("libcuda.so.1")
except OSError as exc:
    print(f"libcuda_load_error={exc!r}", flush=True)
else:
    cuda.cuInit.argtypes = [ctypes.c_uint]
    result = cuda.cuInit(0)
    print(f"cuInit={result}", flush=True)
    if result == 0:
        count = ctypes.c_int()
        cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        print(f"cuDeviceGetCount={cuda.cuDeviceGetCount(ctypes.byref(count))} count={count.value}", flush=True)
        version = ctypes.c_int()
        cuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        print(f"cuDriverGetVersion={cuda.cuDriverGetVersion(ctypes.byref(version))} version={version.value}", flush=True)

try:
    import torch
except Exception as exc:
    print(f"base_torch_import_error={exc!r}", flush=True)
else:
    print(f"base_torch={torch.__version__} built_cuda={torch.version.cuda}", flush=True)
    print(f"base_cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}", flush=True)
    try:
        torch.cuda.init()
        print(f"base_device_name={torch.cuda.get_device_name(0)}", flush=True)
    except Exception as exc:
        print(f"base_cuda_init_error={exc!r}", flush=True)
PY
}

probe 2>&1 | tee "$log_file"
result=${PIPESTATUS[0]}
printf '[finish] exit=%s log=%s\n' "$result" "$log_file" | tee -a "$log_file"
exit "$result"
