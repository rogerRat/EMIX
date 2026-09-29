#!/usr/bin/env bash

# Fail closed: verify that RoboTwin can render with GPU OIDN and cannot use a
# CPU OIDN plugin.  This is intentionally a smoke test, not a data-collection
# command, so it is safe to run before regenerating datasets.
set -Eeuo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[OIDN] Activate the target RoboTwin environment first." >&2
    exit 1
fi

python - <<'PY'
import glob
import os
import sys

import sapien
import torch

sapien_dir = os.path.dirname(sapien.__file__)
oidn_dirs = [
    os.path.join(sapien_dir, "oidn_library"),
    os.path.join(os.environ["CONDA_PREFIX"], "lib"),
]
cpu_plugins = [
    path
    for oidn_dir in oidn_dirs
    for path in glob.glob(os.path.join(oidn_dir, "libOpenImageDenoise_device_cpu.so*"))
]
if cpu_plugins:
    raise RuntimeError(f"CPU OIDN plugin present; refusing fallback: {cpu_plugins}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing CPU fallback")

device = torch.cuda.get_device_name(0)
print(f"[OIDN] Python: {sys.executable}")
print(f"[OIDN] SAPIEN: {sapien.__version__}")
print(f"[OIDN] CUDA device: {device}")
print(f"[OIDN] capability: {torch.cuda.get_device_capability(0)}")
cuda_plugins = [
    path
    for oidn_dir in oidn_dirs
    for path in glob.glob(os.path.join(oidn_dir, "libOpenImageDenoise_device_cuda.so*"))
]
if not cuda_plugins:
    raise RuntimeError("No CUDA OIDN plugin found")
print(f"[OIDN] CUDA plugins: {cuda_plugins}")

sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(1)
sapien.render.set_ray_tracing_path_depth(1)
sapien.render.set_ray_tracing_denoiser("oidn")
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.set_timestep(1 / 60)
scene.add_camera("smoke", 16, 16, 0.1, 10.0, 1.0)
scene.update_render()
scene.get_camera("smoke").take_picture()
print("[OIDN] GPU OIDN render smoke test: PASS")
PY
