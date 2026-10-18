#!/usr/bin/env python3
"""Fast preflight and minimal headless SAPIEN rendering probe."""

import ctypes.util
import os
from pathlib import Path
import subprocess
import sys
import time


def fail(stage: str, detail: str, code: int = 1) -> int:
    print(f"[FAIL] {stage}: {detail}", flush=True)
    return code


def get_driver_capabilities() -> str:
    value = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "").strip()
    if value:
        return value

    # SSH login shells may not inherit environment variables set on PID 1.
    try:
        for entry in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if entry.startswith(b"NVIDIA_DRIVER_CAPABILITIES="):
                return entry.split(b"=", 1)[1].decode("ascii").strip()
    except (OSError, UnicodeError):
        pass
    return ""


def check_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[FAIL] GPU: {exc}", flush=True)
        return False

    name = result.stdout.strip().splitlines()
    if result.returncode != 0 or not name:
        print(f"[FAIL] GPU: {result.stdout.strip() or 'nvidia-smi failed'}", flush=True)
        return False
    print(f"[PASS] GPU: {name[0]}", flush=True)
    return True


def check_graphics() -> bool:
    capabilities = get_driver_capabilities()
    enabled = {item.strip().lower() for item in capabilities.split(",") if item.strip()}
    if capabilities:
        if "all" in enabled or "graphics" in enabled:
            print(f"[PASS] NVIDIA graphics capability: {capabilities}", flush=True)
            return True
        print(
            f"[FAIL] NVIDIA graphics capability: {capabilities}; graphics is not enabled",
            flush=True,
        )
        return False

    # Some runtimes expose graphics via device-request capabilities instead.
    manifests = (
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
    )
    if any(path.is_file() for path in manifests) and ctypes.util.find_library("GLX_nvidia"):
        print("[INFO] NVIDIA graphics capability: runtime variable unset; ICD is present", flush=True)
        return True

    print("[FAIL] NVIDIA graphics capability: not enabled or NVIDIA Vulkan ICD is missing", flush=True)
    return False


def test_sapien_render() -> int:
    probe = r'''import os
from pathlib import Path

for path in (Path("/etc/vulkan/icd.d/nvidia_icd.json"), Path("/usr/share/vulkan/icd.d/nvidia_icd.json")):
    if path.is_file() and not os.environ.get("VK_ICD_FILENAMES"):
        os.environ["VK_ICD_FILENAMES"] = str(path)
        break

import sapien

# Exercise the ray-tracing path used by RoboTwin without loading tasks or assets.
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(1)
sapien.render.set_ray_tracing_path_depth(1)
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
camera = scene.add_camera("probe", 16, 16, 0.1, 10.0, 1.0)
scene.update_render()
camera.take_picture()
image = camera.get_picture("Color")
print("SAPIEN_RENDER=PASS", sapien.__version__, tuple(image.shape))
'''
    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return fail("SAPIEN render", "timed out after 60 seconds", 3)

    output = result.stdout.strip()
    if output:
        print(output, flush=True)
    if result.returncode != 0:
        return fail("SAPIEN render", f"exit={result.returncode}", 3)
    if "SAPIEN_RENDER=PASS" not in output:
        return fail("SAPIEN render", "no rendered camera frame was returned", 3)
    return 0


def main() -> int:
    started = time.monotonic()
    print("[1/3] Checking GPU", flush=True)
    if not check_gpu():
        return 1
    print("[2/3] Checking NVIDIA graphics", flush=True)
    if not check_graphics():
        return 2
    print("[3/3] Testing SAPIEN ray-tracing render", flush=True)
    code = test_sapien_render()
    print(f"Elapsed: {time.monotonic() - started:.1f}s", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
