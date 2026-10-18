#!/usr/bin/env python3
"""Select the single emix-twin runtime and apply Blackwell CUDA path fixes.

The twin installer builds OIDN for visible Blackwell GPUs in ``emix-twin``.
GPU capability detection remains useful to strip an older /usr/local/cuda
cuBLASLt from the runtime path. CUDA enumeration can differ from nvidia-smi.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _conda_envs_path() -> Path:
    override = os.environ.get("EMIX_CONDA_ENVS")
    if override:
        return Path(override).expanduser()
    candidates = [
        Path.home() / "miniconda3" / "envs",
        Path.home() / "miniforge3" / "envs",
        Path.home() / "anaconda3" / "envs",
        Path("/opt/conda/envs"),
        Path("/home/rogerspyke/miniforge3/envs"),
    ]
    return next((p for p in candidates if (p / "emix-twin/bin/python").is_file()), candidates[0])


CONDA_ENVS = _conda_envs_path()
EMIX_TWIN_PYTHON = CONDA_ENVS / "emix-twin" / "bin" / "python"
# First Blackwell compute capability (datacenter parts are 10.0, GeForce 50 is
# 12.0); anything at or above it needs the stale-cuBLASLt path fix.
BLACKWELL_CAPABILITY = (10, 0)
# /usr/local/cuda -> cuda-12.1 ships a stale libcublasLt.so.12.1 that shadows
# torch's bundled 12.8 via LD_LIBRARY_PATH on Blackwell cards.
STALE_CUDA_MARKER = "/usr/local/cuda"

_capability_cache: dict[str, tuple[int, int]] = {}


def gpu_compute_capability(gpu: str) -> tuple[int, int]:
    """CUDA compute capability of one CUDA device index, probed via torch.

    The probe runs under the shared emix-twin interpreter with
    ``CUDA_VISIBLE_DEVICES`` pinned, so it resolves the device exactly as the
    stage itself will -- immune to the CUDA-versus-nvidia-smi order swap.
    Results are cached; each uncached probe costs one torch import.
    """
    if gpu in _capability_cache:
        return _capability_cache[gpu]
    if not gpu:
        raise ValueError("A CUDA device index is required to probe compute capability")
    probe = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "print('%d.%d' % torch.cuda.get_device_capability(0))"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    result = subprocess.run(
        [str(EMIX_TWIN_PYTHON), "-c", probe],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    major, minor = (int(part) for part in result.stdout.strip().splitlines()[-1].split("."))
    _capability_cache[gpu] = (major, minor)
    return (major, minor)


def is_blackwell(gpu: str) -> bool:
    """Whether one CUDA device needs the Blackwell CUDA library path fix.

    An empty device index (CPU-only stage) is never Blackwell.
    """
    if not gpu:
        return False
    return gpu_compute_capability(gpu) >= BLACKWELL_CAPABILITY


def twin_python(gpu: str) -> Path:
    """Interpreter for driving the RoboTwin runtime directly on one GPU."""
    return EMIX_TWIN_PYTHON


def twin_environment_name(gpu: str) -> str:
    """Conda environment name for the RoboTwin runtime on one GPU."""
    return "emix-twin"


def strip_stale_cuda(env: dict[str, str]) -> None:
    """Drop /usr/local/cuda entries from LD_LIBRARY_PATH, in place.

    Only Blackwell cards are hurt by the stale cuBLASLt there, so callers
    apply this selectively (see is_blackwell).
    """
    entries = [
        entry
        for entry in env.get("LD_LIBRARY_PATH", "").split(":")
        if STALE_CUDA_MARKER not in entry
    ]
    if entries:
        env["LD_LIBRARY_PATH"] = ":".join(entries)
    else:
        env.pop("LD_LIBRARY_PATH", None)


def stage_environment(gpu: str) -> dict[str, str]:
    """Full environment for one RoboTwin stage on one GPU.

    On Blackwell cards this also mirrors the emix-twin activate.d hook
    that direct interpreter invocation (scripts/assets.py never activates the
    environment) would otherwise miss: the system libstdc++ lacks
    CXXABI_1.3.15 for curobo's compiled kinematics, and curobo's JIT loader
    needs this environment's ninja on PATH. Stages launched through
    ``conda run`` (scripts/run.py) get the hook from activation; the overlap
    is limited to harmless duplicated entries.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    if is_blackwell(gpu):
        strip_stale_cuda(env)
        prefix = str(EMIX_TWIN_PYTHON.parent.parent)
        env["PATH"] = f"{prefix}/bin:{env.get('PATH', '')}"
        env["CUDA_HOME"] = env.get("CUDA_HOME", prefix)
        env["LD_LIBRARY_PATH"] = (
            f"{prefix}/lib:{prefix}/targets/x86_64-linux/lib"
            + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
        )
        env["LD_PRELOAD"] = (
            f"{prefix}/lib/libstdc++.so.6"
            + (f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else "")
        )
    return env


def main() -> None:
    """Print the detected routing for each given CUDA device index."""
    import argparse

    parser = argparse.ArgumentParser(description="Show per-GPU runtime routing.")
    parser.add_argument("gpus", nargs="*", default=["0"], help="CUDA device indices to probe")
    devices = parser.parse_args().gpus
    for gpu in devices:
        capability = gpu_compute_capability(gpu)
        print(
            f"CUDA {gpu}: sm_{capability[0]}{capability[1]:d} -> "
            f"{twin_environment_name(gpu)} ({twin_python(gpu)})"
        )


if __name__ == "__main__":
    main()
