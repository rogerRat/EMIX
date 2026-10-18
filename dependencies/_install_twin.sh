#!/usr/bin/env bash
# One entry point for the isolated RoboTwin simulation environment.
# Conda's cuda-nvcc/GCC activation hooks reference optional variables. Keep
# errexit and pipefail, but leave nounset off while installing packages.
set -Eeo pipefail

ENV_NAME=emix-twin
PYTHON_VERSION=3.10
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
TORCH_VERSION="2.7.1+cu128"
TORCHVISION_VERSION="0.22.1+cu128"
CUDA_NVCC_VERSION=12.8.93
CUROBO_VERSION=v0.7.8
OIDN_VERSION=v2.4.1
CHECK_ONLY=0
OIDN_MODE=auto

usage() {
    cat <<'EOF'
Usage: bash dependencies/_install_twin.sh [--check] [--skip-oidn|--with-oidn]

Create or reuse emix-twin, install SAPIEN graphics support, CUDA PyTorch,
RoboTwin Python dependencies, Curobo, and on Blackwell GPUs CUDA OIDN.
--check validates an existing environment without installing anything.
--skip-oidn is only for debugging the basic SAPIEN path.
EOF
}

while (($#)); do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --skip-oidn) OIDN_MODE=skip; shift ;;
        --with-oidn) OIDN_MODE=force; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="$ROOT_DIR/third_party/RoboTwin"
CUROBO_DIR="$ROBOTWIN_ROOT/envs/curobo"
if [[ ! -d "$ROBOTWIN_ROOT/envs" ]]; then
    echo "[emix-twin] RoboTwin source is missing: $ROBOTWIN_ROOT" >&2
    exit 1
fi

conda_bin=""
for candidate in "${CONDA_EXE:-}" "$HOME/miniconda3/bin/conda" \
    "$HOME/miniforge3/bin/conda" "$HOME/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        conda_bin="$candidate"
        break
    fi
done
if [[ -z "$conda_bin" ]]; then
    echo "[emix-twin] Conda was not found. Install Conda first." >&2
    exit 1
fi
source "$(dirname "$conda_bin")/../etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    if ((CHECK_ONLY)); then
        echo "[emix-twin] Conda environment is missing: $ENV_NAME" >&2
        exit 1
    fi
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"
PYTHON_BIN="$CONDA_PREFIX/bin/python"

if ((CHECK_ONLY)); then
    bash "$ROOT_DIR/dependencies/_instal_sapien.sh" --python "$PYTHON_BIN" --check
    "$PYTHON_BIN" - <<'PY'
import torch, toppra, warp
from curobo.wrap.reacher.motion_gen import MotionGen
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() > 0
for i in range(torch.cuda.device_count()):
    assert (torch.ones(1, device=f'cuda:{i}') + 1).item() == 2.0
if any(torch.cuda.get_device_capability(i)[0] >= 10
       for i in range(torch.cuda.device_count())):
    from pathlib import Path
    import sapien
    oidn = Path(sapien.__file__).parent / 'oidn_library'
    assert (oidn / 'libOpenImageDenoise_device_cuda.so.2.4.1').is_file(), (
        'Blackwell CUDA OIDN is missing; rerun _install_twin.sh')
print('[emix-twin] CHECK PASS:', torch.__version__, MotionGen.__name__)
PY
    exit 0
fi

# Prove the graphics path before downloading and compiling larger dependencies.
bash "$ROOT_DIR/dependencies/_instal_sapien.sh" --python "$PYTHON_BIN"

# NVCC matches the cu128 wheels. The project needs CUDA extensions for Curobo
# and (on Blackwell) OIDN. Reuse the existing environment; never recreate it.
conda install -y -c nvidia "cuda-nvcc=$CUDA_NVCC_VERSION"
"$PYTHON_BIN" -m pip install \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX_URL"
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/dependencies/requirements-emix-twin.txt"

gpu_arches="$($PYTHON_BIN - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit('No CUDA GPU is visible to emix-twin')
print(';'.join(sorted({f'{a}.{b}' for a, b in
                       (torch.cuda.get_device_capability(i)
                        for i in range(torch.cuda.device_count()))})))
PY
)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$gpu_arches}"
echo "[emix-twin] CUDA architectures: $TORCH_CUDA_ARCH_LIST"

if [[ ! -d "$CUROBO_DIR/.git" ]]; then
    if [[ -e "$CUROBO_DIR" ]]; then
        echo "[emix-twin] Non-Git Curobo directory exists: $CUROBO_DIR" >&2
        exit 1
    fi
    git clone --branch "$CUROBO_VERSION" --depth 1 \
        https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi
CUDA_HOME="$CONDA_PREFIX" MAX_JOBS="${MAX_JOBS:-2}" \
    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$CUROBO_DIR"

needs_oidn="$($PYTHON_BIN - <<'PY'
import torch
print(int(any(torch.cuda.get_device_capability(i)[0] >= 10
              for i in range(torch.cuda.device_count()))))
PY
)"
if [[ "$OIDN_MODE" == force || ( "$OIDN_MODE" == auto && "$needs_oidn" == 1 ) ]]; then
    oidn_arches="${OIDN_ARCHITECTURES:-$(echo "$gpu_arches" | tr ';' '\n' | tr -d '.' | paste -sd ';' -)}"
    oidn_root="${OIDN_BUILD_ROOT:-${TMPDIR:-/tmp}/robotwin-oidn-${OIDN_VERSION#v}}"
    oidn_source="$oidn_root/src"
    oidn_build="$oidn_root/build"
    for tool in git cmake ninja; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            conda install -y cmake ninja
            break
        fi
    done
    if [[ ! -d "$oidn_source/.git" ]]; then
        mkdir -p "$oidn_root"
        git clone --recursive --branch "$OIDN_VERSION" --depth 1 \
            https://github.com/RenderKit/oidn.git "$oidn_source"
    fi
    if [[ "$(git -C "$oidn_source" describe --tags --exact-match 2>/dev/null || true)" != "$OIDN_VERSION" ]]; then
        echo "[emix-twin] OIDN checkout must be $OIDN_VERSION" >&2
        exit 1
    fi
    cmake -S "$oidn_source" -B "$oidn_build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
        -DCMAKE_CUDA_ARCHITECTURES="$oidn_arches" \
        -DOIDN_DEVICE_CPU=OFF -DOIDN_DEVICE_CUDA=ON -DOIDN_FILTER_RT=ON \
        -DOIDN_APPS=OFF -DOIDN_BUILD_TESTING=OFF \
        -DOIDN_INSTALL_DEPENDENCIES=OFF \
        -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCUDAToolkit_ROOT="$CONDA_PREFIX"
    cmake --build "$oidn_build" --parallel "${OIDN_BUILD_JOBS:-2}"
    cmake --install "$oidn_build"
    sapien_oidn_dir="$($PYTHON_BIN - <<'PY'
from pathlib import Path
import sapien
print(Path(sapien.__file__).parent / 'oidn_library')
PY
)"
    mkdir -p "$sapien_oidn_dir"
    for name in libOpenImageDenoise_core.so.2.4.1 \
        libOpenImageDenoise.so.2.4.1 \
        libOpenImageDenoise_device_cuda.so.2.4.1; do
        test -f "$CONDA_PREFIX/lib/$name"
        cp -f "$CONDA_PREFIX/lib/$name" "$sapien_oidn_dir/"
    done
    if compgen -G "$sapien_oidn_dir/libOpenImageDenoise_device_cpu.so*" >/dev/null; then
        echo "[emix-twin] CPU OIDN plugin remains in $sapien_oidn_dir" >&2
        exit 1
    fi
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    cat > "$CONDA_PREFIX/etc/conda/activate.d/robotwin_cuda_runtime.sh" <<'EOF'
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export C_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
EOF
    echo "[emix-twin] Installed CUDA OIDN $OIDN_VERSION for architectures $oidn_arches"
fi

"$PYTHON_BIN" - <<'PY'
import torch, toppra, warp
from curobo.wrap.reacher.motion_gen import MotionGen
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    x = torch.ones(1, device=f'cuda:{i}')
    assert (x + 1).item() == 2.0
    print('[emix-twin]', i, torch.cuda.get_device_name(i),
          torch.cuda.get_device_capability(i), MotionGen.__name__)
PY
# Re-run the render probe after replacing SAPIEN's private OIDN libraries.
bash "$ROOT_DIR/dependencies/_instal_sapien.sh" --python "$PYTHON_BIN" --check
"$PYTHON_BIN" -m pip check
echo "[emix-twin] Installation complete: $CONDA_PREFIX"
