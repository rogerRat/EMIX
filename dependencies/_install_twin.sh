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
CUROBO_SHA256=702415790acf030aaf48c49f3381b8963d24b80bc4ab40d4a10808244ad5ee19
OIDN_VERSION=v2.4.1
OIDN_SHA256=bab9197187a8754cdc0293475a00b7be6a0e967a0da73d6cc86697969cfb0a7e
OIDN_WEIGHTS_COMMIT=28883d1769d5930e13cf7f1676dd852bd81ed9e7
OIDN_WEIGHTS_SHA256=b16dcb836331682219dfb1b57a84ae7a8b622f19d64249ad3fa3ec421626a604
OIDN_CUTLASS_COMMIT=d55f6beeebb6df501a250dc82827db97660f06e0
OIDN_CUTLASS_SHA256=44add0763c3ef6af0f760a487ec5a8c11db87624b213a124b6278a20ddfc02da
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

verify_emix_imports() {
    EMIX_PROJECT_ROOT="$ROOT_DIR" "$PYTHON_BIN" - <<'PY_EMIX_IMPORT'
import os
from pathlib import Path
import embodied_mixup
from embodied_mixup.datasets import require_lerobot_dataset_root
from embodied_mixup.configs.config import resolve_project_output_path
project_root = Path(os.environ["EMIX_PROJECT_ROOT"]).resolve()
package_root = Path(embodied_mixup.__file__).resolve().parent.parent
assert package_root == project_root, (
    f"embodied_mixup resolved from {package_root}, expected {project_root}"
)
print(f"[emix-twin] embodied_mixup: {package_root}")
print("[emix-twin] evaluator helper imports: PASS")
PY_EMIX_IMPORT
}

if ((CHECK_ONLY)); then
    verify_emix_imports
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
    for name in ('libOpenImageDenoise.so', 'libOpenImageDenoise_core.so',
                 'libOpenImageDenoise_device_cuda.so'):
        assert (oidn / f'{name}.2.0.1').resolve() == oidn / f'{name}.2.4.1', (
            'SAPIEN still preloads its old OIDN library; rerun _install_twin.sh')
print('[emix-twin] CHECK PASS:', torch.__version__, MotionGen.__name__)
PY
    if "$PYTHON_BIN" - <<'PY'
import torch
raise SystemExit(not any(torch.cuda.get_device_capability(i)[0] >= 10
                         for i in range(torch.cuda.device_count())))
PY
    then
        "$PYTHON_BIN" "$ROOT_DIR/dependencies/if_visual.py" --denoiser oidn
    fi
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

if [[ -d "$CUROBO_DIR/.git" ]]; then
    if [[ "$(git -C "$CUROBO_DIR" describe --tags --exact-match 2>/dev/null || true)" != "$CUROBO_VERSION" ]]; then
        echo "[emix-twin] Existing Curobo checkout must be $CUROBO_VERSION" >&2
        exit 1
    fi
elif [[ -f "$CUROBO_DIR/.emix-source-version" ]]; then
    if [[ "$(cat "$CUROBO_DIR/.emix-source-version")" != "$CUROBO_VERSION" ]]; then
        echo "[emix-twin] Existing Curobo source must be $CUROBO_VERSION" >&2
        exit 1
    fi
elif [[ -e "$CUROBO_DIR" ]]; then
    echo "[emix-twin] Unrecognized Curobo directory exists: $CUROBO_DIR" >&2
    exit 1
else
    # codeload serves the same tagged source without Git's long HTTPS pack
    # transfer, which is unreliable on some container networks.
    curobo_archive="${TMPDIR:-/tmp}/emix-curobo-${CUROBO_VERSION}.tar.gz"
    if [[ ! -f "$curobo_archive" ]] || \
       ! echo "$CUROBO_SHA256  $curobo_archive" | sha256sum --check --status; then
        curl -fL --retry 3 --retry-delay 2 \
            "https://codeload.github.com/NVlabs/curobo/tar.gz/refs/tags/$CUROBO_VERSION" \
            -o "$curobo_archive"
    fi
    echo "$CUROBO_SHA256  $curobo_archive" | sha256sum --check
    mkdir -p "$CUROBO_DIR"
    tar -xzf "$curobo_archive" -C "$CUROBO_DIR" --strip-components=1
    printf '%s\n' "$CUROBO_VERSION" > "$CUROBO_DIR/.emix-source-version"
fi
if [[ -f "$CUROBO_DIR/.emix-built-arches" ]] && \
   [[ "$(cat "$CUROBO_DIR/.emix-built-arches")" == "$TORCH_CUDA_ARCH_LIST" ]] && \
   "$PYTHON_BIN" - <<'PY'
import torch
from curobo.curobolib import geom_cu, kinematics_fused_cu, lbfgs_step_cu
PY
then
    echo "[emix-twin] Reusing Curobo extensions for $TORCH_CUDA_ARCH_LIST"
else
    CUDA_HOME="$CONDA_PREFIX" MAX_JOBS="${MAX_JOBS:-2}" \
        "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$CUROBO_DIR"
    printf '%s\n' "$TORCH_CUDA_ARCH_LIST" > "$CUROBO_DIR/.emix-built-arches"
fi

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
    if [[ -d "$oidn_source/.git" ]]; then
        if [[ "$(git -C "$oidn_source" describe --tags --exact-match 2>/dev/null || true)" != "$OIDN_VERSION" ]]; then
            echo "[emix-twin] OIDN checkout must be $OIDN_VERSION" >&2
            exit 1
        fi
    elif [[ -f "$oidn_source/.emix-source-version" ]]; then
        if [[ "$(cat "$oidn_source/.emix-source-version")" != "$OIDN_VERSION" ]]; then
            echo "[emix-twin] OIDN source must be $OIDN_VERSION" >&2
            exit 1
        fi
    elif [[ -e "$oidn_source" && ! -f "$oidn_source/.emix-source-incomplete" ]]; then
        echo "[emix-twin] Unrecognized OIDN source directory: $oidn_source" >&2
        exit 1
    else
        # GitHub's Git pack endpoint can close long HTTPS transfers in this
        # container. Fetch the exact tag and its CUDA build submodules from
        # codeload, verifying each archive before extraction.
        oidn_archive="${TMPDIR:-/tmp}/emix-oidn-${OIDN_VERSION}.tar.gz"
        weights_archive="${TMPDIR:-/tmp}/emix-oidn-weights.tar.gz"
        cutlass_archive="${TMPDIR:-/tmp}/emix-oidn-cutlass.tar.gz"
        fetch_archive() {
            local url="$1" archive="$2" digest="$3"
            if [[ ! -f "$archive" ]] || \
               ! echo "$digest  $archive" | sha256sum --check --status; then
                curl -fL --retry 3 --retry-delay 2 "$url" -o "$archive"
            fi
            echo "$digest  $archive" | sha256sum --check
        }
        fetch_archive \
            "https://codeload.github.com/RenderKit/oidn/tar.gz/refs/tags/$OIDN_VERSION" \
            "$oidn_archive" "$OIDN_SHA256"
        fetch_archive \
            "https://codeload.github.com/RenderKit/oidn-weights/tar.gz/$OIDN_WEIGHTS_COMMIT" \
            "$weights_archive" "$OIDN_WEIGHTS_SHA256"
        fetch_archive \
            "https://codeload.github.com/NVIDIA/cutlass/tar.gz/$OIDN_CUTLASS_COMMIT" \
            "$cutlass_archive" "$OIDN_CUTLASS_SHA256"
        mkdir -p "$oidn_source/weights" "$oidn_source/external/cutlass"
        touch "$oidn_source/.emix-source-incomplete"
        if [[ ! -f "$oidn_source/CMakeLists.txt" ]]; then
            tar -xzf "$oidn_archive" -C "$oidn_source" --strip-components=1
        fi
        if [[ ! -f "$oidn_source/weights/rt_hdr.tza" ]]; then
            tar -xzf "$weights_archive" -C "$oidn_source/weights" --strip-components=1
        fi
        if [[ ! -f "$oidn_source/external/cutlass/include/cutlass/cutlass.h" ]]; then
            tar -xzf "$cutlass_archive" -C "$oidn_source/external/cutlass" --strip-components=1
        fi
        # The weights repository stores model blobs in Git LFS. Its codeload
        # archive holds pointers, so fetch the pinned blobs and verify their
        # SHA-256 object IDs before invoking CMake.
        while IFS= read -r -d '' weight; do
            if ! grep -q '^version https://git-lfs.github.com/spec/v1' "$weight"; then
                continue
            fi
            weight_sha="$(awk '$1 == "oid" {sub(/^sha256:/, "", $2); print $2}' "$weight")"
            relative_weight="${weight#"$oidn_source/weights/"}"
            curl -fL --retry 3 --retry-delay 2 \
                "https://media.githubusercontent.com/media/RenderKit/oidn-weights/$OIDN_WEIGHTS_COMMIT/$relative_weight" \
                -o "$weight.download"
            echo "$weight_sha  $weight.download" | sha256sum --check
            mv -f "$weight.download" "$weight"
        done < <(find "$oidn_source/weights" -type f -name '*.tza' -print0)
        printf '%s\n' "$OIDN_VERSION" > "$oidn_source/.emix-source-version"
        rm -f "$oidn_source/.emix-source-incomplete"
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
    # SAPIEN 3.0.0b1 explicitly preloads its private 2.0.1 filenames. Merely
    # copying 2.4.1 next to them leaves the old CUDA plugin in use. Retain
    # the original files for rollback and redirect those exact names.
    for base in libOpenImageDenoise.so libOpenImageDenoise_core.so \
        libOpenImageDenoise_device_cuda.so; do
        old="$sapien_oidn_dir/$base.2.0.1"
        if [[ -f "$old" && ! -L "$old" && ! -e "$old.bak_emix" ]]; then
            cp -a "$old" "$old.bak_emix"
        fi
        ln -sfn "$base.2.4.1" "$old"
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
if [[ "$OIDN_MODE" == force || ( "$OIDN_MODE" == auto && "$needs_oidn" == 1 ) ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/dependencies/if_visual.py" --denoiser oidn
fi
"$PYTHON_BIN" -m pip check
verify_emix_imports
echo "[emix-twin] Installation complete: $CONDA_PREFIX"
