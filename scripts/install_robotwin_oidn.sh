#!/usr/bin/env bash

# Build the exact GPU-only OIDN runtime required by RoboTwin/SAPIEN.
# Run this after activating the RoboTwin environment.  The script never
# removes or recreates a Conda environment.
set -Eeuo pipefail

OIDN_VERSION="${OIDN_VERSION:-v2.4.1}"
OIDN_REPOSITORY="${OIDN_REPOSITORY:-https://github.com/RenderKit/oidn.git}"
OIDN_ARCHITECTURES="${OIDN_ARCHITECTURES:-${CMAKE_CUDA_ARCHITECTURES:-120}}"
OIDN_BUILD_ROOT="${OIDN_BUILD_ROOT:-${TMPDIR:-/tmp}/robotwin-oidn-${OIDN_VERSION#v}}"
OIDN_SOURCE_DIR="${OIDN_BUILD_ROOT}/src"
OIDN_BUILD_DIR="${OIDN_BUILD_ROOT}/build"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[OIDN] Activate the target Conda environment first." >&2
    exit 1
fi

for command_name in git cmake ninja; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "[OIDN] Missing build tool: ${command_name}" >&2
        exit 1
    fi
done

if [[ ! -d "${OIDN_SOURCE_DIR}/.git" ]]; then
    mkdir -p "${OIDN_BUILD_ROOT}"
    git clone --recursive --branch "${OIDN_VERSION}" --depth 1 \
        "${OIDN_REPOSITORY}" "${OIDN_SOURCE_DIR}"
fi

SOURCE_VERSION="$(git -C "${OIDN_SOURCE_DIR}" describe --tags --exact-match 2>/dev/null || true)"
if [[ "${SOURCE_VERSION}" != "${OIDN_VERSION}" ]]; then
    echo "[OIDN] Source checkout is not ${OIDN_VERSION}: ${SOURCE_VERSION:-unknown}" >&2
    exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="${CONDA_PREFIX}"
fi
if [[ ! -f "${CUDA_HOME}/include/cuda_runtime_api.h" && \
      ! -f "${CUDA_HOME}/targets/x86_64-linux/include/cuda_runtime_api.h" ]]; then
    echo "[OIDN] CUDA headers were not found under ${CUDA_HOME}." >&2
    exit 1
fi

cmake -S "${OIDN_SOURCE_DIR}" -B "${OIDN_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
    -DCMAKE_CUDA_ARCHITECTURES="${OIDN_ARCHITECTURES}" \
    -DOIDN_DEVICE_CPU=OFF \
    -DOIDN_DEVICE_CUDA=ON \
    -DOIDN_FILTER_RT=ON \
    -DOIDN_APPS=OFF \
    -DOIDN_BUILD_TESTING=OFF \
    -DOIDN_INSTALL_DEPENDENCIES=OFF \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    -DCUDAToolkit_ROOT="${CUDA_HOME}"
cmake --build "${OIDN_BUILD_DIR}" --parallel "${OIDN_BUILD_JOBS:-$(nproc 2>/dev/null || echo 2)}"
cmake --install "${OIDN_BUILD_DIR}"

# SAPIEN preloads OIDN from its private directory, not from Conda's lib
# directory. Copy the versioned GPU runtime beside SAPIEN's renderer so the
# result is independent of the machine's system OIDN installation.
SAPIEN_OIDN_DIR="$($CONDA_PREFIX/bin/python -c 'import os, sapien; print(os.path.join(os.path.dirname(sapien.__file__), "oidn_library"))')"
mkdir -p "${SAPIEN_OIDN_DIR}"
for library in \
    "${CONDA_PREFIX}/lib/libOpenImageDenoise_core.so.2.4.1" \
    "${CONDA_PREFIX}/lib/libOpenImageDenoise.so.2.4.1" \
    "${CONDA_PREFIX}/lib/libOpenImageDenoise_device_cuda.so.2.4.1"; do
    if [[ ! -f "${library}" ]]; then
        echo "[OIDN] Expected library was not installed: ${library}" >&2
        exit 1
    fi
    cp -f "${library}" "${SAPIEN_OIDN_DIR}/"
done

# Make Curobo's JIT compiler and OIDN's CUDA plugin use this environment's
# runtime.  No CPU OIDN plugin is installed by the build above.
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${ACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/robotwin_cuda_runtime.sh" <<'EOF'
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export C_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
EOF

if compgen -G "${SAPIEN_OIDN_DIR}/libOpenImageDenoise_device_cpu.so*" >/dev/null || \
   compgen -G "${CONDA_PREFIX}/lib/libOpenImageDenoise_device_cpu.so*" >/dev/null; then
    echo "[OIDN] CPU plugin exists; remove it before GPU-only validation." >&2
    exit 1
fi

echo "[OIDN] Installed ${OIDN_VERSION} CUDA-only runtime in ${CONDA_PREFIX}"
echo "[OIDN] CUDA architectures: ${OIDN_ARCHITECTURES}"
