#!/usr/bin/env bash

# Create the isolated RoboTwin runtime used by scripts/run.py.
# Training, conversion, and EMIX index construction remain in the `emix`
# environment; this script intentionally configures only `emix-twin`.

set -euo pipefail

ENV_NAME="emix-twin"
PYTHON_VERSION="3.10"
CUDA_NVCC_VERSION="12.8.93"
TORCH_VERSION="2.7.1+cu128"
TORCHVISION_VERSION="0.22.1+cu128"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
TORCH_ARCH_LIST="8.9;12.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="${PROJECT_ROOT}/third_party/RoboTwin"
CONSTRAINTS_FILE="${PROJECT_ROOT}/requirements-emix-twin.txt"

if [ ! -d "${ROBOTWIN_ROOT}" ]; then
    echo "RoboTwin submodule is missing: ${ROBOTWIN_ROOT}" >&2
    exit 1
fi

if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
# NVIDIA's cuda-nvcc activation hook references an optional variable that is
# unset in a fresh shell, so nounset must be disabled only while activating.
set +u
conda activate "${ENV_NAME}"
set -u

# The environment-local compiler must match the cu128 PyTorch build.  It also
# lets the editable Curobo extensions include code for both installed GPUs.
conda install -y -c nvidia "cuda-nvcc=${CUDA_NVCC_VERSION}"
python -m pip install --upgrade \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

# Do not install RoboTwin's description-generation SDK extras or its obsolete
# cu121 torch pin.  The project-owned runtime requirements are reproducible.
python -m pip install -r "${CONSTRAINTS_FILE}"

CUDA_HOME="${CONDA_PREFIX}" TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST}" MAX_JOBS=2 \
    python -m pip install --no-deps --no-build-isolation -e "${ROBOTWIN_ROOT}/envs/curobo"

for GPU_ID in 0 1; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -c '
import torch
import toppra
from curobo.wrap.reacher.motion_gen import MotionGen
device = torch.device("cuda:0")
assert torch.cuda.is_available()
assert (torch.ones(1, device=device) + 1).item() == 2.0
print(torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device), MotionGen.__name__)
'
done
