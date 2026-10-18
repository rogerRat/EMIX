#!/usr/bin/env bash

set -eo pipefail

ENV_NAME="emix"
PYTHON_VERSION="3.10"

TORCH_CUDA_TAG="cu128"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

TORCH_VERSION="2.10.0"
TORCHAUDIO_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
FILTERED_REQUIREMENTS="$(mktemp "${TMPDIR:-/tmp}/emix-requirements.XXXXXX")"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT

echo "============================================================"
echo "[Step 1/6] Preparing conda environment: ${ENV_NAME}"
echo "============================================================"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook 2>/dev/null || true)"
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Reusing existing environment: ${ENV_NAME}"
else
    echo "Creating environment: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

echo "Active Python: $(which python)"
echo "Python version: $(python --version)"

echo ""
echo "============================================================"
echo "[Step 2/6] Installing PyTorch with CUDA ${TORCH_CUDA_TAG}"
echo "============================================================"
echo "Installing from: ${TORCH_INDEX_URL}"
echo "  torch         == ${TORCH_VERSION}"
echo "  torchaudio    == ${TORCHAUDIO_VERSION}"
echo "  torchvision   == ${TORCHVISION_VERSION}"
echo ""

python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

python -c "
import torch
print(f'torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA is not available. This may indicate a driver or GPU issue.')
"

echo ""
echo "============================================================"
echo "[Step 3/6] Installing lerobot (resolves gymnasium>=1.1.1)"
echo "============================================================"

python -m pip install lerobot==0.4.4

echo ""
echo "============================================================"
echo "[Step 4/6] Installing remaining dependencies from PyPI"
echo "============================================================"

# requirements.txt is a historical pip freeze. Filter packages whose pinned
# versions conflict with the selected cu128 Torch or LeRobot 0.4.4 runtime.
# The temporary file lives outside the repository and is removed on exit.
grep -v -E "^(torch|torchaudio|torchvision|lerobot|modelscope-hub|gymnasium|opencv-python|opencv-python-headless|triton)==" \
    "${REQUIREMENTS_FILE}" \
  | grep -v -E "^nvidia-[[:alnum:]-]+==" \
  | grep -v -E "^(cuda-toolkit|annotated-doc)\b" \
  > "${FILTERED_REQUIREMENTS}"

echo "Filtered $(wc -l < "${FILTERED_REQUIREMENTS}") historical requirements; CUDA wheel pins are managed by the cu128 Torch index."

python -m pip install -r "${FILTERED_REQUIREMENTS}"

echo ""
echo "============================================================"
echo "[Step 5/6] Applying headless NumPy/OpenCV compatibility pins"
echo "============================================================"

# LeRobot requires opencv-python-headless>=4.9,<4.13 and gymnasium>=1.1.1.
# A newer OpenCV wheel can pull NumPy 2, while this project uses NumPy 1.26.
python -m pip install "setuptools<81" "numpy==1.26.4" \
    "opencv-python-headless==4.10.0.84"

echo ""
echo "============================================================"
echo "[Step 6/6] Verifying critical imports"
echo "============================================================"

python -c "
import lerobot;        print(f'lerobot:       {lerobot.__version__}')
import torch;          print(f'torch:         {torch.__version__}')
import torchvision;    print(f'torchvision:   {torchvision.__version__}')
import sapien;         print(f'sapien:        {sapien.__version__}')
import gymnasium;      print(f'gymnasium:     {gymnasium.__version__}')
import h5py;           print(f'h5py:          {h5py.__version__}')
import cv2;            print(f'opencv:        {cv2.__version__}')
import numpy;          print(f'numpy:         {numpy.__version__}')
import diffusers;      print(f'diffusers:     {diffusers.__version__}')
import accelerate;     print(f'accelerate:    {accelerate.__version__}')
import datasets;       print(f'datasets:      {datasets.__version__}')
import sklearn;        print(f'scikit-learn:  {sklearn.__version__}')
import seaborn;        print(f'seaborn:       {seaborn.__version__}')
import umap;           print(f'umap-learn:    {umap.__version__}')
import importlib.metadata as metadata
print(f'transformers:  {metadata.version(\"transformers\")}')
print(f'num2words:     {metadata.version(\"num2words\")}')
"
python -m pip check

echo ""
echo "============================================================"
echo "Setup complete. Environment '${ENV_NAME}' is ready."
echo "Activate it with:  conda activate ${ENV_NAME}"
echo "============================================================"
