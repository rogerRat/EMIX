#!/usr/bin/env bash

set -eo pipefail

ENV_NAME="emix"
PYTHON_VERSION="3.10"

TORCH_CUDA_TAG="cu128"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

TORCH_VERSION="2.10.0"
TORCHAUDIO_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"

# ML libraries to install after the main dependency resolution
ML_LIBS="scikit-learn seaborn umap-learn pynndescent joblib threadpoolctl numba llvmlite"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/../requirements.txt"
FILTERED_REQUIREMENTS="${SCRIPT_DIR}/.requirements_filtered.txt"
EXPORT_FILE="${SCRIPT_DIR}/../emix_requ.txt"

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

# Filter out: torch/lerobot (installed above),
#             cu13 nvidia packages (keep cu12 only),
#             ML libs (installed separately in step 6),
#             annotated-doc / cuda-toolkit (not needed).
grep -v -E "^(torch|torchaudio|torchvision|lerobot|modelscope-hub)\b" \
    "${REQUIREMENTS_FILE}" \
  | grep -v -E "^nvidia-(cublas|cuda-cupti|cuda-nvrtc|cuda-runtime|cudnn-cu13|cufft|cufile|curand|cusolver|cusparse|cusparselt-cu13|nccl-cu13|nvjitlink|nvshmem-cu13|nvtx)==" \
  | grep -v -E "^(cuda-toolkit|annotated-doc)\b" \
  | grep -v -E "^(${ML_LIBS// /|})\b" \
  > "${FILTERED_REQUIREMENTS}"

echo "Filtered requirements (torch/lerobot/cu13/ML libs removed):"
echo "---"
cat "${FILTERED_REQUIREMENTS}"
echo "---"
echo ""

python -m pip install -r "${FILTERED_REQUIREMENTS}"

rm -f "${FILTERED_REQUIREMENTS}"

echo ""
echo "============================================================"
echo "[Step 5/6] Installing ML libraries"
echo "============================================================"
echo ""
echo "Installing: ${ML_LIBS}"
echo ""

python -m pip install ${ML_LIBS}

echo ""
echo "============================================================"
echo "[Step 6/6] Verifying critical imports & exporting requirements"
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

echo ""
echo "Exporting full environment to: ${EXPORT_FILE}"
python -m pip freeze > "${EXPORT_FILE}"
echo "Done."

echo ""
echo "============================================================"
echo "Setup complete. Environment '${ENV_NAME}' is ready."
echo "Activate it with:  conda activate ${ENV_NAME}"
echo "Requirements exported to: ${EXPORT_FILE}"
echo "============================================================"
