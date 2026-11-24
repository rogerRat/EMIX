#!/usr/bin/env bash

set -eo pipefail

ENV_NAME="${EMIX_ENV_NAME:-emix}"
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

# LeRobot 0.4.4 selects TorchCodec when installed. Keep compatible FFmpeg
# shared libraries available; PyAV is much slower for EMIX frame-at-a-time reads.
if command -v apt-get >/dev/null 2>&1 && command -v ldconfig >/dev/null 2>&1; then
    if ! command -v ffmpeg >/dev/null 2>&1 || \
       ! ldconfig -p | grep -qE "libavdevice\.so\.[0-9]+" || \
       ! ldconfig -p | grep -qE "libavfilter\.so\.[0-9]+"; then
        run_privileged() {
            if [[ "$EUID" -eq 0 ]]; then
                "$@"
            elif command -v sudo >/dev/null 2>&1; then
                sudo "$@"
            else
                echo "[emix] FFmpeg shared libraries are missing; rerun as root or install ffmpeg with sudo." >&2
                exit 1
            fi
        }
        run_privileged apt-get update
        run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
        run_privileged ldconfig
    fi
else
    echo "[emix] apt-get/ldconfig unavailable; provide ffmpeg and compatible FFmpeg shared libraries for TorchCodec."
fi

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
grep -v -E "^(torch|torchaudio|torchvision|lerobot|modelscope-hub|gymnasium|opencv-python|opencv-python-headless|triton|numpy|packaging)==" \
    "${REQUIREMENTS_FILE}" \
  | grep -v -E "^nvidia-[[:alnum:]-]+==" \
  | grep -v -E "^(cuda-toolkit|annotated-doc)\b" \
  > "${FILTERED_REQUIREMENTS}"
printf '%s\n' \
    'numpy==2.2.6' \
    'opencv-python==4.12.0.88' \
    'opencv-python-headless==4.12.0.88' \
    'packaging==25.0' >> "${FILTERED_REQUIREMENTS}"

echo "Filtered $(wc -l < "${FILTERED_REQUIREMENTS}") historical requirements; CUDA wheel pins are managed by the cu128 Torch index."

python -m pip install -r "${FILTERED_REQUIREMENTS}"

echo ""
echo "============================================================"
echo "[Step 5/6] Confirming NumPy/OpenCV and LeRobot compatibility pins"
echo "============================================================"

# LeRobot pulls rerun-sdk>=0.24, which requires NumPy 2, and requires
# packaging<26. SAPIEN requires opencv-python while LeRobot uses the headless
# wheel; keep both wheels on the same NumPy-2-compatible release.
python -m pip install -r "${SCRIPT_DIR}/requirements-policy-runtime.txt"
# train/train.py is launched by path, so the project package must be installed.
python -m pip install --no-deps -e "${SCRIPT_DIR}/.."

echo ""
echo "============================================================"
echo "[Step 6/6] Verifying critical imports"
echo "============================================================"

python - <<'PY_IMPORT_CHECK'
import importlib.metadata as metadata
import lerobot
import peft
import torch
import torchcodec
import torchvision
from transformers.models.siglip import check as pi_transformers_check
import sapien
import gymnasium
import h5py
import cv2
import numpy
import diffusers
import accelerate
import datasets
import sklearn
import hdbscan
import seaborn
import umap
for name, module in (
    ('lerobot', lerobot), ('torch', torch), ('torchcodec', torchcodec),
    ('torchvision', torchvision), ('sapien', sapien),
    ('gymnasium', gymnasium), ('h5py', h5py), ('opencv', cv2),
    ('numpy', numpy), ('diffusers', diffusers),
    ('accelerate', accelerate), ('datasets', datasets),
    ('scikit-learn', sklearn), ('seaborn', seaborn),
    ('umap-learn', umap),
):
    print(f'{name}: {module.__version__}')
print('hdbscan: available')
if not pi_transformers_check.check_whether_transformers_replace_is_installed_correctly():
    raise RuntimeError('Install PI-specific Transformers dependency before using PI0.5')
for name in ('transformers', 'num2words'):
    print(f'{name}: {metadata.version(name)}')
PY_IMPORT_CHECK
python -m pip check

python - <<'PY_TORCHCODEC_SMOKE'
import subprocess
import tempfile
from pathlib import Path

from lerobot.datasets.video_utils import decode_video_frames
import torchcodec

with tempfile.TemporaryDirectory(prefix="emix-ffmpeg-smoke-") as temp_dir:
    video_path = Path(temp_dir) / "torchcodec-smoke.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=2:d=1",
            "-an", "-c:v", "mpeg4", str(video_path),
        ],
        check=True,
    )
    frames = decode_video_frames(video_path, [0.0], 0.1, "torchcodec")
    assert tuple(frames.shape) == (1, 3, 64, 64), tuple(frames.shape)
print(f"TorchCodec {torchcodec.__version__}: FFmpeg decode smoke passed")
PY_TORCHCODEC_SMOKE

# Install the browser binary required by scripts/render_vis_png.py.
python -m playwright install --with-deps chromium
python - <<'PY_BROWSER_CHECK'
from playwright.sync_api import sync_playwright
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True, args=["--no-sandbox", "--disable-gpu", "--enable-unsafe-swiftshader"]
    )
    page = browser.new_page()
    page.set_content("<html><body>EMIX browser check</body></html>")
    assert "EMIX browser check" in page.locator("body").inner_text()
    browser.close()
print("playwright chromium: launch and page-render check passed")
PY_BROWSER_CHECK

echo ""
echo "============================================================"
echo "Setup complete. Environment '${ENV_NAME}' is ready."
echo "Activate it with:  conda activate ${ENV_NAME}"
echo "============================================================"
