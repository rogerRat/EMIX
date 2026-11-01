#!/usr/bin/env bash
# Reproducible AutoDL installation of separate policy and simulator environments.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash dependencies/install_isolated.sh [--check]" >&2
    exit 2
fi

DATA_DISK="${EMIX_DATA_DISK:-/root/autodl-tmp}"
if [[ ! -d "$DATA_DISK" ]]; then
    echo "Data disk is unavailable: $DATA_DISK" >&2
    exit 1
fi
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$DATA_DISK/conda-envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_DISK/conda-pkgs:$HOME/miniconda3/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_DISK/pip-cache}"
export EMIX_ENV_NAME="${EMIX_ENV_NAME:-emix}"
export EMIX_TWIN_ENV_NAME="${EMIX_TWIN_ENV_NAME:-emix-twin}"
export EMIX_CUROBO_DIR="${EMIX_CUROBO_DIR:-$DATA_DISK/emix-install-cache/curobo-v0.7.8}"
export TMPDIR="${TMPDIR:-$DATA_DISK/emix-install-cache/tmp}"
mkdir -p "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" "$TMPDIR"

if (( ! CHECK_ONLY )); then
    if [[ -f /etc/network_turbo ]]; then
        source /etc/network_turbo
        export no_proxy="${no_proxy:-},mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn"
    fi
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    bash "$SCRIPT_DIR/_install_twin.sh"
    bash "$SCRIPT_DIR/_install.sh"
fi

bash "$SCRIPT_DIR/_install_twin.sh" --check
twin_python="$CONDA_ENVS_PATH/$EMIX_TWIN_ENV_NAME/bin/python"
"$twin_python" - <<'PY_TWIN_BOUNDARY'
import importlib.util
import numpy
assert int(numpy.__version__.split(".")[0]) == 1, numpy.__version__
assert importlib.util.find_spec("lerobot") is None, "LeRobot leaked into simulator environment"
assert importlib.util.find_spec("rerun") is None, "rerun-sdk leaked into simulator environment"
print("[emix-twin] dependency boundary passed: NumPy", numpy.__version__)
PY_TWIN_BOUNDARY
train_python="$CONDA_ENVS_PATH/$EMIX_ENV_NAME/bin/python"
"$train_python" - <<'PY_POLICY_CHECK'
import torch
import embodied_mixup
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
assert torch.cuda.is_available(), "CUDA is unavailable in policy environment"
print("[emix] policy check passed:", torch.__version__, DiffusionRgbEncoder.__name__, embodied_mixup.__file__)
PY_POLICY_CHECK
"$train_python" -m pip check
"$CONDA_ENVS_PATH/$EMIX_TWIN_ENV_NAME/bin/python" -m pip check
echo "Isolated environments validated: $EMIX_ENV_NAME and $EMIX_TWIN_ENV_NAME"
