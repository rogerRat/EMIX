#!/usr/bin/env bash
# Headless NVIDIA graphics + SAPIEN preflight. This script does not create an
# environment, install a GPU driver, or install the rest of RoboTwin.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SAPIEN_PYTHON:-python}"
CHECK_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash dependencies/_instal_sapien.sh [--python /path/to/python] [--check]

Install missing Vulkan/EGL system packages and the pinned SAPIEN Python stack,
then prove that NVIDIA Vulkan and SAPIEN headless ray tracing work. --check
only validates the existing runtime. Pass the Python interpreter of the target
Conda environment; this script does not activate or create an environment.
EOF
}

while (($#)); do
    case "$1" in
        --python) PYTHON_BIN="${2:?--python needs a path}"; shift 2 ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != Linux ]]; then
    echo "[SAPIEN] This setup targets Linux headless NVIDIA containers." >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[SAPIEN] Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[SAPIEN] nvidia-smi is unavailable; expose the NVIDIA driver first." >&2
    exit 1
fi
if ! command -v dpkg-query >/dev/null 2>&1; then
    echo "[SAPIEN] This installer expects Debian/Ubuntu apt packages." >&2
    exit 1
fi

# libEGL.so.1 was the missing library that prevented the NVIDIA ICD from
# returning vkCreateInstance on the verified Ubuntu 22.04 RTX 5090 container.
packages=(libvulkan1 vulkan-tools libegl1 libgl1 libglx0 libopengl0)
missing=()
for package in "${packages[@]}"; do
    if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" != "install ok installed" ]]; then
        missing+=("$package")
    fi
done
if ((${#missing[@]})); then
    if ((CHECK_ONLY)); then
        echo "[SAPIEN] Missing system packages: ${missing[*]}" >&2
        exit 1
    fi
    if ((EUID == 0)); then
        apt=(apt-get)
    elif command -v sudo >/dev/null 2>&1; then
        apt=(sudo apt-get)
    else
        echo "[SAPIEN] Root or sudo is required for: ${missing[*]}" >&2
        exit 1
    fi
    "${apt[@]}" update
    DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y "${missing[@]}"
fi

if ((!CHECK_ONLY)); then
    "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/dependencies/requirements-sapien.txt"
fi

icd=""
for candidate in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    if [[ -f "$candidate" ]]; then
        icd="$candidate"
        break
    fi
done
if [[ -z "$icd" ]]; then
    echo "[SAPIEN] NVIDIA Vulkan ICD is missing. Expose graphics capability in the container runtime." >&2
    exit 1
fi

if ! vulkan_output="$(VK_ICD_FILENAMES="$icd" vulkaninfo --summary 2>&1)"; then
    echo "$vulkan_output" >&2
    echo "[SAPIEN] NVIDIA Vulkan initialization failed. Check libEGL.so.1 and graphics device access." >&2
    exit 1
fi
if ! grep -Eq 'deviceName[[:space:]]*=[[:space:]]*NVIDIA' <<<"$vulkan_output"; then
    echo "$vulkan_output" >&2
    echo "[SAPIEN] Vulkan did not enumerate an NVIDIA GPU." >&2
    exit 1
fi
echo "[SAPIEN] Vulkan sees: $(grep -E 'deviceName[[:space:]]*=' <<<"$vulkan_output" | head -1 | sed -E 's/^[[:space:]]*deviceName[[:space:]]*=[[:space:]]*//')"
"$PYTHON_BIN" "$ROOT_DIR/dependencies/if_visual.py"
