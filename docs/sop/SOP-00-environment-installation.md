# SOP-00：Linux 无头服务器环境安装与 SAPIEN 验收

本文适用于 Ubuntu 22.04、已由宿主机提供 NVIDIA 驱动、容器具有
`compute,utility,graphics` 能力的无头服务器。已在 RTX 5090 / 驱动
580.105.08 上验证 SAPIEN 3.0.0b1 的物理步进与光线追踪相机渲染。
`DISPLAY` 和 X11 不是本流程的前提；真正的图形验收标准是 NVIDIA Vulkan
ICD 能枚举 GPU，且 SAPIEN 能返回相机帧。

## 1. 三个安装入口与依赖边界

所有命令从仓库根目录运行。三个安装入口、对应的依赖清单与图形探针统一位于
`dependencies/`；脚本通过自身位置定位仓库，可从其他工作目录调用。

| 入口 | 负责 | 对应依赖 |
| --- | --- | --- |
| `bash dependencies/_instal_sapien.sh --python PATH` | 检测或安装系统 Vulkan/EGL/GLVND 包，安装 SAPIEN Python 栈，运行 Vulkan 和 SAPIEN 探针 | `dependencies/requirements-sapien.txt`；Ubuntu `libvulkan1`, `vulkan-tools`, `libegl1`, `libgl1`, `libglx0`, `libopengl0` |
| `bash dependencies/_install.sh` | 创建或复用 `emix`；安装 CUDA PyTorch、LeRobot 和主环境依赖 | `dependencies/requirements.txt` 的兼容子集；PyTorch cu128 索引 |
| `bash dependencies/_install_twin.sh` | 创建或复用 `emix-twin`；调用 SAPIEN 入口，安装 CUDA PyTorch、RoboTwin 运行时与 Curobo；Blackwell 默认构建 CUDA OIDN | `dependencies/requirements-emix-twin.txt`（引用 `dependencies/requirements-sapien.txt`） |

`_instal_sapien.sh` 不创建 Conda 环境，也不安装或更新 NVIDIA 驱动。
`_install.sh` 不改动 `emix-twin`。`_install_twin.sh` 不改动 `emix`。
三个入口均复用已存在的环境，不删除环境或数据。
RoboTwin 的第三方资产和策略检查点需另行准备；安装脚本不下载数据集。

## 2. 安装前检查

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
test -f /etc/vulkan/icd.d/nvidia_icd.json
test -x /root/miniconda3/bin/conda
```

若 SSH 登录 shell 中没有 `NVIDIA_DRIVER_CAPABILITIES`，不要仅凭这个变量判断
graphics 是否可用；`dependencies/if_visual.py` 会读取容器 PID 1 环境，并以真实
渲染作为最终检验。若容器根本未暴露 graphics 或 NVIDIA ICD，需在容器平台
启用该能力，安装 Python 包无法代替驱动映射。

## 3. 安装与只读复核

```bash
# 训练、转换和索引环境；大体量 PyTorch 下载只在首次安装时发生。
bash dependencies/_install.sh

# RoboTwin 仿真环境；在有 graphics 的无头容器中自动检查 EGL/Vulkan。
bash dependencies/_install_twin.sh

# 以后只检查已有环境，不修改系统包或 Python 包。
bash dependencies/_install_twin.sh --check
```

只安装或复核 SAPIEN 基础图形组件时，先提供目标环境的 Python 路径：

```bash
bash dependencies/_instal_sapien.sh \
  --python /root/miniconda3/envs/emix-twin/bin/python
bash dependencies/_instal_sapien.sh \
  --python /root/miniconda3/envs/emix-twin/bin/python --check
```

`--check` 不运行 apt 或 pip 安装，但会运行 Vulkan 枚举与
`dependencies/if_visual.py`。通过时应看到 `SAPIEN_RENDER=PASS 3.0.0b1
(16, 16, 4)`。也可以直接核对 GPU：

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary
/root/miniconda3/envs/emix-twin/bin/python dependencies/if_visual.py
```

`vulkaninfo` 的 `deviceName` 应为 NVIDIA GPU。SAPIEN 探针同时运行 GPU
检测、graphics 检测和一帧光线追踪相机渲染。额外的 20 步刚体物理与 32×32
相机联合探针、实测输出和版本清单见
[2026-10-19 修复记录](../20261019_emix_sapien_runtime_repair.md)。

## 4. 版本与安装顺序

- `dependencies/requirements-sapien.txt` 固定 `sapien==3.0.0b1`、`setuptools<81`、
  `numpy==1.26.4`、`opencv-python==4.10.0.84`。SAPIEN 旧版本仍导入
  `pkg_resources`；新的 OpenCV 轮子会要求 NumPy 2，而 RoboTwin 的 TOPPRA
  扩展使用 NumPy 1.x ABI。
- `dependencies/requirements-emix-twin.txt` 引用上述文件，避免两处维护不同的 SAPIEN
  版本。仿真入口先检查 graphics，再下载 CUDA PyTorch、安装 Curobo。
  GPU 架构从当前可见设备读取，适用于单张 RTX 5090，不假设有 GPU 1。
- Blackwell（计算能力 ≥ 10）由安装脚本自动识别，默认构建 OIDN 2.4.1 CUDA 后端并放入 SAPIEN
  私有库目录，以覆盖其旧版去噪器。仅诊断基础 SAPIEN 时可给
  `_install_twin.sh` 加 `--skip-oidn`；此选项不表示完整 RoboTwin 渲染可用。
- `dependencies/requirements.txt` 是主环境历史锁定快照，包含 CUDA 轮子、旧
  `gymnasium` 与不兼容的 OpenCV 组合。`_install.sh` 先从 PyTorch cu128
  索引安装 Torch 与 LeRobot，再将冲突项过滤到 `/tmp` 临时清单，最后固定
  NumPy 1.26 与兼容的 `opencv-python-headless`。不要直接安装整份快照。

## 5. 故障定位

| 现象 | 检查与处理 |
| --- | --- |
| `No module named sapien` | 指定目标 Conda Python，运行 `_instal_sapien.sh`。 |
| `No module named pkg_resources` | 保持 `setuptools<81`，不要只单独安装 SAPIEN。 |
| `failed to find a rendering device` / `vkCreateInstance` 失败 | 检查 `libEGL.so.1`、`libvulkan.so.1` 和 NVIDIA ICD；运行 `_instal_sapien.sh` 补齐系统包，再用 `vulkaninfo --summary` 验证。 |
| `nvidia-smi` 成功但 Vulkan 失败 | CUDA 可用不等于 graphics 可用；检查容器 graphics 能力和 EGL/GLVND。 |
| `OIDN Error: unsupported device type: CUDA` | 在 Blackwell GPU 上完整运行 `_install_twin.sh`，使 CUDA OIDN 构建完成。 |

本次排查曾发现 `/dev/nvidia-modeset` 不可访问，但补齐 EGL/GLVND 后，
Vulkan 与 SAPIEN 仍正常工作。不要把该节点的缺失直接判为此类错误的根因。

`configs/scripts/run.yaml` 和 `scripts/assets.py` 默认使用 CUDA 0，单张 GPU
即可运行批次入口。`scripts/dataflow.sh --cuda N` 会对指定 CUDA 索引进行可用性
检查，并根据实际计算能力应用 Blackwell 的库路径修复；不再按显卡序号推断型号。

## 6. 验收范围

`SAPIEN_RENDER=PASS` 加上独立物理步进通过，证明当前容器的 GPU、Vulkan
和 SAPIEN 基础仿真路径可用。完整 RoboTwin 任务还需要对象/背景资产、
任务配置、Curobo 运行时和策略检查点；这些不由 SAPIEN 基础探针证明。
