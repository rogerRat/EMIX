# 2026-10-19：EMIX 容器 SSH、SAPIEN/Vulkan 启动修复与复现记录

## 目标和边界

在新容器上克隆 EMIX，验证 RTX 5090 可以运行 SAPIEN 的物理步进与 GPU 光线追踪渲染。本次验证的目标是排除驱动、GPU 和基础仿真运行时故障；不运行完整 RoboTwin 策略评估，不据此推断任务资产、Curobo、训练检查点或端到端评估已经就绪。

- 远端：`root@connect.bjb2.seetacloud.com:24886`，本地 SSH 别名 `emix`。
- 容器：Ubuntu 22.04.5 LTS，NVIDIA GeForce RTX 5090，驱动 `580.105.08`。
- 仓库：`https://github.com/rogerRat/EMIX.git`，容器路径 `/root/EMIX`，验证基线提交 `59aed8f7bb423cad55ffb64333154a94b93638a5`（`main`）。因完整历史下载缓慢，实际使用 `git clone --depth 1 --single-branch`；工作树内容对应上述提交。
- Python 环境：Conda `emix-twin`，Python 3.10。

本地 `~/.ssh/config` 的配置为：

```sshconfig
Host emix
    HostName connect.bjb2.seetacloud.com
    User root
    Port 24886
```

首次连接新端点时，将其 ED25519 主机密钥加入本地 `known_hosts`；它与本机已有记录中的密钥匹配。之后 `ssh emix` 可直接连接。

## 故障链与修复

1. 基础环境运行 `scripts/if_visual.py` 时，GPU 检测通过，随后报 `ModuleNotFoundError: No module named 'sapien'`。新容器只有 Conda `base`，没有项目仿真环境。
2. 创建 `emix-twin` 并安装 `sapien==3.0.0b1` 后，SAPIEN 导入报 `ModuleNotFoundError: No module named 'pkg_resources'`。原因是新建环境的 `setuptools==84.0.0` 已移除旧接口，而此 SAPIEN 版本仍导入它。固定 `setuptools<81`；本次安装版本为 `80.10.2`。
3. 项目依赖要求 `numpy==1.26.4`。当时 pip 自动选择的 OpenCV 新版本要求 NumPy 2；将 `opencv-python` 固定为 `4.10.0.84`，与 NumPy 1.26.4 同时安装。上述两个约束已写入仓库根目录的 `requirements-emix-twin.txt`，避免新环境重建时复发。
4. 此后 SAPIEN 报 `RuntimeError: failed to find a rendering device`。尽管 `nvidia-smi` 正常、`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video`，NVIDIA Vulkan ICD 仍无法提供 `vkCreateInstance`。容器缺少系统 Vulkan loader 和 GLVND/EGL 库；`strace` 发现多次查找 `libEGL.so.1` 返回 `ENOENT`。安装下述 apt 包后，`vulkaninfo` 枚举出 RTX 5090，SAPIEN 渲染成功。
5. 排查中还发现容器没有可访问的 `/dev/nvidia-modeset`。曾临时创建节点进行诊断，但打开它返回 `Operation not permitted`；该节点已经删除。补齐 EGL/GLVND 后，**无需该节点也能成功运行本次 Vulkan 和 SAPIEN 验证**，因此不能将它认定为此次启动失败的根因。

## 从干净容器复现

以下命令以 Ubuntu 22.04、NVIDIA 驱动已由平台注入、Conda 位于 `/root/miniconda3` 为前提。需要 root 权限安装系统包。

```bash
# 在本地机器配置上述 emix SSH 别名，然后进入容器。
ssh emix

git clone --depth 1 --single-branch https://github.com/rogerRat/EMIX.git /root/EMIX
cd /root/EMIX

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libvulkan1 vulkan-tools libegl1 libgl1 libglx0 libopengl0

/root/miniconda3/bin/conda create -n emix-twin python=3.10 -y
/root/miniconda3/envs/emix-twin/bin/python -m pip install \
  -r requirements-emix-twin.txt
/root/miniconda3/envs/emix-twin/bin/python -m pip check
```

本次实际安装的关键版本：`sapien 3.0.0b1`、`numpy 1.26.4`、`opencv-python 4.10.0.84`、`setuptools 80.10.2`、`mplib 0.2.1`。系统包版本为 `libegl1/libgl1/libglx0/libopengl0 1.4.0-1`、`libvulkan1 1.3.204.1-2`、`vulkan-tools 1.3.204.0+dfsg1-1`。`pip check` 输出 `No broken requirements found.`

验证 NVIDIA Vulkan ICD 和项目探针：

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary
cd /root/EMIX
/root/miniconda3/envs/emix-twin/bin/python scripts/if_visual.py
```

预期 `vulkaninfo` 中出现 `deviceName = NVIDIA GeForce RTX 5090`、`driverName = NVIDIA`。本次项目探针输出：

```text
[PASS] GPU: NVIDIA GeForce RTX 5090
[PASS] NVIDIA graphics capability: compute,utility,graphics,video
SAPIEN_RENDER=PASS 3.0.0b1 (16, 16, 4)
Elapsed: 2.3s
```

进一步使用下面的独立探针验证物理引擎步进和 GPU 相机渲染同时工作：

```bash
cd /root/EMIX
/root/miniconda3/envs/emix-twin/bin/python - <<'PY'
import os
os.environ['VK_ICD_FILENAMES'] = '/etc/vulkan/icd.d/nvidia_icd.json'
import numpy as np
import sapien

sapien.render.set_camera_shader_dir('rt')
sapien.render.set_ray_tracing_samples_per_pixel(1)
scene = sapien.Scene()
scene.set_timestep(1 / 240)
scene.add_ground(0)
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
builder.add_box_visual(half_size=[0.05, 0.05, 0.05])
box = builder.build(name='smoke_box')
box.set_pose(sapien.Pose([0, 0, 0.5]))
z0 = float(box.pose.p[2])
for _ in range(20):
    scene.step()
z1 = float(box.pose.p[2])
camera = scene.add_camera('smoke', 32, 32, 0.1, 10, 1.0)
camera.set_pose(sapien.Pose([0, -1.5, 1.0], [0.9238795, 0.3826834, 0, 0]))
scene.update_render()
camera.take_picture()
image = camera.get_picture('Color')
assert z1 < z0, (z0, z1)
assert image.shape == (32, 32, 4), image.shape
assert np.isfinite(image).all()
print('PHYSICS_RENDER_PASS', 'z0', z0, 'z1', z1, 'image', image.shape)
PY
```

本次结果为 `PHYSICS_RENDER_PASS z0 0.5 z1 0.4642343819141388 image (32, 32, 4)`。数值细节可能随引擎版本变化，判定条件是物体下降、图像尺寸正确且像素有限。

## RoboTwin 范围说明

本次没有运行完整 RoboTwin 场景、Curobo 轨迹规划或策略评估。新容器最初缺少 RoboTwin 对象/背景资产、Curobo 和策略检查点，且用户将任务范围明确收束为确认硬件与驱动能启动仿真。排查过程中从可达的 Hugging Face 镜像下载并解压了 `embodiments.zip`；压缩包已删除，解压的机器人资产留在 `third_party/RoboTwin/assets/embodiments/`（被 Git 忽略）。它不是上述 SAPIEN 验证所必需，也不表示 RoboTwin 任务资产已经完整。

因此，已验证的结论是：**这台容器可用 RTX 5090 执行 SAPIEN 物理步进和光线追踪相机渲染；完整 RoboTwin 评估仍需另行准备运行时、资产与检查点。**
