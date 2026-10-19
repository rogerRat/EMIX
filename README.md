# Embodied Mixup (开发中)

## 简介

本项目是一个基于 RoboTwin 和 LeRobot 的 Embodied Mixup 框架。当前活动基准与首轮实验流程见 [docs/ROBOTWIN.md](docs/ROBOTWIN.md)。ManiSkill 历史实现已归档到 `legacy/maniskill/`，不再是运行入口。

### 动机

**[ICLR 2026] When would Vision-Proprioception Policies Fail in Robotic Manipulation?**

| 阶段类型                             | 核心特征                                         | 核心依赖模态                           |
| -------------------------------- | -------------------------------------------- | -------------------------------- |
| 运动一致阶段（motion-consistent phases） | 机器人执行持续、稳定的运动（如匀速向前移动、持续夹取），运动模式无切换          | 本体感受（提供精准的闭环运动控制）                |
| 运动转换阶段（motion-transition phases） | 机器人运动模式发生切换（如从移动转向目标定位、抓取、装配、姿态调整），需要精准的目标定位 | 视觉模态（提供目标物体的位置与环境信息，本体感受无目标相关信息） |

![](https://raw.githubusercontent.com/Koorye/Images/master/20260427200048761.png)

**现象**：
- **运动保持阶段**：“视觉-本体”策略比“纯视觉”策略性能更好
- **运动过渡阶段**：“视觉-本体”策略对性能造成明显退化

**结论**：
- 运动转换阶段，视觉线索往往极其细微，仅存在像素级差异，学习难度高；而本体感受信号是低维和简洁的（如夹爪开合度、位姿变化），对损失下降的贡献更直接。
- 模型为了快速降低训练损失，会天然地倾向于依赖更 “易用” 的本体感受信号，导致本体感受主导了整个优化过程，严重抑制了视觉模态的特征学习。
- 这种抑制在运动一致阶段无负面影响，但在运动转换阶段，视觉线索是唯一的可靠信息来源，导致性能大幅下降。

### 方法设计（开发中）

**核心思想**：捕获训练数据中的本体姿态歧义结点（轨迹间的交点），并在这些结点上进行模态混合（mixup），显式建立动作多峰分布，打破模型的本体感受依赖，促进视觉模态的特征学习。

#### 步骤1: 基于位姿一致性的图像拼贴

- **检索机制：** 建立基于末端执行器（End-effector）6D 位姿的索引表 $\mathcal{T}$。对于锚点样本 $(o, a, s)$，寻找满足 $\|s - s'\| < \epsilon$ 的近邻样本 $(o', a', s')$。
- **图像合成：** $o^{mix} = \alpha o + (1-\alpha) o'$。
    - _优势：_ 由于机器人手臂在两张图中位置几乎重合，叠加后的 $o^{mix}$ 不会出现严重的手臂重影，而环境特征（背景、物体颜色）则通过插值实现了平滑演变。

#### 步骤2: 动作概率预测

**适用于自回归 Policy (Autoregressive / Discrete Tokenization)**

对于类似 RT-2 或 OpenVLA 这种将动作离散化为 Token 的模型，Mixup 转化为**软标签回归（Soft-target Supervision）**
- **输入端：** 图像执行线性插值 $x_{mix} = \alpha x_1 + (1-\alpha) x_2$；语言指令进行拼接或重写
- **输出端：** 不再要求模型预测唯一的动作 Token，而是计算两个原始动作 $a_1, a_2$ 对应 Token 的分布：$$P(a_{target}) = \alpha \cdot \text{OneHot}(a_1) + (1-\alpha) \cdot \text{OneHot}(a_2)$$
- **损失函数：** 使用交叉熵损失（Cross-Entropy）计算预测分布与上述混合分布之间的差异，起到类似“标签平滑”的作用。
![image.png](https://raw.githubusercontent.com/Koorye/Images/master/20260413210801421.png)

**适用于Diffusion Policy (Continuous Action Space)**

由于 Diffusion Policy 预测的是分数场（Score Field），不能直接对预测值做线性平均，否则会退化为单峰均值。
- **策略一：加权分数匹配 (Weighted Score Matching)**。在去噪过程中，针对同一个混合输入 $x_{mix}$，计算其相对于两个目标的梯度合力：$$\mathcal{L}_{diff} = \alpha \| \epsilon_1 - \epsilon_\theta(a_{k,1}, k, x_{mix}) \|^2 + (1-\alpha) \| \epsilon_2 - \epsilon_\theta(a_{k,2}, k, x_{mix}) \|^2$$
- **策略二：多峰轨迹生成**。推理时，由于模型在训练中看到了混合特征，Diffusion过程会自适应的向两个目标靠近。这允许模型在处理模糊指令时，通过随机采样从多个有效路径中选择其一，而非机械地取中间值
![image.png](https://raw.githubusercontent.com/Koorye/Images/master/20260413210751402.png)

## 使用说明

### 安装与环境配置

项目使用两个独立的 Conda 环境。`emix` 用于数据转换、索引与训练；
`emix-twin` 用于 RoboTwin 仿真。Linux 无头服务器只要容器暴露 NVIDIA
`graphics` 能力，就可以用 Vulkan ICD 启动 SAPIEN，不需要 X11 或 `DISPLAY`。
在 Ubuntu 22.04 + RTX 5090 上，缺少系统 `libEGL.so.1` 曾使 Vulkan 无法
创建实例；安装 EGL/GLVND 包后，物理步进和 GPU 相机渲染均通过。
安装脚本与三个依赖清单集中在根目录 `dependencies/`；默认批次配置只使用
CUDA 设备 0。仿真安装会按可见 GPU 的计算能力自动决定是否构建 Blackwell OIDN，
并在 `--check` 中显式运行 OIDN 去噪相机探针。

从仓库根目录选择所需入口：

| 脚本 | 唯一职责 |
| --- | --- |
| `dependencies/_instal_sapien.sh` | 检测、补齐 Vulkan/EGL 与 SAPIEN 组件，并实测无头渲染；不创建 Conda 环境 |
| `dependencies/_install.sh` | 一键创建或复用主环境 `emix` |
| `dependencies/_install_twin.sh` | 一键创建或复用仿真环境 `emix-twin`，调用 SAPIEN 检查并安装仿真依赖 |

```bash
bash dependencies/_install.sh
bash dependencies/_install_twin.sh
# 仿真环境的只读复核：
bash dependencies/_install_twin.sh --check
```

如果只需验证已存在的 Python 环境是否具备 SAPIEN 图形能力：

```bash
bash dependencies/_instal_sapien.sh \
  --python /root/miniconda3/envs/emix-twin/bin/python --check
```

安装顺序、组件边界、无头服务器排障与验收命令见
[SOP-00 环境安装](docs/sop/SOP-00-environment-installation.md)。
本次故障与实测输出见[修复记录](docs/20261019_emix_sapien_runtime_repair.md)。
`dependencies/requirements.txt` 是主环境的历史锁定快照，含需由安装脚本分流的 CUDA 轮子；
不要直接对新环境运行 `pip install -r dependencies/requirements.txt`。

### 数据生成

基于Maniskill Franka Emika Panda机械臂，运行数据生成脚本，以下案例生成“StackCube-v1”任务，带RGB相机的500条轨迹，仅保留成功轨迹：

```bash
python data_collection/panda/run.py \
  -e StackCube-v1 \
  -o rgb \
  -n 500 \
  --only-count-success
```

运行完成后，`demos/{task_name}/motionplanning/`目录下会生成对应的轨迹数据文件，命名格式为`{date}_{time}.h5`和`{date}_{time}.json`。

### 数据重放与控制模式转换（可选，目前暂不使用，后续可以尝试）

为了将轨迹数据转换为适合模型训练的模式，可以使用数据重放脚本，以下案例将`demos/StackCube-v1/motionplanning/20260602_160056.h5`中的轨迹重放为末端控制模式（pd_ee_delta_pose），并保存重放后的轨迹数据：

```bash
python data_processing/replay.py \
  --traj-path demos/StackCube-v1/motionplanning/20260602_160056.h5 \
  -c pd_ee_delta_pose \
  -o rgb \
  --save-traj
```

### 数据格式转换

将h5格式的轨迹数据转换为LeRobot格式，以便模型训练，以下案例将`demos/StackCube-v1/motionplanning/20260602_160056.h5`转换为LeRobot格式，并保存在`data/lerobot/maniskill/stack_cube_rgb_base_hand_episodes_500_84x84`目录下，任务名称为“stack cube”，帧率30fps，图像分辨率84x84：

```bash
python data_processing/convert_to_lerobot_v2.py \
  --h5_path demos/StackCube-v1/motionplanning/20260602_160056.h5 \
  --repo_id stack_cube_rgb_base_hand_episodes_500_84x84 \
  --root data/lerobot/maniskill/stack_cube_rgb_base_hand_episodes_500_84x84 \
  --task "stack cube" \
  --fps 30 \
  --image_width 84 \
  --image_height 84
```

*注意：如果不指定 `--repo_id`，脚本会从 `--root` 的最终目录名推导目标 `repo_id`；`--root` 仍然必须是目标 LeRobot 数据集的精确根目录。*

### 数据后处理

1. 去除部分特征，以下案例将`data/lerobot/maniskill/stack_cube_rgb_base_hand_episodes_500_84x84`目录下的LeRobot格式数据中的`observation.images.hand_camera`特征去除（去除手部相机），并保存到`data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84`目录下：

```bash
python data_processing/remove_feature_lerobot.py \
  --repo_id stack_cube_rgb_base_hand_episodes_500_84x84 \
  --root data/lerobot/maniskill/stack_cube_rgb_base_hand_episodes_500_84x84 \
  --remove_features observation.images.hand_camera \
  --output_dir data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84
```

2. 去除无操作动作（接近静止帧），以下案例将`data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84`目录下的LeRobot格式数据中动作变化小于0.01的相邻帧去除，并保存到`data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84_nonoop`目录下：

```bash
python data_processing/filter_nonoop_lerobot.py \
  --repo_id stack_cube_rgb_base_episodes_500_84x84 \
  --root data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84 \
  --target_repo_id stack_cube_rgb_base_episodes_500_84x84_nonoop \
  --target_root data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84_nonoop \
  --threshold 0.01 
```

*注意：如果不指定 `--target_repo_id`，脚本会从 `--target_root` 的最终目录名推导目标 `repo_id`。*

### Baseline 模型训练（基于LeRobot）

训练LeRobot格式模型需要先准备好配置文件，以下案例使用`configs/policies/diffusion_stack_cube.yaml`配置文件：
```yaml
batch_size: 64
num_workers: 16
steps: 30000
eval_freq: 1000
save_freq: 5000

policy:
  type: diffusion
  vision_backbone: resnet18
  pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
  use_group_norm: false
  optimizer_lr: 1e-4
  push_to_hub: false

dataset:
  repo_id: stack_cube
  root: data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84_no_nonoop

wandb:
  enable: true
```
该配置文件指定了训练的基本参数（如批量大小、训练步数、评估频率等），模型类型为Diffusion Policy，使用ResNet18作为视觉骨干网络，并加载ImageNet预训练权重。数据集来自之前处理好的LeRobot格式数据（使用RGB图像，去除手部相机和无操作帧，共500条轨迹，图像分辨率84x84）。训练过程中启用WandB进行日志记录。

训练和评估入口都会把 `dataset.root` 当作精确的 LeRobot 数据集根目录处理，并要求 `dataset.repo_id` 与该数据集的逻辑标识一致；不会根据目录名自动纠正 `repo_id`。

接下来，运行以下命令开始训练：
```bash
lerobot-train --config configs/policies/diffusion_stack_cube.yaml
```

此外，也可以按类似方式训练`ACT`, `SmolVLA`等模型。

训练完成后，模型权重将保存在`outputs/train/{date}/{time}_diffusion/checkpoints`目录下。

*注意：本配置可以在单张RTX 5060 Laptop GPU (8GB显存)上运行*

### 基于Mixup的模型训练（开发中）

需要封装`LeRobotDataset`的数据加载逻辑，在训练开始之前进行轨迹分析，识别出位姿歧义结点，并在训练过程中动态进行图像Mixup和动作标签的软标签化。

需要实现LeRobot Policy的forward逻辑，例如对于Diffusion Policy，在计算损失时，针对每个混合样本，计算两个目标动作的分数匹配损失，并进行加权求和。

上述功能需要在`embodied_mixup/policies/`中实现，具体细节待开发完成后补充。

### 模型评估（基于LeRobot）

运行以下命令评估训练好的Diffusion Policy模型，指定预训练权重路径、数据集路径等参数：

```bash
python -m emix eval \
  --config configs/eval/default/place_object_scale_left_dp_baseline.yaml
```

`--dataset_root` 必须指向包含 `meta/info.json` 的精确 LeRobot 数据集根目录，`--repo_id` 需要与加载时使用的逻辑数据集标识一致。

*注意：可能需要自行修改`evaluation/policies/lerobot.py`中的特征映射配置，以适配当前数据集的特征结构*
```python
_DEFAULT_OBS_MAPPING = [
    {
        'source': 'sensor_data.base_camera.rgb',
        'target': 'observation.images.base_camera',
        'transform': 'image',
        'transform_kwargs': {
            'target_size': (84, 84),
            'normalize_factor': 255.0,
        }
    },
    {
        'source': 'sensor_data.hand_camera.rgb',
        'target': 'observation.images.hand_camera',
        'transform': 'image',
        'transform_kwargs': {
            'target_size': (84, 84),
            'normalize_factor': 255.0,
        }
    },
    {
        'source': 'agent.qpos',
        'target': 'observation.state',
        'transform': 'tensor',
        'transform_kwargs': {}
    },
]
```

评估完成后，结果将保存在`outputs/evaluation/{task}_lerobot_{policy}_{date}_{time}`目录下，包括评估日志和视频，日志中包含以下指标：
```log
2026-06-01 18:59:29.716 | INFO     | __main__:main:147 - Final Results:
2026-06-01 18:59:29.724 | INFO     | __main__:main:149 - success_once_mean: 0.04999999701976776
2026-06-01 18:59:29.724 | INFO     | __main__:main:149 - return_mean: 44.72536849975586
2026-06-01 18:59:29.724 | INFO     | __main__:main:149 - episode_len_mean: 200.0
2026-06-01 18:59:29.724 | INFO     | __main__:main:149 - reward_mean: 0.22362682223320007
2026-06-01 18:59:29.724 | INFO     | __main__:main:149 - success_at_end_mean: 0.009999999776482582
```

## 致谢

感谢[Maniskill](https://github.com/haosulab/ManiSkill)和[LeRobot](https://github.com/huggingface/lerobot)团队提供的开源工具。
