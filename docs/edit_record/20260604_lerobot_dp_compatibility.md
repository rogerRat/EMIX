# LeRobot Diffusion Policy 与 StackCube 数据集兼容性调研报告

**日期**: 2026-06-04  
**项目**: Embodied-Mixup  
**任务**: 确认 LeRobot Diffusion Policy 是否兼容 EEF-style 数据集，并为两种数据格式创建训练配置

---

## 一、核心结论

**LeRobot Diffusion Policy 对动作格式没有任何先验假设**，完全兼容 EEF delta pose 格式。具体证据：

- **归一化**：纯线性 MIN_MAX / MEAN_STD 映射，基于数据集统计量，无语义假设。无论是绝对关节角度还是 delta pose，归一化公式完全相同：`x_norm = (x - min) / (max - min + 1e-8) * 2 - 1`
- **网络结构**：所有维度由数据集自动推断（`action_feature.shape[0]`），无硬编码。7 维 EEF delta pose 和 8 维 joint_pos 只是维度数不同，网络无需任何修改
- **动作预测头**：`nn.Conv1d(down_dims[0], action_feature.shape[0], 1)` — 输出维度跟随数据集，无关节角度/EEF 特定逻辑
- **无隐含假设**：无 cos/sin 编码、无值域假设、无 delta vs 绝对位置假设

**结论：可直接使用 `lerobot-train --config configs/policies/diffusion_stack_cube_eef_base_nonoop.yaml` 进行训练，无需对 LeRobot 做任何修改。**

---

## 二、调研时间线

### 2.1 初始问题

用户通过以下数据管线处理了 StackCube 演示数据：

```bash
# Step 1: 用 pd_ee_delta_pose 控制器重放轨迹
python data_processing/replay.py \
  --traj-path demos/StackCube-v1/motionplanning/20260603_165716.h5 \
  -c pd_ee_delta_pose -o rgb --save-traj
# 输出: demos/StackCube-v1/motionplanning/20260603_165716.rgb.pd_ee_delta_pose.physx_cpu.h5

# Step 2: 转换为 LeRobot 格式
python data_processing/convert_to_lerobot_v2.py \
  --h5_path demos/StackCube-v1/motionplanning/20260603_165716.h5 \
  --repo_id stack_cube_rgb_base_hand_episodes_500_84x84 ...

# Step 3: 移除 hand_camera
python data_processing/remove_feature_lerobot.py ...

# Step 4: 过滤 no-op 帧
python data_processing/filter_nonoop_lerobot.py ...
```

用户认为最终数据集是 EEF delta pose 格式。

### 2.2 发现问题：旧数据集实际是 joint_pos 格式

逐帧数值对比揭示，最终数据集 `stack_cube_rgb_base_episodes_500_84x84_nonoop` 的 action 与**原始 h5** 完全一致，而非重放后的 h5：

| 数据源 | Action 维度 | 含义 | Action[0] 示例 |
|--------|------------|------|---------------|
| 原始 h5 (`*.h5`) | **8** | 关节角度绝对位置 + 夹爪 | `[0.0353, 0.4007, 0.0196, -1.9187, 0.0374, 2.3366, -0.7664, 1.0]` |
| 重放 h5 (`*.pd_ee_delta_pose.physx_cpu.h5`) | **7** | EE delta pose + 夹爪 | `[1.58e-06, -2.61e-07, 3.87e-06, 0.0, 0.0, -0.0, 1.0]` |
| 旧数据集 | **8** | **= 原始 h5 格式（关节角度）** | `[0.0353, 0.4007, 0.0196, -1.9187, 0.0374, 2.3366, -0.7664, 1.0]` |

**原因**：`convert_to_lerobot_v2.py` 的 `--h5_path` 传入了原始 h5 文件 `20260603_165716.h5`，而非重放后的 `20260603_165716.rgb.pd_ee_delta_pose.physx_cpu.h5`。

### 2.3 用户重新处理数据

用户使用正确的重放 h5 文件重新执行了数据管线，生成了新数据集 `stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop`。

### 2.4 验证新数据集

对新数据集的逐帧验证：

```
Action shape: (7,) = [dx, dy, dz, dax, day, daz, gripper]
Frame 0: [1.58e-06, -2.61e-07, 3.87e-06, 0.0, 0.0, -0.0, 1.0]    # 初始帧 delta 接近零
Frame 1: [-2.39e-03, -4.41e-03, -3.51e-03, 1.10e-03, -6.97e-04, 9.68e-03, 1.0]  # 小幅 delta
Frame 2: [-0.0090, -0.0169, -0.0133, 0.0038, -0.0033, 0.0327, 1.0]
```

**确认**：新数据集是正确的 EEF delta pose 格式。前 6 维为小数值 delta（末端执行器位姿增量），第 7 维为 gripper（-1/1）。

---

## 三、两套数据集对比

| 属性 | 旧数据集 (joint_pos) | 新数据集 (EEF delta pose) |
|------|---------------------|--------------------------|
| 路径 | `stack_cube_rgb_base_episodes_500_84x84_nonoop` | `stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop` |
| Action 维度 | 8 | 7 |
| Action 格式 | `[joint_0..6, gripper]` 绝对位置 | `[dx,dy,dz,dax,day,daz, gripper]` delta pose |
| Action 值域 | 大（如 joint_3: [-2.65, -0.38] rad） | 小（delta 大多在 [-0.3, 0.3]） |
| 总帧数 | 37,508 | 46,245 |
| State 维度 | 9 | 9（相同：7 关节 + 2 手指） |
| Image | 84x84 base_camera | 84x84 base_camera |
| 数据来源 h5 | 原始 h5（错误） | 重放后 h5（正确） |

---

## 四、LeRobot DP 源码兼容性审查

### 4.1 归一化层 (`normalize.py`)

归一化方式由配置 `normalization_mapping` 决定，默认：
- `VISUAL`: `MEAN_STD`（图像用均值/标准差归一化）
- `STATE`: `MIN_MAX`（状态用 min/max 映射到 [-1, 1]）
- `ACTION`: `MIN_MAX`（动作用 min/max 映射到 [-1, 1]）

具体公式：
```python
# Normalize (MIN_MAX):
x_norm = (x - min) / (max - min + 1e-8) * 2 - 1   # 映射到 [-1, 1]

# Unnormalize (MIN_MAX):
x = (x_norm + 1) / 2 * (max - min) + min             # 映射回原始范围
```

**结论**：归一化是纯线性映射，完全基于数据集的统计量（min/max），不依赖 action 的语义。对 EEF delta pose 数据集，min/max 统计量会自动反映 delta 值的范围。

### 4.2 网络结构 (`modeling_diffusion.py`)

- **UNet1D 输入维度** = `config.action_feature.shape[0]`（由数据集自动推断：新数据集为 7）
- **UNet1D 输出维度** = 同上（7）
- **全局条件维度** = `state_dim(9) * n_obs_steps + image_feature_dim * n_obs_steps`
- **无硬编码维度**，完全由配置驱动

### 4.3 动作预测头

```python
self.final_conv = nn.Sequential(
    DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
    nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
)
```

输出维度 = `action_feature.shape[0]` = 7（EEF 数据集），由数据集自动推断，无语义假设。

### 4.4 完整的先验假设检查

| 假设类型 | 是否存在 | 说明 |
|----------|---------|------|
| cos/sin 关节编码 | 否 | 无任何角度编码/解码逻辑 |
| 动作维度硬编码 | 否 | 由数据集自动推断 |
| 值域裁剪假设 | 否 | clip_sample_range=1.0 是归一化后的裁剪，归一化到 [-1,1] 后自然满足 |
| 关节角度特定处理 | 否 | 所有维度一视同仁 |
| Delta vs 绝对位置假设 | 否 | 模型不关心动作的语义 |
| EEF 坐标系假设 | 否 | 无任何笛卡尔/关节空间转换逻辑 |

---

## 五、配置文件

### 5.1 原则：创建新配置文件，不修改原文件

原始配置文件保留不动，为每套数据集创建独立的配置文件。

### 5.2 新数据集（EEF delta pose）配置文件

**`configs/policies/diffusion_stack_cube_eef_base_nonoop.yaml`**

```yaml
# Diffusion Policy config for StackCube
# Dataset: eef_delta_pose (6-DoF EE delta pose + gripper), base_camera only, no-op filtered
# Action dim: 7  [dx, dy, dz, dax, day, daz, gripper]
# State dim: 9   [joint_0..joint_6, finger_left, finger_right]
# Image: 84x84 RGB (base_camera)
# Episodes: 500, Frames: 46245

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
  repo_id: stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop
  root: data/lerobot/maniskill/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop

wandb:
  enable: true
```

**`configs/policies/act_stack_cube_eef_base_nonoop.yaml`**

```yaml
# ACT Policy config for StackCube
# Dataset: eef_delta_pose (6-DoF EE delta pose + gripper), base_camera only, no-op filtered
# Action dim: 7  [dx, dy, dz, dax, day, daz, gripper]
# State dim: 9   [joint_0..joint_6, finger_left, finger_right]
# Image: 84x84 RGB (base_camera)
# Episodes: 500, Frames: 46245

batch_size: 64
num_workers: 16
steps: 100000
eval_freq: 1000
save_freq: 10000

policy:
  type: act
  chunk_size: 16
  n_action_steps: 8
  vision_backbone: resnet18
  pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
  optimizer_lr: 1e-4
  push_to_hub: false

dataset:
  repo_id: stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop
  root: data/lerobot/maniskill/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop

wandb:
  enable: true
```

### 5.3 旧数据集（joint_pos）配置文件

**`configs/policies/diffusion_stack_cube_jointpos_base_nonoop.yaml`**

```yaml
# Diffusion Policy config for StackCube
# Dataset: joint_pos (absolute joint positions + gripper), base_camera only, no-op filtered
# Action dim: 8  [joint_0..joint_6, gripper]
# State dim: 9   [joint_0..joint_6, finger_left, finger_right]
# Image: 84x84 RGB (base_camera)
# Episodes: 500, Frames: 37508

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
  repo_id: stack_cube_rgb_base_episodes_500_84x84_nonoop
  root: data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84_nonoop

wandb:
  enable: true
```

### 5.4 文件名命名规则

命名格式：`{policy_type}_{task}_{action_format}_{camera}_{processing}.yaml`

| 组件 | 含义 | 示例 |
|------|------|------|
| `policy_type` | 策略类型 | `diffusion`, `act` |
| `task` | 任务名 | `stack_cube` |
| `action_format` | 动作格式 | `eef` (EEF delta pose), `jointpos` (关节角度) |
| `camera` | 观测相机 | `base` (仅 base_camera) |
| `processing` | 数据处理 | `nonoop` (过滤no-op帧) |

### 5.5 数据集统计量

#### 新数据集（EEF delta pose, 7维 action）

**Action (7维)**:
| 维度 | 含义 | min | max | mean | std |
|------|------|-----|-----|------|-----|
| 0 | dx (m) | -0.279 | 0.263 | -0.0005 | 0.064 |
| 1 | dy (m) | -0.576 | 0.561 | -0.0006 | 0.118 |
| 2 | dz (m) | -0.325 | 0.289 | -0.026 | 0.096 |
| 3 | dax (rad) | -0.094 | 0.080 | 0.0002 | 0.011 |
| 4 | day (rad) | -0.216 | 0.213 | 0.004 | 0.025 |
| 5 | daz (rad) | -1.000 | 1.000 | 0.002 | 0.238 |
| 6 | gripper | -1.0 | 1.0 | -0.007 | 1.000 |

**State (9维)**: 同旧数据集，7 个关节角度 + 2 个手指位置

**EEF delta pose 特点**：
- 前 6 维（delta pose）值域较小（大部分在 [-0.3, 0.3] 范围内），归一化后分布紧凑
- `daz` 维度 std 较大（0.238），因为包含了大角度旋转的 delta
- `gripper` 维度为离散二值（-1 / 1）

#### 旧数据集（joint_pos, 8维 action）

**Action (8维)**:
| 维度 | 含义 | min | max | mean | std |
|------|------|-----|-----|------|-----|
| 0 | joint_0 (rad) | -0.492 | 0.521 | -0.001 | 0.168 |
| 1 | joint_1 (rad) | 0.062 | 1.473 | 0.593 | 0.220 |
| 2 | joint_2 (rad) | -0.267 | 0.220 | -0.002 | 0.070 |
| 3 | joint_3 (rad) | -2.650 | -0.382 | -1.879 | 0.364 |
| 4 | joint_4 (rad) | -0.210 | 0.254 | 0.002 | 0.065 |
| 5 | joint_5 (rad) | 1.829 | 2.977 | 2.470 | 0.172 |
| 6 | joint_6 (rad) | -1.910 | 0.454 | -0.785 | 0.431 |
| 7 | gripper | -1.0 | 1.0 | -0.028 | 1.000 |

---

## 六、潜在注意事项

### 6.1 离散 Gripper 维度

两个数据集的 gripper 维度都是离散二值（-1/1）。Diffusion Policy 作为连续扩散模型，学习离散分布的效果会劣于连续分布。可能的影响：
- 生成动作在 grip 维度上可能出现中间值（如 0.3），需要在部署时阈值化
- 训练收敛可能稍慢，但不会导致训练失败

### 6.2 EEF vs Joint Pos 的归一化差异

| 特性 | Joint Pos (8维) | EEF Delta Pose (7维) |
|------|----------------|---------------------|
| 值域范围 | 大（如 joint_3: [-2.65, -0.38]） | 小（delta 值大多在 [-0.3, 0.3]） |
| 归一化后分布 | 某些维度偏移较大 | 分布更紧凑、对称 |
| 学习难度 | 可能需要更多 epoch 收敛 | delta 值更平滑，通常更易学习 |

### 6.3 配置中其他参数

当前配置的其他参数沿用 LeRobot 默认值：
- `horizon=16` 和 `n_action_steps=8`：DP 论文推荐值
- `crop_shape` 默认为 `(84, 84)`：恰好匹配图像分辨率，等于不做裁剪
- `down_dims=(512, 1024, 2048)`：默认值，可根据 GPU 显存调整

---

## 七、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `configs/policies/diffusion_stack_cube.yaml` | **保留不动** | 原配置保留 |
| `configs/policies/act_stack_cube.yaml` | **保留不动** | 原配置保留 |
| `configs/policies/diffusion_stack_cube_jointpos_base_nonoop.yaml` | **新建** | 旧数据集（joint_pos 8维）DP 配置 |
| `configs/policies/diffusion_stack_cube_eef_base_nonoop.yaml` | **新建** | 新数据集（EEF delta pose 7维）DP 配置 |
| `configs/policies/act_stack_cube_eef_base_nonoop.yaml` | **新建** | 新数据集（EEF delta pose 7维）ACT 配置 |
| `docs/edit_record/20260604_lerobot_dp_compatibility.md` | **新建** | 本文档 |