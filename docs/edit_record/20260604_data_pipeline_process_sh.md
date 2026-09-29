# Data Processing Pipeline: _process.sh 重构与调研

- **日期**: 2026-06-04
- **修改文件**:
  - `scripts/process_cfg.yaml` (新建)
  - `scripts/_process.sh` (重写)
  - `data_processing/_process.sh` (删除，功能迁移至 `scripts/_process.sh`)

---

## 1. 调研结论：Replay 步骤的必要性

### 1.1 核心问题

README 提到的数据重放（Replay）步骤是否是必须的？不运行 Replay，数据是否完整？

### 1.2 调研过程

检查了 ManiSkill 数据收集脚本 `data_collection/panda/run.py` 的参数和行为：

- **默认 `obs_mode="none"`**：不保存任何观测数据（RGB 图像、qpos 等），仅保存 `actions`、`env_states`、`terminated`、`truncated`、`success`。
- **使用 `-o rgb` 参数时**：保存完整的 RGB 观测数据（`sensor_data/base_camera/rgb`、`sensor_data/hand_camera/rgb`）和本体感受数据（`agent/qpos`、`agent/qvel`）。

验证了当前已有数据的实际结构：

```
demos/StackCube-v1/motionplanning/20260603_165716.h5
├── traj_0/
│   ├── obs/
│   │   ├── sensor_data/
│   │   │   ├── base_camera/rgb: (108, 512, 512, 3)  <- 有观测数据
│   │   │   └── hand_camera/rgb: (108, 512, 512, 3)  <- 有观测数据
│   │   ├── agent/
│   │   │   ├── qpos: (108, 9)                        <- 有本体感受
│   │   │   └── qvel: (108, 9)
│   │   ├── sensor_param/ (相机内外参)
│   │   └── extra/tcp_pose: (108, 7)
│   ├── actions: (107, 8)   <- pd_joint_pos, 8维
│   ├── env_states/
│   ├── terminated, truncated, success
```

JSON 元数据确认收集时的配置：

```json
{
  "env_info": {
    "env_kwargs": {
      "obs_mode": "rgb",
      "control_mode": "pd_joint_pos",
      ...
    }
  }
}
```

### 1.3 结论

**当前数据收集用了 `-o rgb`，h5 文件已包含完整观测数据，不 Replay 也不会缺少观测。**

Replay 步骤的**唯一作用是控制模式转换**：

| 控制模式 | 动作维度 | 含义 | 获取方式 |
|----------|----------|------|----------|
| `pd_joint_pos` | 8 | 7个关节目标位置 + 1个夹爪 | 数据收集时直接产生 |
| `pd_ee_delta_pose` | 7 | dx,dy,dz,droll,dpitch,dyaw + 夹爪 | 必须通过 Replay 重放获得 |

因此，流程设计为**两条可选路径**：

| 路径 | 配置 | 是否 Replay | 动作空间 | 输出命名 |
|------|------|-------------|----------|----------|
| **EEF 路径** | `convert_control=true` | 是 | `pd_ee_delta_pose` (7维) | `stack_cube_rgb_...` |
| **Joint-space 路径** | `convert_control=false` | 否 | `pd_joint_pos` (8维) | `stack_cube_joint_rgb_...` |

> **注意**：如果收集数据时使用默认的 `obs_mode=none`，则必须通过 Replay（带 `-o rgb`）来获取观测数据，此时无论是否转换控制模式都需要 Replay。

---

## 2. 修改内容

### 2.1 新建 `scripts/process_cfg.yaml`

集中管理所有流水线参数，脚本唯一入口为 `bash scripts/_process.sh`，不接受命令行参数。

配置项分 5 个区块：

| 区块 | 关键参数 | 说明 |
|------|----------|------|
| Step 1: 收集 | `task`, `num_traj`, `obs_mode` | ManiSkill 轨迹生成 |
| Step 2: Replay | `replay_enabled`, `convert_control`, `target_control_mode` | 控制模式转换与观测重录制 |
| Step 3: 转换 | `fps`, `image_width`, `image_height` | H5 -> LeRobot 格式 |
| Step 4: 去特征 | `remove_features` | 移除指定特征（如手部相机） |
| Step 5: 去no-op | `noop_threshold`, `noop_norm_type` | 过滤接近静止帧 |

### 2.2 重写 `scripts/_process.sh`

核心设计决策：

1. **纯配置驱动**：所有参数从 `process_cfg.yaml` 读取，`cfg_val()` / `cfg_bool()` 函数解析 YAML 键值对。
2. **两条路径由配置决定**：`convert_control=true` 走 EEF 路径（含 Replay），`false` 走 Joint-space 路径（跳过 Replay）。
3. **校验逻辑**：`convert_control=true` 且 `replay_enabled=false` 时报错终止，因为控制模式转换必须通过 Replay 实现。
4. **命名约定自动推导**：
   - `StackCube-v1` -> `stack_cube`（camelCase 拆分为 snake_case）
   - EEF 路径：`stack_cube_rgb_base_hand_episodes_500_84x84`
   - Joint 路径：`stack_cube_joint_rgb_base_hand_episodes_500_84x84`
   - 去特征后：`base_hand` -> `base`
   - 去no-op后：追加 `_nonoop`
5. **H5 文件自动定位**：在 `demos/{task}/motionplanning/` 目录下查找最新的 `.h5` 文件，优先选择不含 `.rgb.` 的原始文件，Replay 后自动切换到重放输出文件。
6. **每步可独立开关**：5 个步骤各自有 `enabled` 开关，可跳过已完成的步骤。

### 2.3 删除 `data_processing/_process.sh`

旧文件硬编码参数且使用命令行参数，与新的配置驱动设计冲突，已删除。

---

## 3. 流水线执行流程

```
┌─────────────────────────────────────────────────────────────┐
│              process_cfg.yaml (全量参数配置)                   │
└─────────────┬───────────────────────────────────────────────┘
              │ 读取
              v
┌─────────────────────────────────────────────────────────────┐
│              scripts/_process.sh                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 数据收集                                            │
│    run.py -e {task} -o {obs_mode} -n {num_traj}             │
│    ↓ 产出: demos/{task}/motionplanning/*.h5                  │
│                                                             │
│  Step 2: Replay (可选)                                       │
│    ├── convert_control=true:                                │
│    │   replay.py --traj_path {h5} -c pd_ee_delta_pose -o rgb│
│    │   ↓ 产出: *.rgb.pd_ee_delta_pose.physx_cpu.h5          │
│    └── convert_control=false: 跳过                            │
│                                                             │
│  Step 3: 格式转换                                            │
│    convert_to_lerobot_v2.py --h5_path {h5} ...               │
│    ↓ 产出: data/lerobot/maniskill/{repo_id}_base_hand_...    │
│                                                             │
│  Step 4: 去特征                                              │
│    remove_feature_lerobot.py --remove_features hand_camera   │
│    ↓ 产出: data/lerobot/maniskill/{repo_id}_base_...         │
│                                                             │
│  Step 5: 去no-op帧                                           │
│    filter_nonoop_lerobot.py --threshold 0.01                 │
│    ↓ 产出: data/lerobot/maniskill/{repo_id}_nonoop           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 使用方式

```bash
# 编辑配置
vim scripts/process_cfg.yaml

# EEF 路径 (默认配置)
#   replay_enabled: true
#   convert_control: true
bash scripts/_process.sh

# Joint-space 路径
#   replay_enabled: false
#   convert_control: false
bash scripts/_process.sh

# 仅运行后处理（跳过收集）
#   collection_enabled: false
bash scripts/_process.sh
```

---

## 5. 经验记录

- `run.py` 中 `control_mode` 硬编码为 `"pd_joint_pos"`，`-o` 参数仅控制 `obs_mode`。如需 EEF 控制模式必须通过 Replay 转换。
- `convert_to_lerobot_v2.py` 的 `--task` 参数已自动从 `TASK` 配置推导为 snake_case 格式。
- ManiSkill 重放产生的 h5 文件命名格式为 `<原始名>.<obs_mode>.<control_mode>.<backend>.h5`，脚本通过此模式定位重放输出。
- `run.py` 默认 `obs_mode="none"`，如果未来有人用默认参数收集数据，h5 中将**没有**观测数据，必须通过 Replay 获取。