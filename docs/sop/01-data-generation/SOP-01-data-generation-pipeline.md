# SOP-01：RoboTwin 数据生成与 LeRobot 转换

本文档使用 `place_object_scale_left` 作为唯一示例，覆盖 RoboTwin expert 数据
采集、转换为标准 LeRobot 数据集，以及构建 EMIX 所需的 `EmixIndex`。
当前评估协议只支持左臂作为运动臂，右臂不能配置为运动臂。因此，本文档中的
`left` 不是可切换的左右选项，而是单臂任务的统一语义：左臂运动，右臂保持
home qpos。


## RoboTwin 基准与首轮协议背景（按 EXP.md 核对当前参数）

The active benchmark is RoboTwin. Its fixed source is tracked directly under
`third_party/RoboTwin`; it is not a Git submodule. Use `emix` for conversion,
EMIX indexing and training, and `emix-twin` for the simulator. From the project
root, install each environment with its own entry point:

```bash
bash dependencies/_install.sh
bash dependencies/_install_twin.sh
bash dependencies/_install_twin.sh --check
```

On a headless server with NVIDIA graphics exposed, the twin installer first
repairs and verifies Vulkan/EGL and SAPIEN through `dependencies/_instal_sapien.sh`.
It then installs the CUDA runtime and Curobo. On Blackwell GPUs it builds OIDN
2.4.1 for the visible GPU architecture because SAPIEN's bundled OIDN 2.0.1
does not support `sm_120`. Keep `/usr/local/cuda` out of a Blackwell runtime's
library path when it contains an older CUDA build. The installer reuses
existing Conda environments; simulator assets and policy checkpoints are
separate. See [SOP-00](../04-environment-and-rendering/SOP-00-environment-installation.md) for installation
and [the measured repair log](../04-environment-and-rendering/20261019_emix_sapien_runtime_repair.md) for the
RTX 5090 Vulkan/SAPIEN acceptance result.

The first experiment is `click_bell_left`: a dual-Piper scene where the left
Piper is the sole task actor and the right Piper stays at its home qpos. The
only policy image is `head_camera`; the policy state/action are the complete
two-Piper qpos vectors. Right-arm action labels are clamped to its initial home
qpos. The left EEF pose and left gripper signal are index-only features and are
not policy inputs.

The Piper separation is an explicit first-round configuration value of `0.60`
meters. The official Piper asset defines the individual robot pose but does not
publish a two-Piper separation value; the first collection smoke test must
validate this value. Use exactly the same value with `scripts/robotwin_eval.sh`.

Run the closed loop in order:

```bash
scripts/robotwin_collect.sh
scripts/robotwin_convert.sh
scripts/robotwin_train.sh
scripts/robotwin_eval.sh <piper-distance-m> <baseline-checkpoint> <emix-checkpoint>
```

The first protocol collects 200 expert episodes, trains baseline and EMIX
Diffusion Policy for 30,000 steps, and evaluates each checkpoint for 100 valid
RoboTwin seeds. EMIX mixes only `observation.images.head_camera`; state remains
the anchor state and EMIX batch probability is 30 percent.

ManiSkill-only source code and configuration are preserved under
`legacy/maniskill/`; the pre-migration backup is under
`backups/maniskill_migration_20260715045826/`.

本文档中的采集命令使用仓库包装器
[`data_collection/robotwin.py`](../../../data_collection/robotwin.py)，由包装器在
RoboTwin 根目录调用 `script/collect_data.py`，并在成功后将原始 collection
移动到项目根目录的 `data/` 下。RoboTwin 命令使用 `emix-twin` 环境；数据转换
和 EMIX 命令使用 `emix` 环境。两个环境不能混用。

命令环境约定：每个 `bash` 代码块第一行的 `# Environment:` 是该代码块应使用的
Conda 环境；`emix-twin` 仅用于 RoboTwin 采集，`emix` 用于转换、索引和训练侧
工具。Git 命令不依赖这两个环境，可在任一 shell 执行。

## 1. 固定协议

| 项目 | 值 | 说明 |
|------|------|------|
| RoboTwin task | `place_object_scale_left` | 左臂单臂放置物体到电子秤任务 |
| task config | `place_object_scale_left` | `third_party/RoboTwin/task_config/place_object_scale_left.yml` |
| 主动机械臂 | `left` | 唯一支持的运动臂 |
| 非主动机械臂 | 右侧 Piper | 不能设置为运动臂，保持 home qpos |
| embodiment | `[piper, piper, 0.60]` | 两个 Piper，间距为 0.60 米 |
| expert episodes | `500` | 目标成功 episode 数量 |
| 源数据 FPS | `17` | 根据源配置 `save_freq: 15` 得到 |
| 策略图像 | `head_camera` | `D435-W` 头部相机（60° FOV，320×240） |
| 腕部相机 | 关闭 | 不采集 wrist camera |
| 索引特征 | `observation.eef_pose` | 仅用于构建 EMIX 索引 |

物体和电子秤会根据每个 seed 随机选择。`arm_side: left` 表示当前唯一支持的
左臂运动协议；不要将其改为 `right`。

## 2. 初始化仓库和环境

从仓库根目录执行。`third_party/RoboTwin` 已直接纳入仓库；首次使用时按
[SOP-00](../04-environment-and-rendering/SOP-00-environment-installation.md) 安装并验收 `emix-twin`：

```bash
# Environment: 任一 shell；脚本自行创建或复用 emix-twin
bash dependencies/_install_twin.sh
bash dependencies/_install_twin.sh --check
```

`_install_twin.sh --check` 会运行 SAPIEN 图形探针，并检查仿真依赖；
`SAPIEN_RENDER=PASS` 只确认基础图形链路，不代表任务资产已齐全。

采集前激活 RoboTwin 环境：

```bash
# Environment: emix-twin
conda activate emix-twin
```

| 命令或参数 | 类型 | 传入值 | 作用 |
|------------|------|--------|------|
| `conda activate` | Conda 子命令 | - | 激活当前 shell 的 Conda 环境 |
| `emix-twin` | 环境名 | `emix-twin` | 提供 RoboTwin、SAPIEN 和 MPLib 运行时依赖 |

RoboTwin 的安装和资产下载不属于本数据闭环。执行采集前，必须确认该环境可以
正常导入 RoboTwin 运行时依赖。

## 3. RoboTwin 任务配置

采集命令读取以下现有配置文件：

```text
third_party/RoboTwin/task_config/place_object_scale_left.yml
```

以下表格列出该配置文件的全部参数。修改任意配置都代表一个新的数据协议，
不同配置产生的数据不能合并到同一个输出目录。

### 3.1 顶层参数

| 参数 | 值 | 说明 |
|------|------|------|
| `render_freq` | `0` | 不保存渲染帧 |
| `episode_num` | `500` | 目标成功 episode 数量 |
| `use_seed` | `false` | 不使用已有 seed 列表 |
| `save_freq` | `15` | 源轨迹保存频率 |
| `embodiment` | `[piper, piper, 0.60]` | 左 Piper、右 Piper 及两臂间距 |
| `language_num` | `1` | 为每个 episode 生成至少一个 RoboTwin 文字提示 |
| `pcd_down_sample_num` | `1024` | 点云下采样数量；当前点云已关闭 |
| `pcd_crop` | `true` | 点云裁剪开关；当前点云已关闭 |
| `save_path` | `./data` | RoboTwin 内部临时输出根目录；采集成功后由包装器移出 |
| `clear_cache_freq` | `5` | 每 5 个 episode 清理一次缓存 |
| `collect_data` | `true` | 保存 HDF5 数据 |
| `eval_video_log` | `true` | 保留评估视频日志 |

### 3.2 `domain_randomization` 参数

| 参数 | 值 | 说明 |
|------|------|------|
| `domain_randomization.random_background` | `false` | 关闭背景随机化 |
| `domain_randomization.cluttered_table` | `false` | 不增加桌面杂物 |
| `domain_randomization.clean_background_rate` | `1` | 干净背景比例 |
| `domain_randomization.random_head_camera_dis` | `0` | 固定头部相机距离 |
| `domain_randomization.random_table_height` | `0` | 固定桌面高度 |
| `domain_randomization.random_light` | `false` | 关闭光照随机化 |
| `domain_randomization.crazy_random_light_rate` | `0` | 强随机光照比例 |

### 3.3 `camera` 参数

| 参数 | 值 | 说明 |
|------|------|------|
| `camera.head_camera_type` | `D435-W` | 头部相机型号（60° FOV，320×240） |
| `camera.wrist_camera_type` | `D435` | 腕部相机型号；采集开关为 false |
| `camera.collect_head_camera` | `true` | 采集头部相机图像 |
| `camera.collect_wrist_camera` | `false` | 不采集腕部相机图像 |

### 3.4 `data_type` 参数

| 参数 | 值 | 说明 |
|------|------|------|
| `data_type.rgb` | `true` | 保存 RGB 图像 |
| `data_type.third_view` | `false` | 不保存第三视角 |
| `data_type.depth` | `false` | 不保存深度图 |
| `data_type.pointcloud` | `false` | 不保存点云 |
| `data_type.observer` | `false` | 不保存 observer 数据 |
| `data_type.endpose` | `true` | 保存末端位姿，供 EMIX 建立索引 |
| `data_type.qpos` | `true` | 保存双 Piper qpos |
| `data_type.mesh_segmentation` | `false` | 不保存 mesh segmentation |
| `data_type.actor_segmentation` | `false` | 不保存 actor segmentation |

## 4. 采集 expert episodes

从仓库根目录在已激活的 `emix-twin` 环境中调用仓库包装器：

```bash
# Environment: emix-twin
python data_collection/robotwin.py place_object_scale_left \
  --task-config place_object_scale_left \
  --cuda 0 --episodes 300
```

`--task-config` 可以省略；省略时自动使用与 task 同名的配置。若使用自定义
配置，配置文件必须存在于 `third_party/RoboTwin/task_config/`，且输出目录必须
尚未存在。采集前可使用 `--dry-run` 检查命令和路径。

### 4.1 Shell 和运行时参数

| 命令或参数 | 类型 | 值 | 作用 |
|------------|------|----|------|
| `python data_collection/robotwin.py` | Python 入口 | 仓库根目录 | 校验任务、调用 RoboTwin 并移动输出 |
| `--task-config` | 可选参数 | `place_object_scale_left` | task config 文件名，不含 `.yml` 后缀 |
| `--cuda` | 可选参数 | `0` | 设置 `CUDA_VISIBLE_DEVICES` |
| `--episodes` | 可选参数 | `300`（默认） | 设置目标成功 episode 数，并覆盖 RoboTwin 模板值 |
| `--python` | 可选参数 | 当前 Python | 指定 RoboTwin 使用的 Python 可执行文件 |
| `--dry-run` | flag | - | 只打印命令和路径，不启动采集 |

### 4.2 包装器与 `collect_data.py` 参数

包装器的 `task` 是位置参数，RoboTwin 内部入口仍接收两个位置参数：

| 参数 | 类型 | 值 | 说明 |
|------|------|----|------|
| `task_name` | 位置字符串 | `place_object_scale_left` | 加载 `envs/place_object_scale_left.py` |
| `task_config` | 位置字符串 | `place_object_scale_left` | 加载对应 `.yml` 文件，不包含 `.yml` 后缀 |

输出路径由配置中的 `save_path`、task name 和 task config 拼接得到：

```text
data/place_object_scale_left/place_object_scale_left/
  data/episode0.hdf5
  data/episode1.hdf5
  ...
  seed.txt
  scene_info.json
```

采集完成后必须确认 `data/` 中恰好有 500 个 `episode*.hdf5` 文件，并确认每个
episode 都包含以下字段：

```text
observation/head_camera/rgb
joint_action/vector
endpose/left_endpose
```

如果数量不足或存在失败 episode，应修复配置并输出到新的目录后重新采集。
不能通过修改 HDF5、`seed.txt` 或元数据伪造成功数量。

## 5. 转换为 LeRobot

打开新的 shell，激活 `emix` 环境，并从仓库根目录直接运行转换程序：

```bash
# Environment: emix
python data_processing/convert_to_lerobot.py \
  --src data/robotwin/place_object_scale_left/place_object_scale_left_wide \
  --tgt data/lerobot/place_object_scale_left_wide
```

```bash
# Environment: emix
cd <repository-root>
conda activate emix
python data_processing/convert_to_lerobot.py \
  --src data/place_object_scale_left/place_object_scale_left \
  --tgt data/lerobot/place_object_scale_left_piper_200_head_84 \
  --instruction-type seen \
  --image-width 84 \
  --image-height 84 \
  --fps 17 \
  --left-qpos-dim 7
```

### 5.1 Shell 和运行时参数

| 命令或参数 | 类型 | 值 | 作用 |
|------------|------|----|------|
| `cd` | Shell builtin | `<repository-root>` | 使 `data_processing/` 和 `data/` 从仓库根目录解析 |
| `conda activate` | Conda 子命令 | `emix` | 激活 LeRobot、OpenCV 和 EMIX 依赖 |
| `python` | 程序 | `emix` 环境中的 Python | 运行转换入口 |

### 5.2 `convert_to_lerobot.py` CLI 参数

| 参数 | 类型 | 值 | 说明 |
|------|------|----|------|
| `--src` | 必填路径 | `data/place_object_scale_left/place_object_scale_left` | RoboTwin collection 源目录，必须包含 `data/episode*.hdf5` |
| `--tgt` | 必填路径 | `data/lerobot/place_object_scale_left_piper_200_head_84` | 新 LeRobot 数据集目标目录，不能包含已有数据 |
| `--repo-id` | 可选字符串 | 无 | LeRobot repository ID；未提供时从 `--tgt` 最后一级目录名推断 |
| `--instructions-root` | 可选路径 | 默认 `<data-root>/instructions` | RoboTwin episode instruction JSON 目录 |
| `--instruction-type` | 可选枚举 | `seen` | 选择 `seen` 或 `unseen` 提示 |
| `--instruction-index` | 可选整数 | `0` | 每个 episode 的提示列表索引 |
| `--task` | 可选字符串 | 无 | 仅在 instruction sidecar 缺失/为空时的旧数据回退文本 |
| `--expected-episodes` | 已弃用 | - | 兼容旧命令但不生效；转换器会将源目录中的全部 episode 转换 |
| `--image-width` | 可选整数 | `84` | 输出图像宽度 |
| `--image-height` | 可选整数 | `84` | 输出图像高度 |
| `--fps` | 可选整数 | `17` | LeRobot 数据集 FPS |
| `--left-qpos-dim` | 可选整数 | `7` | 左 Piper qpos 维度；右臂标签固定为初始 home qpos |

`repo_id` 可以通过 `--repo-id` 显式传入；未传入时从 `--tgt` 的最后一级目录名
推断。例如上述命令默认推断为 `place_object_scale_left_piper_200_head_84`。

转换器默认从 `<data-root>/instructions/episodeN.json` 读取 RoboTwin 为每个
episode 生成的文字提示，并将所选提示写入该 episode 的每一帧 `task` 字段。
`--instruction-type seen` 使用训练提示，`--instruction-type unseen` 使用未见提示；
每个 JSON 中默认使用第一个提示，可用 `--instruction-index` 选择其他提示。

如果 instruction sidecar 缺失或为空且没有提供回退，转换器会直接失败。只有显式
提供 `--task` 时，才会将它作为旧数据兼容回退，并输出使用回退的 episode 数量；
它不应再作为正常 RoboTwin 数据的主任务文本来源。当前协议必须在采集阶段生成
instruction sidecar（`language_num` 不能为 `0`）。

转换结果必须是一个完整的 LeRobot 数据集根目录：

```text
data/lerobot/place_object_scale_left_piper_200_head_84/
  meta/info.json
  meta/stats.json
  data/
  videos/observation.images.head_camera/
```

转换器会创建以下特征：

| 特征 | 用途 |
|------|------|
| `observation.images.head_camera` | 策略图像输入 |
| `observation.state` | 完整双 Piper qpos 状态 |
| `action` | 下一帧 qpos 动作；右臂固定为 home qpos |
| `observation.eef_pose` | EMIX 交点构建所需的左末端位姿 |
| `observation.left_gripper` | EMIX 索引辅助特征 |

## 6. 构建 EmixIndex

在仓库根目录、`emix` 环境中直接调用 EMIX CLI：

```bash
# Environment: emix
conda activate emix
python -m emix build \
  --root data/lerobot/ \
  --output_dir data/emix_index/
```

### 6.1 `python -m emix build` CLI 参数

| 参数 | 类型 | 值 | 说明 |
|------|------|----|------|
| `build` | 子命令 | `build` | 构建 `EmixIndex` |
| `--repo_id` | 可选字符串 | `place_object_scale_left_piper_200_head_84` | 必须与 LeRobot 数据集 ID 一致 |
| `--root` | 必填路径 | `data/lerobot/place_object_scale_left_piper_200_head_84` | 精确的 LeRobot 数据集根目录 |
| `--output_dir` | 可选路径 | `data/emix_index/place_object_scale_left_piper_200_head_84` | 索引输出目录 |
| `--training_candidates_include_noise` | Flag | 启用 | 将 noise group 标记为可训练 |
| `--training_candidates_seed` | 整数 | `0` | 记录在候选元数据中的随机种子 |

命令中未显式传入的候选生成参数使用
`embodied_mixup/configs/config.py` 中的默认值。如果需要修改 `xyz_eps`、
`rpy_eps`、未来动作过滤或 HDBSCAN 参数，必须将其显式加入命令并记录实验值，
不要只修改结果文件。

命令必须生成：

```text
data/emix_index/place_object_scale_left_piper_200_head_84/emix_index.json
```

该文件中的 `public.metadata.source_dataset_fingerprint` 必须与当前 LeRobot
数据集根目录一致。训练时使用的 `dataset.root`、`dataset.repo_id` 和
`emix.index_path` 必须与本文档保持一致。

## 7. 闭环检查和后续 SOP

本 SOP 完成的数据闭环为：

```text
RoboTwin task config
  -> place_object_scale_left expert HDF5
  -> LeRobot dataset
  -> EmixIndex
```

开始训练前检查：

| 检查项 | 预期值 |
|--------|--------|
| native episode 数量 | `500` |
| 主动机械臂 | `left`，不能改为 `right` |
| Piper 间距 | `0.60` |
| 数据集 FPS | `17` |
| 策略图像 | `observation.images.head_camera`，尺寸 `84 x 84` |
| 索引辅助特征 | 存在 `observation.eef_pose` |
| 索引 fingerprint | 与当前数据集根目录一致 |

交点构建和可视化见
[SOP-02](../02-emix-processing/SOP-02-intersection-clustering-visualization.md)，策略训练见
[SOP-03](../03-train-and-inference/SOP-03-model-training.md)，RoboTwin 在线评估见
[SOP-04](../03-train-and-inference/SOP-04-evaluation.md)。这些下游文档中的 task、dataset root、
repository ID 和评估 task config 也必须使用本文档的
`place_object_scale_left` 协议。
