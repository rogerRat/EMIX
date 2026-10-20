# SOP-03 模型训练

本 SOP 中所有训练命令均在 `emix` 环境运行；`emix-twin` 不参与训练。

本文档描述基于 RoboTwin 数据的当前训练流程。训练入口既可以读取普通 LeRobot 数据集，也可以直接从 RoboTwin 转换得到的源 LeRobot 数据集 + EmixIndex 进行 EMIX 混合训练，无需物化中间数据集。

## 1. 目标

本 SOP 只描述当前可复现的训练闭环，不描述旧的物化 (`materialize`) 路径。

训练目标是确认以下内容能够闭环：

- 普通策略可以继续读取标准 LeRobot 数据集
- `emix_diffusion`、`emix_smolvla`、`emix_act` 直接从源数据集 + EmixIndex 读取训练样本
- `emix.training.enabled` 只在 EMIX 策略上启用时才会走混合训练分支
- EmixIndex fingerprint 不一致时加载阶段直接失败

## 2. 必需输入

运行训练前，必须准备以下内容。

- 已激活的 `emix` 环境
- 标准 LeRobot 数据集根目录
- 如果要训练 `emix_*` 策略，还需要一个由 `python -m emix build` 生成的 `emix_index.json`
- 位于 `configs/train/` 下的训练配置文件
- 配置中必须正确填写 `dataset.root` 和 `dataset.repo_id`
- 如果是 `emix_*` 策略，配置中还必须填写 `emix.index_path`

训练配置只描述训练本身，不放 `evaluation:` 区块。评估参数属于
`configs/eval/`，由单独的 `python -m emix eval --config ...` 入口读取；训练入口
不会读取它们。

统一训练入口是：

```bash
# Environment: emix
python train/train.py --config <config_path>

# 等价的兼容入口
python -m emix train --config <config_path>
```

### 2.1 命令行参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `--config` | 是 | 本地 YAML 配置文件路径 |
| `--resume_from` | 否 | Checkpoint 目录或 `pretrained_model` 路径，用于断点续训 |
| `--output_dir` | 否 | 覆盖配置中的输出目录 |
| `--seed` | 否 | 全局随机种子 |
| `--start_step` | 否 | 恢复训练时的起始步数 |
| `--total_steps` | 否 | 训练总步数 |

代码里会在训练开始前创建日志文件，并把控制台输出同时写入 `logs/train_<YYYYMMDDHHMMSS>.log`。
如果 `output_dir` 已存在且非空，训练会直接拒绝继续。

### 2.2 当前可直接使用的配置

当前 `place_object_scale_left` 的训练配置为：

```text
configs/train/default/place_object_scale_left_dp_baseline.yaml
configs/train/default/place_object_scale_left_dp_emix.yaml
configs/train/default/place_object_scale_left_smolvla_emix.yaml
```

三者都固定 `policy.n_obs_steps: 1`。Baseline 不包含 `emix` 区块；DP/SmolVLA
使用 `emix.training.training_mode: mixed`，其中 `batch_prob: 0.30` 表示约 30%
的 batch 来自 EMIX。也可以用 `scheduler.type: step` 按训练百分比切换多个
常数比例，例如：

```yaml
scheduler:
  type: step
  steps:
    - start_percentage: 0.0
      end_percentage: 0.4
      ratio: 0.40
    - start_percentage: 0.4
      end_percentage: 1.0
      ratio: 0.10
```

`start_percentage` 和 `end_percentage` 是 `[0, 1]` 的训练进度比例，区间按
`[start_percentage, end_percentage)` 解释；覆盖到 `1.0` 的最后一个区间也会包含
训练最后一步。区间必须按顺序排列且不能重叠；未覆盖的训练百分比使用比例 `0.0`。

### 2.3 高斯噪声训练与可控评估

实验身份、固定任务、参数和执行顺序以 [唯一实验协议](../../EXP.md) 为准。
单次训练 YAML 顶层添加（也支持 `train.image_noise`）：

```yaml
image_noise:
  enabled: true
  variance: 0.0025
```

对解码/缩放到 `[0,1]` 的图像执行 `clip(x + ε, 0, 1)`，R/G/B 各像素独立
采样 `ε ~ N(0, variance)`。噪声先于模型标准化，EMIX 则在混合之后加噪声。
`0.0025` 是方差，标准差为 `0.05`，在 255 色域等效标准差为 `12.75`；
255 色域标准差 `s` 对应归一化方差 `(s/255)²`。截断会改变边界区域实测分布。
默认不启用；`enabled: false` 或 `variance: 0` 精确关闭。

评估单独读取评估 YAML 的配置，训练噪声不会随 checkpoint 隐式开启：

```yaml
evaluation:
  image_noise:
    enabled: true  # false 表示原始图像评估
    variance: 0.0025
  image_noise_seed: 0
```

批量 YAML 对应 `training.image_noise` 和 `evaluation.image_noise`。
`v0-noise` 的两个条件共用训练 checkpoint，各自写入独立评估目录；
运行命令、任务选择、历史 OIDN 证据和算力安排统一见 [docs/EXP.md](../../EXP.md)。
噪声参数和协议写入训练快照、checkpoint 训练元数据和评估 metrics。
磁盘数据、状态和动作不变。训练 debug 显示实际输入噪声；评估视频保存原始渲染画面，
视频本身不能用于判定策略输入是否已加合成噪声。

## 3. 数据契约

### 3.1 `dataset.root`

`dataset.root` 必须指向精确的标准 LeRobot 数据集根目录。

这个根目录至少要包含：

```text
{root}/
  meta/
    info.json
  data/
  videos/
```

当前训练代码会直接读取 `meta/info.json`，不会猜父目录，也不会自动修正目录名。

### 3.2 策略输入由数据集特征自动推断

当前 LeRobot 策略创建流程会从 `dataset.root/meta/info.json` 的 `features` 自动推断 policy 的 `input_features` 和 `output_features`。
本项目训练入口调用 `make_policy(cfg.policy_cfg, ds_meta=ds_meta)`，底层会把数据集中除 `action` 以外的 policy feature 作为候选输入。

因此，训练配置中的注释如“base camera only + qpos”只有在训练数据集确实只包含这些 observation feature 时才成立。
如果数据集同时包含 `observation.eef_pose`，且没有额外过滤机制，checkpoint 的 `pretrained_model/config.json` 会记录并使用该输入。

这个行为有两个直接后果：

- 数据集字段是模型输入边界的一部分，不只是存储格式
- 评估阶段必须提供 checkpoint `input_features` 中的所有 observation key，否则会产生训练/评估输入不一致

如果某个实验要求策略不能看到 `observation.eef_pose`，推荐使用删除该字段后的训练数据集，而不是把带 `observation.eef_pose` 的 full 数据集直接作为 `dataset.root`。
full 数据集仍可保留给 `python -m emix build` 使用，因为 EMIX index 构建需要末端位姿。

### 3.2.1 强制的单点观测契约

当前 EMIX-LeRobot 训练协议是**点观测协议**，不是历史轨迹输入协议。所有
`emix_*` 策略必须满足：

```yaml
policy:
  n_obs_steps: 1
```

这意味着模型每次 forward 只能接收 member 的当前帧：

```text
observation.state:          (B, 1, state_dim)
observation.images.<camera>: (B, 1, C, H, W)
```

单环境推理的无显式时间维形式 `(B, state_dim)` 和 `(B, C, H, W)` 也表示一个当前帧，
但禁止出现第二个 observation frame。`observation.eef_pose` 等辅助字段如果存在，
同样不能携带多帧输入。

未来动作窗口仍然允许作为监督目标：

```text
action / emix_actions: (B, action_steps, action_dim)
```

它们不是模型 observation，不得回灌到 observation 输入。当前代码会在 EMIX loader、
训练 batch 入口以及 EMIX policy 的 forward/select_action 入口进行断言；违反单帧契约
会直接抛出 `AssertionError`，不会自动截断历史帧。

因此，禁止通过增加 `n_obs_steps`、把 member 点展开为历史观测窗口、或把未来帧拼入
`observation.*` 的方式改变实验语义。若要研究历史轨迹输入，必须另行定义实验协议，
不能与当前 EMIX 单点协议混用。

### 3.3 `emix.index_path`

`emix.index_path` 是训练 YAML 中的新配置键：

- 它必须指向由 `python -m emix build` 生成的 `emix_index.json` 文件
- 训练 loader 直接读取源 `LeRobotDataset`（由 `dataset.root` 和 `dataset.repo_id` 指定），加载 EmixIndex 中引用的帧
- 对非 `emix_*` baseline 策略，这个字段可以不存在，也不会被读取
- 旧字段 `policy.emix_files_root` 和 `emix.files_root` 在加载阶段直接报错

### 3.4 EmixIndex 结构

`emix_index.json` 是训练侧的 EMIX 索引工件（schema v2）。

```text
data/emix_data/<repo_id>/
  emix_index.json
```

它的 `public.metadata` 包含：

- `source_dataset_fingerprint` - sha256 全目录哈希，用于验证源数据集完整性

`content.cross_point_groups[]` 中的每个 group 至少包含：

- `group_id`
- `member_count` (必须为 2)
- `state`
- `members` - 每个 member 含 `source_repo_id`、`source_dataset_index`、`source_frame_index`

训练 loader 按 `source_frame_index` 直接从源数据集读取帧，span 长度由
`max(n_obs_steps, action_steps)` 决定，不跨 episode 边界；在当前协议中，只有
span 的第 0 帧可以进入 observation，后续帧只能用于构造未来 action supervision。

## 4. 推荐执行顺序

如果要做一轮真实验证，建议按以下顺序执行。

1. 准备一个真实的源 LeRobot 数据集。
2. 运行 `build` 生成 `emix_index.json`。
3. 检查 `emix_index.json` 的 `source_dataset_fingerprint` 和至少一个训练 group。
4. 在训练 YAML 中填写 `emix.index_path`。
5. 对 `emix_diffusion` 跑最小训练（验证 fingerprint 通过、loader 加载成功）。
6. 对 `emix_smolvla` 跑最小训练。
7. 如有需要，再在同一个源数据上跑 baseline。

这个顺序比先跑训练再回头查数据更稳，因为 fingerprint 验证会在加载阶段就暴露数据不一致。

## 5. 示例命令

### 5.1 Baseline Diffusion

仓库当前已有的 baseline 示例配置是：

```bash
# Environment: emix
conda run -n emix python -m emix train \
  --config configs/train/default/place_object_scale_left_dp_baseline.yaml \
  --total_steps 10
```

这个配置只读取普通 LeRobot 数据集，不依赖物化后的 `emix/` 目录。

### 5.2 EMIX Diffusion

仓库当前已有的 EMIX 示例配置是：

```bash
# Environment: emix
conda run -n emix python -m emix train \
  --config configs/train/default/place_object_scale_left_dp_emix.yaml \
  --total_steps 10
```

注意：

- 该配置里的 `emix.index_path` 必须与当前源数据集对应
- `dataset.root` 和 `dataset.repo_id` 必须与构建 EmixIndex 时使用的源数据集完全一致
- 旧的 `policy.emix_files_root` 已移除，出现在 YAML 中会直接报错

### 5.3 SmolVLA

当前仓库提供了可运行的
`configs/train/default/place_object_scale_left_smolvla_emix.yaml`，以及用于说明字段
含义的 `configs/train/templates/template_SmolVLA.yaml` 模板。

如果你准备了一个 `policy.type: emix_smolvla` 的配置，训练方式和 Diffusion 一样：

- `dataset.root` 指向源数据根目录
- `emix.index_path` 指向 `emix_index.json`
- 如果启用 `emix.training.enabled`，只能使用 EMIX 相关策略类型

### 5.4 命令行参数约束

| 参数 | 说明 |
|------|------|
| `--resume_from` | 可以是 checkpoint 目录，也可以是 `pretrained_model` 路径 |
| `--output_dir` | 必须是新目录或空目录，不能覆盖已有非空目录 |
| `--start_step` | 恢复训练时的起始步数 |
| `--total_steps` | 本次运行的总步数预算 |
| `--seed` | 控制训练和混合采样随机性 |

## 6. Loader 语义

训练阶段实际使用的组件是：

- `EmixIndexReader`
- policy loader registry (`make_emix_loader`)
- `EmixPolicyDataset`

`EmixIndexReader` 在构造时执行以下验证：

1. 验证 `emix_index.json` 的 `schema_version == 2`。
2. 验证源数据集 `repo_id` 和 `root` 与 loader 参数匹配。
3. 用 sha256 全目录哈希验证源数据集 fingerprint。
4. 加载源 `LeRobotDataset` 并建立 episode 边界映射。

对于每个可训练 group，loader 会执行以下动作。

1. 选择 state 为 `training` 的 group。
2. 取出 group 内的两个 member。
3. 按 `source_frame_index` 直接从源数据集读取帧，span 长度为
   `max(n_obs_steps, action_steps)`，不跨 episode 边界。
4. 只把第 0 帧作为 observation；后续帧只能用于未来动作标签。
5. 解析窗口 payload。
6. 采样 `alpha`。
7. 混合当前帧图像观测。
8. 暴露未来动作、mask 和 EMIX 相关辅助字段。

当前实现中，state-like 观测键的解析优先级是：

1. `observation.state`
2. `observation.eef_pose`
3. `observation.environment_state`

`emix_smolvla` 的训练视图会额外保留 `task`，而 `emix_act` 和 `emix_diffusion` 默认不强制暴露 `task`。

如果 `mixup_alpha_mode == "beta"`，loader 会用 Beta 分布采样 `alpha`。
如果 `mixup_alpha_mode == "fixed"`，则必须提供 `mixup_alpha_value`。

`emix.training.scheduler` 目前支持两种类型：

- `constant`
- `linear`

## 7. 失败条件

训练和加载会在以下情况直接失败。

- `dataset.root/meta/info.json` 缺失
- `emix.index_path` 文件不存在
- EmixIndex `schema_version != 2`
- EmixIndex 中的 `source_dataset_fingerprint` 与实际源数据集不一致
- EmixIndex 的 `source_repo_id` 或 `source_root` 与 loader 参数不匹配
- YAML 中出现 `emix.files_root` 或 `policy.emix_files_root`（已移除的旧字段）
- EmixIndex 中没有任何 group
- EmixIndex 中没有 `training` 状态的 group
- member window 为空（帧索引越界或 episode 边界导致 span 为零）
- 任意 `emix_*` 配置的 `n_obs_steps != 1`
- EMIX observation batch 含有多于一个时间帧
- 窗口 payload 中缺少 `action`
- 窗口 payload 中缺少任何可用的 state-like 观测键

这些情况都应被视为工件损坏或布局不支持，而不是训练阶段可以自动修复的问题。

## 8. 断点续训

当前训练入口支持断点续训。

- `--resume_from` 会被解析为 checkpoint 目录或 `pretrained_model` 路径
- 如果没有显式给 `--start_step`，训练器会尝试从 checkpoint 目录推断起始步数
- `--output_dir` 依然必须是空目录或新目录
- `--total_steps` 可以覆盖配置里的 `steps`

如果你要比较 baseline 和 EMIX 分支，建议保证以下条件一致：

- 同一个恢复起点
- 同一个 `--start_step`
- 同一个 `--total_steps`
- 同一个 `--seed`

唯一允许变化的因素，是数据分支本身。

## 9. 训练前检查

建议在正式提交训练前检查以下内容。

- `dataset.root` 是否是精确的源 LeRobot 根目录。
- `dataset.repo_id` 是否与构建 EmixIndex 时使用的 `repo_id` 完全一致。
- 如果是 `emix_*` 策略，`emix.index_path` 是否指向有效的 `emix_index.json`。
- EmixIndex 的 `schema_version` 是否为 2。
- EmixIndex 中是否不存在 `emix.files_root` 配置键（已移除）。
- 至少一个 group 是否处于 `training` 状态。
- 至少一个窗口是否包含 `action`。
- 至少一个窗口是否包含可用的 state-like 观测键。
- 配置中的 `policy.n_obs_steps` 是否为 `1`。
- 训练 batch 的 `observation.state` 是否为 `(B,1,D)`，图像是否为
  `(B,1,C,H,W)`；未来 action 维度不属于 observation 检查范围。

## 10. 已知现状

当前代码层已经支持以下训练类型：

- baseline `diffusion`
- baseline `act`
- `emix_diffusion`
- `emix_smolvla`
- `emix_act`

其中 `emix.training.enabled` 只允许 `emix_diffusion` 和 `emix_smolvla`。

仓库当前只提供了 Diffusion 的可直接复制示例 YAML。其他 EMIX 类型如果要跑，需要你自己准备对应配置，但数据契约和加载逻辑是一样的。
