# SOP-04 评估流程

环境约定：离线 LeRobot checkpoint 评估在 `emix` 环境运行；调用 RoboTwin/SAPIEN
的在线闭环评估在 `emix-twin` 环境运行。每个命令块均标注环境。

本文档描述当前代码真实支持的评估流程。当前唯一支持的 benchmark 实现是
RoboTwin；这里的"评估"指的是使用训练完成的策略，在 RoboTwin 环境中运行
固定数量的 episode，并记录成功率、回报、回合长度和视频结果。

本 SOP 只覆盖当前仓库里的 `evaluation/` 路径，不覆盖训练侧的数据构建、`EmixIndex` 生成或离线可视化。

## 0. 本流程所涉及的代码文件

- [evaluation/evaluate.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/evaluation/evaluate.py)
- [evaluation/policies/base.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/evaluation/policies/base.py)
- [evaluation/policies/utils.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/evaluation/policies/utils.py)
- [evaluation/policies/lerobot.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/evaluation/policies/lerobot.py)
- [evaluation/policies/constants.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/evaluation/policies/constants.py)
- [scripts/robotwin_eval.sh](/home/rogerspyke/projects/EMIX/Embodied-Mixup/scripts/robotwin_eval.sh)

## 1. 这条流程能做什么

执行这条流程后，你会得到：

1. 一个按时间戳命名的评估输出目录
2. 评估日志 `eval.log`
3. 每个环境实例对应的视频文件
4. 汇总指标，包括：
   - `success_once_mean`
   - `return_mean`
   - `episode_len_mean`
   - `reward_mean`
   - `success_at_end_mean`（如果环境返回该字段）

如果你只需要跑一个最小闭环，`evaluation/evaluate.py` 已经足够。
如果你要比较不同 checkpoint，通常只需要重复运行同一命令，修改 `--pretrained_path` 即可。

## 2. 适用范围

当前评估入口只支持 `LeRobotPolicyConfig` 对应的策略包装器，也就是：

- `--policy lerobot`

`evaluation/policies/utils.py` 里的工厂函数目前只会返回 `LeRobotPolicy`。
如果传入其他 policy 名称，代码会直接报错。

这意味着当前评估流程主要用于：

1. 从 LeRobot 格式数据集读取观测统计信息
2. 加载 LeRobot 预训练策略
3. 在 RoboTwin 环境里执行 rollout

它不是一个通用的多策略评测框架。

### 2.1 并行评估与 reset 语义

当前评估支持 `num_envs > 1` 的并行 rollout，但不使用 partial reset。
所有环境实例会作为一个 batch 同时 reset、同时运行 `max_episode_steps`，然后统一进入下一批 episode。

这样做的原因是 LeRobot DiffusionPolicy 是有状态策略：

- policy 内部维护 observation queue 和 action chunk queue
- queue 中保存的是整批 tensor，形状包含 `num_envs`
- `policy.reset()` 只能清空整个 batch queue，不能只清某一个 env 的队列行

如果某个子环境 done 后立刻 `env.reset(options={"env_idx": ...})`，该子环境会进入新 episode，但 policy queue 对应行仍可能保留上一 episode 的历史或未来 action chunk。
这会造成同一个 env 的新旧 episode 历史污染。

因此，当前评估采用同步批次语义：

1. reset 全部 env
2. `policy.reset()`
3. 并行运行到 `max_episode_steps`
4. 记录每个 env 在本批次内的 `success_once`、return、episode length
5. 下一批重新 reset 全部 env 和 policy

这个实现保留 batch policy 推理和 RoboTwin vectorized env 的速度收益，同时避免 partial reset 污染。

这个方案有一个前提：环境在某个子环境提前 done 后，仍然允许整个 vectorized env 继续 `step()` 到统一的 `max_episode_steps`。
如果具体 RoboTwin 环境或 wrapper 不支持 done 后继续 step，则不能使用这种同步批次实现。
可选替代方案是：

- 使用 fixed-horizon / `ignore_terminations` 语义，让环境不因为 success 或 failure 提前终止
- 改为 `num_envs=1` 串行评估，每个 episode 后完整 reset env 和 policy
- 实现 per-env policy state，使 partial reset 时只清理对应 env 的 policy queue 行

## 3. 输入要求

运行评估前，必须准备以下内容。

- 已激活的 `emix` 环境
- 一个可用的训练权重目录，通常是 `pretrained_model/`
- 精确的标准 LeRobot 数据集根目录
- 与数据集匹配的 `repo_id`
- 可用的 RoboTwin task/config

### 3.0 单点观测强制约束

当前评估必须与训练使用相同的单点观测协议。每次策略调用只能提供当前环境帧，
不得向 policy queue 或 observation 字段注入历史帧：

```text
observation.images.head_camera: (B, C, H, W)
observation.state:               (B, state_dim)
```

如果经过 LeRobot temporal processor 保留显式时间维，则该时间维必须为 1：

```text
observation.images.head_camera: (B, 1, C, H, W)
observation.state:               (B, 1, state_dim)
```

当前协议的未来 action chunk 是模型输出/训练监督，不是输入观测。评估阶段不得
使用历史 observation 来补齐输入，也不得把已生成的未来 action 当作 observation。
EMIX DP 和 SmolVLA 的配置、loader、模型入口都会检查该约束；多帧输入会直接失败。

评估前必须确认 checkpoint 是按 `n_obs_steps: 1` 重新训练得到的。旧的
`n_obs_steps: 2` checkpoint 不属于当前协议，不能通过评估脚本兼容使用。

推荐入口是：

```bash
# Environment: emix
python -m emix eval --config configs/eval/default/place_object_scale_left_dp_baseline.yaml
```

对应的 EMIX 配置是
`configs/eval/default/place_object_scale_left_dp_emix.yaml`。训练配置和评估配置
故意分开：`configs/train/` 只给 `emix train` 使用，`configs/eval/` 才包含
`evaluation.checkpoint`、`episodes`、`device` 等评估字段。

也可以直接使用底层脚本：

```bash
# Environment: emix-twin
python evaluation/evaluate.py --task click_bell_left --piper-distance 0.60
```

### 3.1 数据契约

`evaluation/policies/lerobot.py` 里会直接调用：

```python
LeRobotDataset(repo_id=self.repo_id, root=self.root)
```

因此 `--dataset_root` 必须是精确的 LeRobot 数据集根目录，并且根目录下必须直接存在：

```text
{root}/
  meta/
    info.json
```

当前实现不会自动猜测父目录，也不会自动修正 `repo_id`。

### 3.2 权重契约

`--pretrained_path` 必须指向 Hugging Face LeRobot 预训练权重目录，也就是能被：

```python
PreTrainedConfig.from_pretrained(pretrained_path)
```

正常读取的位置。

如果路径不完整，或者 checkpoint 目录结构和预期不一致，评估会在加载阶段直接失败。

### 3.3 环境契约

`evaluation/evaluate.py` 默认使用：

- `task = click_bell_left`
- `num_envs = 5`
- `num_eval_episodes = 100`
- `max_episode_steps = 200`
- `camera = head_camera`
- `control = dual-Piper qpos`

这些值由 RoboTwin task config 和评估脚本参数共同决定。checkpoint、输出目录、seed
以及 Piper 间距必须显式传入或在对应配置中确认。

需要注意的是，评估代码不会自动检查策略输出是否和 RoboTwin 环境动作空间严格一致。
它依赖于训练时和评估时使用同一套：

1. 策略 checkpoint
2. 数据集元信息
3. 观测映射
4. 控制模式

如果这四项不一致，结果通常不是"自动修复"，而是直接表现异常或报错。

### 3.4 控制模式说明

RoboTwin 的控制契约必须与数据集采集时使用的控制模式完全一致，否则策略输出维度或数值量纲会与环境不匹配，导致运动异常或报错。

#### RoboTwin Piper 控制契约

当前评估使用 RoboTwin Piper qpos 控制契约。动作维度、qpos 排列、相机字段和
控制参数以对应 task config 与 `docs/sop/01-data-generation/SOP-01-data-generation-pipeline.md` 为准；不要使用其他 benchmark
的控制模式表推断 RoboTwin 动作。

第一轮 `click_bell_left` 协议使用完整双 Piper qpos；左侧 Piper 执行任务，右侧
Piper 保持 home qpos。策略图像输入为 `observation.images.head_camera`，左侧
末端位姿和夹爪信号不作为策略输入。动作维度和 qpos 排列必须直接从 RoboTwin
task config 与训练 checkpoint 的 `input_features` / `output_features` 验证。

当前 checkpoint 还必须满足 `policy.n_obs_steps == 1`。评估调用只映射一个当前
`head_camera` 帧和一个当前 `observation.state`，不能使用过去小轨迹作为输入。

## 4. 命令形式

当前评估入口是 `python -m emix eval`：

```bash
# Environment: emix
python -m emix eval --config configs/eval/default/place_object_scale_left_dp_baseline.yaml
```

配置文件结构如下：

```yaml
dataset:
  repo_id: place_object_scale_left
  root: data/lerobot/place_object_scale_left

evaluation:
  checkpoint: <pretrained_model_dir>
  output_dir: outputs/eval/place_object_scale_left
  benchmark_root: third_party/RoboTwin
  task_name: place_object_scale_left
  task_config: place_object_scale_left
  piper_distance: 0.60
  episodes: 100
  seed: 0
  device: cuda
```

### 4.1 参数表

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--config` | 必填 | 本地 YAML 评估配置 |
| `--policy` | `policy.type` | 覆盖 policy wrapper 类型，当前只支持 `lerobot` |
| `--pretrained_path` | `policy.pretrained_path` | 覆盖预训练模型目录 |
| `--output_dir` | `eval.output_dir` | 覆盖输出根目录 |
| `--seed` | `eval.seed` | 覆盖随机种子 |

## 5. 评估数据流

当前流程分成三段：

1. 构建环境
2. 构建策略
3. 运行 rollout 并聚合指标

### 5.1 构建环境

当前评估入口会加载 RoboTwin task，并使用其环境和 task config 创建 rollout 环境。

它固定使用：

- `reconfiguration_freq=0`
- `render_mode="rgb_array"`
- `sensor_configs=dict(shader_pack="default", width=512, height=512)`
- `sim_backend="auto"`

如果 `save_video=True`，环境会再包一层 `RecordEpisode`，把视频写到：

```text
{output_dir}/videos
```

### 5.2 构建策略

`evaluation/policies/lerobot.py` 会：

1. bootstrap LeRobot 运行时
2. 加载源 LeRobot 数据集
3. 从数据集统计信息里读取 `observation.state` 和 `action` 的 min/max
4. 通过 `PreTrainedConfig.from_pretrained()` 读取 checkpoint 配置
5. 构建 LeRobot policy
6. 构建 preprocessor 和 postprocessor

其中 `ObservationMapper` 会按照 `_DEFAULT_OBS_MAPPING` 做观测映射。

当前默认映射包含：

- `sensor_data.base_camera.rgb -> observation.images.base_camera`
- `sensor_data.hand_camera.rgb -> observation.images.hand_camera`
- `agent.qpos -> observation.state`
- `extra.tcp_pose -> observation.eef_pose`

这意味着如果你的数据集特征结构不同，必须调整 `evaluation/policies/constants.py` 里的映射，否则策略输入会不匹配。

### 5.3 运行 rollout

`main()` 按同步批次循环：

1. reset 全部环境实例
2. `policy.reset()`
3. `policy.get_action(obs)`
4. `eval_envs.step(action)`
5. 累积 active env 的回报、步数和成功信号
6. 记录每个 env 第一次 done 时的 episode 信息
7. 不做局部 reset，继续运行到统一的 `max_episode_steps`
8. 批次结束后进入下一批 reset

这里采用的是同步批次并行 rollout，不是单环境逐条执行，也不是谁 done 谁立即 reset 的 auto-reset 模式。
如果某个 env 提前 done，它不会在同一批内开始新的 episode；统计只累计到它第一次 done 的时刻。
因此，当前实现要求环境和 wrapper 能接受 done 后继续 step 到本批次结束。

## 6. 指标定义

当前评估代码记录的主要指标如下。

### 6.1 `success_once`

含义：一个 episode 内只要成功信号出现过一次，就记为 `True`。

来源：`info["success"]` 的逐步累积。

### 6.2 `return`

含义：整条 episode 的累计 reward。

### 6.3 `episode_len`

含义：该 episode 实际经历的步数。

### 6.4 `reward`

含义：`return / episode_len`，也就是平均每步 reward。

### 6.5 `success_at_end`

含义：episode 结束时环境返回的 `success` 值。

这个字段不是强制存在的，只有 `info` 里包含 `success` 时才会写入。

## 7. 输出结构

评估输出目录由以下三部分拼接而成：

```text
{output_dir}/{env_id}_{policy_name}_{policy_cfg.name}_{timestamp}
```

例如：

```text
outputs/eval/click_bell_left_lerobot_diffusion_<timestamp>
```

输出目录内通常包含：

```text
{run_dir}/
  eval.log
  videos/
```

其中：

- `eval.log` 记录配置和最终指标
- `videos/` 保存每个环境实例的评估视频

如果启用轨迹或视频保存，输出文件由 RoboTwin 评估流程写入指定输出目录。

## 8. 运行示例

当前 RoboTwin 闭环使用仓库脚本完成：

```bash
# Environment: emix-twin
scripts/robotwin_eval.sh <piper-distance-m> <baseline-checkpoint> <emix-checkpoint>
```

该命令调用 `evaluation/evaluate.py`，并使用 `third_party/RoboTwin` benchmark 根目录。

### 8.1 基本评估

```bash
# Environment: emix-twin
scripts/robotwin_eval.sh 0.60 <baseline-checkpoint> <emix-checkpoint>
```

### 8.2 批量评估多个 checkpoint

批量评估时重复调用 `scripts/robotwin_eval.sh`，只替换 checkpoint 路径；RoboTwin
任务、Piper 间距、有效 seed 数和其他配置保持一致，便于比较训练过程中的性能变化。

## 9. 失败条件

评估和加载会在以下情况直接失败。

- `--dataset_root/meta/info.json` 缺失
- `LeRobotDataset(repo_id, root)` 无法打开
- `--pretrained_path` 不是有效的预训练权重目录
- `--policy` 不是 `lerobot`
- 数据集观测映射和策略预期不一致
- checkpoint 的 `n_obs_steps` 不是 `1`
- 评估 observation 含有多于一个时间帧
- RoboTwin task/config 不存在或环境无法创建
- `RecordEpisode` 包装失败

这些情况应当视为输入不合法，而不是评估阶段可以自动修复的问题。

## 10. 评估前检查

正式运行前，建议确认以下内容。

- `dataset_root` 是否是精确的数据集根目录。
- `repo_id` 是否和训练时使用的数据集逻辑名一致。
- `pretrained_path` 是否可被 `PreTrainedConfig.from_pretrained()` 读取。
- `obs_mode` 和 `control_mode` 是否与训练保持一致。
- checkpoint 配置中的 `n_obs_steps` 是否严格为 `1`。
- 每次送入 policy 的 image/state 是否只对应当前环境帧；未来 action chunk
  只能作为 policy 输出，不能出现在 observation 字段中。
- `_DEFAULT_OBS_MAPPING` 是否适配当前数据集的特征结构。
- `env_id` 是否确实是目标任务名。
- `save_video` 是否按需开启。

## 11. 已知现状

当前评估实现有几个明确边界：

1. 只支持 `lerobot` policy。
2. 评估是同步批次并行 rollout，不是严格的单 episode 顺序执行。
3. 不使用 partial reset；每批 episode 结束后统一 reset env 和 policy。
4. 同步批次并行依赖环境支持 done 后继续 step，或使用 fixed-horizon / `ignore_terminations` 语义。
5. 代码不会自动修正数据集路径或 repo id。
6. 当前日志输出集中在 `eval.log`，不会额外写复杂的汇总表格。

如果你后面扩展新的 policy 类型，评估文档和 `evaluation/policies/utils.py` 都需要同步更新。
