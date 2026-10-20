# 数据结构与转换

当前 benchmark 是 RoboTwin。本文描述 RoboTwin 转换产物使用的 LeRobot 与
`EmixIndex` 数据契约，以及它们和训练 batch 之间的转换边界。

本文档描述当前代码实际使用的离线工件流、`EmixIndex` 的目录与字段契约，以及它们和训练 batch 之间的转换边界。

当前架构（schema v2）直接从源 `LeRobotDataset` 加载训练帧，不需要物化中间数据集。`materialize` 命令已移除。

## 1. 标准离线流程

当前仓库支持的标准离线工件流是：

1. 源 `LeRobotDataset`
2. `EmixIndex` (schema v2，含 sha256 源数据集 fingerprint)
3. `EmixIndexReader` 直接加载源帧
4. training

对应命令如下。

- `python -m emix build`
- `python -m emix visualize`
- `python -m emix tune`

其中：

- `build` 只负责生成 canonical `EmixIndex`，其中包含 sha256 全目录哈希
- `visualize` 负责生成可视化结果
- `tune` 负责交互式调参和重建 `EmixIndex`

补充说明：

- 本项目唯一允许的原始数据结构，是 `data_processing/convert_to_lerobot_v2.py` 转换出来的标准 `LeRobotDataset`
- 只要目录树、`meta/info.json`、`data/`、`videos/` 或特征列结构和该脚本产物不一致，就不能直接作为当前管线输入
- 源数据根目录必须是精确根目录，不能通过父目录猜测来修正
- 转换后的 episode 数量必须与 RoboTwin 采集结果和 `--expected-episodes` 一致

## 2. `EmixIndex` 输出

`build` 只写一个核心文件：

- `{output_dir}/emix_index.json`

`EmixIndex` 保存的是索引信息，不是训练输入本体。它包含以下部分：

- 源数据声明
- 媒体契约
- cross-point groups
- group state
- 生成元数据

`EmixIndex` 不应包含以下内容：

- 展开的训练窗口
- 物化后的训练 batch
- 物化后的 LeRobot 目录树

### 2.1.1 `EmixIndex` 数据结构

`EmixIndex` 是一个 Python dataclass 根对象，定义在 `embodied_mixup/candidates/emix_index.py`（schema v2）。它通过 `to_dict()` 和 `from_dict()` 做确定性序列化与反序列化，`save_emix_index(...)` 负责把它写成 `emix_index.json`，`load_emix_index(...)` 负责从 JSON 读回对象。

`public.metadata` 中必须包含 `source_dataset_fingerprint`，记录源数据集的 sha256 全目录哈希，用于在训练加载时严格校验数据完整性。

`EmixIndex` 的根结构固定为：

```text
EmixIndex
  public: EmixIndexPublic
  content: EmixIndexContent
```

`public` 固定包含：

```text
EmixIndexPublic
  id: str
  schema_version: int
  created_at: str
  source_datasets: list[EmixSourceDatasetDeclaration]
  source_dataset_version: str
  metadata: dict
```

`source_datasets[]` 中的每个源数据声明固定包含：

```text
EmixSourceDatasetDeclaration
  repo_id: str
  root: str
  hash: str | None
  media: EmixSourceMediaContract
```

`media` 固定包含：

```text
EmixSourceMediaContract
  camera_keys: list[str]
  frame_alignment: str
  video_path_pattern: str
  storage_format: str
```

`content` 固定包含：

```text
EmixIndexContent
  cross_point_groups: list[EmixCrossPointGroup]
```

`cross_point_groups[]` 中的每个 group 固定包含：

```text
EmixCrossPointGroup
  group_id: str
  member_count: int
  members: list[EmixGroupMemberRef]
  state: EmixGroupState
  group_metrics: dict
  build_trace: dict
```

`members[]` 中的每个 member 固定包含：

```text
EmixGroupMemberRef
  source_dataset_repo_id: str
  source_episode_index: int
  source_frame_index: int
```

`state` 固定包含：

```text
EmixGroupState
  state: str
  abort_reasons: list[str]
```

`EmixGroupState.state` 只能取 `training` 或 `aborted`。

`build` 阶段的输出文件 `emix_index.json` 就是上述对象的 JSON 表示。

### 2.1 源数据声明

源数据必须是一个精确的 `LeRobotDataset` 根目录，并且根目录下必须存在：

- `meta/info.json`

不支持父目录猜测，也不支持 legacy 目录自动适配。

`meta/info.json` 中的 `total_episodes` 是后续 `build` 和 training 使用的 episode 数量。
进入 `python -m emix build` 或训练前，必须确认它与 RoboTwin 采集目标和转换命令的
`--expected-episodes` 一致。

`convert_to_lerobot_v2.py` 输出的标准结构如下：

```text
{root}/
  meta/
    info.json
    tasks.parquet
    episodes/
      chunk-000/
        file-000.parquet
    stats.json
  data/
    chunk-000/
      file-000.parquet
  videos/
    observation.images.<camera_key>/
      chunk-000/
        file-000.mp4
```

其中，`chunk-000` 只是分片目录名的示例。实际导出时，`meta/`、`data/` 和 `videos/` 下面都可能出现 `chunk-001`、`chunk-002` 等后续分片；目录名本身不承载语义，只表示按 chunk 切分后的物理存储单元。

各文件的职责如下：

- `meta/info.json`
  - 数据集级别的总元信息文件，顶层键固定为：
    - `codebase_version`
    - `robot_type`
    - `total_episodes`
    - `total_frames`
    - `total_tasks`
    - `total_videos`
    - `total_chunks`
    - `chunks_size`
    - `fps`
    - `data_files_size_in_mb`
    - `splits`
    - `data_path`
    - `video_path`
    - `features`
  - `splits` 当前写入的是 `{"train": "0:<episode_count>"}` 形式的训练切分范围
  - `data_path` 固定为 `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
  - `video_path` 固定为 `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`
  - `features` 的键固定由当前数据集中实际存在的特征构成，字段定义如下：
    - `action`
    - `observation.state`，当源数据包含状态时存在
    - `timestamp`
    - `frame_index`
    - `episode_index`
    - `index`
    - `task_index`
    - `task`
    - `observation.images.<camera_key>`，对每个相机键各写一项
  - 每个 `action` / `observation.state` / `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` / `task` / `observation.images.<camera_key>` 条目都包含其 `dtype`、`shape`、`names`，视频条目还包含 `info`
  - 这些 `features` 也是 LeRobot 训练时自动推断 policy `input_features` 和 `output_features` 的来源；数据集中存在的 `observation.*` 字段可能直接成为模型输入

- `meta/tasks.parquet`
  - 任务索引表，落盘为一个 parquet 两列表
  - 行索引是任务名
  - 列名固定为 `task_index`
  - 当前脚本在单任务场景下只写一行，索引值固定为 `0`

- `meta/episodes/chunk-xxx/file-yyy.parquet`
  - episode 级元数据表，每一行对应一个 episode
  - 顶层列固定为：
    - `episode_index`
    - `data/chunk_index`
    - `data/file_index`
    - `dataset_from_index`
    - `dataset_to_index`
    - `tasks`
    - `length`
    - `meta/episodes/chunk_index`
    - `meta/episodes/file_index`
  - 每个 camera 还会生成一组视频定位列，列名固定为：
    - `videos/observation.images.<camera_key>/chunk_index`
    - `videos/observation.images.<camera_key>/file_index`
    - `videos/observation.images.<camera_key>/from_timestamp`
    - `videos/observation.images.<camera_key>/to_timestamp`
  - 还会生成统计列，列名前缀固定为 `stats/`
  - `stats/action` 始终存在，字段为 `min`、`max`、`mean`、`std`、`count`
  - 当源数据包含 `observation.state` 时，会额外生成 `stats/observation.state`，字段为 `min`、`max`、`mean`、`std`、`count`
  - 对每个 camera，会额外生成 `stats/observation.images.<camera_key>`，字段为 `min`、`max`、`mean`、`std`、`count`
  - 对以下帧级字段，还会生成统计列：
    - `timestamp`
    - `frame_index`
    - `episode_index`
    - `index`
    - `task_index`
  - 每个上述帧级字段都对应 `stats/<field>/min`、`stats/<field>/max`、`stats/<field>/mean`、`stats/<field>/std`、`stats/<field>/count`

- `meta/stats.json`
  - 全数据集统计摘要文件
  - 顶层键固定为：
    - `action`
    - `observation.state`，仅当源数据包含状态时存在
    - `observation.images.<camera_key>`，对每个 camera_key 各写一项
    - `timestamp`
    - `frame_index`
    - `episode_index`
    - `index`
    - `task_index`
  - 每个键都包含 `min`、`max`、`mean`、`std`、`count`
  - `observation.images.*` 的统计值必须来自解码后的实际图像内容，并在浮点类型上计算；错误的 image std 会直接影响视觉 policy 的 `MEAN_STD` 输入归一化

- `data/chunk-xxx/file-yyy.parquet`
  - 逐帧结构化数据表，每一行对应一个 frame
  - parquet 列固定为：
    - `action`
    - `observation.state`，仅当源数据包含状态时存在
    - `timestamp`
    - `frame_index`
    - `episode_index`
    - `index`
    - `task_index`
    - `task`
    - `observation.eef_pose`，仅当启用提取时存在
    - `observation.images.<camera_key>`，对每个 camera_key 各写一列
  - `action` 和 `observation.state` 的每个元素都以向量形式存储
  - `timestamp` 是 `float32`
  - `frame_index`、`episode_index`、`index`、`task_index` 是 `int64`
  - `task` 是字符串列
  - `observation.eef_pose` 是 `float32` 向量列，维度为 6
  - `observation.images.<camera_key>` 是视频特征列，存储为视频写入对应 `videos/` 路径
  - 缺少 `action` 或 `episode_index` 时，后续 `build` 无法继续

- `videos/observation.images.<camera_key>/chunk-xxx/file-yyy.mp4`
  - 每个相机一个视频目录
  - 保存对应帧序列的 MP4 编码结果
  - `camera_key` 直接对应字段名后缀，例如 `observation.images.base_camera`、`observation.images.hand_camera`
  - 训练或可视化时需要图像输入时，就从这里按帧对齐读取

总结来说：

- `data/` 里的 parquet 保存逐帧标量与结构化特征
- `videos/` 里的 mp4 保存逐帧图像特征
- `meta/` 里的文件保存数据集元信息、episode 切分和统计信息

当 `convert_to_lerobot_v2.py` 开启 `observation.eef_pose` 提取时，该字段也会作为逐帧特征进入标准 `LeRobotDataset`。

### 2.1.1 图像统计修复

当前项目提供 `data_processing/fix_stats.py` 用于修复 LeRobot 数据集中的图像统计。
该脚本会自动读取数据集当前的图像保存方式：

- 如果图像是 video feature，则通过 LeRobotDataset 解码对应视频帧
- 如果图像是 image feature，则通过 LeRobotDataset 读取对应图像帧

脚本随后用 `float32` 图像重新计算每个 `observation.images.*` 的 episode 级统计和全局统计，并覆盖：

- `meta/episodes/chunk-*/file-*.parquet`
- `meta/stats.json`

覆盖前会创建同目录时间戳备份。

这个修复是必要的，因为当前 LeRobot 版本的部分 stats 计算路径会在 `uint8` 图像上执行平方操作。
`uint8` 平方会溢出，导致 episode image std 被写成 0，进一步导致全局 image std 极小。
视觉策略训练时如果使用 dataset `MEAN_STD`，会把图像输入放大几十到上百倍。

`convert_to_lerobot_v2.py` 已在数据集转换完成后自动调用该修复步骤。
已有旧数据集可以手动执行：

```bash
# Environment: emix
python data_processing/fix_stats.py \
  --root data/lerobot/<dataset_id> \
  --repo_id <dataset_id>
```

### 2.1.2 数据字段与模型输入边界

当前训练代码不会把“用于分析的 observation 字段”和“允许策略网络使用的 observation 字段”自动区分开。
LeRobot 会根据数据集 metadata 中的 feature 列表自动构造 policy features：

- `action` 会成为 `output_features`
- `observation.images.*`、`observation.state`、`observation.eef_pose` 等 observation 字段会成为候选 `input_features`

因此，`observation.eef_pose` 的存在有双重含义：

- 对 `python -m emix build`，它是位姿近邻搜索和交点构建所需字段
- 对模型训练，它也可能成为策略网络输入，并被写入 checkpoint 的 `pretrained_model/config.json`

如果实验设计要求模型只使用 camera + qpos，需要在训练数据层面移除 `observation.eef_pose`，或使用明确支持输入过滤的训练实现。
在当前保守流程中，推荐维护两个数据集：

- full 数据集：保留 `observation.eef_pose`，用于构建 EMIX index
- train 数据集：删除不允许进入网络的 observation 字段，用于 baseline 或策略训练

### 2.2 媒体契约

媒体契约固定为：

- `frame_alignment == "per_frame"`
- `storage_format == "mp4"`

媒体契约还会记录：

- `camera_keys`
- `video_path_pattern`

### 2.3 group 语义

`cross_point_groups` 保存的是候选组及其状态。每个 group 关注的是一组在位姿空间中可比较的 member，而不是最终训练 batch。group 代表交点，也就是多个点被判定为轨迹的近似交点组。

每个 group 包含以下字段：

- `group_id`: 该交点组的唯一标识。
- `member_count`: 该组里 member 的数量。
- `members`: 该组包含的 member 列表。
- `state`: 该组的状态记录，决定它是否可进入训练侧。
- `group_metrics`: 该组的统计指标或聚类指标。
- `build_trace`: 该组是如何构建出来的过程记录。

`state.state` 只能取以下值：

- `training`: 该组可用于训练。
- `aborted`: 该组被中止，不应进入训练。

`state.abort_reasons` 为空时，group 才会被视为可训练。

## 3. 转换边界

### 3.1 `LeRobotDataset` -> `EmixIndex`

`build` 阶段负责把源数据转换为索引信息。

这个阶段会做以下事情：

- 读取末端位姿
- 读取动作
- 读取 episode 边界
- 搜索邻域候选
- 按动作差异过滤
- 使用 HDBSCAN 聚类
- 组装 cross-point groups

当前实现中，`training_candidates_total_budget`、`training_candidates_cap_per_cluster`、`training_candidates_include_noise` 和 `training_candidates_seed` 只会进入配置或元数据，不会触发真正的类级平衡采样。

换句话说，`EmixIndex` 目前保存的是”筛选后全量候选 + 聚类标记”，不是”经过 cluster 平衡后的子集”。

### 3.2 `EmixIndex` -> training batch

训练 loader 负责把 `EmixIndex` 直接变成 batch，无需物化中间数据集。

`EmixIndexReader` 在构造时会：

1. 验证 `schema_version == 2`
2. 验证 `source_repo_id` 和 `source_root` 与 loader 参数匹配
3. 用 sha256 全目录哈希验证源数据集 fingerprint，哈希不匹配时立即失败
4. 加载源 `LeRobotDataset`，建立 episode 边界映射

对于每个可训练 group，loader 会执行以下动作：

- 选出所有 `state == “training”` 的 group
- 读取两个 member 的 `source_frame_index`
- 从源数据集直接读取帧，span 长度为 `max(n_obs_steps, action_steps)`，不跨 episode 边界；
  当前 EMIX 协议强制 `n_obs_steps=1`，只有第 0 帧进入 observation，后续帧只能作为
  future action supervision
- 解析窗口 payload
- 采样 `alpha`
- 混合图像观测
- 暴露 action、mask 和策略侧观测张量

`EmixIndexReader.training_samples()` 只是把所有可训练 group 直接转成训练样本，`EmixPolicyDataset` 再按两个 member 的 pair 形式消费它们。

## 4. 严格契约

### 4.1 源数据根目录

CLI 的 `--root` 和训练 YAML 的 `dataset.root` 必须是精确的数据集根目录，并且必须包含：

- `meta/info.json`

不支持父目录猜测。

### 4.2 EmixIndex 路径

训练 YAML 的 `emix.index_path` 必须指向有效的 `emix_index.json`，且：

- `schema_version` 必须为 2
- `source_dataset_fingerprint` 必须与实际源数据集一致
- 不允许出现 `emix.files_root` 或 `policy.emix_files_root`（已移除）

### 4.3 损坏工件语义

loader 遇到以下情况必须立即失败：

- `emix_index.json` 文件不存在
- `schema_version != 2`
- `source_dataset_fingerprint` 与实际源数据集不一致
- `source_repo_id` 或 `source_root` 与 loader 参数不匹配
- EmixIndex 中没有 group
- group 没有 training state
- member window 为空
- 窗口 payload 中缺少 `action`
- 窗口 payload 中缺少任何可用的 state-like 观测键

这些情况都应视为工件损坏或不支持的布局，而不是可自动修复的正常输入。

### 4.4 当前未实现的采样约束

尽管配置层已经预留了以下字段：

- `training_candidates_total_budget`
- `training_candidates_cap_per_cluster`
- `training_candidates_include_noise`
- `training_candidates_seed`

但截至目前，这些字段只会被解析并写入配置或元数据，不会触发真正的采样逻辑。

如果后续需要引入 cluster 平衡采样，应明确新增独立阶段，并同步更新本节契约。

## 5. 已废弃布局

以下布局属于历史遗留，不再属于当前支持行为。

- 物化 LeRobot 数据集（`emix/emix_index.json` sidecar 结构）
- `emix.files_root` / `policy.emix_files_root` 配置键
- `materialize` CLI 命令
- `EmixFilesReader`（已替换为 `EmixIndexReader`）
- `EmixMaterializedIndex`
- `emix_files/` 目录树
- 旧 build-result pickle snapshots

如果这些工件仍在本地存在，它们只能视为归档数据，不能作为当前管线输入。

## 6. 代码入口对应关系

这部分用于快速定位”某种转换是谁做的”。

- `embodied_mixup/candidates/emix_index_utils.py`
  - `inspect_source_dataset(...)`
  - `compute_source_dataset_fingerprint(...)`
  - `validate_source_dataset_fingerprint(...)`
  - `build_emix_index(...)`
  - `save_emix_index(...)`
  - `load_emix_index(...)`

- `embodied_mixup/policies/utils/emix_loader_registry_utils.py`
  - `EmixIndexReader` - 直接从源数据集加载训练帧
  - `EmixPolicyDataset`
  - 加载和 batch 组装的硬失败校验

- `embodied_mixup/policies/utils/emix_config.py`
  - `apply_emix_schema(...)` - 验证 emix YAML payload，设置 `emix_index_path`

- `embodied_mixup/policies/train.py`
  - `train(...)`
  - `main()`
  - 训练配置加载、输出目录准备、断点续训和快照保存
