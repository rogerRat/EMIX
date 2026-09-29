# 开发总览

本文档是仓库的开发总说明。它用于描述当前 EMIX 项目的目录结构、核心代码文件职责、关键函数入口、数据契约和已废弃路径。  
SOP 文档负责操作步骤和复现流程，理论文档负责方法动机与算法背景；本文件负责“代码是什么、各文件做什么、数据怎么流动”。

## 1. 项目分层

当前仓库的主动开发面可以分成四层。

1. 源数据处理层：把原始轨迹、LeRobot 数据集和局部子集整理成可计算的输入。
2. 候选发现层：从源数据中寻找位姿接近但动作可能分叉的样本对，并构建 `EmixIndex`（含 sha256 源数据集 fingerprint）。
3. 可视化与调试层：生成交互式图和调参面板，帮助检查候选分布与聚类效果。
4. 策略训练层：直接从源 `LeRobotDataset` + `EmixIndex` 加载训练 batch，驱动 `Diffusion Policy`、`SmolVLA` 等策略训练，无需物化中间数据集。

这四层共同构成当前唯一支持的 EMIX 离线主路径。

## 2. 目录结构

下面是当前仓库中与 EMIX 主流程直接相关的目录。

```text
embodied_mixup/
  __main__.py
  cli.py
  configs/
  candidates/
  visualization/
  policies/
data_processing/
evaluation/
data_collection/panda/
docs/
```

### 2.1 `embodied_mixup/`

这是核心 Python 包，包含构建、可视化和训练的全部主逻辑。

### 2.2 `data_processing/`

这里放源数据准备脚本，例如 LeRobot 转换、特征裁剪、子集抽取、回放检查等。

### 2.3 `evaluation/`

这里放评估相关代码，主要用于策略侧的环境评测和接口适配。

### 2.4 `embodied_mixup/configs/`

这里放 EMIX 的候选生成配置定义和默认配置。

- `config.py`：定义 `EmbodiedMixupConfig` 与 `DEFAULT_EMBODIED_MIXUP_CONFIG`，负责所有候选生成参数的默认值、约束校验和属性别名。

### 2.5 `docs/`

这里放开发文档、SOP、理论稿和历史记录。  
其中：

- `docs/DEV.md`：开发总说明，描述目录、模块、数据契约和运行约束。
- `docs/sop/`：操作型 SOP，描述具体流程和复现步骤。
- `docs/theory/`：理论说明，描述 EMIX 的方法背景和实验设想。

当前唯一支持的 benchmark 实现是 RoboTwin。ManiSkill 已舍弃，不再属于
开发、采集、转换、训练或评估流程；相关代码仅作为历史归档保留在
`legacy/maniskill/`，不得作为当前入口或新实验依赖。

### 2.6 `data_collection/panda/`

这里放历史 ManiSkill Panda 机械臂的数据采集与运动规划脚本，以及逐任务的手写解法。
该目录仅供历史代码查阅，当前数据采集使用 `third_party/RoboTwin` 和
`scripts/collect.sh`。

## 3. 核心执行入口

仓库对外最重要的入口推荐使用 `python -m emix`，同时保留完整包名 `python -m embodied_mixup` 作为兼容入口。

### 3.1 `embodied_mixup/__main__.py`

这是模块级入口，负责把 `python -m embodied_mixup` 转到 CLI。`emix/__main__.py` 提供同样的转发入口，推荐优先使用 `python -m emix`。

### 3.2 `embodied_mixup/cli.py`

这是离线管线的主命令行入口，提供 `build`、`visualize` 和 `tune`。`materialize` 已移除。

关键函数如下。

- `_setup_logging(level)`：配置 CLI 日志。
- `_default_emix_output_dir(repo_id)`：给出默认输出目录。
- `_ensure_local_plotly_bundle(output_path)`：在 HTML 旁边写入本地 `plotly.min.js`。
- `_load_dataset(repo_id, root)`：加载精确的 LeRobotDataset 根目录，并检查 `meta/info.json`。
- 该加载函数会先记录 `repo_id` 和 `root`，再调用 `LeRobotDataset(repo_id=..., root=...)`，不会从路径反推 `repo_id`。
- `_resolve_config(args)`：把命令行参数转成 `EmbodiedMixupConfig`。
- `cmd_build(args)`：构建并保存 `EmixIndex`（含 sha256 源数据集 fingerprint）。
- `cmd_visualize(args)`：构建 `EmixIndex` 并导出可视化结果。
- `cmd_tune(args)`：启动交互式调参面板。
- `build_parser()`：组装参数解析器。
- `main(argv=None)`：CLI 主入口。

### 3.2.1 本流程所涉及的代码文件

- [embodied_mixup/cli.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/cli.py)
- [embodied_mixup/datasets/path_utils.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/datasets/path_utils.py)
- [embodied_mixup/candidates/emix_index_utils.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/candidates/emix_index_utils.py)
- [embodied_mixup/visualization/utils.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/visualization/utils.py)
- [embodied_mixup/visualization/plot_intersections.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/visualization/plot_intersections.py)
- [embodied_mixup/visualization/plot_projections.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/visualization/plot_projections.py)
- [embodied_mixup/visualization/tuning_dashboard.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/visualization/tuning_dashboard.py)

### 3.3 命令边界

当前支持的标准命令是：

- `python -m emix build`
- `python -m emix visualize`
- `python -m emix tune`
- `python -m emix train`

其中 `build` 负责生成 `emix_index.json`，`visualize` 和 `tune` 负责调试，`train` 负责训练。`materialize` 已移除。
其中数据处理类命令统一采用同一套契约：`--root` 必选，`--repo_id` 可选；如果不传 `--repo_id`，就按 `root` 的最后一级目录名自动推导。对带目标目录的命令，`target_repo_id` 也遵循同样的推导规则。

## 4. 数据与工件契约

当前 EMIX 的唯一永久工件是：

- `EmixIndex`（schema v2，含 sha256 源数据集 fingerprint）

训练直接从源 `LeRobotDataset` + `EmixIndex` 加载帧，不需要物化中间数据集。任何旧的 `EmixFiles`、sample 目录树、pickle 结果或旧 loader 结构都不属于当前支持路径。

### 4.1 源数据契约

源数据必须满足以下条件：

- 是一个精确的 `LeRobotDataset` 根目录。
- 根目录下必须存在 `meta/info.json`。
- 不允许通过父目录猜测、repo id 尾缀猜测或别名路径自动修正。

本项目唯一允许的标准 `LeRobotDataset` 布局，是 `data_processing/convert_to_lerobot_v2.py` 产出的布局。其核心树形结构如下：

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

其中：

- `data/` 保存逐帧标量和结构化特征
- `videos/` 保存逐帧图像特征
- `meta/` 保存数据集元信息、episode 切分和统计信息

如果输入目录树和 `convert_to_lerobot_v2.py` 的产物不一致，就不能直接作为本项目输入。

这是整个管线的第一个硬约束。

### 4.2 `EmixIndex`

`EmixIndex` 是训练侧唯一工件，主要保存”从哪里来、为什么被选中、组是怎么形成的”。

它的结构分为两部分。

- `public`：公开元数据，例如 `id`、`schema_version`（必须为 2）、`created_at`、`source_datasets`、`source_dataset_version`、`metadata`。
- `content`：内容区，主要是 `cross_point_groups` 和组状态。

`public.metadata` 必须包含 `source_dataset_fingerprint`，字段结构为：

- `algorithm`: `”sha256”`
- `included_roots`: 参与哈希的子目录列表（`meta/`, `data/`, `videos/`）
- `file_count`: 参与哈希的文件数
- `total_bytes`: 总字节数
- `dataset_sha256`: 全目录哈希值（16 进制 64 字符）
- `files`: 各文件的哈希记录

当前实现的稳定契约是：

- `frame_alignment == “per_frame”`
- `storage_format == “mp4”`

`EmixIndex` 不保存展开后的训练窗口，也不保存合成的 pair 数据集。训练 loader 在运行时直接从源数据集读取帧。

### 4.3 训练契约

训练侧通过 `EmixIndexReader` 直接从源 `LeRobotDataset` 加载帧。

必须满足以下约束：

- `emix.index_path` 必须指向有效的 `emix_index.json`（schema v2）。
- `dataset.root` 和 `dataset.repo_id` 必须与构建 EmixIndex 时使用的源数据集完全一致。
- EmixIndex 中的 `source_dataset_fingerprint` 必须与实际源数据集匹配。
- 不允许使用 `emix.files_root` 或 `policy.emix_files_root`（已移除，出现时立即报错）。

如果 fingerprint 不匹配、schema version 不对、repo_id/root 不匹配，loader 必须立即失败，而不是尝试容错修复。

## 5. 核心模块职责

下面按目录说明关键文件和函数作用。

### 5.1 `embodied_mixup/candidates/`

这一层负责候选发现、索引构建和物化。

- `emix_index.py`
  - 定义 `EmixSourceMediaContract`、`EmixSourceDatasetDeclaration`、`EmixGroupState`、`EmixGroupMemberRef`、`EmixCrossPointGroup`、`EmixIndexPublic`、`EmixIndexContent`、`EmixIndex`。
  - 作用是把索引工件的数据模型固定下来，并支持确定性序列化。

- `emix_index_utils.py`
  - `inspect_source_dataset(dataset, root)`：检查源数据声明。
  - `build_emix_index(dataset, config)`：从源数据构建 canonical `EmixIndex`。
  - `_cluster_candidate_records(...)`：把候选 pair 聚成组。
  - `_build_cluster_info(...)`：补充聚类信息。
  - `save_emix_index(index, path)` / `load_emix_index(path)`：读写 JSON 索引。
  - `build_emix_index` 是 build 阶段的核心函数。

- `loaders.py`
  - `load_eef_poses(dataset)`：读取末端位姿。
  - `load_episode_indices(dataset)`：读取 episode 索引。
  - `load_actions(dataset)`：读取动作。
  - `load_gripper_signal(actions)`：提取夹爪信号。
  - `load_optional_feature_window(dataset, feature_key, ...)`：读取可选窗口特征。

- `episode_utils.py`
  - `build_episode_start_end(episode_indices)`：构建 episode 起止区间。
  - `build_episode_bound_maps(...)`：构建边界映射。
  - `safe_future_index(...)`：安全计算未来索引。
  - `compute_window_bounds(...)`：计算局部窗口范围。

- `pairs.py`
  - `run_neighbor_search(...)`：执行邻域搜索。
  - `build_frame_to_neighbors(...)`：把 pair 结果整理成 frame 到 neighbor 的映射。

- `action_filter.py`
  - `run_action_filter(...)`：按当前动作分歧和未来运动角度过滤候选。
  - `compute_action_filter_metrics(...)`：统计过滤指标。
  - `compute_gripper_transition(...)`：计算夹爪转换信号。

- `ids.py`
  - `make_emix_sample_id(...)`
  - `make_emix_group_id(...)`
  - `make_emix_index_id(...)` — 现在使用 sha256 + source_dataset_sha256 生成 16 位十六进制前缀
  - 作用是生成确定性 ID。

- `schemas.py`
  - 定义候选记录相关的结构体，例如 `CandidatePublic`、`CandidateParticipant`、`CandidateRelation`、`CandidateMediaAsset`、`CandidateMedia`、`CandidateRecord`。
  - 作用是把“候选发现前的中间态”结构化。
  - 这些结构体当前主要作为构建阶段的内部中间表达，不是训练侧永久工件。

### 5.2 `embodied_mixup/visualization/`

这一层负责把 `EmixIndex` 和源数据转成可视化结果。

- `utils.py`
  - `prepare_visualization_data(index, dataset)`：准备 Plotly 数据。
  - `_get_cluster_colors(n_clusters)`：生成聚类颜色。
  - `_compute_local_point_density(...)` / `_estimate_density_radius(...)`：估计局部密度。

- `plot_intersections.py`
  - `plot_intersections_3d(...)`：绘制 3D 相交/聚类视图。

- `plot_projections.py`
  - `plot_2d_projections(...)`：绘制 2D 投影视图。

- `tuning_dashboard.py`
  - `create_tuning_dashboard(...)`：创建交互式 Dash 面板。
  - `make_rebuild_fn(dataset, config=None)`：生成重建 `EmixIndex` 的回调。

- `camera_presets.py`
  - `set_camera_view(...)`：统一相机视角。

### 5.3 `embodied_mixup/policies/`

这一层负责训练和 loader 适配。

- `train.py`
  - `EmixTrainConfig`、`SaveableTrainConfig`：训练配置结构。
  - `_configure_logging()`：初始化训练日志。
  - `_make_output_dir(...)`、`_prepare_output_dir(...)`：准备输出目录。
  - `_load_yaml_config(...)`：读取 YAML 配置。
  - `_load_local_dataset_metadata(...)`：读取本地数据元信息。
  - `_build_train_config(...)`：构造训练配置。
  - `_create_dataset_and_policy(...)`：创建数据集与策略对象。
  - `_write_train_config_snapshot(...)`：写训练快照。
  - `_build_resume_config(...)`：构造断点续训配置。
  - `_snapshot_payload(...)`：组装快照内容。
  - `train(config_path, resume_cfg=None)`：训练主入口。
  - `main()`：CLI 入口。

- `utils/bootstrap.py`
  - `bootstrap_lerobot_runtime()`：补齐或修正 LeRobot 运行时环境。
  - `maybe_bootstrap_from_policy_type(policy_type)`：按策略类型选择性启动 bootstrap。
  - 作用是适配当前环境里的 LeRobot 包布局差异。

- `utils/emix_loader_registry.py`
  - `EmixLoaderRegistration`：loader 注册结构。
  - `make_emix_loader(...)`：按策略创建 loader。

- `utils/emix_loader_registry_utils.py`
  - `EmixIndexReader`：加载 `EmixIndex`，验证 fingerprint，直接从源 `LeRobotDataset` 读帧。
  - `EmixPolicyDataset`：面向训练的 dataset 封装。
  - `EmixTrainingSample`：训练样本结构。
  - `_frame_dict_from_window(...)`：把 window payload 还原成 frame dict。
  - 这一层是训练时最关键的校验入口，负责硬失败语义。

- `utils/emix_config.py`
  - `apply_emix_schema(policy_cfg, emix_payload)`：验证 YAML emix 块，拒绝 `files_root`，设置 `policy_cfg.emix_index_path`。
  - 这是训练配置里 EMIX 语义的唯一入口，避免旧字段继续渗透到策略配置中。

- `utils/batch.py`
  - `PolicyTemporalSpec`：策略时间规格。
  - `get_policy_temporal_spec(policy_cfg)`：解析策略时间维度。
  - `stack_tensor_list(samples)`：把样本堆叠成 batch。

- `utils/checkpoint.py`
  - `ResumeConfig`：断点续训配置。
  - `set_global_seed(...)`：设置随机种子。
  - `resolve_checkpoint_dir(...)`：解析 checkpoint 目录。
  - `infer_start_step(...)`：推断起始 step。
  - `load_train_state(...)` / `save_config_snapshot(...)`：读写训练状态。

- `ACT/`、`DP/`、`SmolVLA/`
  - 这些目录保存对应策略的 EMIX 适配实现。
  - 当前 SOP 的验证重点放在 `DP/` 和 `SmolVLA/`；`ACT/` 仅保留为扩展实现，不纳入当前训练与评估流程。
  - `cfg_emix_*.py` 负责配置适配。
  - `emix_*.py` 负责模型/forward/loader 侧适配。
  - 其中 `cfg_emix_*` 只做配置继承和少量字段补齐，不应引入新的行为分支。

### 5.4 `configs/policies/`

仓库当前没有单独的 `configs/policies/` 目录。策略训练配置实际由外部 YAML 提供，训练代码在运行时读取并归一化。

当前建议遵循的配置约定是：

- 源数据输入写在 `dataset.root` 和 `dataset.repo_id`。
- EMIX 索引路径写在 `emix.index_path`。
- 不再使用旧键 `policy.emix_files_root`、`emix.files_root` 或 `mix_training`（出现时立即报错）。

### 5.5 `data_processing/`

这里放源数据准备脚本，常见用途如下。

- `convert_to_lerobot.py` / `convert_to_lerobot_v2.py`：把原始数据转换成 LeRobot 数据集。
- `filter_nonoop_lerobot.py`：过滤无动作或低有效动作样本。
- `sample_subset_lerobot.py`：抽取小规模可验证子集。
- `remove_feature_lerobot.py`：裁剪不需要的特征。
- `replay.py`：回放检查数据质量。

### 5.6 `evaluation/`

这里放策略评测辅助代码。

- `evaluate.py`：评测入口。
- `policies/`：面向具体策略的评测适配层。
  - `base.py`：定义评测侧通用配置与策略抽象。
  - `utils.py`：根据配置实例化策略对象。
  - `lerobot.py`：提供 LeRobot 策略配置、观测映射和图像预处理。

### 5.7 `data_collection/panda/`

这里放 Panda 机械臂的数据采集与运动规划代码。

- `run.py`：批量运行已写好的 motion planning 解法并记录轨迹。
- `motionplanner.py`：Panda 机械臂的运动规划求解器封装。
- `solutions/*.py`：每个任务对应一个 `solve(env, seed, debug, vis)` 实现，用于生成演示轨迹。

### 5.8 `docs/`

这里放文档本体与过程记录。

- `docs/DEV.md`：当前总开发说明，负责目录、模块、函数、数据契约和废弃路径。
- `docs/sop/*.md`：面向执行步骤的 SOP。
- `docs/theory/*.md`：方法背景、理论说明和文献整理。
- `docs/edit_record/*.md`：历史修改记录和过程笔记。

## 6. 关键运行流程

### 6.1 `build`

`build` 的职责是从源数据构建 `EmixIndex`（schema v2）。

核心步骤是：

- 读取精确的 LeRobotDataset 根目录。
- 计算 sha256 全目录哈希（fingerprint），写入 `public.metadata.source_dataset_fingerprint`。
- 解析 episode、末端位姿、动作和可选特征。
- 运行邻域搜索。
- 按未来动作差异做过滤。
- 通过聚类形成 cross-point groups。
- 将结果保存成 `emix_index.json`。

### 6.2 `visualize` 和 `tune`

这两个命令的共同点是都依赖 `EmixIndex`。

- `visualize` 负责把当前索引和源数据转成静态 HTML 图。
- `tune` 负责打开交互式调参面板，并在参数变化时重建索引。

### 6.3 `policies.train`

训练入口通过 `EmixIndexReader` 直接从源 `LeRobotDataset` + `EmixIndex` 加载帧。

当前训练流程要求：

- 读取 `emix.index_path` 指向的 `emix_index.json`。
- 验证 schema version、source_repo_id、source_root 和 fingerprint。
- 按策略类型实例化 `EmixIndexReader` 和 `EmixPolicyDataset`。
- 把两个 member 的帧窗口作为一组 pair 送入策略。

如果 fingerprint 不匹配或 EmixIndex 缺少必要字段，必须立即报错。

## 7. 验证与测试

当前验证重点不在“能不能跑一个假样例”，而在“真实工件是否闭环”。

### 7.1 自动化关注点

应优先覆盖以下内容。

- `EmixIndex` JSON round-trip（schema v2，含 fingerprint）
- fingerprint 稳定性（同一数据集两次哈希一致）
- fingerprint 变化（文件内容改变后哈希不一致）
- `EmixIndexReader` 对 fingerprint 不匹配的硬失败行为
- `apply_emix_schema` 对 `emix.files_root` 的拒绝行为

### 7.2 真实数据验证

真实验证应使用 `emix` conda 环境和本地真实 fixture。

建议的验证序列是：

1. 用真实 LeRobotDataset 跑 `build`，生成 `emix_index.json`。
2. 检查 `emix_index.json` 的 `schema_version == 2` 和 `source_dataset_fingerprint`。
3. 检查至少一个 training group。
4. 在训练 YAML 中设置 `emix.index_path`。
5. 用 `EmixIndexReader` 跑 loader smoke。
6. 用 `emix_diffusion` 跑一轮最小训练 smoke。

## 8. 文件职责清单

本节按实际文件列出职责，便于直接对照代码定位。这里优先覆盖当前主动维护路径，避免把历史旧路径写成现状。

### 8.1 `embodied_mixup/`

- `__init__.py`：包标记文件，无运行逻辑。
- `__main__.py`：模块入口，把 `python -m embodied_mixup` 转发到 `cli.main()`；`emix/__main__.py` 提供推荐别名入口。
- `cli.py`：离线主命令入口，负责参数解析、日志初始化、数据集加载、`build`、`visualize`、`tune` 和 `train`。
  - `_setup_logging()`：创建 `logs/` 日志文件并接入控制台和文件输出。
  - `_default_emix_output_dir()`：为 `emix_index.json` 提供默认输出目录。
  - `_ensure_local_plotly_bundle()`：把 `plotly.min.js` 写到 HTML 同目录，避免依赖外部 CDN。
  - `_load_dataset()`：强制使用精确的 LeRobot 数据集根目录加载源数据。
  - `_resolve_config()`：把 CLI 参数组装成 `EmbodiedMixupConfig`。
  - `cmd_build()`：构建并持久化 `EmixIndex`。
  - `cmd_visualize()`：构建索引并生成 3D/2D HTML 可视化。
  - `cmd_tune()`：启动 Dash 调参面板。
  - `build_parser()`：定义子命令和参数。
  - `main()`：CLI 统一入口。
- `configs/__init__.py`：导出配置对象。
- `configs/config.py`：定义 `EmbodiedMixupConfig` 与默认配置。
  - `__post_init__()`：对所有候选生成参数做范围校验。
  - `xyz_thresh` / `rot_thresh`：向后兼容的属性别名。
- `datasets/__init__.py`：导出数据集路径工具。
- `datasets/path_utils.py`：做 LeRobot 数据集根目录校验和默认 `repo_id` 推导。
  - `require_lerobot_dataset_root()`：检查 `meta/info.json` 是否存在，确保输入是精确根目录。
  - `default_repo_id_from_dataset_root()`：从目录名推导 `repo_id`。
- `candidates/__init__.py`：候选模块导出。
- `candidates/emix_index.py`：定义 `EmixIndex` 及其全部 schema 数据类。
  - `EmixSourceMediaContract`：描述视频/帧对齐契约。
  - `EmixSourceDatasetDeclaration`：描述源数据集声明。
  - `EmixGroupState`：描述 group 是否可训练。
  - `EmixGroupMemberRef`：描述单个帧成员引用。
  - `EmixCrossPointGroup`：描述一个交叉点 group。
  - `EmixIndexPublic` / `EmixIndexContent` / `EmixIndex`：索引的公开区、内容区和总对象。
- `candidates/emix_index_utils.py`：构建、校验、保存和读取 `EmixIndex`。
  - `compute_source_dataset_fingerprint()`：对源数据集 `meta/`、`data/`、`videos/` 做全文件 sha256 指纹。
  - `validate_source_dataset_fingerprint()`：对照期望指纹做硬校验。
  - `inspect_source_dataset()`：提取 `repo_id`、`root`、媒体契约和 fingerprint。
  - `build_emix_index()`：完整构建索引的核心流程。
  - `_cluster_candidate_records()`：对候选 pair 聚类。
  - `_compute_progress_interval()` / `_should_log_progress()`：构建进度日志控制。
  - `_build_cluster_info()`：补充聚类统计信息。
  - `save_emix_index()` / `load_emix_index()`：索引 JSON 读写。
  - `_validate_group_members()`：验证 group 成员引用是否合法。
- `candidates/loaders.py`：从 LeRobotDataset 读取轨迹、动作和可选窗口特征。
  - `load_eef_poses()`：读取末端位姿。
  - `load_episode_indices()`：读取 frame 对应 episode 编号。
  - `load_actions()`：读取动作数组。
  - `load_gripper_signal()`：从动作中提取夹爪信号。
  - `load_optional_feature_window()`：读取可选特征窗口。
  - `_get_feature_meta()` / `_load_video_backed_feature_window()`：特征元数据与视频特征窗口读取辅助。
- `candidates/episode_utils.py`：处理 episode 边界和未来帧索引。
  - `build_episode_start_end()`：生成每个 episode 的起止位置。
  - `build_episode_bound_maps()`：生成快速查找边界映射。
  - `safe_future_index()`：在不越界前提下计算未来索引。
  - `compute_window_bounds()`：计算局部窗口边界。
- `candidates/pairs.py`：邻域搜索与 pair 映射整理。
  - `run_neighbor_search()`：执行候选邻域搜索。
  - `build_frame_to_neighbors()`：把 pair 列表整理成 frame 到 neighbor 的映射。
  - `_query_neighbor_pairs()`：内部查询函数。
- `candidates/action_filter.py`：按当前动作分歧、未来运动角度和夹爪变化过滤候选。
  - `run_action_filter()`：主过滤流程。
  - `compute_action_filter_metrics()`：统计过滤结果。
  - `compute_gripper_transition()`：计算夹爪状态转换。
- `candidates/ids.py`：生成确定性 ID。
  - `make_emix_sample_id()`：生成 sample ID。
  - `make_emix_group_id()`：生成 group ID。
  - `make_emix_index_id()`：生成 index ID。
- `candidates/schemas.py`：候选发现阶段的中间态数据结构。
  - `CandidatePublic`、`CandidateParticipant`、`CandidateRelation`、`CandidateMediaAsset`、`CandidateMedia`、`CandidateRecord`：描述候选记录、参与者和媒体信息。
- `visualization/__init__.py`：可视化模块导出。
- `visualization/utils.py`：把 `EmixIndex` 和源数据整理成 Plotly 友好的中间数据。
  - `prepare_visualization_data()`：构建轨迹、midpoint、密度、cluster label 等数据。
  - `_get_cluster_colors()`：返回稳定色板。
  - `_format_member_id()`：生成成员展示字符串。
  - `_compute_local_point_density()`：计算局部点密度。
  - `_estimate_density_radius()`：估计密度半径。
  - `_compute_local_point_density_bruteforce()`：无 SciPy 时的回退实现。
- `visualization/plot_intersections.py`：生成 3D 交叉点和轨迹图。
  - `plot_intersections_3d()`：主 3D 图入口。
- `visualization/plot_projections.py`：生成 XY/XZ/YZ 三视图。
  - `plot_2d_projections()`：主 2D 图入口。
- `visualization/camera_presets.py`：管理 3D 相机预设。
  - `_spherical_to_cartesian()`：把角度转成 Plotly eye 坐标。
  - `set_camera_view()`：应用预设或显式角度。
- `visualization/tuning_dashboard.py`：Dash 调参面板。
  - `create_tuning_dashboard()`：创建页面、滑条和回调。
  - `make_rebuild_fn()`：根据新参数重建 `EmixIndex`。
- `policies/__init__.py`：策略训练模块导出。
- `policies/train.py`：本地训练实现，支持普通训练和 EMIX 训练/断点续训。
  - `EmixTrainConfig`：训练配置对象。
  - `SaveableTrainConfig`：checkpoint 兼容的配置包装。
  - `_configure_logging()`：训练日志初始化。
  - `_make_output_dir()`：计算输出目录。
  - `_load_yaml_config()`：读取 YAML 配置。
  - `_reject_legacy_policy_emix_keys()`：拒绝旧版 policy 下 EMIX 字段。
  - `_get_emix_training_cfg()`：读取 `emix.training` 子配置。
  - `_validate_training_mode()` / `_resolve_training_mode()`：校验并解析训练模式。
  - `_load_dataset_metadata()`：读取 LeRobot 元信息。
  - `_build_train_config()`：把 YAML 归一化成 `EmixTrainConfig`。
  - `_create_dataset_and_policy()`：创建数据集和策略对象。
  - `_build_emix_dataset()`：单独构建 EMIX loader。
  - `_build_delta_timestamps()`：为非 EMIX 训练构造时间偏移查询。
  - `_prepare_policy_batch()`：对 batch 做最小兼容修正。
  - `_write_train_config_snapshot()`：写训练配置快照。
  - `_prepare_output_dir()`：保证输出目录为空。
  - `_build_resume_config()`：组装断点续训配置。
  - `_snapshot_payload()`：拼装训练快照 JSON。
  - `train()`：训练主循环。
  - `main()`：命令行入口。
- `policies/utils/bootstrap.py`：对 LeRobot 运行时做最小补丁，修正环境差异。
  - `bootstrap_lerobot_runtime()`：应用全部补丁。
  - `maybe_bootstrap_from_policy_type()`：按策略类型选择性 bootstrap。
  - `_ensure_package_stub()`、`_patch_lerobot_dataset_utils()`、`_patch_policy_factory()`、`_patch_scripts_if_present()`：内部补丁辅助。
- `policies/utils/registry.py`：EMIX 策略注册和类型识别。
  - `EmixPolicyRegistration`：注册信息结构。
  - `is_emix_policy_type()`：判断策略类型是否属于 EMIX 体系。
- `policies/utils/emix_config.py`：把 YAML 中的 `emix` 块注入策略配置。
  - `apply_emix_schema()`：验证字段并设置 `emix_index_path`。
- `policies/utils/emix_loader_registry.py`：根据策略类型和 EMIX 索引创建 loader。
  - `EmixLoaderRegistration`：注册结构。
  - `make_emix_loader()`：返回对应的 EMIX loader。
- `policies/utils/emix_loader_registry_utils.py`：EMIX loader 的核心实现。
  - `EmixLoaderSettings`：loader 参数集合。
  - `EmixTrainingSample`：训练样本结构。
  - `EmixIndexReader`：读取索引、验证 fingerprint、从源数据直接拉帧。
  - `EmixPolicyDataset`：训练侧 dataset 封装。
  - `_frame_dict_from_window()`：把窗口 payload 还原成 frame 字典。
  - `_stack_column()`：列式堆叠辅助。
  - `_get_lerobot_dataset_class()`：解析可用的 LeRobotDataset 类。
- `policies/utils/batch.py`：批处理与时间规格工具。
  - `PolicyTemporalSpec`：策略时间规格结构。
  - `get_policy_temporal_spec()`：从 policy config 读取时间规格。
  - `stack_tensor_list()`：把样本列表堆成 batch。
- `policies/utils/mix_scheduler.py`：构建 mix scheduler。
  - `make_mix_scheduler()`：根据配置创建 scheduler。
- `policies/utils/mix_scheduler_utils.py`：mix scheduler 协议与基础实现。
  - `MixSchedulerProtocol`：协议定义。
  - `ConstantMixScheduler` / `LinearMixScheduler`：基础调度器。
  - `register_mix_scheduler()`：注册新的 scheduler。
- `policies/utils/checkpoint.py`：训练状态、断点和配置快照。
  - `ResumeConfig`：断点续训配置。
  - `set_global_seed()`：设随机种子。
  - `resolve_checkpoint_dir()`：解析 checkpoint 目录。
  - `infer_start_step()`：推断起始 step。
  - `_load_policy_weights()` / `_load_training_state()`：内部加载辅助。
  - `load_train_state()`：恢复训练状态。
  - `save_config_snapshot()`：保存配置快照。
  - `log_resume_metadata()`：输出续训元信息。
  - `describe_resume_config()`：把续训配置转成可序列化字典。
- `policies/DP/`
  - `cfg_emix_diffusion.py`：Diffusion Policy 的 EMIX 配置继承。
  - `emix_diffusion.py`：Diffusion Policy 的 EMIX 模型/前向适配。
- `policies/ACT/`
  - `cfg_emix_act.py`：ACT 的 EMIX 配置继承。
  - `emix_act.py`：ACT 的 EMIX 适配实现。
- `policies/SmolVLA/`
  - `cfg_emix_smolvla.py`：SmolVLA 的 EMIX 配置继承。
  - `emix_smolvla.py`：SmolVLA 的 EMIX 适配实现。

### 8.2 `data_processing/`

- `convert_to_lerobot.py`：把原始 H5 演示转成旧版 LeRobot 数据布局。
  - `load_metadata()`：读取 H5 元信息。
  - `detect_rgb_cameras()`：识别 RGB 相机。
  - `load_trajectory_from_h5()`：读轨迹帧。
  - `parse_image_size()`：解析图像尺寸。
  - `create_directory_structure()`：创建输出目录树。
  - `resize_image_with_padding()`：等比缩放并补边。
  - `create_video_from_frames()`：编码视频。
  - `process_episode()`：处理单个 episode。
  - `calculate_statistics()`：统计数据集信息。
  - `create_meta_files()`：写 meta 文件。
  - `main()`：命令行入口。
- `convert_to_lerobot_v2.py`：把原始轨迹转成当前主用的 LeRobot V2 布局。
  - `detect_rgb_cameras()`：识别 RGB 相机。
  - `resize_image_with_padding()`：图像缩放补边。
  - `quaternion_to_axis_angle()`：四元数转轴角。
  - `parse_trajectory()`：解析轨迹并提取特征。
  - `create_lerobot_dataset()`：创建目标数据集结构。
  - `main()`：命令行入口。
- `filter_nonoop_lerobot.py`：过滤无效或重复帧。
  - `EpisodeLeRobotDataset`：按 episode 读取数据的 dataset 包装。
  - `compute_action_difference()`：计算动作差异。
  - `filter_redundant_frames()`：过滤冗余帧。
  - `main()`：命令行入口。
- `sample_subset_lerobot.py`：抽取子集数据集。
  - `parse_args()`：解析参数。
  - `validate_args()`：校验参数。
  - `compute_keep_count()`：计算保留 episode 数量。
  - `create_subset_atomically()`：原子创建子集。
  - `main()`：命令行入口。
- `remove_feature_lerobot.py`：删除不需要的特征列。
  - `main()`：命令行入口。
- `replay.py`：历史 ManiSkill 演示轨迹回放和检查，仅供归档代码查阅。
  - `_UnwrappedAttrProxy`：用于透传底层环境属性。
  - `Args`：回放参数。
  - `ReplayResult`：回放结果结构。
  - `sanity_check_and_format_seed()`：格式化并检查 seed。
  - `replay_parallelized_sim()` / `replay_cpu_sim()`：并行或 CPU 回放。
  - `_main_helper()` / `_main()`：内部执行入口。
  - `parse_args()` / `main()`：命令行入口。

### 8.3 `evaluation/`

- `evaluate.py`：评测总入口。
  - `EvalConfig`：评测环境与日志配置。
  - `format_dict_for_logging()`：把字典格式化成日志文本。
  - `add_dataclass_args()`：把 dataclass 字段挂到 CLI 参数。
  - `build_eval_env()`：创建评测环境并包装录像。
  - `main()`：执行评测循环。
- `policies/base.py`：定义评测策略的抽象配置和接口。
  - `PolicyConfig`：策略配置抽象类。
  - `BasePolicy`：策略接口抽象类。
- `policies/utils.py`：把策略配置实例化为可执行策略。
  - `make_policy_from_config()`：根据配置创建策略对象。
- `policies/lerobot.py`：LeRobot 评测策略适配。
  - `LeRobotPolicyConfig`：LeRobot 策略配置。
  - `resize_image_with_padding()`：图像补边缩放。
  - `identity_transform()`：空变换。
  - `image_transform_with_padding()`：图像预处理管线。
  - `get_nested_value()`：读取嵌套观测字段。
  - `ObservationMapper`：观测映射器。
  - `LeRobotPolicy`：LeRobot 评测策略实现。

### 8.4 `data_collection/panda/`

- `run.py`：批量执行 Panda 任务的 motion planning 方案并记录轨迹。
  - `parse_args()`：解析命令行参数。
  - `_main()`：单进程采集执行。
  - `main()`：多进程聚合和轨迹合并入口。
- `motionplanner.py`：Panda 机械臂 motion planning 求解器。
  - `PandaArmMotionPlanningSolver`：历史 ManiSkill 运动规划器适配，仅供归档代码查阅。
- `solutions/*.py`：每个任务一个 `solve()`，对接对应环境并输出轨迹。
  - 这些文件的职责都很单一，只负责给定环境下的人工编写解法，不负责数据转换或训练。

### 8.5 `docs/`

- `docs/sop/*.md`：操作步骤。
- `docs/theory/*.md`：理论说明。
- `docs/edit_record/*.md`：修改记录。

### 8.6 其他顶层文件

- `README.md`：仓库入口说明。
- `AGENTS.md`：当前协作约束和开发规则。

## 8. 已废弃或不再支持的路径

以下内容视为历史路径，不属于当前主动支持范围。

- `python -m embodied_mixup materialize` 命令（已移除）
- `EmixFiles` / `EmixFilesReader`（已替换为 `EmixIndexReader`）
- `EmixMaterializedIndex` / `export_materialized_lerobot_dataset`
- `policy.emix_files_root` / `emix.files_root` 配置键（出现时立即报错）
- `window_len` / `raw_candidate_feature_keys` CLI 参数
- pickle build snapshots
- synthetic pair datasets
- sample-centric sidecars
- legacy loaders
- `CandidateBuildResult`
- `RawCandidateDataset`
- `mixup_lerobot_dataset`
- `sample_manifest.json`
- `mixup_spec.json`
- `provenance.json`
- `media_manifest.json`
- `lerobot_dataset/` mini pair datasets
- `--dbscan_eps`
- `--dbscan_min_samples`
- `policy.mixup_data_root`

如果本地仍存在这些旧路径，它们只能作为历史数据或归档数据看待，不能作为当前管线输入。

## 9. 与其他文档的关系

为了避免重复，后续文档建议这样分工。

- `docs/DEV.md`：总开发说明，覆盖目录、模块、函数、数据契约和遗留约束。
- `docs/sop/*.md`：操作 SOP，覆盖执行步骤、命令、检查点和交付标准。
- `docs/theory/*.md`：方法论和论文草稿，覆盖 EMIX 的动机、推导和实验设想。

如果实现、SOP 和理论同时发生变化，优先保证 `DEV.md` 与代码保持同步，再补 SOP，最后补理论稿。
