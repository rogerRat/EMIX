# SOP-02 交点聚类与可视化

本 SOP 中所有 `build`、`visualize`、`tune` 命令均在 `emix` 环境运行。

当前 benchmark 是 RoboTwin。本文处理由 RoboTwin 生成并转换得到的标准
LeRobot 数据集，示例数据路径统一使用 `data/lerobot/<dataset_id>` 命名。

本文档描述基于 canonical `EmixIndex` 的可视化和调参流程。它的作用是帮助检查候选组、聚类结果和视觉分布，而不是替代正式训练或数据验证。
可视化的目的是进行超参数的选择。在确定超参数之后，将 `EmixIndex` 直接用于训练，无需物化中间数据集。
调参时既可以使用 CLI 方式，也可以使用非 CLI 方式。所谓非 CLI 方式，就是直接修改 [`embodied_mixup/configs/config.py`](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/configs/config.py) 中的 `EmbodiedMixupConfig` 默认值，例如 `xyz_epsilon`、`rpy_epsilon`、`future_horizon`、`future_angle_thresh`、`min_future_disp`、`action_delta_threshold`、`window_len`、`hdbscan_min_cluster_size_ratio`、`hdbscan_min_samples`、`hdbscan_cluster_selection_epsilon`、`hdbscan_alpha` 等字段。修改这些默认值后，后续命令如果不再显式传参，就会直接使用新的默认值，从而控制聚类、筛选和近邻查找行为。

## 0.1 EMix 筛选参数应该在哪里设置

当前这一步的 EMix 筛选参数，不是写在训练侧 YAML 配置文件里，而是通过 `embodied_mixup/cli.py` 的命令行参数传入。对外使用时推荐直接调用 `python -m emix`。

- 运行 `build`、`visualize`、`tune` 时，CLI 会先把参数解析成 `EmbodiedMixupConfig`
- `build` / `visualize` / `tune` 的参数入口在 `embodied_mixup/cli.py`
- 真正生效的默认值也定义在 `embodied_mixup/cli.py`，并由 `embodied_mixup/configs/config.py` 的 `EmbodiedMixupConfig` 接收和校验
- `configs/policies/*.yaml` 只用于后续训练阶段，不负责这一步的候选筛选与聚类调参

也就是说，如果你要调 `xyz_eps`、`rpy_eps`、`future_horizon`、`future_angle_thresh`、`min_future_disp`、`action_delta_threshold`、`cluster_selection_epsilon`、`hdbscan_min_cluster_size_ratio`、`hdbscan_min_samples`、`hdbscan_alpha` 等参数，可以优先在 CLI 命令里设置；如果想把某一组参数长期固定下来，则可以直接改 `embodied_mixup/configs/config.py` 里的 `EmbodiedMixupConfig` 默认值。两种方式本质上是同一套参数来源，区别只是一个是”单次覆盖”，一个是”修改默认值”。

## 0. 本流程所涉及的代码文件

- [embodied_mixup/cli.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/cli.py)
- [embodied_mixup/datasets/path_utils.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/datasets/path_utils.py)
- [embodied_mixup/candidates/emix_index_utils.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/candidates/emix_index_utils.py)
- [embodied_mixup/visualization/utils.py](/home/rogerspyke/projects/EMIX/Embodied-Mixup/embodied_mixup/visualization/utils.py)

## 1. 输入要求

运行可视化前，需要准备以下输入。

- 精确的源 `LeRobotDataset` 根目录，通过 `--root` 指定
- 源数据集 `repo_id`，可通过 `--repo_id` 指定；不传时按 `--root` 的最后一级目录名自动推断
- 一组聚类和邻域搜索超参数

关键约束是：

- `--root` 必须自身包含 `meta/info.json`
- CLI 不会猜测父目录，也不会自动切换到 legacy 布局
- 输入数据必须和 `data_processing/convert_to_lerobot_v2.py` 的产物一致，不能混用其他非标准 `LeRobotDataset` 布局
- 只要 `data/`、`videos/` 或 `meta/` 结构不一致，就应当先转换成标准结构再进入本 SOP
- `build`、`visualize` 和 `tune` 都会先验证 `root`；如果未显式提供 `repo_id`，会根据目录名自动推断

## 1.1 CLI 命令概览

当前 CLI 推荐通过 `python -m emix` 调用，支持三个主要命令：

| 命令 | 用途 | 输出 |
|------|------|------|
| `build` | 构建标准 `EmixIndex` | `emix_index.json` |
| `visualize` | 生成交点分布的交互式可视化 | Plotly HTML 文件 |
| `tune` | 启动交互式调参仪表板 | Dash web 服务器 |

`materialize` 命令已移除。训练直接从源 `LeRobotDataset` 加载，无需物化中间数据集。

## 2. StackCube 示例

下面以仓库里当前的 StackCube 数据链路为例。

- 源轨迹任务：`click_bell_left`
- 转换后的 LeRobot 数据集：`click_bell_left_piper_200_head_84`
- 数据集根目录：`data/lerobot/click_bell_left_piper_200_head_84`
- `repo_id`：`click_bell_left_piper_200_head_84`

这个例子假设你已经完成了 SOP-01 中的生成、重放、转换、去特征和过滤步骤，并且源数据已经满足当前训练侧要求的精确根目录契约。

### 2.1 计算 `EmixIndex`

```bash
# Environment: emix
python -m emix build \
  --repo_id stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --root data/lerobot/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --output_dir data/emix_data/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop
```

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--repo_id` | `root.name` | 指定源 LeRobot 数据集名称；不传时按 `--root` 最后一级目录名推断 |
| `--root` | 必需 | 指定精确的数据集根目录 |
| `--output_dir` | `data/emix_data/<repo_id>` | 指定 `EmixIndex` 输出目录 |
| `--xyz_eps` | `0.02` | 位姿空间近邻阈值（米） |
| `--rpy_eps` | `0.01` | 姿态角近邻阈值（弧度） |
| `--future_horizon` | `8` | 未来动作比较步长（帧数） |
| `--future_angle_thresh` | `π/6` | 未来位移方向角阈值（弧度） |
| `--min_future_disp` | `0.002` | 最小未来位移阈值（米） |
| `--action_delta_threshold` | `0.05` | 当前动作分歧阈值（L2 距离） |
| `--disable_gripper_action_filter` | 不设置 | 禁用夹爪转移诊断 |
| `--cluster_selection_epsilon` | `0.05` | HDBSCAN 聚类选簇阈值 |
| `--hdbscan_min_cluster_size_ratio` | `0.01` | 最小簇大小比例（总候选对数的百分比） |
| `--hdbscan_min_samples` | `3` | HDBSCAN 最小样本数 |
| `--hdbscan_alpha` | `1.0` | HDBSCAN 聚类稳定性参数 |

上述参数都来自 CLI。当前实现里没有单独的 `emix_index.yaml` 或同类配置文件入口；如果不在命令中显式传入，就会使用 `embodied_mixup/cli.py` 中定义的默认值。

这个命令会生成：

- `data/emix_data/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop/emix_index.json`

## 3. 可视化命令

`visualize` 命令生成交互式 Plotly 可视化，展示候选交点在3D空间中的分布和2D投影。

```bash
# Environment: emix
python -m emix visualize \
  --repo_id stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --root data/lerobot/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --xyz_eps 0.02 \
  --rpy_eps 0.01 \
  --cluster_selection_epsilon 0.05 \
  --hdbscan_min_samples 3
```

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--repo_id` | `root.name` | 指定源 LeRobot 数据集名称；不传时按 `--root` 最后一级目录名推断 |
| `--root` | 必需 | 指定精确的数据集根目录 |
| `--output` | `outputs/intersections/emix_visualization_{repo_id}.html` | 输出 HTML 文件路径 |
| `--xyz_eps` | `0.02` | 位姿空间近邻阈值（米） |
| `--rpy_eps` | `0.01` | 姿态角近邻阈值（弧度） |
| `--cluster_selection_epsilon` | `0.05` | HDBSCAN 聚类选簇阈值 |
| `--hdbscan_min_samples` | `3` | HDBSCAN 最小样本数 |
| `--future_horizon` | `8` | 未来动作比较步长（帧数） |
| `--future_angle_thresh` | `π/6` | 未来位移方向角阈值（弧度） |
| `--min_future_disp` | `0.002` | 最小未来位移阈值（米） |
| `--hdbscan_min_cluster_size_ratio` | `0.01` | 最小簇大小比例 |
| `--hdbscan_alpha` | `1.0` | HDBSCAN `alpha` 参数 |
| `--camera` | `isometric` | 3D 摄像机预设（`front`, `top`, `side`, `isometric`, `close`） |
| `--vis-obj` | 不启用 | RoboTwin 任务文件路径（可省略 `.py`）；将任务 manifest 中每个非机械臂 actor 的 XY 随机化范围叠加到 3D 图和 XY/XZ/YZ 投影。 |
| `--grid_resolution` | `30` | 密度云渲染的网格分辨率 |
| `--density_sigma` | `1.5` | 密度云高斯平滑 sigma 值 |
输出内容：

- `emix_visualization_{repo_id}.html` - 包含3D交点散点图和2D投影的交互式可视化
- `plotly.min.js` - 本地 Plotly 库（自动下载）

例如，为 `blocks_ranking_rgb` 数据集显示其三个 block 的定义域：

```bash
python -m emix visualize \
  --root data/lerobot/<dataset> \
  --vis-obj third_party/RoboTwin/envs/left/cen_arm_right_wide_cam/blocks_ranking_rgb_left_cen_arm_right_wide_cam
```

`--vis-obj` 不启动 RoboTwin 或构建仿真场景；它读取 EMIX 左臂任务声明的
`manifest.actor_specs`。橙色线框是每个 actor 的随机化 XY 范围，3D 图中以桌面高度
附近的薄盒表示。LeRobot 转换数据默认不保存每个物体的逐帧 pose，因此该功能展示的是
**定义范围**，不是某次采集实际抽样到的位置。若需要绘制实际位置，采集/转换流程必须额外
写入每个 actor 的名称和初始/逐帧 world pose，再由可视化读取该字段。

## 4. 调参命令

`tune` 命令启动一个交互式 Dash 仪表板，允许实时调整聚类超参数并查看结果变化。

```bash
# Environment: emix
python -m emix tune \
  --repo_id stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --root data/lerobot/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop
```

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--repo_id` | `root.name` | 指定源 LeRobot 数据集名称；不传时按 `--root` 最后一级目录名推断 |
| `--root` | 必需 | 指定精确的数据集根目录 |
| `--host` | `127.0.0.1` | Dash 服务器绑定地址 |
| `--port` | `8050` | Dash 服务器端口 |
| `--debug` | 不设置 | 启用 Dash 调试模式（热重载、错误显示） |
| `--xyz_eps` | `0.02` | 位姿空间近邻阈值（米） |
| `--rpy_eps` | `0.01` | 姿态角近邻阈值（弧度） |
| `--future_horizon` | `8` | 未来动作比较步长（帧数） |
| `--future_angle_thresh` | `π/6` | 未来位移方向角阈值（弧度） |
| `--min_future_disp` | `0.002` | 最小未来位移阈值（米） |
| `--cluster_selection_epsilon` | `0.05` | HDBSCAN 聚类选簇阈值 |
| `--hdbscan_min_cluster_size_ratio` | `0.01` | 最小簇大小比例 |
| `--hdbscan_min_samples` | `3` | HDBSCAN 最小样本数 |
| `--hdbscan_alpha` | `1.0` | HDBSCAN `alpha` 参数 |
| `--grid_resolution` | `30` | 密度云渲染的网格分辨率 |
| `--density_sigma` | `1.5` | 密度云高斯平滑 sigma 值 |

仪表板会在参数改变时实时重建 `EmixIndex`，并显示当前的聚类结果和组数量统计。

启动后访问 `http://127.0.0.1:8050` 即可打开调参界面。

## 5. 当前语义

### 5.1 `build`

- 从源 `LeRobotDataset` 构建标准 `EmixIndex`
- 执行邻域搜索、动作过滤、聚类等核心分析
- 只输出索引工件，不产出训练数据

### 5.2 `visualize`

- 构建 `EmixIndex`
- 基于 `EmixIndex` 和源数据张量准备可视化数组
- 写出交互式 Plotly HTML，展示3D交点分布和2D投影
- 可选导出物化数据集

### 5.3 `tune`

- 构建初始 `EmixIndex`
- 显示当前簇数量和 group 数量
- 启动 Dash 仪表板
- 在参数变化时按需重建 `EmixIndex`

## 6. 支持参数

所有命令（`build`、`visualize`、`tune`）都支持以下超参数：

这些命令最终都会收敛到 `embodied_mixup/configs/config.py` 里的 `EmbodiedMixupConfig`。该配置类定义了各参数的默认值和合法性校验；CLI 只是把命令行输入覆盖到这份配置上。因此，如果你不想每次都在命令行重复输入同一组参数，可以直接修改 `config.py` 中的默认值。下面各参数名称同时给出 CLI 常用写法和配置字段写法，方便你在两种入口之间对应。

### 6.1 邻域搜索参数

- `--xyz_eps` / `xyz_epsilon` - 位姿空间距离阈值（米）。值越小，越严格要求平移位置接近，候选交点会更少但更精确。
- `--rpy_eps` / `rpy_epsilon` - 姿态角距离阈值（弧度）。值越小，越严格要求姿态一致，适合抑制姿态差异较大的误匹配。

### 6.2 动作过滤参数

- `--future_horizon` / `future_horizon` - 未来动作比较的帧数。值越大，越关注更远未来的动作趋势，但也更容易引入长时序噪声。
- `--future_angle_thresh` / `future_angle_thresh` - 未来位移方向的角度阈值（弧度）。用于判断未来运动方向是否一致，阈值越小，方向一致性要求越高。
- `--min_future_disp` / `min_future_disp` - 稳定角度估计的最小位移阈值（米）。只有当未来位移足够大时，方向判断才更可靠。
- `--action_delta_threshold` / `action_delta_threshold` - 当前帧动作分歧阈值（L2 距离）。用于判断 anchor 和 neighbor 的当前动作是否足够不同，只有达标后才会继续和未来角度条件一起通过筛选。
- `--disable_gripper_action_filter` / `gripper_action_filter_enabled` - 禁用夹爪转移诊断。需要注意，`config.py` 里对应的是 `gripper_action_filter_enabled`，默认值为 `True`，如果想通过配置关闭它，需要把它改成 `False`。

### 6.3 聚类参数

- `--cluster_selection_epsilon` / `hdbscan_cluster_selection_epsilon` - HDBSCAN 聚类合并阈值（标准化 6D 空间）。值越大，越容易把相近的小簇合并成更大的簇。
- `--hdbscan_min_cluster_size_ratio` / `hdbscan_min_cluster_size_ratio` - 最小簇大小占候选对总数的比例。值越大，越偏向保留大簇并过滤小簇。
- `--hdbscan_min_samples` / `hdbscan_min_samples` - HDBSCAN 核心点分类的最小样本数。值越大，聚类越保守，对稀疏区域更不敏感。
- `--hdbscan_alpha` / `hdbscan_alpha` - HDBSCAN 聚类稳定性参数。它影响簇的稳定性判定，通常在保持其他参数不变时再做微调。

### 6.4 训练候选参数（仅用于元数据记录，当前未实现采样）

- `--training_candidates_total_budget` / `training_candidates_total_budget` - 训练样本总预算。当前主要用于元数据记录，暂不直接参与采样逻辑。
- `--training_candidates_cap_per_cluster` / `training_candidates_cap_per_cluster` - 每个簇的硬上限。用于限制单簇贡献的候选规模，避免少数大簇主导结果。
- `--training_candidates_include_noise` / `training_candidates_include_noise` - 是否将噪声组标记为可训练。默认 `False`，表示噪声一般不进入训练候选。
- `--training_candidates_seed` / `training_candidates_seed` - 随机种子。用于保证采样或记录过程可复现。

旧的 CLI 别名不再支持。

## 7. 输出工件

### 7.1 `build` 输出

```text
--output_dir/
  emix_index.json
```

### 7.2 `visualize` 输出

```text
outputs/intersections/
  emix_visualization_{repo_id}.html
  plotly.min.js
```

### 7.3 `tune` 输出

- Dash web 服务器（默认 http://127.0.0.1:8050）
- 不产生文件输出

## 8. 推荐工作流

推荐先停留在 `EmixIndex` 阶段做可视化和参数收敛。建议先用一组参数算出索引，然后在 `visualize` 或 `tune` 阶段反复观察聚类效果；如果发现某些参数需要长期固定，可以直接修改 `config.py` 的默认值。当可视化结果和聚类边界都满意之后，直接将 `emix_index.json` 路径填入 `emix.index_path` 启动训练，无需物化中间数据集。

### 8.1 快速可视化

```bash
# Environment: emix
python -m emix visualize \
  --repo_id <repo_id> \
  --root <dataset_root> \
  --output outputs/intersections/my_viz.html
```

用于快速查看候选分布，不产出训练数据。

### 8.2 完整流程（从源数据到训练）

```bash
# Environment: emix
# 第1步：构建索引
python -m emix build \
  --repo_id <repo_id> \
  --root <dataset_root> \
  --output_dir data/emix_data/<repo_id>

# Environment: emix
# 第2步：可视化验证
python -m emix visualize \
  --repo_id <repo_id> \
  --root <dataset_root> \
  --output outputs/intersections/final_check.html

# 第3步：在训练 YAML 中设置 emix.index_path，直接训练
# configs/policies/your_config.yaml:
#   emix:
#     index_path: data/emix_data/<repo_id>/emix_index.json
```

### 8.3 调参流程

```bash
# Environment: emix
# 启动调参仪表板
python -m emix tune \
  --repo_id <repo_id> \
  --root <dataset_root> \
  --port 8050 \
  --debug
```

- 在浏览器中调整 slider
- 观察聚类结果变化
- 记下最优超参数
- 关闭仪表板后，先用相同参数重新运行 `visualize` 或 `build` 验证 `EmixIndex`
- 如果后续想长期复用这组参数，可以把最优值写回 `embodied_mixup/configs/config.py`
- 最后将生成的 `emix_index.json` 路径填入训练 YAML 的 `emix.index_path`

## 9. 与其他文档的关系

- **数据准备**：`docs/sop/01-data-generation/SOP-01-data-generation-pipeline.md` - 如何从演示轨迹生成标准 LeRobot 数据集
- **数据结构**：`docs/sop/02-emix-processing/data-structure-and-transforms.md` - LeRobotDataset、EmixIndex 的详细格式说明
- **模型训练**：`docs/sop/03-train-and-inference/SOP-03-model-training.md` - 如何使用 EmixIndex 直接进行策略训练
- **总开发说明**：`docs/dev/DEV.md` - 项目全景和开发指南

## 10. 常见问题

### Q: 调参时应该重点调整哪些参数？

A: 通常优先调整：
- `--xyz_eps` 和 `--rpy_eps` - 控制邻域搜索的宽松程度
- `--cluster_selection_epsilon` - 控制微簇合并程度
- `--hdbscan_min_samples` - 控制稀疏区域的敏感性

### Q: 训练如何加载 EMIX 数据？

A: 训练直接从源 `LeRobotDataset` 加载。`emix_index.json` 中记录的 `source_frame_index` 指向源数据集的帧，训练 loader 直接读取这些帧，无需物化中间数据集。

### Q: `emix.index_path` 是什么？

A: 是训练 YAML 中的配置键，指向 `build` 命令生成的 `emix_index.json` 文件路径。旧的 `emix.files_root` / `policy.emix_files_root` 已移除，会在加载阶段直接报错。
