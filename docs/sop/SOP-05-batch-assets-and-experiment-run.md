# SOP-05：批量资产构建（assets.py）与实验流水线（run.py）

环境安装以 [SOP-00](SOP-00-environment-installation.md) 为准。下文的
4070/5060 双卡分工是旧实验主机的批次配置；当前默认 `gpu_batches` 和
`assets.py --gpus` 均使用 CUDA 0，适用于单张 RTX 5090。当前安装入口将 Blackwell 所需 OIDN
构建在同一个 `emix-twin` 环境中。

本文档说明两个批量驱动脚本的当前用法：

- [`scripts/assets.py`](../../scripts/assets.py)：GPU worker 从共享工作清单动态抢单、每卡任一时刻至多一个仿真器，批量构建
  RoboTwin 任务资产（采集、转换、索引、可视化），面向居中臂相机变体
  （`cen_arm_*`），默认小 episode 数（5）。
- [`scripts/run.py`](../../scripts/run.py)：固定的双 GPU 左臂 DP/EMIX 实验
  矩阵，配置全部来自 [`configs/scripts/run.yaml`](../../configs/scripts/run.yaml)，
  覆盖 prepare -> index -> vis -> train-eval 四个全局阶段。

单任务手动流程见 [SOP-01](SOP-01-data-generation-pipeline.md)；单任务
端到端脚本见 `scripts/dataflow.sh`。三者共享同一套数据协议（FPS 17、
`--left-qpos-dim 7`、`observation.eef_pose` 索引特征）和环境约定。

命令环境约定：两个驱动脚本本身不要求激活特定 Conda 环境——`assets.py`
只依赖标准库，`run.py` 只额外要求启动它的 Python 装有 PyYAML（如 `emix`）。
所有子命令由脚本自身通过绝对路径 Python 或 `conda run` 在正确的环境中执行
（见 3.4 节环境矩阵），因此不要在已激活错误环境的 shell 中手动改写子命令。

## 1. 两个入口的选择

| 入口 | 任务集 | episodes | 图像尺寸 | 阶段 | 典型用途 |
|------|--------|----------|----------|------|----------|
| `scripts/assets.py` | 10 个活跃 family × `cen_arm_*` 变体 | 默认 `5` | 默认 `224x224` | collect / convert / index / vis | 为新相机变体快速产出资产 |
| `scripts/run.py` | `run.yaml` 的 `gpu_batches` 中固定的 10 个 `*_left_central_wide_cam` 任务 | `500` | `84x84` | prepare / index / vis / train-eval | 完整基线/EMIX 对照实验 |
| `scripts/dataflow.sh` | 单个任务 | 自定 `--num` | `224x224` | collect -> convert -> build -> vis(HTML+PNG) | 单任务端到端补数据 |

注意任务命名差异：`assets.py` 构建的是
`<family>_left_cen_arm_{right_wide,cen_side,front}_cam`；`run.py` 消费的是
`run.yaml` 里的 `<family>_left_central_wide_cam` 数据。两套命名指向不同的
task config，不能混用。

## 2. scripts/assets.py

### 2.1 命令

```bash
# Environment: 任一有标准库的 Python
python scripts/assets.py
```

默认对全部 10 个活跃 family、`cen_arm_right_wide_cam` 变体，GPU 0/1 两个
worker 从一条共享工作清单抢单（静态 round-robin 分配已废弃：任务时长相差
10 倍时会让某张卡空转），每个任务跑全部四个阶段、5 个 episode、224x224
图像。

只重建某个 family 的可视化与索引：

```bash
python scripts/assets.py --variant cen_arm_right_wide_cam \
  --gpus 0 1 \
  --episodes 200 \
  --image-width 84 \
  --image-height 84
```

### 2.2 CLI 参数

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `--tasks` | 可选，多值 | 全部 `ACTIVE_FAMILIES` | 任务 family 基名列表（不带变体后缀） |
| `--variant` | 可选枚举 | `cen_arm_right_wide_cam` | 相机变体，决定任务名后缀 |
| `--gpus` | 可选，多值 | `0 1` | 参与的 CUDA 设备号（CUDA 编号，非 nvidia-smi 编号） |
| `--episodes` | 可选整数 | `5` | 每个任务的 episode 数，必须为正 |
| `--stages` | 可选，多值 | `collect convert index vis` | 要执行的阶段子集 |
| `--image-width` | 可选整数 | `224` | 转换输出图像宽 |
| `--image-height` | 可选整数 | `224` | 转换输出图像高 |

变体后缀映射（`VARIANT_SUFFIXES`）：

| `--variant` | 任务名后缀 |
|-------------|------------|
| `cen_arm_right_wide_cam` | `_left_cen_arm_right_wide_cam` |
| `cen_arm_near_side_cam` | `_left_cen_arm_near_side_cam` |
| `cen_arm_side_cam` | `_left_cen_arm_side_cam` |
| `cen_arm_front_cam` | `_left_cen_arm_front_cam` |

任务名 = `<family><后缀>`。`--tasks` 传入的 family 必须属于脚本内的
10 个 `ACTIVE_FAMILIES`（blocks_ranking_rgb、blocks_ranking_size、
place_bread_basket、place_bread_skillet、place_burger_fries、
place_can_basket、place_cans_plasticbox、place_dual_shoes、
stack_blocks_two、stack_bowls_two），否则直接报错退出。

### 2.3 阶段与产物

| 阶段 | 完成判据（ reused 的条件） | 产物路径 |
|------|----------------------------|----------|
| `collect` | `data/robotwin/<task>/<task>/data/episode{0..N-1}.hdf5` 齐全 | 原生 RoboTwin collection |
| `convert` | `meta/info.json` 的 `total_episodes`、图像 shape、parquet 与 mp4 齐全 | `data/lerobot/<task>`（fps 17、`--left-qpos-dim 7`、repo-id 即任务名） |
| `index` | `emix_index.json` 存在 | `data/emix_index/<task>/emix_index.json` |
| `vis` | `<task>.html`、`<task>_first_frame_range.png` 与 `<task>_pick_place.{html,png}` 均存在 | `outputs/assets_vis/intersections/<task>.html`（`--skip-click-images`）+ `outputs/assets_vis/first_frame/<task>_first_frame_range.png`（占据范围热图）+ `outputs/assets_vis/pick_place/<task>_pick_place.{html,png}`（抓/放点分布，根目录可用 `--output-root` 覆盖） |

断点续跑语义：每个阶段开始前检查完成判据；已完成的阶段直接复用，未完成
（或部分完成）的产物目录会先被整体删除再重跑。某个任务在任何阶段抛错时，
该 GPU 队列停止，最后统一汇报所有失败任务。

`vis` 阶段额外调用
`embodied_mixup/visualization/first_frame_average.py`：取数据集内每个
episode 视频的第一帧，生成**占据范围热图**（默认 `--mode range`）：以逐像素
历史最大亮度估计静态背景，统计每个像素"比背景暗超过阈值"的 episode 占比，
用 turbo colormap 叠加在灰度背景上——高频放置点呈暖色、偶发点呈冷色，
近乎必现的静态元素（桌面、夹具、机械臂初始位）压暗为灰色。旧的白底
`1/N` 平均因背景主导而看不出分布，保留为 `--mode average`。该模块只在
单个数据集内部操作，所有 episode 索引都有越界断言；也可单独运行：

```bash
# Environment: emix
python -m embodied_mixup.visualization.first_frame_average \
  --root data/lerobot/<task> \
  --output outputs/assets_vis/first_frame/<task>_first_frame_range.png
```

**查看 HTML 的注意事项**：页面是全自包含的（plotly 已内联），直接双击或
`file://` 打开即可，不需要 Live Server。若用 Live Server 打开，注意它监听
整个 `outputs/assets_vis/` 目录——批次运行期间其它任务写入兄弟文件会触发已打开
页面的整页刷新；HTML/PNG 现在都经临时文件+原子重命名落盘（不会出现半写
状态），等批次跑完再打开即可稳定浏览。

### 2.4 GPU 路由与环境

本机 CUDA 设备编号与 nvidia-smi 相反：CUDA 0 是 RTX 4070，CUDA 1 是
RTX 5060。`--gpus` 使用 CUDA 编号。

| GPU | 物理 卡 | 采集环境 | 说明 |
|-----|---------|----------|------|
| `0` | RTX 4070 | `emix-twin` | 直接调用 |
| `1` | RTX 5060 (sm_120) | `emix-twin` | Blackwell 库路径处理（见下） |

环境选择**不按 GPU 序号硬编码**，而是由 [`scripts/oidn_env.py`](../../scripts/oidn_env.py)
在运行时用 torch 探测每张卡的 compute capability（可用
`python scripts/oidn_env.py 0 1` 单独查看）。Blackwell 卡（>= sm_100）的
OIDN 由 `_install_twin.sh` 在 `emix-twin` 内构建；Curobo 编译运动学需要该
环境的新 libstdc++ 和 ninja，所以运行时会在子进程环境里剥离
`LD_LIBRARY_PATH` 中的 `/usr/local/cuda` 条目（旧 cuBLASLt 会炸 GEMM），并补上
该环境的 `PATH`/`CUDA_HOME`/`LD_LIBRARY_PATH`/`LD_PRELOAD`。

## 3. scripts/run.py

### 3.1 配置文件 run.yaml

所有参数在 `configs/scripts/run.yaml`（入库跟踪）。文件只写操作者需要调的
键；派生路径（`data/robotwin|lerobot|emix_index/{task}`、
`outputs/train|eval/{task}` 等）和始终开启的 debug 开关由 `run.py` 的
`normalize_config` 按约定补全。关键小节：

| 小节 | 内容 |
|------|------|
| `project` | `robotwin_root`、`runtime_config_root`（生成配置目录）、`log_root` |
| `gpu_batches` | 每张 CUDA 设备的任务列表（当前 0:4、1:6），即任务集与分卡方案 |
| `environments` | 各阶段 Conda 环境名（RoboTwin 阶段统一使用 `emix-twin`） |
| `pipeline` | 每任务 500 episodes、84x84、fps 17、评估 100 episodes seed 0 等 |
| `training` | batch_size 64、30000 steps、save_freq 5000 等（`encoder_debug` 默认开启） |
| `emix` | alpha_mode/beta、loss_weight、mixup scheduler（`debug`/`visual_keys` 默认补全） |
| `visualization` | `output_root`、相机 preset、PNG 窗口尺寸、Chrome 渲染参数 |
| `logging` | worker 日志轮转字节数（默认 50 MiB） |

修改 `run.yaml` 中任何影响数据的参数（episodes、图像尺寸、环境名）都代表
新协议；已有的 collection/dataset 会被完成判据判为不完整并删除重跑。

### 3.2 命令

完整流程（四个阶段按全局屏障顺序执行）：

```bash
# Environment: 任一装有 PyYAML 的 Python（如 emix）
python scripts/run.py
```

单阶段回填（跳过环境/GPU 校验，直接在已有数据上跑一个阶段）：

```bash
python scripts/run.py --phase vis
python scripts/run.py --phase prepare
```

覆盖设备列表：

```bash
python scripts/run.py --cuda 0
```

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `--phase` | 可选枚举 | 无（跑全部） | `prepare` / `index` / `vis` / `train-eval` 四选一 |
| `--cuda` | 可选，多值 | `run.yaml` 的 `gpu_batches` 设备 | 覆盖设备列表（必须是 `gpu_batches` 中出现的设备的子集），如 `--cuda 0 1` |

### 3.3 阶段顺序与调度

| 阶段 | 内容 | 调度 |
|------|------|------|
| `prepare` | 缺失时采集（RoboTwin）+ 转换（LeRobot 84x84） | 每卡一个 worker，按 `gpu_batches` 分任务 |
| `index` | 为每个数据集构建 EmixIndex（CPU，独占主机核/内存） | 单个串行 worker，全部任务按平均帧数升序 |
| `vis` | 从 EmixIndex 渲染 HTML + headless Chrome PNG | 同上，单串行 worker |
| `train-eval` | 每任务先 baseline 后 emix：训练 -> 评估 | 每卡一个 worker，队列内按平均帧数升序 |

阶段之间是强制全局屏障；`index`/`vis` 必须在 `prepare` 之后，`vis` 消费
`index` 的产物。

### 3.4 环境矩阵

| 阶段（`environments` 键） | GPU 0 (4070) | GPU 1 (5060) |
|---------------------------|--------------|--------------|
| `robotwin`（采集、GPU 探针） | `emix-twin` | `emix-twin` |
| `preprocessing`（转换、索引、vis） | `emix` | `emix` |
| `training`（训练） | `emix` | `emix` |
| `evaluation`（在线评估） | `emix-twin` | `emix-twin` |

`robotwin`/`evaluation` 两个键统一使用 `emix-twin`；Blackwell 卡仍剥离
子进程 `LD_LIBRARY_PATH` 中的
`/usr/local/cuda`（防止系统 cuBLASLt 覆盖环境内构建）。环境切分是有意的
（RoboTwin 运行时只在 `emix-twin`），不要合并。

### 3.5 运行前校验（仅完整流程）

不带 `--phase` 启动时，`run.py` 在任何采集前做四类校验，任一失败立即退出：

1. `run.yaml` 结构（`project`/`gpu_batches` 等）、conda 可执行文件、四个环境键齐全；
2. 每个被用到的 Conda 环境 `python --version` 可运行；
3. 每张卡的 `robotwin` 环境能完成 CUDA 初始化并导入该任务的
   `envs.import_task_env`（提前暴露 torch/curobo/NumPy ABI 问题）;
4. `evaluation` 环境能导入
   `lerobot.policies.diffusion.modeling_diffusion`（提前暴露
   diffusers/huggingface_hub 版本漂移），以及每个任务的
   `task_config/left/<variant>/<task>.yml` 的 `episode_num` 与
   `pipeline.episodes_per_task` 一致且采集 `D435-W` 头部相机。

### 3.6 复用与断点续跑

- collection / dataset / index / vis 各有完成判据，完整即复用，不完整先删
  再重跑（`train.py` 拒绝非空 output_dir）。
- 训练：最终 checkpoint
  `outputs/train/<task>_dp_<kind>/checkpoints/<steps:06d>/pretrained_model/model.safetensors`
  存在即跳过。
- 评估：`outputs/eval/<task>_dp_<kind>/` 下有 `metrics.json` **且**对应
  checkpoint 仍在才跳过，否则视为过期重跑。

崩溃后直接重跑同一条命令即可，只有缺失的阶段会执行。

### 3.7 日志与锁

- 实例互斥：`logs/.flow-cuda-<devices>.lock`（fcntl 排他锁）。两个实验
  不能并发消耗主机内存；`--phase` 模式同样要拿锁。
- 每次运行创建 `logs/flow-<时间戳>-cuda-<devices>/`，worker 日志按
  `{worker:03d}-{序号:04d}.log` 轮转（默认 50 MiB 一卷）。
- 运行中的子命令绑定在独立进程组，Ctrl+C / SIGTERM 会整组终止，不会留下
  孤儿模拟器或训练进程。

### 3.8 产物路径

| 产物 | 路径 |
|------|------|
| 运行期生成的训练/评估配置 | `configs/scripts/run/<task>/{baseline,emix}_{train,eval}.yaml`（不入库） |
| 训练输出（含 encoder_debug） | `outputs/train/<task>_dp_{baseline,emix}/` |
| 评估输出（metrics.json、视频） | `outputs/eval/<task>_dp_{baseline,emix}/` |
| 可视化 | `outputs/intersections/<task>.html` + `<task>.png` |
| 数据与索引 | `data/robotwin/`、`data/lerobot/`、`data/emix_index/` |

### 3.9 绕过锁回填可视化

`python scripts/run.py --phase vis` 需要排他锁，可能被长训练阻塞数小时。
如只需在已有数据上补渲染，用 `scripts/temp/backfill_vis.py`（临时工具，
不入库）：它执行与 vis 阶段完全相同的命令和产物，但不获取 flow 锁
（纯 CPU，一次一个任务）。

## 4. 配置文件放置约定（采集与 train-eval 阶段）

规则：凡是脚本自动生成的配置文件（非人工编写），必须放在对应驱动的生成
目录中（gitignore 覆盖，不入库），不得与人工配置混放：

| 配置来源 | 允许位置 | 现状 |
|----------|----------|------|
| `run.py` train-eval 阶段生成的 `{baseline,emix}_{train,eval}.yaml` | `configs/scripts/run/<task>/` | 已由 `run.yaml` 的 `project.runtime_config_root` 固定 |
| `assets.py` 各阶段生成的运行期配置 | `configs/scripts/assets/<task>/` | 当前 `assets.py` 不落盘配置，仅传 CLI 参数；未来若引入生成配置必须写入此目录 |

- 采集（collect/prepare）阶段不生成项目侧配置：`assets.py`/`run.py` 直接
  传 CLI 参数；RoboTwin 的 task config
  （`third_party/RoboTwin/task_config/left/<variant>/<task>.yml`，按 envs/left
  的变体子目录组织）属于子模块，由
  `scripts/generate_left_task_assets.py` 生成或人工维护，不属于 `configs/`。
- `configs/train/`、`configs/eval/` 等目录只存放人工编写的配置；脚本不得
  向其写入生成配置。
- 生成配置会被驱动脚本随时覆盖重写，不要手改；需要调整参数时改
  `run.yaml`（`run.py`）或脚本内常量（`assets.py`）后重新生成。
- 复用生成配置的脚本（如 temp 里的 `eval_reseed.sh`）必须从
  `configs/scripts/run/<task>/` 读取，不得复制到别处再改。

## 5. 相关入口

| 入口 | 与本 SOP 的关系 |
|------|-----------------|
| [SOP-01](SOP-01-data-generation-pipeline.md) | 单任务手动版采集/转换/索引协议 |
| `scripts/dataflow.sh` | 单任务端到端（含 HTML+PNG），支持 `--seed` 固定场景种子 |
| `scripts/temp/backfill_vis.py`、`scripts/temp/eval_reseed*.sh` | 临时工具（不入库）：免锁回填 vis、新种子重评估队列 |
| [SOP-03](SOP-03-model-training.md) / [SOP-04](SOP-04-evaluation.md) | 训练与评估的手动协议 |
