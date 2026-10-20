# SOP-05：批量资产构建（assets.py）与实验流水线（run.py）

环境安装以 [SOP-00](../04-environment-and-rendering/SOP-00-environment-installation.md) 为准。下文的
当前默认 `gpu_batches` 和 `assets.py --gpus` 均使用 CUDA 0，适用于
单张 RTX 5090。安装入口按实际计算能力构建 Blackwell 所需 OIDN，
并统一使用 `emix-twin` 环境。

本文档说明两个批量驱动脚本的当前用法：

- [`scripts/assets.py`](../../../scripts/assets.py)：GPU worker 从共享工作清单动态抢单、每卡任一时刻至多一个仿真器，批量构建
  RoboTwin 任务资产（采集、转换、索引、可视化），面向居中臂相机变体
  （`cen_arm_*`），默认小 episode 数（5）。
- [`scripts/run.py`](../../../scripts/run.py)：可按配置使用单张或多张 GPU 的左臂 DP/EMIX 实验
  矩阵，配置全部来自 [`configs/scripts/run.yaml`](../../../configs/scripts/run.yaml)，
  覆盖 prepare -> index -> vis -> train-eval 四个全局阶段。

单任务手动流程见 [SOP-01](../01-data-generation/SOP-01-data-generation-pipeline.md)；单任务
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

默认对全部 10 个活跃 family、`cen_arm_right_wide_cam` 变体，在 CUDA 0
启动一个 worker；指定多张卡时，worker 从一条共享工作清单抢单。每个任务跑全部四个阶段、5 个 episode、224x224
图像。

只重建某个 family 的可视化与索引：

```bash
python scripts/assets.py --variant cen_arm_right_wide_cam \
  --gpus 0 \
  --episodes 200 \
  --image-width 84 \
  --image-height 84
```

### 2.2 CLI 参数

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `--tasks` | 可选，多值 | 全部 `ACTIVE_FAMILIES` | 任务 family 基名列表（不带变体后缀） |
| `--variant` | 可选枚举 | `cen_arm_right_wide_cam` | 相机变体，决定任务名后缀 |
| `--gpus` | 可选，多值 | `0` | 参与的 CUDA 设备号 |
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

`--gpus` 使用 CUDA 编号；当前单卡服务器只需 `--gpus 0`。
采集环境统一为 `emix-twin`。

环境选择**不按 GPU 序号硬编码**，而是由 [`scripts/oidn_env.py`](../../../scripts/oidn_env.py)
在运行时用 torch 探测每张卡的 compute capability（可用
`python scripts/oidn_env.py 0` 单独查看）。Blackwell 卡（>= sm_100）的
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
| `--cuda` | 可选，多值 | `run.yaml` 的 `gpu_batches` 设备 | 覆盖设备列表（必须是 `gpu_batches` 中出现的设备的子集），如 `--cuda 0` |

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

| 阶段（`environments` 键） | 环境 |
|---------------------------|------|
| `robotwin`（采集、GPU 探针） | `emix-twin` |
| `preprocessing`（转换、索引、vis） | `emix` |
| `training`（训练） | `emix` |
| `evaluation`（在线评估） | `emix-twin` |

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
| [SOP-01](../01-data-generation/SOP-01-data-generation-pipeline.md) | 单任务手动版采集/转换/索引协议 |
| `scripts/dataflow.sh` | 单任务端到端（含 HTML+PNG），支持 `--seed` 固定场景种子 |
| `scripts/temp/backfill_vis.py`、`scripts/temp/eval_reseed*.sh` | 临时工具（不入库）：免锁回填 vis、新种子重评估队列 |
| [SOP-03](../03-train-and-inference/SOP-03-model-training.md) / [SOP-04](../03-train-and-inference/SOP-04-evaluation.md) | 训练与评估的手动协议 |

---

### 6. 批次实验启动与归档规范

面向执行代理（Agent）的流程规则层：启动一次 DP-vs-EMIX 完整批次实验、以及
批次完成后归档时，**必须**遵守本文约定，避免行为偏移。机制细节（CLI 参数、
阶段流水线、环境矩阵）见本 SOP 第 1–5 节；
本文件只定义"哪些参数要问、什么必须报、东西放哪、留什么删什么"。

### 1. 启动新批次实验

#### 1.1 必须向用户确认的参数（每次启动前，不允许自行默认）

即使与上一批次完全相同，也必须列出让用户明确确认（"与 top_cam 500ep 批完全
相同，确认？"不算违规，静默开跑算违规）。

| 参数 | 说明 | 常见取值 |
|---|---|---|
| `variant`（相机视角） | cen_arm 系列变体 | `cen_arm_top_cam` / `cen_arm_right_wide_cam` / `cen_arm_near_side_cam` / `cen_arm_front_cam` / `cen_arm_side_cam` |
| `--episodes` | 每任务 accepted episode 数 | `200` / `500` |
| 任务集合 | 默认 10 个 ACTIVE_FAMILIES | 增减任何任务必须明示并获确认 |
| `--image-width/height` | 训练分辨率 | 默认 `84 84` |
| `--gpus` | 采集/评测使用的 CUDA 设备 | 默认 `0`；多卡时显式列出设备编号 |

#### 1.2 隐性参数（沿用默认，但每次启动必须把数值报给用户）

报告格式：一行清单收在启动说明里，并注明"以
`configs/scripts/run_<cfg>.yaml` 与各任务 `train_config_snapshot.json` 为准"。

- 训练：`steps 30000`、`batch_size 64`、`optimizer_lr 1e-4`、`resnet18`、
  `horizon 16`、`n_obs_steps 1`、`n_action_steps 8`、`seed 0`、`save_freq 5000`
- EMIX 混合：`alpha beta(2)`、`state_mode anchor`、`loss_weight 1.0`、
  scheduler `logarithmic 0.6→0.0`
- 配对（决定 index 质量，2026-09-04 起为新默认）：`xyz_eps 0.02 m`、
  `rpy_eps 0.26 rad(15°)`、`exclude_tail_seconds 0.1 s @17fps`、
  `cross_episode_only`、`gripper_consistency`
- 评测：每任务 `100 episodes`、`seed 0`、保存视频
- 环境：robotwin/evaluation = `emix-twin`，training/preprocessing = `emix`；
  Blackwell GPU 的 OIDN 在安装时构建，不手动改写子命令环境

#### 1.3 启动前检查

1. 变体接线四件套齐全：RoboTwin `envs/left/<variant>/` 任务文件、
   `data_collection/robotwin.py` 注册、`configs/{train,eval}/<variant>/`、
   `left_task_base.CAMERA_SPECS`。缺哪件补哪件，不要裸跑。
2. run yaml：`configs/scripts/run_<variant>[_<ep>ep].yaml`。新批次从最近的
   同类 yaml 用 sed 替换任务名/episodes 生成；**该目录按 .gitignore 约定
   不入库**，不要 `git add -f`。
3. 工作区干净（未提交改动先按语义批次提交，见 1.5）、所选 GPU 空闲
   （`nvidia-smi`）、live `data/` 与 `outputs/{eval,train,intersections}`
   无同名残留（归档后应为空）。

#### 1.4 启动与记录（约定俗成）

- tmux 会话：单字母递增（`d`/`e`/`f` 已用）或语义名；
  `tmux new-session -d -s <id> -c <repo-root>`
- 命令模板：

  ```bash
  python scripts/assets.py --variant <V> --gpus 0 --episodes <N> \
    --image-width 84 --image-height 84 \
  && python scripts/run.py --config configs/scripts/run_<V>[_<N>ep].yaml
  ```

- **run ID** = flow 自动生成的 `logs/flow-<YYYYMMDD>-HHMMSS-cuda-0/` 目录名，
  在启动说明里报给用户并记录
- 启动后验证：每个所选 GPU 至多一个仿真进程（`nvidia-smi --query-compute-apps`）；
  `[assets] gpu=X claimed task=...` 行出现；开头 `simulate data episode N
  fail!` 是专家失败换 seed 重试，属正常两阶段采集，**不要**当成故障干预
- 必须记录到两处：`docs/PROG.md`（带日期小节）与代理记忆。内容：
  启动时间戳、启动时 commit ID、tmux 会话、run ID、1.1/1.2 的参数、
  预计时长、对照基线（哪个归档批次）

#### 1.5 启动前如工作区不干净

按语义分批提交（本次改动一个 commit；非本次的遗留改动总结为另外的语义
commit），全部 push 后再开跑。commit message 遵循仓库现有
`type(scope): summary` 风格并附 Co-Authored-By 尾注。

### 2. 归档规范

#### 2.1 时机与前提

- 前提：20 个 eval run 全部完成、每个 `metrics.json` 齐全且 episodes=100；
  经用户确认后执行（用户说"归档"即确认）
- 归档前先记录 **commit ID**（`git rev-parse --short HEAD`）——SUMMARY.md
  要写

#### 2.2 位置与命名（严格）

```
archive/DP-10tasks-<ep>ep-emix-baseline-<variant-kebab>-<YYYYMMDD>/
```

- 例：`DP-10tasks-200ep-emix-baseline-cen-arm-top-cam-20260904`
- **时间戳必须有**，取归档当日；variant 用 kebab-case 短名（near-side-cam /
  right-wide-cam / top-cam）

#### 2.3 结构与保留/删除规则（严格）

| 子目录 | 内容 | 保留 | 删除 |
|---|---|---|---|
| `data/robotwin` `data/lerobot` `data/emix_index` | 10 任务三件套 | 按原规则整体 `mv` 进归档 | live 对应目录随后为空 |
| `eval/` | 20 个 run | 每个 run 的 `metrics.json` 全保；评估视频默认保留该 run 视频数的 **5%**（100 episodes 默认 5 条），在整个 run 中均匀抽样；其他非大文件保留 | 其余评估视频及其他大文件删除 |
| `train/` | 20 个任务 | 所有非大文件（包括 JSON、TXT 等配置与元数据） | 所有 `checkpoints/` / `ckpt` 目录及其他大文件一律删除 |
| `vis/` | 可视化 | 非大文件按原归档路径保留，包括 HTML、PNG、`*_assets`、`plotly.min.js` 中符合大小要求的文件 | 大文件删除 |

通用保留规则：除 `data/{robotwin,lerobot,emix_index}` 按原规则整体归档外，
归档中只保留非大文件；所有模型 checkpoint 无论大小都必须删除。评估视频按
每个 run 单独计算 5%，不足 100 个 episode 时向上取整且至少保留 1 条；均匀抽样
覆盖 run 的前、中、后段，并在 `SUMMARY.md` 记录保留比例和视频文件清单。其他
非大文件（如 JSON、TXT 和小型调试产物）保留。

归档后 live 的 `data/{robotwin,lerobot,emix_index}`、
`outputs/{eval,train,intersections}` 应全部为空；用 `df -h` 核对释放量。归档前
清点每个目录的大小，确认所有 checkpoint 和其他大文件已排除，再移动非大文件
及规定保留的数据、评估视频和可视化文件。

#### 2.4 SUMMARY.md（必须，放归档根目录）

必须包含，缺一不可：

1. **归档时间戳**（YYYY-MM-DD）与**归档时 commit ID**——硬性要求
2. 批次背景：variant、1.1/1.2 参数、采集规模与每任务专家成功率
   （`collection_metrics.json` 的 accepted/attempted）、训练/eval 规模
3. 复现命令（assets.py + run.py 两行）
4. 结果表：逐任务 baseline vs EMIX（百分比 + n/100）+ delta + TOTAL 行
5. 与可比批次的对比表（如同机位异 episode 数、同参数异机位）
6. 数据异常备注（命名不一致、缺失任务、采集事故等）
7. 评估视频保留比例及保留文件清单

#### 2.5 归档后

更新 `docs/PROG.md`（归档时间戳 + SUMMARY 里的 TOTAL 数字）与代理记忆
（归档路径、结果一句话、机位/批次对比结论）。prog.md 的日期小节**倒序排列：
最近的日期在最上面**，新条目插在文件头部最旧的日期之前。

### 3. 速查

- 结果解读：100-ep eval 的逐任务 delta 噪声约 ±5pt，±3pt 以内的组间差异
  应读作持平
- 三机位扫描现状（200ep）：near_side +1 / right_wide +0.6 / top −2.9；
  500ep + 新配对判据（rpy 15° + 去尾）首个批次已归档：500ep top_cam
  Δ=+2.3（archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911）
- 历史归档索引：`archive/*/SUMMARY.md`，每个归档一份，先读它再比对
