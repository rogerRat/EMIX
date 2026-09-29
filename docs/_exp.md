# 实验协议真相文档（_exp.md）

更新：2026-09-28。**本文档是当前项目实验的唯一认证来源和实施记录**：实验命名、固定条件、运行顺序和状态的唯一权威入口。YAML 是协议的可执行表示；TODEV 是开发清单，SOP 是操作说明，归档 SUMMARY 是历史证据。

文档结构：前半部分为**现行协议**（V0 总协议及其下属实验，固定当前要进行的实验、实施方法与全部参数）；后半部分为**历史更新记录**（按日期升序排列，越早的越靠前）。

发生冲突先核对实际配置/快照/metrics，再更新本文档，不能把计划写成已完成结果。

---

# 第一部分：现行协议

## 1. V0 总协议

V0 是当前固定执行的实验协议族：RoboTwin 左臂单任务（一个任务独立训练一个模型）、`cen_arm_top_cam` 顶视角、DP 系列模型。V0 下属实验只改变数据侧条件（如显式图像噪声），**不改变模型与训练架构**。

### 1.1 模型代号

| 代号 | 模型 | 状态 |
|---|---|---|
| `defaultDP` | Diffusion Policy vanilla（`policy.type=diffusion`），无 EMIX batch | 现行 |
| `emixDP` | EMIX 混合训练 DP（`policy.type=emix_diffusion`） | 现行 |
| `emixPatchDP` | EMIX 拼贴（Collage）混合 DP | 预留代号，未排期（见 TODEV 条目 2） |
| `smolVLA` | smolVLA | 预留代号，未排期 |
| `emixSmolVLA` | EMIX + smolVLA | 预留代号，未排期 |

预留代号仅登记命名，不代表已立项；启用时必须先在本文件补全该模型的固定条件，再建立实例。

### 1.2 固定条件（V0 全部下属实验共用）

| 项目 | 固定值 |
|---|---|
| 机器人 / 场景 | RoboTwin 左臂任务，Piper ×2，piper_distance 0.6 m；右臂目标冻结为初始 qpos |
| 相机视角 | `cen_arm_top_cam`，仅 head_camera RGB；D435-W，fovy 60°，原渲染 320×240 |
| 相机外参 | position `(0,-0.10,1.50)`，forward `(0,0,-1)`，left `(-1,0,0)` |
| 相机/背景随机化 | random_head_camera_dis=0、random_table_height=0、random_light=false、random_background=false、cluttered_table=false |
| 渲染请求 | RT，32 spp，path depth 8，OIDN 请求固定为 `oidn`；实际初始化结果须看日志，不以请求值冒充成功 |
| 专家数据 | 每任务 500 条已接受轨迹；失败 seed 被跳过；17 FPS；500ep top baseline 原归档数据直接复用 |
| 数据位置 | `archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911/data/lerobot/<任务>/` |
| 图像存储 / 输入 | 归档视频 84×84 RGB（AV1/yuv420p），加载后 float `[0,1]`；噪声在解码/缩放后添加，磁盘数据不修改 |
| 状态 / 动作 | 原生 14 维 qpos（左 7 + 右 7），不把输出改成仅左 7 维；执行前右臂 suffix 冻结 |
| 图像编码器 | ResNet18，`ResNet18_Weights.IMAGENET1K_V1`，use_group_norm=false；像素进入编码器前执行 ImageNet 标准化 |
| 时序 | horizon=16，n_obs_steps=1，n_action_steps=8；采样时间间隔使用数据的 17 FPS |
| 训练 | 30,000 optimizer steps，batch=64，num_workers=16，shuffle=true，seed=0；save_freq=5000，log_freq=100 |
| 优化器 | Adam，lr=1e-4，betas=(0.95,0.999)，eps=1e-8，weight_decay=1e-6，grad_clip_norm=10 |
| LR scheduler | cosine，warmup=500 steps，沿用 Diffusion training preset |
| Diffusion 网络 | down_dims=(512,1024,2048)，kernel_size=5，n_groups=8，step embedding=128 |
| Diffusion 过程 | DDPM，100 train timesteps、100 inference timesteps，squaredcos_cap_v2，prediction_type=epsilon，clip_sample=true |
| 裁剪 | crop_shape=null（不启用裁剪）；保留 crop_is_random=true 默认值，但无裁剪尺寸时不生效 |
| 评估 | 每任务/条件 100 个专家有效场景，evaluation seed=0，从场景 seed 100000 起，使用最终 030000 checkpoint |
| 视频 | save_video=true，10 FPS；记录的是原渲染画面，含噪条件的噪声只加在策略输入，视频本身不是含噪输入证据 |
| 环境 | 预处理/训练=`emix`；采集/评估=`emix-twin`，Blackwell 自动路由 `emix-twin-5060` |

默认参数已在执行 YAML 中显式固化，完整实际策略配置随 checkpoint 保存。
评估接受 seed 集在不同 run 间不完全相同，同 evaluation seed 不等于场景集合严格配对；
每组保留 `records[].seed`，报告原始 100ep 结果，同时可报告共同 seed 集比较；
没有多训练 seed 的结论不扩展为统计显著性。

### 1.3 实例命名规则

运行实例统一命名：`<实验>-<条件>-<模型代号>-<任务集>-seed<训练seed>`，输出目录为 `outputs/run_<实例>/`。
实验与条件命名见下属实验小节；任务集 `s5` 为固定五任务子集（选择依据见实验 1），完整任务名为 `<任务>_left_cen_arm_top_cam`。
已发布实例的固定条件不得就地改写；任务集、模型、方差、数据或 seed 改变须创建新实例并使用新输出目录。

### 1.4 命名未覆盖的控制维度（完整性声明）

当前实例代号只编码了实验、噪声条件、模型、任务集和 seed。**以下控制维度同样影响实验结果，但尚未编入代号**；
在纳入命名前，每个实例的这些维度必须保持与本文档记录一致，变更任一维度都视同改变协议、须另立实例：

| 控制维度 | 当前固定值 | 当前控制方式 | 命名编码状态 |
|---|---|---|---|
| 物体摆放随机化 | 由任务代码决定（例：`place_can_basket` 随机选 basket_id、罐头种类、臂 tag；各任务 init 位姿范围写在任务代码/配置里），当前固定为采集所用 RoboTwin（emix-twin）代码版本的行为 | 手动修改任务代码/配置，以 RoboTwin commit 版本固定 | **未编码**；后续以任务代码 commit（或显式随机化档位）标识 |
| 相机视角 | `cen_arm_top_cam`（顶视角，外参见 1.2）；RoboTwin 已开发视角：`central_cam`、`left_oppo_cam`、`central_wide_cam`、`cen_arm_top_cam`、`cen_arm_side_cam`、`cen_arm_front_cam`、`cen_arm_near_side_cam`、`cen_arm_right_wide_cam`（见 `configs/left_task_registry.yaml`） | 任务全名后缀携带视角 | 任务名已含；协议代号未单列，跨视角实验须显式声明 |
| DP 输入图像尺寸 | 84×84（原渲染 320×240 降采样后送入策略） | 采集/转换参数 | **未编码**（当前所有实例一致） |
| EMIX 近邻筛选参数 | 见 1.5，当前全部为默认档 | `emix build` CLI / `DEFAULT_EMBODIED_MIXUP_CONFIG` | **未编码**；emixDP 臂隐含"索引默认档"，改参数须另建索引并另立实例 |

### 1.5 EMIX 控制参数（当前全部默认档）

索引构建（近邻筛选，"有效点"判据；CLI 默认值即当前使用值，索引已物化在归档 `data/emix_index/`）：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `xyz_eps` | 0.02 m | 近邻位置相似阈值 |
| `rpy_eps` | 0.26 rad（约 15°） | 近邻姿态相似阈值 |
| `action_delta_threshold` | 0.05 | 当前帧动作 L2 分歧下限（动作过滤：分歧不足不成对） |
| `future_horizon` | 8 帧 | 未来运动方向前瞻窗口 |
| `future_angle_thresh` | π/6 rad | 未来 xyz 方向角下限（未来动作分歧判据） |
| `min_future_disp` | 0.002 m | 估计方向角所需的最小未来位移 |
| `exclude_tail_seconds` | 0.1 s（17 fps 约 2 帧） | 剔除每集末尾回 home 段，避免与集首错误配对 |
| gripper 动作过滤 | 默认开启 | 夹爪切换不一致的诊断性过滤 |
| `cluster_selection_epsilon` | 0.05 | HDBSCAN 后处理合并阈值（标准化 6D 空间） |
| `hdbscan_min_cluster_size_ratio` | 0.01 | HDBSCAN min_cluster_size（占总对数比例） |
| `hdbscan_min_samples` | 3 | HDBSCAN min_samples |

混合训练（执行 YAML `emix:` 段，随 checkpoint 保存）：`alpha_mode=beta`、`beta=2`、`loss_weight=1.0`、
`state_mode=anchor`、调度 logarithmic `start_batch_prob=0.6 → end_batch_prob=0.0`。

## 2. 实验 1：v0-singleTask（单任务基线，已完成归档）

- **含义**：V0 协议的单任务情况；每任务独立训练一个模型，模型臂为 `defaultDP` / `emixDP`。
- **状态**：已完成、已归档。来源：[`DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911`](../archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911/SUMMARY.md)。父仓库 commit `3d3c51e`，RoboTwin commit `61798c9`；历史 flow `logs/flow-20260907-222045-cuda-0-1`，2026-09-07 启动、09-10 完成、09-11 归档。

历史十任务结果（成功数已核对归档 DP/EMIX 的 `eval/<完整任务名>_dp_<kind>/*/metrics.json`；任务筛选指标为历史 EMIX 成功率减去 DP 成功率的百分点差，原始 100ep 口径）：

| 任务 | DP SR | EMIX SR | EMIX−DP（百分点） | v0-noise 子集 |
|---|---:|---:|---:|---|
| place_can_basket | 49% | 64% | +15 | good-1 |
| place_dual_shoes | 33% | 44% | +11 | good-2 |
| place_bread_skillet | 87% | 92% | +5 | — |
| place_cans_plasticbox | 20% | 25% | +5 | — |
| place_burger_fries | 36% | 37% | +1 | — |
| blocks_ranking_size | 3% | 3% | +0 | — |
| stack_bowls_two | 88% | 88% | +0 | — |
| blocks_ranking_rgb | 28% | 26% | -2 | poor-3 |
| stack_blocks_two | 49% | 45% | -4 | poor-2 |
| place_bread_basket | 45% | 37% | -8 | poor-1 |
| 全部十任务 | 43.8% | 46.1% | +2.3 | — |

`s5` 任务集已由用户定稿：EMIX 提升最高的两个任务（good-1/2）和提升最低（退化最大）的三个任务（poor-1/2/3）。
这是基于历史结果的任务筛选，不声称差值具有统计显著性。五任务结果仅报告为 `s5`，不能替代十任务总体结论；
历史 s5 的 DP 为 204/500=40.8%，EMIX 为 216/500=43.2%，差 +2.4 点。不得根据本轮结果替换任务，其他任务留待后续扩展。

## 3. 实验 2：v0-singleTask-noise（当前实验：显式图像高斯噪声）

### 3.1 定义与命名

- 在实验 1 基础上**仅增加数据侧显式高斯噪声**；模型与训练架构不变，模型代号仍为 `defaultDP` / `emixDP`。
- 噪声定义：`ε[c,h,w] ~ N(0, variance)`，所有像素/通道独立，`x' = clip(x+ε,0,1)`。无整图缩放；
  不改状态、动作标签或相机位姿。注入位置为解码/缩放为 `[0,1]` 之后、策略标准化之前（EMIX 混合之后）；
  每次观测重新采样，不物化噪声视频。方差描述截断前的 Gaussian，黑白边界附近截断后实测方差可能较小。
- 训练噪声由训练 seed 和绝对 optimizer step 决定（独立于模型/加载器全局 RNG）；评估噪声 seed=0，
  结合场景 seed 与 take_action_cnt 使用独立 RNG，不扰动模型采样/专家规划 RNG。
- **档位**（2026-09-28 用户确认两档均完全足够；同日最终决定**本轮（第一个噪声轮次）沿用已执行的 var0025，不强制切换**）：

| 档位后缀 | 方差 | σ（[0,1]） | σ（255 域） | 定位 |
|---|---:|---:|---:|---|
| `noiseVar0025` | 0.0025 | 0.05 | 12.75 | **本轮执行档位**（defaultDP 已训练/评估，emixDP 臂同档补齐） |
| `noiseVar000625` | 0.000625 | 0.025 | 6.4 | 已确认有效；留待后续轮次使用（另立实例，不复用本轮 checkpoint） |

- **侧别后缀**：`Tr` = 仅训练含噪（评估原始图像）；`TrEv` = 训练与评估均含噪。
  完整条件名示例：`v0-singleTask-noiseVar0025TrEv` 表示本轮档位双侧。
- **训练复用规则**：同一档位、同一模型臂的 `Tr` 与 `TrEv` 训练侧完全相同，共用同一组 checkpoint，
  只改评估注入；两组 checkpoint 路径必须相同，eval_root 必须不同。
- **旧代号映射**（2026-09-28 命名规范化前的执行文件仍使用旧名）：`Tr` ↔ `tn-ec`，`TrEv` ↔ `tn-en`；
  配置与目录名中的 `run_v0_noise_*`、`run_emix_noise_*` 为旧命名遗留，映射如下表。
- 比较解释约束：要严谨隔离噪声因果，需同代码 clean 对照（`tc-ec`，即无噪声重训），本轮未自动扩展；
  归档历史 EMIX/DP 为旧代码快照结果，只能作趋势参考，历史实际噪声分布记为 **unknown**。

### 3.2 实例登记与状态

| 实例 | 模型臂 | 状态 | 旧代号 / 目录 | 结果（s5 任务序：stack_blocks / bread_basket / rgb / can_basket / dual_shoes） |
|---|---|---|---|---|
| v0-singleTask-noiseVar0025Tr-defaultDP-s5-seed0 | defaultDP | ✅ 完成 | v0-noise-tn-ec；`outputs/run_v0-noise-tn-ec-s5-var0025-seed0/` | 40 / 44 / 21 / 65 / 35（均值 41.0%） |
| v0-singleTask-noiseVar0025TrEv-defaultDP-s5-seed0 | defaultDP | ✅ 完成 | v0-noise-tn-en；`outputs/run_v0-noise-tn-en-s5-var0025-seed0/` | 43 / 37 / 30 / 65 / 39（均值 42.8%） |
| v0-singleTask-noiseVar0025Tr-emixDP-s5-seed0 | emixDP | 📋 配置就绪待启动 | `configs/scripts/run_emix_noise_tn_ec.yaml` | — |
| v0-singleTask-noiseVar0025TrEv-emixDP-s5-seed0 | emixDP | 📋 配置就绪待启动 | `configs/scripts/run_emix_noise_tn_en.yaml`，复用上一行 checkpoint | — |

本轮四个臂固定 s5 五任务（沿用实验 1 筛选，不替换）、var0025、seed 0；`noiseVar000625` 臂留待后续轮次另立实例。
emixDP 臂的 vanilla 对照直接引用同代码、同 seed、同数据的 defaultDP 结果，不重训；
缺同代码 clean 对照（无噪声重训的 defaultDP / emixDP）是已知边界，跨快照比较不得严格归因噪声。

### 3.3 对比方案（本轮四臂齐备后执行）

**第一级（内部比较，同代码同 seed 同数据）**：在 `v0-singleTask-noise` 内部，逐任务与 s5 均值比较
emixDP 与 defaultDP 的成功率，差值记为 **EMIX−DP 提升（百分点）**，分别在 `Tr`（训噪评净）与
`TrEv`（训噪评噪）两种条件下各算一次。回答：含噪训练下 emixDP 是否比 defaultDP 效果更好。

**第二级（跨实验比较）**：将本轮（v0-singleTask-noise，var0025）的 EMIX−DP 提升与
`v0-singleTask`（实验 1，历史十任务顶部相机版本，clean）的提升对比，回答：
**含噪条件下 EMIX 相对 DP 的提升是否比干净条件下更显著**。
参照基准为实验 1 历史结果（s5 子集：逐任务 +15 / +11 / −2 / −4 / −8，均值 +2.4 点）。

**解释约束（必须随结果一并报告）**：
- 实验 1 历史结果为旧代码快照（父仓库 `3d3c51e`、RoboTwin `61798c9`），与本轮存在软件/环境漂移，
  第二级比较是**趋势级参照**，不能严格归因于噪声；要严格归因需补同代码 clean 对照臂（已知边界）。
- 所有比较为单训练 seed、100ep 口径，只做描述性对比（逐任务差值、s5 均值），
  无多 seed 重复不得声称统计显著性；可另报共同 evaluation seed 集上的配对比较。
- 历史实际噪声分布为 unknown，本轮噪声是显式高斯可控注入，不声称复现历史渲染噪声。

### 3.4 可执行配置与运行顺序

当前执行 YAML（旧命名遗留，映射见 3.2）：defaultDP 臂 `configs/scripts/run_v0_noise_tn_{ec,en}.yaml`（var0025）；
emixDP 臂 `configs/scripts/run_emix_noise_tn_{ec,en}.yaml`（已规范命名为 v0-singleTask-noiseVar0025Tr/TrEv-emixDP，variance 0.0025）。只执行 `train-eval`，不重新采集、转换或构建 EMIX 索引。
顺序执行器 [`scripts/run_noise_experiment.py`](../scripts/run_noise_experiment.py) 按 Tr → TrEv 自动执行，
每 30 秒更新 3.4 节台账和 status.json；`--phase summary` 可生成实例结果摘要；
训练快照、checkpoint 训练元数据、评估 metrics 均记录协议标识和噪声参数。

```bash
# 在 emix 环境执行。第一条成功后才执行第二条；不要同时启动两条。
python scripts/run.py --config configs/scripts/run_v0_noise_tn_ec.yaml --phase train-eval
python scripts/run.py --config configs/scripts/run_v0_noise_tn_en.yaml --phase train-eval
```

固定任务 GPU 分配与归档保持一致：CUDA 0 为 blocks/rgb、bread_basket、stack_blocks_two；CUDA 1 为 can_basket、dual_shoes。
2026-09-28 在训练环境实测 CUDA 0=4070、CUDA 1=5060；nvidia-smi 的编号相反。设备枚举可能改变，
启动时以 torch CUDA 设备名称与自动环境路由为准，记录实际 GPU；两个评估条件的同一任务保持同一实际 GPU/环境。
`pipeline.rt_denoiser=oidn` 显式覆盖继承 shell 的旧 OIDN-off 环境变量，避免隐含第二变量。

### 3.5 自动运行状态

<!-- v0-noise-run-status:start -->
### 自动运行状态

- 状态：`complete`；条件：`tn_ec`（exit 0，2026-09-28T17:34+08:00）→ `tn_en`（exit 0，2026-09-29T05:16+08:00）。
- 更新：2026-09-29T05:16:07+08:00；开始：2026-09-28T03:01:25+08:00。
- 运行目录：`outputs/run_v0-noise-suite-20260928-030125`；监督进程 PID 3757142（已退出）。
- 状态与完整结果：`outputs/run_v0-noise-suite-20260928-030125/status.json`；实例摘要 `outputs/run_v0-noise-tn-{ec,en}-s5-var0025-seed0/summary/RESULTS.md`。
- 按用户定稿顺序自动执行：单侧训练/评估 → 双侧复用训练/评估。

| 任务 | 训练完成 | 单侧 (Tr) DP SR | 双侧 (TrEv) DP SR |
|---|---|---:|---:|
| stack_blocks_two_left_cen_arm_top_cam | 是 | 40/100 (40.0%) | 43/100 (43.0%) |
| place_bread_basket_left_cen_arm_top_cam | 是 | 44/100 (44.0%) | 37/100 (37.0%) |
| blocks_ranking_rgb_left_cen_arm_top_cam | 是 | 21/100 (21.0%) | 30/100 (30.0%) |
| place_can_basket_left_cen_arm_top_cam | 是 | 65/100 (65.0%) | 65/100 (65.0%) |
| place_dual_shoes_left_cen_arm_top_cam | 是 | 35/100 (35.0%) | 39/100 (39.0%) |
| **s5 均值** | — | **41.0%** | **42.8%** |
<!-- v0-noise-run-status:end -->

<!-- emix-noise-run-status:start -->
### 自动运行状态

- 状态：`running`；当前条件：`tn_en`。
- 更新：2026-09-29T23:18:15+08:00；开始：2026-09-29T06:28:28+08:00。
- 运行目录：`outputs/run_emix-noise-suite-20260929-062827`；监督进程 PID：2242451。
- 状态与完整结果：`outputs/run_emix-noise-suite-20260929-062827/status.json`。
- 按用户定稿顺序自动执行：单侧训练/评估 → 双侧复用训练/评估。

| 任务 | 训练完成 | 单侧 EMIX SR | 双侧 EMIX SR |
|---|---|---:|---:|
| stack_blocks_two_left_cen_arm_top_cam | 是 | 37/100 (37.0%) | 待完成 |
| place_bread_basket_left_cen_arm_top_cam | 是 | 39/100 (39.0%) | 待完成 |
| blocks_ranking_rgb_left_cen_arm_top_cam | 是 | 31/100 (31.0%) | 待完成 |
| place_can_basket_left_cen_arm_top_cam | 是 | 58/100 (58.0%) | 待完成 |
| place_dual_shoes_left_cen_arm_top_cam | 是 | 28/100 (28.0%) | 待完成 |
<!-- emix-noise-run-status:end -->

## 4. 后续实施计划（现行排期）

按优先级排列；启动任一批次前须先在本文件登记实例并确认固定条件。

1. **完成本轮四臂（实验 2，var0025）**：defaultDP 臂 TrEv 正在运行；待其完成、GPU 空闲后启动
   emixDP 臂（`--status-prefix emix-noise`，Tr → TrEv，配置已就绪）。
2. **本轮分析**：按 3.3 两级对比方案出报告——第一级含噪内部比较（emixDP vs defaultDP，Tr/TrEv），
   第二级与实验 1 历史干净提升的趋势级对比；同时报告噪声对 defaultDP 自身的 Tr vs TrEv 退化幅度。
3. **严格归因对照（待排期）**：补同代码 clean 对照臂（无噪声重训的 defaultDP / emixDP，`tc` 条件），
   消除与历史 v0-singleTask 比较时的代码/环境漂移混淆。
4. **var000625 轮次（待排期）**：已确认有效的低档位，另立实例跑四臂，与 var0025 轮次对比噪声强度效应。
5. **命名扩展（随下一次实例建立时实施）**：把 1.4 所列未编码维度纳入实例命名——物体摆放随机化版本
   （任务代码 commit 或显式档位）、相机视角、DP 输入图像尺寸、EMIX 索引筛选参数档。
6. **方法演进（TODEV 优先级）**：`emixPatchDP` 拼贴混合（P1，TODEV 条目 2）→ 分阶段增强（P2，条目 3，
   依赖拼贴）→ push/pull/open/close 任务扩展（P3，条目 4）；`smolVLA` / `emixSmolVLA` 代号已预留，
   启用前须补全各自固定条件。

---

# 第二部分：历史更新记录

（按日期升序，越早的越靠前；只记录已发生的事实与已批准的决策，不记录计划。）

## 2026-09-22

- TODEV 条目 1 开发：OIDN（Open Image Denoise）实现 `ROBOTWIN_RT_DENOISER` 环境变量开关（`third_party/RoboTwin` `2de39e7`）。
- 实测发现：32spp、路径深度 8 下 svulkan2 RT 渲染在平坦区域完全收敛（禁用 OIDN 后平坦区 std=0.00，
  两次独立渲染逐像素一致），关闭 OIDN 不会产生可见噪点；JPEG(q95) 编码 + 84×84 降采样进一步抹掉残余颗粒。
  emix-twin 环境的 OIDN 插件在本机 4070 上一直初始化失败（"OIDN Error: invalid handle/unsupported device type"），
  历史批次中经 4070 采集的任务实为无降噪渲染。详见 [TODEV](TODEV.md)。

## 2026-09-28

- 登记 v0；从归档 metrics 提出任务子集；完成历史 OIDN 审计：

| 问题 | 已核实证据 | 判定 |
|---|---|---|
| 历史是否主动添加高斯噪声？ | 历史训练配置没有噪声项；commit `3d3c51e` 的评估只 resize、除以 255 | 未发现主动合成噪声 |
| 采集/评估是否请求 OIDN？ | RoboTwin `61798c9` 的 `setup_scene` 共用 RT：32 spp、path depth 8、denoiser `oidn` | 两侧均请求 OIDN |
| 环境是否分侧修复？ | 历史 `scripts/oidn_env.py` 对采集和评估共用按 GPU 架构的环境路由，Blackwell 使用 `emix-twin-5060` | 不是统一的"训练失效、评估正常"配置 |
| 当时 OIDN 两侧实际是否失败？ | 两份历史 flow 日志未检出 `OIDN`/`invalid handle`/`unsupported device type`；采集在 flow 开始前已完成/复用 | 日志不足以判定实际初始化状态 |
| OIDN 失败是否证明图像有可见噪声？ | 2026-09-22 探测：32 spp 平坦区收敛，开关可能逐像素相同，编码/降采样进一步弱化颗粒 | 不能由错误/开关直接推出可见噪声 |

  历史实际噪声分布记为 **unknown**，不得写成已证实双侧或单侧噪声；高斯噪声是可控替代实验，
  不能称为统计上精确复现路径追踪噪声。详见 [历史审计](analysis/20260911-top500-emix-regression-audit.md)。
- 实现独立训练/评估高斯噪声注入与两种条件的 checkpoint 复用（`embodied_mixup/policies/utils/image_noise.py`；
  `image_noise.enabled` 默认 false，`variance` 默认 0.0025，variance=0 精确关闭）；
  首档定为 var0025。开发使用方式见 [SOP-03](sop/SOP-03-model-training.md#23-高斯噪声训练与可控评估)。
- 用归档 bread_basket 真实数据验证训练链路（batch=2、2 steps，`outputs/gaussian_noise_smoke/baseline/`）；
  同一冒烟 checkpoint 分别接收原始与含噪归档帧，动作输出均为有限 14 维向量
  （`outputs/gaussian_noise_smoke/evaluation_input_smoke.json`、`input_comparison.png`）；仅冒烟，不属于正式结果。
- 用户定稿：任务筛选按历史 EMIX−DP 百分点（最高 2 个 + 最低 3 个，固定 s5）；执行顺序单侧先、双侧后；
  模型 DP vanilla 单任务、30k steps、100ep/条件、seed 0；无噪声重训对照（tc-ec）本轮不加；
  两个评估条件共用任务 checkpoint，只训练五个模型。
- 8 项噪声/配置/评估边界测试通过（`tests/test_image_noise.py`）；两个 GPU 的正式运行前提检查通过。
- 启动 var0025 正式顺序实验：`outputs/run_v0-noise-suite-20260928-030125`，监督进程 PID 3757142。
  Tr（旧称 tn-ec）于 17:34 完成，defaultDP s5 结果 40/44/21/65/35（均值 41.0%）；TrEv 评估进行中。
- 生成噪声可视化对比图 `outputs/analysis/noise_visual_compare.png`（3 个 s5 任务真实归档帧 ×
  var 0.000625/0.0025/0.01 三档，使用与训练完全相同的加噪代码路径；脚本 `scripts/temp/noise_visual_compare.py`）。
- **用户确认 var000625 与 var0025 两档噪声均完全足够，主要使用 var000625（主档）**；
  本文档重构为「V0 现行协议 + 历史更新记录」结构，实验 2 规范命名为 `v0-singleTask-noise`；
  模型代号登记 defaultDP / emixDP（现行），emixPatchDP / smolVLA / emixSmolVLA（预留）；
  emix-noise 批次配置由 var0025 切换到主档 var000625（实例 v0-singleTask-noiseVar000625Tr/TrEv-emixDP），待启动；
  defaultDP 主档两实例配置待建。
- **用户最终决定：本轮（第一个噪声轮次）沿用已执行的 var0025，不强制切换到 var000625**；
  本轮固定为 s5 五任务 × var0025 × seed 0 的四个臂：defaultDP/emixDP × Tr/TrEv
  （`TrEv` 复用同臂 `Tr` checkpoint，只改评估注入；emixDP 的 defaultDP 对照引用同代码结果）。
  实验 2 增补 3.3 对比方案：第一级内部比较（含噪下 emixDP vs defaultDP 提升，Tr 与 TrEv 各算）；
  第二级跨实验比较（本轮提升 vs 实验 1 v0-singleTask 历史干净提升，趋势级参照，受旧代码漂移约束）。
  emixDP 臂配置已回切 var0025 并按新命名规范为 v0-singleTask-noiseVar0025Tr/TrEv-emixDP-s5-seed0；
  var000625 臂留待后续轮次另立实例。
- 增补实验细节与控制维度登记：V0 总协议新增 1.4「命名未覆盖的控制维度」（物体摆放随机化=任务代码
  版本控制、RoboTwin 已开发 8 个相机视角、DP 输入图像尺寸 84×84、EMIX 近邻筛选参数）与
  1.5「EMIX 控制参数」（索引构建近邻/动作分歧/HDBSCAN 各参数当前默认档及混合训练超参）；
  新增第 4 节「后续实施计划」（完成本轮四臂 → 两级对比分析 → clean 对照 → var000625 轮次 →
  命名扩展 → TODEV 方法演进路线）。
- 用户批准按 v0-noise 协议继续推进 var0025 本轮完整实验：挂载自动衔接启动器
  `scripts/temp/chain_launch_emix_noise.sh`（setsid 脱离终端），等待 defaultDP 臂监督进程退出并核验
  全部条件 exit 0、TrEv 五任务 metrics 齐备后，自动启动 emixDP 臂
  （`--status-prefix emix-noise`，Tr → TrEv）；核验不通过则不启动并记录原因。

## 2026-09-29

- defaultDP 臂（var0025）全部完成：Tr 于 09-28 17:34 收官（s5 均值 41.0%），TrEv 于 05:16 收官
  （逐任务 43/37/30/65/39，均值 42.8%；两个条件共用同一组含噪训练 checkpoint）。
  含噪评估相对干净评估未造成平均退化（42.8% vs 41.0%，逐任务 +3/−7/+9/0/+4），
  说明 var0025 训练注入下 defaultDP 对同分布含噪输入具有鲁棒性；单 seed、100ep 描述性结果。
- 衔接启动器首次核验因状态字符串误判（`completed` vs 实际 `complete`）未自动启动，
  已修正 `scripts/temp/chain_launch_emix_noise.sh` 后于 05:26 人工重挂，核验通过，
  emixDP 臂（v0-singleTask-noiseVar0025Tr/TrEv-emixDP-s5-seed0）正式开训，
  监督进程 PID 2163765，运行目录 `outputs/run_emix-noise-suite-20260929-052651`。
- emixDP 臂首次开训失败（05:27，两任务同因）：归档 EMIX 索引声明的 source root 是索引构建时的
  `data/lerobot/<task>`，与训练配置指向的归档数据路径严格不匹配（`EmixIndexReader` root 校验）。
  处置：不改动归档原件，将 s5 五任务索引复制到 `data/emix_index_archive_root/<task>/` 并把
  `public.source_datasets[0].root` 重定向到归档数据绝对路径；内容指纹校验五任务全部通过；
  两份 emixDP 执行 YAML 的 `index_root` 同步指向该目录。05:29 重启监督进程
  （PID 见 `outputs/run_emix-noise-suite-20260929-062827/status.json`），EMIX 训练正常开跑
  （双卡满载，loss 正常）。历史 EMIX 训练配置使用 `data/lerobot/` 根即因此通过校验。
- 深挖 08 月大提升批（skillet 32→95、bowls 31→61、can_basket 44→66）的布局与设置（证据：RoboTwin
  提交历史、left_task_base.py、left_task_manifests.py、08-18/08-20 归档 train 配置、history_20260911 首帧对照图）：
  ① 渲染时间线——08-13 前 OIDN 报错失效（图像损坏，0813 归档地板）；08-13（854651f）~08-21（2198f37）
  OIDN 显式禁用（32spp RT 直接出图，TODEV 实测平坦区收敛无可见噪点）；08-21 起重申请求 oidn；
  大提升批恰好落在"OIDN 禁用"窗口，属"干净无降噪"而非"含噪"图像。
  ② 布局——旧批：左臂原生安装位 x=-0.30、双臂同屏（右臂未隐藏）、central_wide 斜视相机
  (0.268,-0.45,1.35)、物体散布范围大（如积木拾取带 x∈[-0.45,-0.16]、yaw≤0.75）；新批（cen_arm）：
  左臂补偿到 x=0、右臂移至 x=5 出画、顶视相机 (0,-0.10,1.50) 垂直向下、共享居中包络
  x∈[-0.20,0.20]×y∈[-0.15,0.15]×yaw≤0.5。
  ③ EMIX 参数——08-20 批 scheduler 0.8→0（当前 0.6→0）、无索引留存无法核验筛选阈值；200ep 批
  rpy_eps=0.01 rad（0.57°，极严）；09-04 起放宽到 0.26 rad 并加 exclude_tail 0.1s、09-06 加 50 万对 cap。
  ④ 数据规模——08-18 批 200ep；08-20 批声称约 500ep 但归档无 data/index 无法核验；当前 500ep 仅已接受轨迹。
  结论：大提升批与当前批在相机视角、工作区几何、臂可见性、物体分布范围、EMIX 调度/筛选参数上
  全部不同，不能将历史提升归因于任何单一变量；与噪声的关系仅是时间线巧合（同属 OIDN 禁用窗口），
  该窗口图像经实测为干净收敛渲染。
