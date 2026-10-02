# 实验协议真相文档（_exp.md）

更新：2026-10-01。**本文档是当前项目实验的唯一认证来源和实施记录**：实验命名、固定条件、运行顺序和状态的唯一权威入口。YAML 是协议的可执行表示；TODEV 是开发清单，SOP 是操作说明，归档 SUMMARY 是历史证据。

文档结构：前半部分为**版本边界、已归档 V0 协议和当前 V1 实现**；后半部分为**历史更新记录**（按日期升序排列，越早的越靠前）。

发生冲突先核对实际配置/快照/metrics，再更新本文档，不能把计划写成已完成结果。

---

# 第一部分：版本协议与归档结果

## 0. 当前版本边界

| 名称 | 任务实现和数据边界 | 状态 |
|---|---|---|
| `v0` / `v0-noise` | 归档顶视单任务实现与既有 500ep 数据；评估使用当时的 `legacy` 摆放逻辑。下文 §2/§3 的所有 SR 均属于这一实现与数据。 | 已完成并封存 |
| `v1` | 2026-10-01 更新后的当前任务实现：基座相对的逐任务摆放范围、footprint 间距和经 pilot 终选的矩形/椭圆采样；默认 `ROBOTWIN_CENTERED_PROTOCOL=v0` 选择空间参数表。 | 当前代码版本；尚无按 V1 采集和训推的正式结果 |

`v0` 出现在 `CENTERED_V0_OVERRIDES` 和 `ROBOTWIN_CENTERED_PROTOCOL` 中时，是**空间参数预设的历史名字**；它不能替代任务实现版本。当前代码用 `TASK_IMPLEMENTATION_VERSION=v1` 标识任务实现。今后的新实验命名以 `v1-...` 开头，配置中显式写入任务实现版本、空间参数预设、任务集与数据版本；V1 **必须重新采集十任务各 500 条轨迹**，不得复用 V0 数据。V0 的 YAML、JSON、日志等文本结果保持原名；旧 checkpoint 删除，旧二进制数据/视频逐目录保留约 10% 用于回查。

V1 数据采集配置为 `configs/scripts/run_v1_cen_arm_top_cam_500ep.yaml`：十任务、顶视角、新数据与独立输出目录；其中 `task_implementation_version=v1`、`spatial_preset=v0` 显式分开，`collect_instance_ids=true` 同步记录原始 actor ID、物体名表，并在 84×84 转换时生成压缩归属图。十任务各两帧的真实仿真采集冒烟已逐一验证 RGB/ID 帧数和任务对象名表；84×84 归属图转换另通过合成帧测试。

新的 `pre_vis` 事前可视化要求适用于此后启动的训推；此前已经完成的 `v0-noise` 四臂不追溯标为符合新要求。

## 1. V0 归档总协议

V0 是已完成的实验协议族：RoboTwin 左臂单任务（一个任务独立训练一个模型）、`cen_arm_top_cam` 顶视角、DP 系列模型。V0 下属实验只改变数据侧条件（如显式图像噪声），**不改变模型与训练架构**。

### 1.1 模型代号

| 代号 | 模型 | 状态 |
|---|---|---|
| `defaultDP` | Diffusion Policy vanilla（`policy.type=diffusion`），无 EMIX batch | 现行 |
| `emixDP` | 旧式 EMIX 线性叠加，现统一称 `Emix-overlay`（`policy.type=emix_diffusion`、`emix.mixup.method=overlay`） | V0 已完成；V1 对照排期 |
| `emixPatchDP` | `Emix-collage`：从第二图中选择一个可见任务物体，按归属图贴入第一图；双分支 `loss1+loss2`，不除以 2 | V1 已立项，待训推 |
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

### 1.2.1 训推前可视化（2026-10-01 起新增）

V0 每个实际执行训练或评估的任务、模型臂，默认必须先执行 `pre_vis`，即使数据集是旧数据或复用数据。编排脚本全流程在 `vis` 之后、`train-eval` 之前执行该阶段；单独运行 `--phase train-eval` 时仍逐任务自动补齐。只有调用者显式传入 `--skip-pre-vis` 才跳过。单独补图用 `--phase pre_vis`。

每个训练根目录和评估根目录下各保存一套 `pre_vis/`：`input_samples.png`（固定随机种子抽样的实际训练图像，经当前混合/噪声处理、策略标准化前）、`intersections.png`（摆放/交点图）、`pick_place.png`（抓放轨迹点）、`first_frame_range.png`（跨 episode 首帧物体摆放范围热力图），并保留交互 HTML 和简要参数记录。评估运行时另保存 `actual_eval_input.png`，取真实仿真首个送入策略的图像。四类图任一失败即阻止后续训推，不能以别的任务或旧实验的图片代替。

图像直接放在对应任务与模型的 checkpoint 所在训练目录 `pre_vis/`，评估侧放在同一任务与模型的评估根目录 `pre_vis/`。训练器只允许先存在该 `pre_vis` 目录，其余已有训练产物仍拒绝覆盖。旧实验不追溯声称满足这一新规；本条适用于此后启动的编排训推。首帧热力图属于 RGB 差分估计，不能解读为精确语义物体标注。

### 1.3 实例命名规则

运行实例统一命名：`<实验>-<条件>-<模型代号>-<任务集>-seed<训练seed>`，输出目录为 `outputs/run_<实例>/`。
实验与条件命名见下属实验小节；任务集 `s5` 为固定五任务子集（选择依据见实验 1），完整任务名为 `<任务>_left_cen_arm_top_cam`。
已发布实例的固定条件不得就地改写；任务集、模型、方差、数据或 seed 改变须创建新实例并使用新输出目录。

### 1.4 命名未覆盖的控制维度（完整性声明）

已归档 V0 实例代号只编码了实验、噪声条件、模型、任务集和 seed。**以下控制维度同样影响实验结果，但尚未编入代号**；
在纳入命名前，每个实例的这些维度必须保持与本文档记录一致，变更任一维度都视同改变协议、须另立实例：

| 控制维度 | 当前固定值 | 当前控制方式 | 命名编码状态 |
|---|---|---|---|
| 物体摆放随机化 | **V0 参数**（§1.6，逐任务定制、x/y 各自对称；历史参数保留在 `CENTERED_WIDE_WORKSPACE` / `CENTERED_FAMILY_OVERRIDES` / 基线 `MANIFESTS` 作对照，不删除）；V0 之前由任务代码决定（例：`place_can_basket` 随机选 basket_id、罐头种类、臂 tag） | 任务代码版本 + `ROBOTWIN_CENTERED_PROTOCOL` 开关（`legacy`/`v0`）；当前运行中的实例在 TrEv 评估完成前保持 `legacy` | **未编码**；以任务代码版本 + 协议开关标识 |
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

### 1.6 V0 摆放随机化范围（cen_arm_top_cam 十任务，2026-09-30 立项）

本节标题的 V0 是 `ROBOTWIN_CENTERED_PROTOCOL=v0` 的**空间预设名**，终版范围属于现在的 V1 任务实现；它不属于已完成的 `v0-noise` 训练/评估数据分布。下方旧 pilot 日期和试验名保留原状。

> **坐标语义变更（2026-09-30 用户决断）**：V0 参数改为**基座相对描述**——x 关于臂基座 x 对称
> （居中批基座 x=0，数值与世界坐标相同），y 为**基座前方距离带**（基座在世界 (0,-0.45)，
> 桌子只存在于基座前方，y 无法基座对称；绝对 y = -0.45 + y_rel）。这正是历史批次的语义
> （旧批摆放带关于臂位置对称，如 blocks x(-0.45,-0.16) 关于 x=-0.30）。变更动机与几何分析
> 见 2026-09-30 历史记录；下方"表心对称轮"参数表保留作为对照记录，现行轮为 **base-1（基座相对）**。

**动机**：历史批次（central_wide 双臂斜视布局）的物体摆放绝对尺度显著大于当前 cen_arm_top_cam 居中批次
（当前共享包络仅 x∈[-0.20,0.20]×y∈[-0.15,0.15]）。用户决断（2026-09-29，TODEV 优先 TODO）：
逐任务扩大随机化范围、x 和 y 各自对称、不得超出桌面/相机画幅，并以 pilot 成功率把关。

**实现**（`third_party/RoboTwin/envs/left/left_task_manifests.py`）：
- 新注册表 `CENTERED_V0_OVERRIDES`：逐任务、逐 actor 的 `XYYawRange`，x/y 均关于 0 对称；
- 历史参数全部保留（`CENTERED_WIDE_WORKSPACE`、`CENTERED_FAMILY_OVERRIDES`、基线 `MANIFESTS`），作 legacy 回退与对照，不删除；
- 协议开关 `CENTERED_ACTIVE_PROTOCOL`，由环境变量 `ROBOTWIN_CENTERED_PROTOCOL` 控制（`legacy`/`v0`）。
  V0-noise TrEv 收官前默认是 `legacy`；2026-10-01 已翻转为 `v0` 空间预设。旧 V0 结果仍归属 `legacy`；
- 硬边界：桌面 x∈[-0.60,0.60]、y∈[-0.35,0.35]；顶视画幅 y≤+0.34、x≤+0.49（`CEN_ARM_TOP_CAM` 注释）；
  各家族 impl 注释中的实测可达/IK 证据（如 stack_bowls_two 的 place 目标 x>0.12 或 y>0.05 零成功）。

**验收规则（用户规定）**：每任务 100-seed pilot（`scripts/temp/pilot_v0_ranges.py`，
结果落盘 `results/pilot/v0_ranges/<task>.pilot.jsonl`），成功率 ≥0.20 通过；<0.20 则收敛范围重测；
明显高于 0.20 可尝试更激进范围。逐任务单独调参，不共用统一范围。

**表心对称轮最终参数表**（2026-09-30 上午定稿，全部 10 任务完成 pilot 验收；该轮语义已被上方基座相对决断取代，本表保留作对照；证据归档 `results/pilot/v0_ranges/*.d*.jsonl`）：

| 任务 | actor | x | y | yaw(rad) | pilot SR | legacy 采集基线 |
|---|---|---|---|---|---|---|
| stack_blocks_two | block1/2 | ±0.30 | ±0.22 | 0~0.60 | 0.35（四档加宽 0.35→0.38→0.42→0.35，进入平台期） | 0.61 |
| blocks_ranking_rgb | block1/2/3 | ±0.25 | ±0.20 | 0~0.75 | 0.22（贴线=max 加宽） | 0.41 |
| blocks_ranking_size | block1/2/3 | ±0.25 | ±0.20 | 0~0.75 | 0.21（贴线） | 0.38 |
| place_bread_skillet | bread/skillet | ±0.25 | ±0.18 | 0~0.50 | 0.41（加宽且不低于基线，最佳案例） | 0.44 |
| place_bread_basket | breadbasket/bread0/bread1 | ±0.22 | ±0.15 | 0~0.50 | 0.20（y±0.20 时 0.18 不过线，收 y） | 0.27 |
| place_can_basket | basket/can | ±0.22 | ±0.15 | 0（锁） | 0.22 | 0.24 |
| place_burger_fries | tray | ±0.18 | ±0.12 | 0（锁） | 0.21（结构性天花板家族，比 legacy 窄才能上线） | 0.17 |
| place_burger_fries | hamburg/frenchfries | ±0.18 | ±0.12 | 0~0.20 | 同上 | 同上 |
| place_cans_plasticbox | plasticbox | ±0.12 | ±0.08 | -0.60~0.60 | 0.19（≈0.2；可达窗 0.35-0.5m 约束，draft 轨迹 0.09→0.13→0.19） | 0.48 |
| place_cans_plasticbox | can1/can2 | ±0.15 | ±0.10 | 0 | 同上 | 同上 |
| place_dual_shoes | shoe_box | ±0.20 | -0.15~-0.02 | 0（锁） | 0.24（x±0.25 时 0.17，UnStableError 约束，无加宽空间） | 0.20 |
| stack_bowls_two | bowl1/2 | **例外**：历史非对称带 x(-0.20,0.05)×y(-0.15,0.00) | 0 | 例外保留（见下） | 0.10 |

**终版参数表（基座相对 + 形状终选，2026-10-01 定稿；y 为基座前方距离，绝对 y=-0.45+y_rel）**：

| 任务 | actor | x（±，关于基座） | y_rel（前方距离带） | yaw(rad) | 形状 | 终版 pilot SR | legacy 基线 |
|---|---|---|---|---|---|---|---|
| stack_blocks_two | block1/2 | ±0.40 | 0.15~0.50 | 0~0.60 | 矩形 | 0.78（四档加宽 0.81→0.77→0.78→0.78 平台期） | 0.61 |
| blocks_ranking_rgb | block1/2/3 | ±0.32 | 0.15~0.50 | 0~0.75 | 矩形 | 0.60 | 0.41 |
| blocks_ranking_size | block1/2/3 | ±0.30 | 0.16~0.48 | 0~0.75 | 矩形 | 0.53 | 0.38 |
| place_bread_skillet | bread/skillet | ±0.32 | 0.26~0.50 | 0~0.50 | 矩形 | 0.63 | 0.44 |
| place_bread_basket | 三 actor 同带 | ±0.30 | 0.22~0.48 | 0~0.50 | **椭圆** r=(0.30,0.13) c=0.35 | 0.58（矩形 0.47） | 0.27 |
| place_can_basket | basket/can | ±0.22 | 0.24~0.42 | 0（锁） | **椭圆** r=(0.22,0.09) c=0.33 | 0.27（矩形 0.23） | 0.24 |
| place_burger_fries | tray/食物 | ±0.22 | 0.22~0.40 | 食物 0~0.20 | 矩形 | 0.39 | 0.17 |
| place_cans_plasticbox | plasticbox | ±0.12 | 0.22~0.38 | -0.60~0.60 | **椭圆** r=(0.12,0.08) c=0.30 | 0.39（矩形 0.29） | 0.48 |
| place_cans_plasticbox | can1/can2 | ±0.18 | 0.15~0.45 | 0 | **椭圆** r=(0.18,0.15) c=0.30 | 同上 | 同上 |
| place_dual_shoes | shoe_box | ±0.22 | 0.30~0.43（=impl 硬编码） | 0（锁） | 矩形 | 0.29（椭圆 0.20） | 0.20 |
| stack_bowls_two | 例外不变：历史非对称带（绝对坐标 x(-0.20,0.05)×y(-0.15,0.00)） | | | | — | — | 0.10 |

形状终选依据（2026-10-01 椭圆 A/B 全部 100-seed，证据 `results/pilot/v0_ranges/*.ellipse/base*.pilot.jsonl`）：
椭圆胜——bread_basket(+11)、plasticbox(+10)、can_basket(+4，其矩形远角在罐-篮运输死区）；
矩形胜——burger_fries(+10)、dual_shoes(+9，椭圆 x 边际中心聚集使鞋盒贴近鞋行、UnStableError 升高）;
stack_blocks 同包围盒平局且矩形可推得更宽（x±0.40）→ 矩形。机制：`CENTERED_V0_FINAL_ELLIPSE_TASKS`
为最终椭圆任务集，`ROBOTWIN_V0_ELLIPSE` 环境变量可强制全开/全关供未来 A/B。

三轮迭代对比（同 pilot 方法、legacy 对照校准）：

| 任务 | 表心对称轮 | 基座相对终版 | legacy 基线 |
|---|---|---|---|
| stack_blocks_two | 0.35 | **0.78**（x±0.40） | 0.61 |
| bread_skillet | 0.47 | **0.63** | 0.44 |
| blocks_ranking_rgb | 0.22 | **0.60** | 0.41 |
| blocks_ranking_size | 0.21 | **0.53** | 0.38 |
| bread_basket | 0.20 | **0.58** | 0.27 |
| burger_fries | 0.21 | **0.39** | 0.17 |
| plasticbox | 0.19 | **0.39** | 0.48 |
| dual_shoes | 0.24 | **0.29** | 0.20 |
| can_basket | 0.24 | **0.27** | 0.24 |
| bowls | 例外 | 例外 | 0.10 |

设计依据：可靠可达半径 ~0.5-0.55m（角点距基座）；近臂条带（y_rel 0.13-0.30）为最高成功率区；
y_rel 下界 footprint 感知（0.11 + 物体半宽 + 0.01，保证全 footprint 在桌面，杜绝非桌面采样浪费）；
画幅约束 y_rel≤~0.79 不活跃。

**验收基准说明（重要）**：pilot 方法经 legacy 对照校准——legacy 协议 pilot 结果（rgb 0.42、stack_blocks 0.52）
与 09-11 归档采集成功率（0.408、0.608）一致，故 v0 pilot 成功率与 legacy 采集基线直接可比。
按用户规则（绝对 0.20 线，贴线即最大加宽），rgb/size/stack_blocks 等以成功率换范围；
若后续改为相对线（如不低于 legacy 的某个比例），这些任务需回退。

**已知例外**（物理锁定，不做对称化）：
1. **place_dual_shoes**：鞋 y 锁定在 lift 验证行 y∈{0.13,0.14}（diag_ds_fullmap.py 0.02m 网格探测，
   带外为 grasp-only 不可 lift 区），鞋的 manifest x/y 在居中批次被 impl 硬编码覆盖，不参与 v0 对称化。
2. **stack_bowls_two**：对称化×实测 place-IK 可达边界（目标 x>0.12 或 y>0.05 零成功）×impl 三点 0.13m
   间隙几何三元约束不可同时满足（draft-1 x±0.12/y±0.05 几何不可行全 SceneInitReject；draft-2
   x±0.15/y±0.08 可采样但 0.10 不过线）。两协议下均保留历史非对称家族带。
3. **place_cans_plasticbox（部分）**：历史非对称近臂箱带（y(-0.28,-0.20)，pilot 0.47）与对称化冲突，
   v0 取对称可达窗近似（0.19），是"臂偏心布局 vs 关于 0 对称"矛盾的典型。

**失败模式总结**：所有家族的失败以 plan 失败（可达性）为主、UnStableError（近臂碰撞/不稳定）次之；
距臂基座 (0,-0.45) ≥0.5m 成功率系统性崩落——y 方向（臂方位）是主要约束，"收 y 保 x"是通用收敛方向。
这同时回答了"历史批为何摆放尺度更大"：旧批臂旁置布局的拾取带天然贴近臂基座，居中对称布局必有一半
概率采样到远侧低效区。

**去 hash 化（用户决断 2026-09-30）**：协议/数据不使用内容哈希校验，全部改为字段级校验——
manifest 哈希冻结测试替换为逐 actor 字段断言（`tests/test_left_task_base.py`）；
`scripts/run_noise_experiment.py` 的源码 sha256 门禁替换为协议关键字段快照比对
（`protocol_snapshot()`），数据集溯源改记文件数/字节数字段（`dataset_summary()`）；
EMIX 数据集指纹改字段级（`emix_index_utils.compute_source_dataset_fingerprint`，
逐文件 relative_path+size_bytes；对 legacy sha256 指纹向后兼容，已实测通过）。

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

## 3. 实验 2：v0-singleTask-noise（已完成归档：显式图像高斯噪声）

### 3.1 定义与命名

- 在实验 1 基础上**仅增加数据侧显式高斯噪声**；模型与训练架构不变，模型代号仍为 `defaultDP` / `emixDP`。
- 噪声定义：`ε[c,h,w] ~ N(0, variance)`，所有像素/通道独立，`x' = clip(x+ε,0,1)`。无整图缩放；
  不改状态、动作标签或相机位姿。注入位置为解码/缩放为 `[0,1]` 之后、策略标准化之前（EMIX 混合之后）；
  每次观测重新采样，不物化噪声视频。方差描述截断前的 Gaussian，黑白边界附近截断后实测方差可能较小。
- 训练噪声由训练 seed 和绝对 optimizer step 决定（独立于模型/加载器全局 RNG）；评估噪声 seed=0，
  结合场景 seed 与 take_action_cnt 使用独立 RNG，不扰动模型采样/专家规划 RNG。
- **本实验执行约束**：2026-10-01 起启动的本实验任何 `defaultDP` / `emixDP` 训推任务，
  均先执行 V0 `pre_vis`，无论使用新数据还是复用归档数据。四类图与 checkpoint 同任务、
  同模型归档，评估时另存真实策略输入截图。历史上已完成的运行不追溯补称为事前完成；
  如需补图，须标记为运行后回填。仅显式 CLI `--skip-pre-vis` 可绕过，见 1.2.1 节。
- **档位**（2026-09-28 用户确认两档均完全足够；同日最终决定**本轮（第一个噪声轮次）沿用已执行的 var0025，不强制切换**）：

| 档位后缀 | 方差 | σ（[0,1]） | σ（255 域） | 定位 |
|---|---:|---:|---:|---|
| `noiseVar0025` | 0.0025 | 0.05 | 12.75 | **已完成**（defaultDP 与 emixDP 四臂齐备） |
| `noiseVar000625` | 0.000625 | 0.025 | 6.4 | **已冻结**（用户决定不继续噪声方向） |

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
| v0-singleTask-noiseVar0025Tr-emixDP-s5-seed0 | emixDP | ✅ 完成 | `outputs/run_v0-singleTask-noiseVar0025Tr-emixDP-s5-seed0/` | 37 / 39 / 31 / 58 / 28（均值 38.6%） |
| v0-singleTask-noiseVar0025TrEv-emixDP-s5-seed0 | emixDP | ✅ 完成 | `outputs/run_v0-singleTask-noiseVar0025TrEv-emixDP-s5-seed0/`（监督目录 `outputs/run_emix-noise-suite-20260929-235027/`） | 42 / 43 / 35 / 62 / 34（均值 43.2%） |

本轮四个臂固定 s5 五任务（沿用实验 1 筛选，不替换）、var0025、seed 0；`noiseVar000625` 臂已冻结。
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

归档执行 YAML（旧命名遗留，映射见 3.2）：defaultDP 臂 `configs/scripts/run_v0_noise_tn_{ec,en}.yaml`（var0025）；
emixDP 臂 `configs/scripts/run_emix_noise_tn_{ec,en}.yaml`（已规范命名为 v0-singleTask-noiseVar0025Tr/TrEv-emixDP，variance 0.0025）。历史运行只执行了 `train-eval`，未重新采集、转换或构建 EMIX 索引。
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

- 状态：`complete`；当前条件：`tn_en`。
- 更新：2026-10-01T00:36:44+08:00；开始：2026-09-29T23:50:28+08:00。
- 运行目录：`outputs/run_emix-noise-suite-20260929-235027`；监督进程 PID：3930382。
- 状态与完整结果：`outputs/run_emix-noise-suite-20260929-235027/status.json`。
- 按用户定稿顺序自动执行：单侧训练/评估 → 双侧复用训练/评估。

| 任务 | 训练完成 | 单侧 EMIX SR | 双侧 EMIX SR |
|---|---|---:|---:|
| stack_blocks_two_left_cen_arm_top_cam | 是 | 37/100 (37.0%) | 42/100 (42.0%) |
| place_bread_basket_left_cen_arm_top_cam | 是 | 39/100 (39.0%) | 43/100 (43.0%) |
| blocks_ranking_rgb_left_cen_arm_top_cam | 是 | 31/100 (31.0%) | 35/100 (35.0%) |
| place_can_basket_left_cen_arm_top_cam | 是 | 58/100 (58.0%) | 62/100 (62.0%) |
| place_dual_shoes_left_cen_arm_top_cam | 是 | 28/100 (28.0%) | 34/100 (34.0%) |
<!-- emix-noise-run-status:end -->

### 3.6 噪声方向阶段性结论（2026-09-29 用户决断）

- **历史 EMIX 提升与原始 OIDN 噪声的关联判断**：历史上 v0 归档中 EMIX 的正提升
  （can_basket 49→64、place_dual_shoes 33→44）**很可能与 50 系显卡（5060）OIDN 错误产生的含噪数据有关**——
  该两任务的数据正是 5060 采集（GPU 分配与归档一致：CUDA 1=5060 负责 can_basket、dual_shoes；
  CUDA 0=4070 采集的另三任务为干净图）。
  **含噪数据批次：`archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911` 中的
  `place_can_basket`、`place_dual_shoes` 两任务数据**（OIDN 请求但 50 卡报错、明暗偏移噪声入图）。
- **该原始噪声的特性（用户人工目检）**：幅度小、为随机明暗偏移、**不太影响任务相关物体的识别**；
  与本轮注入的逐像素独立高斯噪声**机理不同**。**在当前高斯噪声（var0025）下无法复现该优势**
  （本轮 emixDP 在这两个任务上反而 −7/−7）。
- **并注意到**：历史批与当前批的物体摆放分布在**绝对范围尺度**上存在差异（旧批拾取带如
  x∈[-0.45,-0.16]，当前居中包络仅 x∈[-0.20,0.20]×y∈[-0.15,0.15]；详见历史更新记录 2026-09-29 条目）。
- **资源决断**：TrEv 已完成，**后续不再对噪声相关内容投入实验和计算资源**
  （var000625 轮次、clean 对照臂一并冻结，见第 4 节）。噪声方向的收官分析在 TrEv 完成后按 3.3 出具。

## 4. V1 后续实施计划

`v0-noise` 的四臂结果仅属 V0。V1 使用当前任务代码重新采集十任务各 500 条已接受轨迹；首轮正式训练前采集、转换和索引必须完成。所有 V1 条件共享这一份**新采集**的数据和索引，不共享 V0 checkpoint。

| 顺序 | 协议代号 | 任务/模型 | 状态 |
|---|---|---|---|
| 1 | `v1-data-all10` | 当前 V1 实现、顶视、十任务各 500ep；同步实例归属图，84×84 输入 | **完成**；十项 500ep 数据/标注、all10 索引与可视化完成 |
| 2 | `v1-collage-s5` | 历史 Emix 提升最大 2 任务（can_basket、dual_shoes）与较差 3 任务（bread_basket、stack_blocks_two、blocks_ranking_rgb）；各跑 defaultDP 与 Emix-collage | **运行中**；can_basket 59/100 vs defaultDP 67/100，stack_blocks_two 17/100 vs 26/100；bread_basket Emix 31/100 vs defaultDP 43/100；dual_shoes defaultDP 29/100、Emix 评估 73/100；blocks_ranking_rgb defaultDP 评估 90/100 |
| 3 | `v1-collage-all10` | 同参数扩至全部十任务，复用 s5 已完成模型 | 已排队 |
| 4 | `v1-overlay-s5/all10` | 同数据、同训练参数的 Emix-overlay 方法对照，先五后十 | 已排队 |
| 5 | `v1-noise-s5/all10` | 后续排期：逐通道高斯噪声训练，先 clean 评估、再同噪声评估；先五后十；defaultDP 与 Emix-overlay | 已排队 |

`Emix-overlay` 保留旧图像线性叠加 `αI1+(1-α)I2` 与 `αloss1+(1-α)loss2`。`Emix-collage` 在第二图可见的 `task:` 对象中选一个，使用同帧的紧凑归属图把该对象像素粘入第一图，其余像素保持第一图；损失为 `loss1+loss2`，不乘 α、不除以 2。背景、机器人和墙不参与对象选择。首轮不加入阶段判断；阶段规则是 TODEV 条目 3 的后续独立变量。84×84 缩小时小物体和边缘归属的不可消除歧义保留为边界，不声称修复。

固定条件沿用 §1.2 的 DP/相机/训练/评估参数：Piper 双臂、左臂操作、top head RGB、84×84、17 FPS、14 维状态动作、ResNet18、30,000 steps、batch64、seed0、100 个评估场景。V1 空间预设 `ROBOTWIN_CENTERED_PROTOCOL=v0`，任务实现版本 `v1`；OIDN 请求为 `oidn`。噪声方差固定 `0.0025`（归一化 RGB），只有 `v1-noise` 的训练启用，评估 clean/noisy 分别显式关闭/启用。所有训推默认执行 `pre_vis`，同一模型的四类图放在对应训练/评估输出目录。

`stack_bowls_two` 的 V1 顶视 bowl **初始位置**默认 x∈[-0.33,0.33]、y∈[-0.15,0.15]（标称矩形 0.198 m²，原生 0.18 m²）；**堆叠放置目标**仍受 place IK 限制，保留 x∈[-0.20,0.05]、y∈[-0.15,0]。两套范围独立。100-seed 试验（不同 seed 区间）分别为 ±0.30 m 起点成功 77/100、专家可采集 71/100；±0.33 m 起点成功 63/100、专家可采集 61/100；±0.33 m 的 200 个 bowl 起点中 185 个超出旧窄带。选择 ±0.33 m 是以更大范围换取较低可采集率；**标称矩形不能等同实际接受轨迹的均匀覆盖，更不能推断十任务所有放置目标都大于原生**。原始逐 seed 证据：`docs/evidence/v1_bowls/wide_spawn_100.pilot.jsonl` 与 `wide_spawn_x033_100.pilot.jsonl`。

执行器 `scripts/run_v1_queue.py` 顺次启动十任务新采集和索引、拼贴五/十、overlay 对照五/十、噪声五/十；监督进程 PID 与当前阶段以 `outputs/v1_experiment_queue/status.json` 为准，完整输出在同目录 `supervisor.log`，各实例 YAML 位于 `configs/scripts/run_v1_*.yaml`。运行中以实际 metrics 和状态文件更新本表，不把排队或运行中写成已完成。

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
- 用户人工目检修正历史噪声判定：**OIDN 错误产生可见噪点的是 50 系卡（5060）**，为随机明暗偏移、
  不损害物体辨识、影响较轻；40 系卡（4070）OIDN 失败时 RT 收敛为干净图（与 TODEV 09-22 探测一致）。
  原始 OIDN 噪声与本轮显式高斯噪声机理不同（亮度级偏移 vs 逐像素独立像素值扰动）。
  含义：归档数据的历史噪声状态应分卡判定（5060 采集的任务可能自带轻度明暗噪声，4070 采集的干净），
  历史噪声状态由 unknown 细化为"分卡部分已知"；第二级跨实验比较需考虑此差异。
- **用户决断两项，记入现行协议**：
  ① 噪声方向收官判断——历史 EMIX 正提升（can_basket +15、dual_shoes +11）很可能来自 50 系卡
  OIDN 错误的含噪数据（批次：`archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911` 的
  can_basket、dual_shoes 两任务，5060 采集）；该噪声为小幅随机明暗偏移、不影响物体辨识、
  机理不同于高斯噪声，var0025 高斯噪声下无法复现该优势；另确认历史与当前物体摆放分布在绝对
  尺度上存在差异。决断：除在跑的 TrEv 收尾外，**噪声方向不再投入实验和计算资源**
  （var000625 轮次与 clean 对照臂冻结）。
  ② **优先 TODO：扩大物体摆放随机化范围尺度**——当前居中包络均匀但绝对尺度不足（对比旧批
  x∈[-0.45,-0.16]），允许按任务定制随机化范围以保证采集成功率；列入 `_exp.md` 第 4 节第 1 条。

## 2026-09-30

- **GPU 资源重排（用户指示）**：TrEv 评估迁移到单卡（CUDA 1 / RTX 5060）顺序执行
  （`configs/scripts/run_emix_noise_tn_en.yaml` 的 `gpu_batches` 全部任务集中到 `'1'`；
  迁移时 stack_blocks 评估进度 77/100、place_can_basket 63/100，重启后经 resume 跳过已完成任务，
  评估从头重跑这两个任务）。腾出的 CUDA 0 / RTX 4070 专用于测试与开发。
  新监督目录 `outputs/run_emix-noise-suite-20260929-235027`，cron 巡检已指向新目录。
- **去 hash 化（用户决断）**：删除协议相关内容哈希校验，替换为字段级校验，覆盖四处——
  manifest 哈希冻结测试 → 逐 actor 字段断言；`run_noise_experiment.py` 源码 sha256 门禁 →
  协议关键字段快照比对（`protocol_snapshot()`）；数据集溯源哈希 → 文件数/字节数字段；
  EMIX 数据集指纹 → 逐文件 relative_path+size_bytes 比对（对 legacy sha256 指纹向后兼容，实测通过）。
  详见 §1.6 末节。`manifest_hash()` 函数已删除；impl docstring 中的历史哈希字符串作为 pilot 证据记录保留。
- **V0 摆放随机化范围立项（决断②实施）**：`CENTERED_V0_OVERRIDES` 注册表 + `ROBOTWIN_CENTERED_PROTOCOL`
  开关落地，draft-1 逐任务参数表入 §1.6；第一批 5 任务 100-seed pilot 在 4070 上运行中
  （`results/pilot/v0_ranges/`），验收规则 ≥0.20。

## 2026-09-30（pilot 战役定稿）

- **V0 摆放随机化十任务全部定稿**：共跑 17 个 100-seed pilot（含 2 个 legacy 对照）。
  pilot 方法经对照校准（legacy pilot 0.42/0.52 vs 采集基线 0.41/0.61，一致）。
  定稿值与逐任务 pilot 结果见 §1.6 最终参数表；证据按 draft 版本归档于
  `results/pilot/v0_ranges/*.d{1,2,3,4}.pilot.jsonl`（legacy 对照为 `.legacy.pilot.jsonl`）。
- **主要结论**：①失败以 plan（可达性）为主，距臂基座 (0,-0.45) ≥0.5m 成功率系统性崩落，
  "收 y 保 x"是通用收敛方向；②stack_blocks_two 加宽至 x±0.30/y±0.22（成功率 0.35 进入平台期，
  四档 0.35→0.38→0.42→0.35）；③bread_skillet 是唯一"加宽且成功率不低于基线"的任务
  （x±0.25/y±0.18，0.41 vs 基线 0.44）；④burger_fries 为结构性天花板家族（legacy 本身 0.17），
  v0 取值（x±0.18/y±0.12）比 legacy 略窄才上线（0.21）；⑤三条物理例外定案：
  dual_shoes 鞋行 y∈{0.13,0.14}、stack_bowls_two 保留历史非对称带（三元约束不可同时满足）、
  plasticbox 取对称可达窗（0.19≈0.2，历史非对称近臂带 0.47 不可对称化）。
- **遗留动作**：TrEv 评估完成（cron 托管中）后翻转 `ROBOTWIN_CENTERED_PROTOCOL` 默认值为 `v0`，
  之后新采集批次使用 v0 布局。

## 2026-09-30（基座相对语义变更，用户决断）

- **用户指出并确认**：旧批次的"正负范围"语义是相对于摆放中心（臂基座），不是相对于桌面中心；
  当前代码中的 v0 变量写的是相对整个坐标系（桌心）正负。用户决断：**立刻转向基座描述**。
- **几何事实**（核对 piper config `robot_pose [[0,-0.45,0.75]]` 后确认）：两批臂基座 y 都在 -0.45
  （桌子近缘外侧），"居中"只居中了 x。工作空间是以基座为圆心、半径 ~0.55-0.6m 的可达圆盘；
  旧批摆放带（如 blocks x(-0.45,-0.16)）关于臂 x 对称、距臂 0.27-0.55m，全在圆盘内——
  旧批"范围大"的真正原因是**臂被摆放区包围**，而非对称性本身。
- **实施**：`CENTERED_V0_BASE_XY=(0,-0.45)`；`CENTERED_V0_OVERRIDES` 全部改为基座相对
  （x 关于基座 x 对称；y 为基座前方距离带，解析时平移为绝对坐标）；表心对称轮定稿值保留在
  注释与 §1.6 对照表中。y 无法基座对称（桌子只在基座前方），这是物理约束而非妥协。
- **base-1 轮启动**：9 任务（bowls 例外不变）100-seed pilot 在 4070 上运行中，
  验收规则不变（≥0.20，贴线=max 加宽）。

## 2026-10-01（基座相对终版定稿 + 椭圆 A/B 终选）

- **base-2/base-3/base-4 迭代定稿**：矩形臂 9 任务全部完成 100-seed 验收；
  stack_blocks_two 四档加宽（x±0.35→±0.38→±0.40，SR 0.81→0.77→0.78→0.78）进入平台期，
  终版取 x±0.40（角点距基座 0.64m）。
- **椭圆/圆盘 A/B 完成**（用户建议的第二种范围形状）：新机制
  `ActorSamplingSpec.ellipse` + 面积均匀采样（r=√u）+ `CENTERED_V0_ELLIPSES` 注册表
  （基座相对，内接于可达圆盘∩桌面，worst-case 距基座 ≤0.51m）。
  终选：椭圆——bread_basket(0.58>0.47)、plasticbox(0.39>0.29)、can_basket(0.27>0.23)；
  矩形——burger_fries(0.39>0.29)、dual_shoes(0.29>0.20，椭圆中心聚集加剧鞋盒-鞋行不稳定)、
  stack_blocks(同盒平局且矩形可更宽)。固化于 `CENTERED_V0_FINAL_ELLIPSE_TASKS`。
- **y 下界 footprint 感知规则落实**（用户重述的工作流）：y_rel_lo = 0.11 + 物体半宽 + 0.01，
  保证整个 footprint 在桌面，杜绝非桌面位置参与采样。
- **终版成绩**（对比表见 §1.6）：9 个非例外任务基座相对终版 SR 0.27-0.78，
  全部 ≥0.20，且 8/9 达到或超过 legacy 采集基线（burger_fries 0.39 vs 0.17 翻倍）。
  相对表心对称轮全面提升（0.19-0.47 → 0.27-0.78）。
- 例外不变：stack_bowls_two 保留历史非对称带；dual_shoes 鞋行 y∈{0.13,0.14} 物理锁定。

## 2026-10-01（TrEv 收官，v0-singleTask-noise 四臂齐备）

- **emixDP TrEv 完成**（监督目录 `outputs/run_emix-noise-suite-20260929-235027`，迁移单卡后重启）：
  42 / 43 / 35 / 62 / 34（均值 **43.2%**）。四臂齐备，var0025 轮正式收官。
- **两级对比结论**：
  - 内部（var0025 噪声，emixDP − defaultDP）：Tr **−2.4**（38.6 vs 41.0），TrEv **+0.4**（43.2 vs 42.8）
    ——高斯噪声下 EMIX 无正提升，与 §3.6 决断一致（无法复现历史 OIDN 噪声优势）；
  - 外部（相对 v0-singleTask 干净协议 09-11 批）：defaultDP 干净 40.8% → Tr 41.0 / TrEv 42.8
    （噪声训练对 defaultDP 无实质影响）；emixDP 干净 43.2% → Tr 38.6（−4.6）/ TrEv 43.2（±0.0）
    （单侧噪声训练使 emixDP 暂时掉点，双侧评估噪声补齐）。干净批的历史 EMIX 边缘正提升
    （can_basket +15、dual_shoes +11，p≈0.12-0.17）在 var0025 高斯噪声下消失。
- **噪声方向自此完全收官**：不再投入噪声实验资源（var000625 轮次与 clean 对照臂保持冻结）。
- **协议默认值翻转**：TrEv 完成后，`ROBOTWIN_CENTERED_PROTOCOL` 生产默认由 `legacy` 翻转为
  **`v0`**（基座相对 + footprint 感知 + 逐任务矩形/椭圆终选，见 §1.6 终版参数表）；
  之后的新采集/训练批次使用 v0 布局，legacy 仍可用环境变量回退。

### 1.7 v0 终版 vs RoboTwin 原生任务的范围对比（2026-10-01）

原生实现（双臂改造前，`envs/<task>.py` 的 init 范围）：stack_blocks/rgb 0.56×0.13 m²、
skillet 0.56×0.25、bread_basket 篮子钉死仅面包 0.54×0.25、burger_fries/can_basket/
plasticbox 容器钉死且食物/罐带极窄（0.004-0.008 m²）、dual_shoes 0.10×0.15、bowls 0.60×0.30。
v0 终版 8/10 任务明确更大（stack_blocks 3.8×、rgb 3.1×、容器任务 >10× 且容器本身也参与随机，
dual_shoes 鞋盒 x 向 4×）；skillet 与原生相当；**唯一反例 stack_bowls_two**（例外带 0.037 m²
< 原生 0.18 m²，受实测 place-IK 死区约束）——对外 claim "随机化范围高于原生实现"时必须除外该任务。


## 2026-10-01（版本升至 V1，V0 噪声结果封存）

- 用户将当前大幅更新的任务实现定为 `v1`；`v0-singleTask-noise` 四臂结果全部归属于先前的 `v0` 实现与归档数据，不因后续代码默认摆放预设变化而重新命名。
- `v1` 当前使用基座相对范围预设（代码标识 `v0`）；版本号与预设名分开维护。正式 V1 实验须建立独立实例、记录新数据版本，并遵守 §1.2.1 的 `pre_vis` 规则。
- **bowls 修订**：此前 bowl 与堆叠目标共享 0.0375 m² 窄带，不足原生 0.18 m²。现把抓取起点与目标解耦，顶视默认起点标称 0.198 m²，目标仍为原窄带；100-seed 可采集率 61%，证据及边界见 §4。
- **队列接管快照（08:03 CST）**：监督进程 PID 3136298 存活，正在转换首批两个任务，各约 160/500 episode；本阶段由 CPU 转码，双 GPU 暂无训练/推理负载。后续按小时巡检；状态 JSON 中旧 `exit_code`/`ended_at` 字段与当前 `running` 状态不一致，按存活 PID 和持续前进的监督日志判定当前运行状态。
- **队列巡检（09:03 CST）**：监督进程 PID 3136298 持续运行；首批两个数据集已进入 LeRobot 转换目录，双卡当前分别采集 `place_bread_skillet`（5060，约 81%）和 `place_bread_basket`（4070，约 44%），日志持续写入 episode。状态文件中的旧退出字段仍残留，当前阶段以存活进程和日志为准。
- **队列巡检（10:04 CST）**：监督进程 PID 3136298 持续运行，四个任务已进入转换目录；当前采集 `place_can_basket`（5060，约 88%）和 `place_burger_fries`（4070，约 56%）。失败 episode 后仍有成功记录，采集继续推进；未见需重启的异常。
- **队列巡检（11:05 CST）**：监督进程 PID 3136298 和两路采集子进程持续运行；转换目录仍有前四个任务。日志显示 `place_can_basket` 已写出 episode 206 视频并继续采集，另一任务持续尝试并有成功 episode；GPU 利用率约 82%/55%。
- **队列巡检（12:05 CST）**：监督进程 PID 3136298 持续运行，五个任务已进入转换目录；双卡当前分别采集 `place_dual_shoes`（5060，约 43%）和 `place_burger_fries`（4070，约 78%），episode 帧保存日志继续增长。
- **队列巡检（13:06 CST）**：监督进程 PID 3136298 持续运行，六个任务已进入转换目录；双卡当前分别采集 `place_dual_shoes`（5060，约 72%）和 `place_cans_plasticbox`（4070，约 43%）。dual_shoes 多次出现 `041_shoe` 不稳定失败，但 episode 编号递增且仍有成功记录，采集继续。
- **队列巡检（14:06 CST）**：监督进程 PID 3136298 与两路采集进程持续运行；转换目录仍有六个任务，活动任务为 `place_dual_shoes`、`place_cans_plasticbox`。双卡利用率约 82%/83%，日志持续写入帧。
- **队列巡检（15:07 CST）**：监督进程 PID 3136298 持续运行；前六个任务已完成转换，`place_dual_shoes` 原始采集完成、转换统计到 75/500，`place_cans_plasticbox` 仍在采集。转换使用 CPU，5060 暂空闲、4070 利用率约 45%；队列每卡串行，未抢跑下一任务以避免和监督 worker 争用其数据目录。
- **队列巡检（16:09 CST）**：监督进程 PID 3136298 持续运行，前七个任务已完成转换；双卡分别采集 `stack_bowls_two`（5060，约 83%）和 `place_cans_plasticbox`（4070，约 73%），日志持续写入帧。
- **队列巡检（17:09 CST）**：`stack_bowls_two` 转换完成，八个任务已完成并进入转换目录；`place_cans_plasticbox` 正在采集，日志 episode 约至 470，随后还需处理 `stack_blocks_two`。4070 当前利用率约 72%；5060 对应 worker 的 5/5 项已完成并等待全局屏障，保持队列分卡顺序以避免清理另一 worker 的进行中数据。
- **队列巡检（18:10 CST）**：`place_cans_plasticbox` 转换完成，九个任务已进入转换目录；最后任务 `stack_blocks_two` 已开始采集并写到 episode 7。仅 4070 当前承担这项最后的串行任务，5060 worker 已完成五项并等待全局屏障；待十个数据集齐备后再启动索引。
- **队列巡检与数据校验（19:11 CST）**：十项采集与 84×84 转换全部完成，各 `info.json` 均报告 500 episodes，每项均有 500 个连续编号的归属图 HDF5。逐任务抽查 episode 0，归属图为 `uint8 [N,84,84]`，`frame_ids` 从 0 连续递增，N 与该 episode 的 RGB/状态/动作共同有效帧数一致，十项通过。all10 EmixIndex 正在 CPU 邻居过滤（约 26%）；两卡空闲，索引完成前不启动训练。
- **队列巡检（20:12 CST）**：all10 索引与可视化阶段于 11:37 完成，`collage_s5` 正在运行；两卡分别训练 `place_can_basket` 和 `stack_blocks_two` 的 defaultDP 基线，进度约 7,100/30,000 与 10,170/30,000 steps，GPU 利用率约 97%/99%。
- **队列巡检（21:13 CST）**：`collage_s5` 继续运行；`place_can_basket` defaultDP 训练约 29,400/30,000 steps，`stack_blocks_two` defaultDP 评估中，日志曾显示单条 rollout 动作步数约 447/800（不是 episode 计数）。5060 利用率约 99%，4070 约 37%。
- **队列巡检（22:14 CST）**：`collage_s5` 继续运行，`place_can_basket` 与 `stack_blocks_two` 的 defaultDP 评估进程均在运行；GPU 利用率约 24%/54%，当前无需干预。
- **队列巡检（23:15 CST）**：两项 defaultDP 评估进程仍运行；`stack_blocks_two` 日志中的单条 rollout 动作步数曾到 615/800（不是 episode 计数），GPU 利用率约 91%/100%，监督队列无异常。

## 2026-10-02（V1 collage s5 继续运行）

- **队列巡检（00:16 CST）**：V1 数据与 all10 索引阶段已完成；`collage_s5` 继续运行。`place_can_basket` defaultDP 评估仍在执行，`stack_blocks_two` 的 Emix-collage 训练约 2,956/30,000 steps；双卡利用率约 99%/100%。
- **队列巡检（01:17 CST）**：`place_can_basket` Emix-collage 训练约 7,595/30,000 steps，`stack_blocks_two` Emix-collage 评估日志中的单条 rollout 动作步数约 175/700；双卡利用率约 99%/100%。
- **队列巡检（02:17 CST）**：`place_can_basket` Emix-collage 训练约 27,683/30,000 steps，`stack_blocks_two` Emix-collage 评估单条 rollout 动作步数约 735/800；双卡利用率约 99%/100%，完整指标尚未落盘。
- **队列巡检（03:18 CST）**：两项 Emix-collage 评估进程均在运行；日志中的单条 rollout 动作步数曾到 631/700，GPU 利用率约 99%/83%，未见错误。
- **队列巡检与指标更新（04:18 CST）**：defaultDP 评估文件已落盘：`place_can_basket` 67/100，`stack_blocks_two` 26/100。对应 Emix-collage 评估仍在运行，当前接受 episode 进度分别为 54/100、90/100；双卡利用率约 99%/100%。`step: N/700` 是单条 rollout 的动作计数，不能当作 700 个评估 episode。
- **服务器重启恢复巡检（04:45 CST）**：`stack_blocks_two` Emix-collage 评估完成 17/100，对比 defaultDP 26/100（−9 个百分点）；`place_can_basket` Emix-collage 评估推进至 66/100；GPU worker 已开始 `place_bread_basket` defaultDP 训练。监督 PID 3136298 存活，双卡利用率约 99%/99%。
- **队列巡检（05:46 CST）**：`stack_blocks_two` Emix 结果为 17/100；`place_can_basket` Emix 评估推进至 94/100，`place_bread_basket` defaultDP 评估为 6/100。监督进程存活，双卡利用率约 52%/100%。
- **队列巡检与结果更新（06:47 CST）**：`place_can_basket` Emix 评估完成 59/100，对比 defaultDP 67/100（−8 个百分点）；`stack_blocks_two` 为 17/100，对比 26/100（−9 个百分点）。`place_bread_basket` defaultDP 评估推进至 40/100，`place_dual_shoes` defaultDP 训练至约 17,186/30,000 steps；GPU 利用率约 99%/77%。
- **队列巡检（07:48 CST）**：`place_bread_basket` defaultDP 评估推进至 76/100，`place_dual_shoes` defaultDP 评估推进至 9/100。5060 利用率约 99%，4070 约 3%；后者评估进程仍运行且持有显存，本次采样负载偏低。
- **队列巡检（08:49 CST）**：`place_bread_basket` defaultDP 评估完成 43/100，Emix-collage 训练约 9,200/30,000 steps；`place_dual_shoes` defaultDP 评估推进至 29/100。5060 利用率约 42%，4070 约 98%。
- **队列巡检（09:49 CST）**：`place_bread_basket` defaultDP 结果 43/100，Emix-collage 评估推进至 9/100；`place_dual_shoes` defaultDP 评估推进至 45/100。双卡利用率约 99%/100%。
- **队列巡检（10:50 CST）**：`place_bread_basket` Emix-collage 评估推进至 42/100，`place_dual_shoes` defaultDP 评估推进至 67/100；这两项结果文件尚未生成。双卡利用率约 54%/74%，评估进程仍在运行。
- **队列巡检（11:51 CST）**：`place_bread_basket` Emix-collage 评估推进至 77/100，`place_dual_shoes` defaultDP 评估推进至 86/100；结果文件尚未生成。监督 PID 3136298 与两个评估进程均存活，双卡利用率约 99%/100%，日志和评估视频持续增长。
- **队列巡检与结果更新（12:52 CST）**：bread_basket Emix-collage 评估完成 31/100（defaultDP 43/100），dual_shoes defaultDP 完成 29/100；GPU worker 已进入 `blocks_ranking_rgb` defaultDP 训练（约 10,606/30,000 steps）与 `dual_shoes` Emix-collage 训练（约 4,727/30,000 steps）。监督 PID 3136298 存活，双卡利用率约 97%/99%。
- **队列巡检（13:53 CST）**：`blocks_ranking_rgb` defaultDP 评估推进至 7/100，最新 episode 视频正在写入；`dual_shoes` Emix-collage 训练推进至约 24,384/30,000 steps。监督与 worker 进程存活、日志持续增长，双卡约 99%/100%。
- **队列巡检（14:54 CST）**：`blocks_ranking_rgb` defaultDP 评估推进至 29/100，`dual_shoes` Emix-collage 评估推进至 16/100；两路评估进程和监督进程均存活，日志持续增长。GPU 利用率采样为约 2%/27%，负载低于训练阶段；当前两个 worker 都在推进评估，无空闲 worker 可接队列后续任务。
- **队列巡检（15:54 CST）**：`blocks_ranking_rgb` defaultDP 评估推进至 51/100，`dual_shoes` Emix-collage 评估推进至 34/100，最新视频持续写入。监督和两路评估进程存活；双卡利用率采样约 99%/5%，4070 的评估仍有 episode 进展，负载呈间歇性。
- **队列巡检（16:55 CST）**：`blocks_ranking_rgb` defaultDP 评估推进至 71/100，`dual_shoes` Emix-collage 评估推进至 54/100；视频和日志持续更新，监督与两路评估进程存活。双卡利用率约 99%/100%。
- **队列巡检（17:55 CST）**：`blocks_ranking_rgb` defaultDP 评估推进至 90/100，`dual_shoes` Emix-collage 评估推进至 73/100；最新 episode 视频持续写入，两个评估和监督进程存活。GPU 利用率采样约 57%/100%。
