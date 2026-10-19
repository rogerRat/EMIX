# SOP-06：批次实验启动协议与归档规范

面向执行代理（Agent）的流程规则层：启动一次 DP-vs-EMIX 完整批次实验、以及
批次完成后归档时，**必须**遵守本文约定，避免行为偏移。机制细节（CLI 参数、
阶段流水线、环境矩阵）见 [SOP-05](SOP-05-batch-assets-and-experiment-run.md)；
本文件只定义"哪些参数要问、什么必须报、东西放哪、留什么删什么"。

## 1. 启动新批次实验

### 1.1 必须向用户确认的参数（每次启动前，不允许自行默认）

即使与上一批次完全相同，也必须列出让用户明确确认（"与 top_cam 500ep 批完全
相同，确认？"不算违规，静默开跑算违规）。

| 参数 | 说明 | 常见取值 |
|---|---|---|
| `variant`（相机视角） | cen_arm 系列变体 | `cen_arm_top_cam` / `cen_arm_right_wide_cam` / `cen_arm_near_side_cam` / `cen_arm_front_cam` / `cen_arm_side_cam` |
| `--episodes` | 每任务 accepted episode 数 | `200` / `500` |
| 任务集合 | 默认 10 个 ACTIVE_FAMILIES | 增减任何任务必须明示并获确认 |
| `--image-width/height` | 训练分辨率 | 默认 `84 84` |
| `--gpus` | 采集/评测使用的 CUDA 设备 | 默认 `0`；多卡时显式列出设备编号 |

### 1.2 隐性参数（沿用默认，但每次启动必须把数值报给用户）

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

### 1.3 启动前检查

1. 变体接线四件套齐全：RoboTwin `envs/left/<variant>/` 任务文件、
   `data_collection/robotwin.py` 注册、`configs/{train,eval}/<variant>/`、
   `left_task_base.CAMERA_SPECS`。缺哪件补哪件，不要裸跑。
2. run yaml：`configs/scripts/run_<variant>[_<ep>ep].yaml`。新批次从最近的
   同类 yaml 用 sed 替换任务名/episodes 生成；**该目录按 .gitignore 约定
   不入库**，不要 `git add -f`。
3. 工作区干净（未提交改动先按语义批次提交，见 1.5）、所选 GPU 空闲
   （`nvidia-smi`）、live `data/` 与 `outputs/{eval,train,intersections}`
   无同名残留（归档后应为空）。

### 1.4 启动与记录（约定俗成）

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
- 必须记录到两处：`docs/prog/prog.md`（带日期小节）与代理记忆。内容：
  启动时间戳、启动时 commit ID、tmux 会话、run ID、1.1/1.2 的参数、
  预计时长、对照基线（哪个归档批次）

### 1.5 启动前如工作区不干净

按语义分批提交（本次改动一个 commit；非本次的遗留改动总结为另外的语义
commit），全部 push 后再开跑。commit message 遵循仓库现有
`type(scope): summary` 风格并附 Co-Authored-By 尾注。

## 2. 归档规范

### 2.1 时机与前提

- 前提：20 个 eval run 全部完成、每个 `metrics.json` 齐全且 episodes=100；
  经用户确认后执行（用户说"归档"即确认）
- 归档前先记录 **commit ID**（`git rev-parse --short HEAD`）——SUMMARY.md
  要写

### 2.2 位置与命名（严格）

```
archive/DP-10tasks-<ep>ep-emix-baseline-<variant-kebab>-<YYYYMMDD>/
```

- 例：`DP-10tasks-200ep-emix-baseline-cen-arm-top-cam-20260904`
- **时间戳必须有**，取归档当日；variant 用 kebab-case 短名（near-side-cam /
  right-wide-cam / top-cam）

### 2.3 结构与保留/删除规则（严格）

| 子目录 | 内容 | 保留 | 删除 |
|---|---|---|---|
| `data/robotwin` `data/lerobot` `data/emix_index` | 10 任务三件套 | 整体 `mv` 进归档 | live 对应目录随后为空 |
| `eval/` | 20 个 run | `metrics.json` 全保；每 run 视频只留**前 10 个**（`episode_0000-0009`） | 其余 90 个 mp4 `rm` |
| `train/` | 20 个任务 | `*.json`（train_config_snapshot / emix_train_config）+ `encoder_debug/`、`debug/` | `checkpoints/` 一律删除（每任务 ~18G） |
| `vis/` | 可视化 | `outputs/intersections` 的 html/png/`*_assets` + `plotly.min.js` | — |

归档后 live 的 `data/{robotwin,lerobot,emix_index}`、
`outputs/{eval,train,intersections}` 应全部为空；`df -h` 核对释放量。
`rm -rf checkpoints` 前先确认 20 个 train 目录里只剩 checkpoints（config 已
mv 走）。

### 2.4 SUMMARY.md（必须，放归档根目录）

必须包含，缺一不可：

1. **归档时间戳**（YYYY-MM-DD）与**归档时 commit ID**——硬性要求
2. 批次背景：variant、1.1/1.2 参数、采集规模与每任务专家成功率
   （`collection_metrics.json` 的 accepted/attempted）、训练/eval 规模
3. 复现命令（assets.py + run.py 两行）
4. 结果表：逐任务 baseline vs EMIX（百分比 + n/100）+ delta + TOTAL 行
5. 与可比批次的对比表（如同机位异 episode 数、同参数异机位）
6. 数据异常备注（命名不一致、缺失任务、采集事故等）

### 2.5 归档后

更新 `docs/prog/prog.md`（归档时间戳 + SUMMARY 里的 TOTAL 数字）与代理记忆
（归档路径、结果一句话、机位/批次对比结论）。prog.md 的日期小节**倒序排列：
最近的日期在最上面**，新条目插在文件头部最旧的日期之前。

## 3. 速查

- 结果解读：100-ep eval 的逐任务 delta 噪声约 ±5pt，±3pt 以内的组间差异
  应读作持平
- 三机位扫描现状（200ep）：near_side +1 / right_wide +0.6 / top −2.9；
  500ep + 新配对判据（rpy 15° + 去尾）首个批次已归档：500ep top_cam
  Δ=+2.3（archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911）
- 历史归档索引：`archive/*/SUMMARY.md`，每个归档一份，先读它再比对
