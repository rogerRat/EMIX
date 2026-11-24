# Experiment Results

正式实验结果汇总索引。每个结果页记录协议、数据来源、逐任务 metrics 和结论边界。

## 统一结果汇报格式

对于包含 Baseline、Mix、Patch 三个实验臂的实验，每个任务及总体汇总都必须同时报告五项：Baseline 原始结果、Mix 原始结果、Patch 原始结果、Mix − Baseline、Patch − Baseline。不得只报三种原始结果或只报差值。原始结果需包含成功次数/总评估次数及成功率；差值以百分点（pp）表示，正值代表高于 Baseline，负值代表低于 Baseline，零代表持平。其他实验臂也按同一规则报告：给出每个实验臂自身的结果，并给出该实验臂减去 Baseline 的差值。后续口头进度和结果汇报沿用此格式，并说明汇总样本数。

| 实验 | 日期 | 状态 / 结论 | 详情 |
|---|---|---|---|
| v0-pi05-paramsearch：π0.5 batch 1 试运行 | 2026-11-07–09 | 已停止并归档；10k 为 0.4%，20k 为 0.7%，30k 仅五任务完成；旧权重已删除。 | [结果报告](results/20261109-pi05-batch1-pilot.md)；原始指标：`outputs/v0-pi05-paramsearch/eval/` |
| v0-pi05-accum300k：π0.5 有效 batch 32 + LR 修复 | 2026-11-15– | 训练中；100k/200k/300k 各十任务 100ep；尚无正式结果。 | [结果报告](results/20261115-pi05-accum300k.md)；原始指标：`outputs/v0-pi05-accum300k/eval/` |
| multi-v0-searchparam：SmolVLA 微调步数探索 | 2026-11-05–06 | 已完成；200k 的十任务平均成功率最高，为 8.8%；单 seed 描述性结果。 | [结果报告](RESULT.md)；原始指标：`outputs/multi-v0-searchparam/eval/` |
| sin-v0-patch，left_oppo_cam | 2026-10-30 | 有效十任务批次；BaselineDP 38.7%，Mix DP 38.3%，Patch DP 38.3%。单 seed 描述性结果。 | [Patch 批次结果](results/20261030-sin-v0-patch-results.md) |
| sin-v0-patch，场景背景被误当作 Patch 前景 | 2026-10-22 | 无效批次，已作废，不纳入比较。 | [错误批次记录](results/20261022-sin-v0-patch-scene-background-bug-invalid-run.md) |
| sin-v0-patch，早期掩码错误运行 | 2026-10-22 | 无效批次，已作废，不纳入比较。 | [早期错误记录](results/20261022-sin-v0-patch-bug-invalid-run.md) |
| 10-task 500ep EMIX 回归审计 | 2026-09-11 | 完成；结果限制和逐任务分析见报告。 | [回归审计](results/20260911-top500-emix-regression-audit.md) |
| EMIX 历史结果比较 | 2026-09-11 | 历史跨批次比较，受配置差异限制。 | [历史比较](results/20260911-emix-historical-comparison.md) |
| 消融任务级结果 | 2026-09-16 | 完成；描述性结果。 | [消融结果](results/20260916-ablation-task-level-results.md) |
| 视角实验汇总 | 历史 | 汇总见报告。 | [视角实验](results/viewpoint_experiment_summary.md) |
| TorchCodec / FFmpeg 运行环境 | 2026-10-29 | 环境修复记录，非实验结果。 | [运行环境记录](results/20261029-torchcodec-ffmpeg-runtime.md) |
# sin-v0 实验结果

最后修改：2026-11-02 19:22 CST。

## 统计口径与条件

- 成功率差值为实验组成功率减去同任务 Baseline 成功率，单位为百分点（pp），不是相对百分比。每份正式评估均为 100 回合。
- Patch 正式批次使用 left_oppo_cam、每任务 300 条专家轨迹、训练 seed 0、30,000 步、评估 seed 0；Baseline DP、图像 Mix DP 和实例掩码 Patch DP 各有 10 个任务的完整评估。Patch 使用修正后的任务物体实例掩码。
- Seg 批次使用相同的视角、训练步数和评估回合数；Seg-Mix 与 Seg-Patch 启用 segment_wise。Overlap 0.33 两组还启用重叠检测与 refuse_ratio=0.33。2026-11-02 在 place_burger_fries 两项 Overlap 评估完成后停止任务队列，不再运行后续单任务。

## 有效正式结果：sin-v0-patch-oppo

| 任务 | Baseline DP | 图像 Mix DP | Mix − Baseline | Patch DP | Patch − Baseline |
|---|---:|---:|---:|---:|---:|
| blocks_ranking_rgb | 11/100 (11%) | 14/100 (14%) | +3 pp | 11/100 (11%) | 0 pp |
| blocks_ranking_size | 6/100 (6%) | 2/100 (2%) | -4 pp | 2/100 (2%) | -4 pp |
| place_bread_basket | 22/100 (22%) | 31/100 (31%) | +9 pp | 18/100 (18%) | -4 pp |
| place_bread_skillet | 87/100 (87%) | 93/100 (93%) | +6 pp | 94/100 (94%) | +7 pp |
| place_burger_fries | 18/100 (18%) | 11/100 (11%) | -7 pp | 18/100 (18%) | 0 pp |
| place_can_basket | 56/100 (56%) | 50/100 (50%) | -6 pp | 52/100 (52%) | -4 pp |
| place_cans_plasticbox | 12/100 (12%) | 11/100 (11%) | -1 pp | 4/100 (4%) | -8 pp |
| place_dual_shoes | 30/100 (30%) | 44/100 (44%) | +14 pp | 37/100 (37%) | +7 pp |
| stack_blocks_two | 65/100 (65%) | 43/100 (43%) | -22 pp | 59/100 (59%) | -6 pp |
| stack_bowls_two | 80/100 (80%) | 84/100 (84%) | +4 pp | 88/100 (88%) | +8 pp |
| **10 任务合计** | **387/1000 (38.7%)** | **383/1000 (38.3%)** | **-0.4 pp** | **383/1000 (38.3%)** | **-0.4 pp** |

来源：/root/autodl-tmp/outputs/sin-v0-patch-oppo/formal/eval/ 下 30 份 metrics.json；批次配置见 configs/scripts/run_sin_v0_patch_oppo.yaml，既有归档说明见 docs/results/20261030-sin-v0-patch-results.md。

## 有效正式结果：sin-v0-seg

Seg 没有重新训练 Baseline；下表的 Baseline 均取自上方有效 Patch 正式批次。— 表示没有完整的 100 回合 metrics.json。

| 任务 | 历史 Baseline | Seg-Mix | 差值 | Seg-Patch | 差值 | Seg-Mix + Overlap 0.33 | 差值 | Seg-Patch + Overlap 0.33 | 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blocks_ranking_rgb | 11/100 (11%) | 12/100 (12%) | +1 pp | 12/100 (12%) | +1 pp | 11/100 (11%) | 0 pp | 5/100 (5%) | -6 pp |
| blocks_ranking_size | 6/100 (6%) | — | — | — | — | — | — | — | — |
| place_bread_basket | 22/100 (22%) | — | — | — | — | — | — | — | — |
| place_bread_skillet | 87/100 (87%) | 84/100 (84%) | -3 pp | — | — | — | — | — | — |
| place_burger_fries | 18/100 (18%) | 20/100 (20%) | +2 pp | 16/100 (16%) | -2 pp | 14/100 (14%) | -4 pp | 11/100 (11%) | -7 pp |
| place_can_basket | 56/100 (56%) | 45/100 (45%) | -11 pp | — | — | — | — | — | — |
| place_cans_plasticbox | 12/100 (12%) | — | — | — | — | — | — | — | — |
| place_dual_shoes | 30/100 (30%) | — | — | — | — | — | — | — | — |
| stack_blocks_two | 65/100 (65%) | 52/100 (52%) | -13 pp | — | — | — | — | — | — |
| stack_bowls_two | 80/100 (80%) | — | — | — | — | — | — | — | — |

| Seg 实验组 | 完整任务数 | 成功回合／评估回合（成功率） | 对应任务历史 Baseline | 差值 |
|---|---:|---:|---:|---:|
| Seg-Mix | 5 | 213/500 (42.6%) | 237/500 (47.4%) | -4.8 pp |
| Seg-Patch | 2 | 28/200 (14.0%) | 29/200 (14.5%) | -0.5 pp |
| Seg-Mix + Overlap 0.33 | 2 | 25/200 (12.5%) | 29/200 (14.5%) | -2.0 pp |
| Seg-Patch + Overlap 0.33 | 2 | 16/200 (8.0%) | 29/200 (14.5%) | -6.5 pp |

同两个已完成任务（blocks_ranking_rgb、place_burger_fries）内，Seg-Mix 无 Overlap 为 32/200（16.0%），Overlap 0.33 为 25/200（12.5%），差值 −3.5 pp；Seg-Patch 无 Overlap 为 28/200（14.0%），Overlap 0.33 为 16/200（8.0%），差值 −6.0 pp。这是单次运行的描述性比较。

Seg-Mix 的 stack_bowls_two 已训练但评估中断，仅有 48 段回合视频，没有完整 metrics.json；不计入成功率。其余未出现在表中的 Seg 任务未取得完整结果。

来源：/root/autodl-tmp/outputs/sin-v0-seg/formal/ 下 11 份 metrics.json；配置见 configs/scripts/run_sin_v0_seg_formal_seg_mix.yaml、configs/scripts/run_sin_v0_seg_formal_seg_patch.yaml 及两个 overlap033 配置。

## 作废批次（不得进入方法比较）

| 批次 | 错误原因 | 停止前的完整评估（仅识别作废运行） |
|---|---|---|
| 2026-10-22 首次错误运行 | Patch 只复制单个 patch_target 的像素，与要求的对象集合不符 | place_bread_skillet：Baseline 89/100、Mix 84/100、错误 Patch 87/100；place_can_basket：Baseline 56/100 |
| 2026-10-22 第二次错误运行 | labels != 0 将桌面、地面和机械臂等场景像素也选入，Patch 近似整张 B 图替换 | place_bread_skillet：Baseline 85/100、Mix 95/100、错误 Patch 94/100；place_can_basket：Baseline 50/100 |

来源：docs/results/20261022-sin-v0-patch-bug-invalid-run.md 与 docs/results/20261022-sin-v0-patch-scene-background-bug-invalid-run.md。两批均整体作废；上方所有有效结果不包含这些数字。

## 可比性与证据边界

- 所有正式指标均为单训练 seed 0 和单评估 seed 0 的结果，不支持跨 seed 的稳定性或显著性结论。
- Seg 使用 lerobot_v3 数据目录与重建环境，Patch 正式批次使用较早的 lerobot 数据目录；评估实际接受的场景 seed 也不完全相同。Seg 对历史 Baseline 的百分点差值是跨批次参考，不是严格配对消融。
- 各 Seg 组的完整任务数不同。汇总行各自只统计已有完整评估的任务，不应将不同任务集合的均值直接排序。

## 收尾与产物状态

- 两项 place_burger_fries Overlap 0.33 评估分别于 2026-11-02 17:41 和 18:07 CST 生成完整 metrics.json；两条评估及其 run.py 子进程均已退出，任务队列已停止，未启动后续单任务。
- scripts/run.py 在每项评估完成后自动删除 7 个模型／优化器大文件，每组 8,351,776,600 字节，包含最终 030000 的 model.safetensors 和 optimizer_state.safetensors。因此这两组最终模型权重在人工清理前已不存在，无法按原计划保留；最终模型配置仍在。
- 2026-11-02 收尾时，另清理这两组各 5 个中间检查点目录及最终训练状态元数据。完整 metrics.json、评估视频、配置、日志、原始数据和索引保留。
