# sin-v0-patch（left_oppo_cam）正式批次结果

归档时间：2026-10-30。状态：**有效正式结果**。2026-10-22 两个掩码错误批次均已作废；本报告只包括按任务物体实例掩码实现修正后完成的正式 Patch 批次。

## 实验条件

- 十个任务，每任务使用正式批次的 300 条专家轨迹；训练 seed 0，30,000 steps，batch size 64。
- 三组策略：BaselineDP（vanilla）、EMIX-DP（线性图像 Mix）、EMIX-PatchDP（实例掩码硬拼贴）。每任务评估 100 episodes，evaluation seed 0。
- 对象 Patch 只使用任务物体掩码，排除 background、桌面场景与机器人部件。错误掩码批次不进入汇总。
- 结果为单训练 seed、单评估 seed 的描述性成功率，不据此声称统计显著或跨种子稳定提升。

## 逐任务结果

| 任务 | BaselineDP | EMIX Mix DP | Patch DP | Mix − Baseline（pp） | Patch − Baseline（pp） |
|---|---:|---:|---:|---:|---:|
| blocks_ranking_rgb | 11/100 (11%) | 14/100 (14%) | 11/100 (11%) | +3 pp | +0 pp |
| blocks_ranking_size | 6/100 (6%) | 2/100 (2%) | 2/100 (2%) | -4 pp | -4 pp |
| place_bread_basket | 22/100 (22%) | 31/100 (31%) | 18/100 (18%) | +9 pp | -4 pp |
| place_bread_skillet | 87/100 (87%) | 93/100 (93%) | 94/100 (94%) | +6 pp | +7 pp |
| place_burger_fries | 18/100 (18%) | 11/100 (11%) | 18/100 (18%) | -7 pp | +0 pp |
| place_can_basket | 56/100 (56%) | 50/100 (50%) | 52/100 (52%) | -6 pp | -4 pp |
| place_cans_plasticbox | 12/100 (12%) | 11/100 (11%) | 4/100 (4%) | -1 pp | -8 pp |
| place_dual_shoes | 30/100 (30%) | 44/100 (44%) | 37/100 (37%) | +14 pp | +7 pp |
| stack_blocks_two | 65/100 (65%) | 43/100 (43%) | 59/100 (59%) | -22 pp | -6 pp |
| stack_bowls_two | 80/100 (80%) | 84/100 (84%) | 88/100 (88%) | +4 pp | +8 pp |
| **平均** | **387/1000 (38.7%)** | **383/1000 (38.3%)** | **383/1000 (38.3%)** | **−0.4 pp** | **−0.4 pp** |

## 相对 Baseline 的差值

| 对比 | 总体结果（1000 次评估） | 逐任务方向 |
|---|---:|---|
| Mix − Baseline | **−0.4 pp**（383/1000 − 387/1000） | 5 个提高、0 个持平、5 个降低 |
| Patch − Baseline | **−0.4 pp**（383/1000 − 387/1000） | 3 个提高、2 个持平、5 个降低 |

差值单位为百分点（pp）；逐任务的两列差值见上表。

## 结果归纳

- Patch DP 相对 BaselineDP：3 个任务提高、2 个持平、5 个降低；总体均值低 0.4 个百分点。
- EMIX Mix DP 相对 BaselineDP：5 个任务提高、0 个持平、5 个降低；总体均值低 0.4 个百分点。
- Patch 在 bread skillet、dual shoes、stack bowls 上高于 Baseline；在 RGB blocks 与 burger fries 上持平。不同任务间方向不一致，因此总体结果不支持“Patch 在这批设置下稳定提高成功率”的结论。

## 结果归档

- 远端实体归档：/root/autodl-tmp/archive/sin-v0-patch-oppo-20261030/。
- 归档含 outputs（train/eval 产物与 30 份 metrics）、运行日志及本批 root/per-task YAML 配置。归档时 outputs 约 4.0 GB、logs 约 89 MB；训练评估后已清理大部分 checkpoint，归档保留当时实际存在的文件。
- 可复现数据与索引仍在 /root/autodl-tmp/sin-v0-patch-oppo/formal/lerobot/ 与 .../emix_index/，原件未移动。输出原目录也保留，归档为独立复制。
- 归档 manifest：/root/autodl-tmp/archive/sin-v0-patch-oppo-20261030/ARCHIVE.md。
- 十任务原始 metrics 路径：/root/autodl-tmp/sin-v0-patch-oppo/formal/outputs/eval/<task>_dp_{baseline,emix,patch}/<run>/metrics.json。
