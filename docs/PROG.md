最后修改：2026-11-05

## 2026-11-05 — multi-v0 评估并发扩容

- 单评估时 RTX 5090 使用约 6.7/32.6 GiB 显存、GPU 利用率约 10%，CPU 与内存有充足余量。已暂停顺序队列启动器 PID 1423（当前评估子进程继续运行），由 `scripts/parallel_multi_v0_eval.py` 调度其余 11 个未完成的 mix/patch 评估，避免同一任务被重复启动。
- 当前为原有 1 个评估加调度器 3 个评估，共 4 个并发。稳态采样显存约 23.9/32.6 GiB，GPU 利用率 81%–98%；剩余约 8.2 GiB，不继续增加第 5 个评估以免超出显存。调度器在启动前要求至少 8 GiB 空闲、每次启动间隔 30 秒；并发数由 `outputs/multi-v0/parallel/max_jobs.txt` 控制，当前值 3。
- 调度器逐项检查完成指标，失败任务最多重试一次；全部并行任务结束后自动恢复顺序启动器，使其核对剩余任务并写入总完成标记。日志为 `outputs/multi-v0/parallel/scheduler.log`，任务日志为同目录 `<arm>_<task>.log`。训练步数、评估 seed、episode 数和模型配置未改。 原顺序评估进程 PID 987120 完成后，`scripts/watch_multi_v0_capacity.py` 会确认其 100 回合指标并将调度器并发上限从 3 调到 4，维持总共 4 个评估；若该任务失败则不扩容，由恢复后的顺序启动器处理。

## 2026-11-05 — GIL 故障归因与评估器对齐决策

- 2026-10-30 至 10-31 的 `sin-v0-seg` 评估曾在 `stack_bowls_two_left_oppo_cam` 出现 pybind11 `inc_ref` / `PyGILState_Check` 错误、退出码 139 和长时间卡住。此前为保留进度而加入十次 seed 尝试后重启解释器、超时栈转储、原生异常捕获钩子，以及只放行少数专家异常的定向处理。下方旧记录保留当时事实，不代表当前推荐流程。
- 后续重建并隔离策略与 RoboTwin 仿真环境（`emix-reinstall` / `emix-twin-reinstall`）。截至本次核查，11 月 1 日以来可见的 14 份近期远端日志、包括 `multi-v0` 评估日志，均未出现 GIL / `PyGILState_Check` 崩溃特征；baseline 曾有一次 `UnStableError`，属于物体稳定性异常，不是 GIL 故障。
- 因故障只在旧环境中观察到，重装后未再复现，当前将其归类为**环境侧故障**；具体二进制库或 ABI 根因没有被证明。停止 GIL 定向捕获与十次尝试后的进程轮换，专家异常按 RoboTwin `main` 的跳过 seed 规则处理。评估任务对象复用、专家成功条件在 `close_env()` 后检查，也与该分支对齐。保留历史日志作为追溯依据；若新环境再次出现原生崩溃，重新独立诊断。
- 回滚点：RoboTwin 子仓库空提交 `ee27b45` 与存档分支 `archive/20261105-pre-official-eval-alignment`（代码树仍为 `a02b3e9`），主仓指针提交 `3a7ae9c`。旧评估器另存于 `/root/autodl-tmp/archive/evaluate-pre-official-align-20261105.py`。
- 切换边界：`multi-v0/emix-mix/place_burger_fries_left_oppo_cam` 按旧评估器完整结束并写出 100 回合指标后，才切换为 `robotwin-main-aligned-20261105`。队列下一任务 `place_cans_plasticbox_left_oppo_cam` 开始使用新评估器；旧任务结果不因代码切换而改写。

## 2026-11-01 18:01 CST — 双线方案 A 验证与正式恢复

- 17:32 暂停单线正式队列；`seg_patch` 评估保留 53/100 条记录，下一种子 100163。暂停后无遗留训练/评估进程。
- 独立诊断目录 `outputs/diagnostics/parallel-a-20261101` 中，两条训练各跑 800 步，保持正式 batch size 64 和各 16 个数据加载进程；均完成并产生 checkpoint。并行稳态各约 8.1 步/秒，合计约 16.2 步/秒；此前正式单训练约 13.1 步/秒。短跑与完整 30,000 步的吞吐未必完全相同。
- 两条诊断评估各完成 2 回合，合计 4 回合耗时 9 分 37 秒，峰值显存约 12.6/32 GB；未观察到 OOM 或进程错误。与先前单评估约 3.1 分钟/回合相比，合计吞吐有提升，但任务种子和短样本不同，不把诊断成功率当作正式结果。
- 训练＋评估同时运行的诊断也分别写出 800 步 checkpoint 和完整 1 回合指标。随后误重复启动相同诊断输出路径，训练器按设计以 `FileExistsError` 拒绝覆盖；正式输出未受影响。
- 因方案 A 的训练、评估及混合阶段均可运行且观察到合计吞吐收益，跳过方案 B。正式编排已设为每个任务最多并行两个臂，四臂完成后才进入下一个任务。18:01 重启后，已完成的 `seg_mix` 被复用；`seg_patch` 续评估与 `seg_mix_overlap033` 训练同时运行。完整结果仍以各臂 100 回合 metrics 为准。

## 2026-11-01 11:20 CST — 当前五任务四臂正式队列

用户已将范围改为现有配置的前五个任务，并要求一个任务的四臂依次完成。
远端 tmux `sin-v0-seg-task-first` 运行
`scripts/run_seg_task_first.py`，使用隔离的 `emix-reinstall` 策略环境与
`emix-twin-reinstall` 仿真环境。旧十任务、按臂推进的巡检记录仅是历史状态；
本机旧小时巡检和 handoff 已关闭。

首任务 `blocks_ranking_rgb_left_oppo_cam` 的 `seg_mix` 训练已完成。
评估在 43/100 回合、种子 100125 处因空抓取目标断言退出。核对显示
RoboTwin 10 月 28 日的变更只涉及分割对象命名；近期收紧的 EMIX 评估异常
处理误将预期的空移动目标视作致命异常。已恢复该类无效专家种子的跳过逻辑，
GIL 和其他未知异常继续上报。11:20 重启正式队列，复用训练检查点和评估
进度；当前完整指标尚未产生。以远端日志和 metrics 为准。

## 当前实验交接：sin-v0-seg（核验于 2026-10-30 13:32 CST）

**目标与背景。** 本批评估阶段与操作对象约束是否改善 EMIX 增强。阶段边界按夹爪动作结束点标注：闭合稳定点结束 PICK，放松稳定点结束 PLACE；阶段包含动作过程及相应搬运过程，Segment 对应的 object 是该阶段实际操作物体。阶段定义根据十个任务程序审阅并标注，不由夹爪信号单独猜对象。EmixIndex 使用规范化 v3，增强只从同 `segment.id`、同 `segment.object` 的候选中选取。

**实验臂与参照。** 四个新处理臂按单 GPU 串行运行：`seg_mix`、`seg_patch`、`seg_mix_overlap033`、`seg_patch_overlap033`，每臂十任务。segment-wise Mix 只对阶段目标对象区域做线性混合；Patch 对该对象区域硬拼贴。overlap033 臂在候选对象掩码 IoU > 0.33 时拒绝并顺延候选；候选均超限时选最低 IoU 并记录 fallback。BaselineDP 不重训，复用 sin-v0-patch 十任务的 100 集结果（387/1000，38.7%）。因此结果比较应同时列各臂原始成功数/评估数、成功率，以及该臂减 Baseline 的百分点差值。

**固定条件与运行顺序。** 每个训练任务 seed 0、30,000 steps；评估 100 episodes、seed 0。四臂顺序为 `seg_mix → seg_patch → seg_mix_overlap033 → seg_patch_overlap033`，每臂内按任务配置顺序串行。正式配置和 runbook 位于 `configs/scripts/run_sin_v0_seg_formal*.yaml` 及对应任务子目录；输出在 `outputs/sin-v0-seg/formal/`。数据副本与 v3 索引在 `/root/autodl-tmp/sin-v0-seg/formal/`，不修改 Patch 基线归档或原始数据。

**当前进度。** 启动器 PID 147276，主日志 `/root/autodl-tmp/sin-v0-seg/formal/logs/run_all_arms.log`。截至 13:32，只有 `seg_mix` 在运行，完成 2/10 个任务（2/40 份新臂指标）：`place_bread_skillet` 为 84/100 (84%)，对应复用 Baseline 87/100 (87%)，差值 −3 pp；`place_can_basket` 为 45/100 (45%)，Baseline 56/100 (56%)，差值 −11 pp。`stack_blocks_two` 已训练完成，正在评估，flow 日志到 episode 22/100；`seg_patch` 及两个 overlap 臂尚未开始。该进度是部分结果，不能代表整批结论。核验时评估进程活跃、日志更新，GPU 0 利用率 13%、显存约 2.0 GiB；评估仿真有 CPU 开销，低 GPU 利用率本身不等于训练故障。

**接手检查入口。** 先检查上述启动器、主日志、对应 `outputs/.../metrics.json` 和当前进程，再确认运行臂/任务是否按串行顺序推进。最新状态以远端进程、flow 日志和完整 metrics 为准；`docs/PROG.md` 下方按时间追加巡检记录。不得重复启动实验或并行占用 GPU；若修复影响四臂协议一致性，先记录并比较四份配置与快照。实验巡检器每小时在本机触发 Codex，通过 `ssh emix` 检查远端；WSL Codex 因服务端 403 未承担该任务。


## 2026-10-30：Patch 批次结果归档；sin-v0-seg 小时巡检已挂载

sin-v0-patch 有效正式批次的 BaselineDP / EMIX Mix DP / Patch DP 十任务指标已统一核对，每臂均为 1,000 次评估。总体成功数为 387/1000、383/1000、383/1000（38.7%、38.3%、38.3%）；Patch 相对 baseline 3 胜、2 平、5 负，单 seed 结果只作描述性报告。详情与结论边界见 docs/RESULTS.md 和 docs/results/20261030-sin-v0-patch-results.md。30 份 metrics、当时留存的训练/评估产物、日志和配置已独立复制到 /root/autodl-tmp/archive/sin-v0-patch-oppo-20261030/；源输出、数据与索引未移动。

sin-v0-seg 小时巡检在 WSL 中无法使用 Codex：Codex CLI 登录刷新被服务端以 unsupported_country_region_territory（403）拒绝，因此未在 WSL 安装无效巡检任务，也保留其原有 crontab。按用户的本机回退要求，已在本机 scratch 目录运行 Codex 的只读 SSH 检查，并建立 tmux 会话 sin-v0-seg-watch。WSL 本身可用，但 macOS crontab 安装在当前执行环境挂起；现改用用户级 LaunchAgent，每 3600 秒触发同一个 Codex SSH 巡检脚本。脚本只通过 ssh emix 访问/修复项目；本机文件仅保存 prompt、日志和调度配置。会话可用 tmux attach -t sin-v0-seg-watch 查看。巡检规则保留四臂配置和训练协议一致性，不为追 GPU 利用率而改动 batch/seed/optimizer/num_workers 等控制条件。

## 2026-10-30：sin-v0-seg 四臂训练/评估启动

本批固定四个新处理臂（每臂十任务、seed 0、30,000 steps、每任务评估 100 集）：Segment-wise Mix、Segment-wise Patch、Segment-wise Mix + overlap refuse 0.33、Segment-wise Patch + overlap refuse 0.33。四臂都以 v3 索引筛选同阶段且 segment.object 相同的候选；Mix 仅对阶段目标物体区域做线性混合，Patch 对该物体区域硬拼贴。overlap 指两帧该对象掩码的 IoU；高于阈值即拒绝并继续找候选，候选均超阈值时取最低 IoU 并记录 fallback。实现和四个 runbook 已提交为 commit 33e6579；输出根位于 outputs/sin-v0-seg/formal/。

BaselineDP 不重训、不重评，直接引用 sin-v0-patch 正式批次的十个 100 集 metrics（共 387/1000，38.7%）：RGB 11%、size 6%、bread basket 22%、bread skillet 87%、burger fries 18%、can basket 56%、cans plasticbox 12%、dual shoes 30%、stack blocks 65%、stack bowls 80%。指标文件位于 Patch 批次 formal/outputs/eval/*_dp_baseline/*/metrics.json。

四臂启动器 PID 147276，日志为 /root/autodl-tmp/sin-v0-seg/formal/logs/run_all_arms.log；2026-10-30 03:56 CST 从 seg_mix 开始，当前训练首任务 place_bread_skillet_left_oppo_cam。首次启动因模型输出目录落在项目 outputs 根外，被路径保护拒绝且未开始训练；配置已改到 /root/EMIX/outputs/sin-v0-seg/formal/ 并重启。已进入训练优化步骤：seg_mix 首任务约 400/30,000 steps，RTX 5090 利用率约 66%、显存约 5.9 GiB，loss 持续更新；按当前速度首任务训练约需 40 分钟（不含评估）。EXP.md 登记了比较矩阵与共享 baseline。

## 2026-10-30：阶段标签端点修正、EmixIndex v3 规范化与十任务索引重建

阶段边界按用户修正为夹爪命令结束点：PICK 的闭合信号首次到达 0 且下一帧完全不变；PLACE 的放松信号首次到达 1 且下一帧不变。端点帧归入该 PICK/PLACE 阶段，下一阶段从下一帧开始。每个任务按 task code 的固定操作顺序，为每个对象生成 PICK 与 PLACE 两段；最终 PLACE 后回程留空。

commit f5929d1 已提交阶段标签生成器、十任务对象顺序配方与标注说明。全部 3000 个 episode 的动作序列均与对应任务代码预期 PICK/PLACE 次数和顺序一致；每个端点信号均有稳定平台，自动阶段边界可由动作数据确定。新标签写入独立目录 formal/segment_annotations/<task>/；先前使用中点边界的草稿保留在 formal/segment_annotations_midpoint_20261030/，不可用于后续构建。

为避免改写 Patch 批次原件，以硬链接复制十个 LeRobot 数据集到 formal/lerobot_v3/<task>/，并在副本中将历史渲染 actor 名称映射到任务的规范对象角色、加入 segment_id 与 segment_object。映射依据任务代码 actor 顺序及原始 instance_objects 表；10 个任务每个 300 集的渲染 actor 映射一致。原始数据和已有 v2 索引保持不变。

新增规范模块 embodied_mixup/candidates/emix_index_spec.py，以 TypedDict、JSON Schema 与增量 validator 定义 v3；emix_index.py、streaming builder 与保存/加载路径接入校验。当前构建器只写 v3、v2 保留为历史只读格式；说明见 docs/dev/EMIX_INDEX_SPEC.md。docs/EXP.md 新增实验代号 sin-v0-seg 的对照计划和开训门槛。

十任务索引已按原有构建参数重建到 formal/emix_index_v3/<task>/emix_index.json，并逐个通过 EmixIndex v3 完整结构校验。每个索引均含 300,000 组、正确的 V3.0 数据来源、启用阶段标注的配置和 2–6 个阶段定义；所有任务均有可用的同阶段候选。十任务合计 3,000,000 组，其中 1,286,164 组的全部成员具有完全匹配的 segment.id 与 segment.object。训练尚未启动，需继续完成训练配置差异审计。阶段标签实现已提交为 f5929d1；v3 规范、数据准备器与进展记录已提交为 6e4ca4a。

逐任务 segment-wise 候选组数：blocks_ranking_rgb 95,981；blocks_ranking_size 95,825；place_bread_basket 125,281；place_bread_skillet 88,315；place_burger_fries 128,278；place_can_basket 37,801；place_cans_plasticbox 130,614；place_dual_shoes 134,382；stack_blocks_two 215,177；stack_bowls_two 234,510。原始数据及 v2 索引保留不变；v3 数据副本和索引分别位于 formal/lerobot_v3/ 与 formal/emix_index_v3/。


## 2026-10-22 12:34 CST：Patch 第二次 BUG 批次作废并重启

本批曾将 B 图的 `labels != 0` 当作“全部非背景物体”。真实标注中的桌面、地面与机器人等场景像素也是非零；十任务 episode 0 的抽查帧均为 100% 非零，因此 Patch 近似整张 B 图，不能作为有效拼贴实验。2026-10-22 02:26 开始的正式批次已停止，整个批次标记为 BUG 错误结果。完整 100-episode 数字仅供识别：`place_bread_skillet_left_oppo_cam` vanilla/mix/错误 Patch 为 85/95/94；`place_can_basket_left_oppo_cam` vanilla 为 50。第二任务 mix 仅训练到约 19,355/30,000 步。简要记录见 `docs/results/20261022-sin-v0-patch-scene-background-bug-invalid-run.md`，不得用于方法比较。

最新提交 `5fc08ec` 按实例名称选择任务物体，排除场景表面与机器人；十任务样本帧的新掩码面积约 0.7%–6.6%，均同时含拼贴物体与保留的 A 图背景。相关 9 项测试通过。旧 checkpoint 与评估产物清理后，使用原 `sin-v0-patch` / `261020` 协议、原训练数据与索引，从首任务重新训练三臂。旧巡检关闭，建立新版半小时巡检。下方 02:13 记录描述的是更早的单物体 BUG，已非当前实现。

## 2026-10-22 02:13:08 CST
**Patch 拼贴的目标范围与遮挡关系问题（已观察，尚未修复）**

**当前实现说明：**每个任务通过 segment_object_actors() 给被操作对象提供规范逻辑名；patch_target_actor() 仅标记 Patch 实验选定对象，不另造特殊对象名。新采集对象表用这些规范名记录任务物体，场景表面和机器人部件仍用 scene_* 名称。Patch loader 对所有规范任务对象生成合并掩码，历史标注中的 patch_target 不再兼容；转换器和读取器会要求重新生成规范对象名标注。训练加载器以配对 member 0 的图像为 A（底图），member 1 的图像为 B（来源图），然后执行 image_out = torch.where(mask_B, image_B, image_A)。掩码内 B 直接覆盖 A 的同坐标像素，没有跨图深度比较、遮挡顺序判断或物体完整形状重建；交换 A/B 会改变结果。

**简单直觉版本：**先选定 B 图里的一个物体，只把它露在画面中的部分剪出来，在原位置盖到 A 图上。不会把 B 中所有物体都剪来，也不会判断 A 中已有物体是否应该挡在它前面。谁作为底图、谁提供剪下的像素，会影响结果。

**对应观察：**有些物体或物体区域没有被拼上；拼上的目标像素又可能盖住 A 中其他物体。边缘差异还可能来自 RGB 双线性缩放与实例标签最近邻缩放。若目标 actor 的大块可见区域缺失，仍需核对该帧实例标签才能定位具体原因。

## 0916
**消融套件三批正式归档(SOP-06):过滤消融全部持平,默认构建仍最优**

三个消融批次(homefilter、timeratio、combo)按 SOP-06 §2 完成归档。注意:今日早间条目所记「已于 18:24 整体归档」当时**并未落盘**——本会话开始时 outputs/run_top500-emix-* 与各批 176G checkpoints 均仍在,归档目录只有 SUMMARY 骨架;实际搬移与删除在本会话执行完毕。`archive/DP-10tasks-500ep-emix-{homefilter,timeratio,combo}-cen-arm-top-cam-20260911/`(目录名沿用批次启动日)现各含 eval(train 后视频每 run 只留前 10)/train(删 checkpoints,3×176G≈528G)/vis/summary/数据索引(各批自己的 emix_index 归入 data/)与补全的 SUMMARY.md(归档 commit `9e7fe8c`)。live data/ 已清空(data/lerobot 与 baseline 归档批逐文件相同,删冗余副本),df 1.1T→500G used。

结果(100ep seed 0,对照 top500-emixfix):

| 批次 | 唯一变量 | EMIX TOTAL | vs emixfix 46.1 | vs DP baseline 43.8 |
|---|---|---|---|---|
| top500-emixfix(主对照) | 无(默认构建) | 46.1% | — | +2.3 |
| homefilter | exclude_start_radius 0.02 | 43.7% | -2.4 | -0.1 |
| timeratio | max_pair_time_ratio 0.2 | 45.2% | -0.9 | +1.4 |
| combo | 两者叠加 | 44.4% | -1.7 | +0.6 |

结论:三种过滤配置 TOTAL 差异均在 ±5pt 噪声带内、互为持平——home 簇配对与长跨度配对在 500ep+修复构建下**均无实质危害**,叠加也无叠加收益;0904 归因的两类"伪样本"假设不成立。**top500-emixfix 默认构建仍是当前最优,后续批次沿用**。单点观察(不下结论):timeratio 的 can_basket(移除率 81%)Δ+5 方向符合假设;combo 的 burger_fries −14 为最大单任务偏离。

遗留两件:① outputs/intersections{,_timeratio} 为今日重渲染产物,内容与归档 vis 重复(196M),未删,待手动清理;② 早间条目的 run.py max_points/backfill_vis.py 改动仍未提交;~~intersections_combo 渲染缺失~~已补齐:渲染与 data/lerobot 清空并发,后 15 组改读 baseline 归档数据集(`archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911/data/lerobot`,与原 data/lerobot 同一份数据,只读),backfill_vis.py 为此加了 `--dataset-root` 覆盖;现三目录各 10/10 组 HTML+PNG 校验齐全。

## 0916
**消融套件归档,可视化改为索引落盘即出、并加了性能闸门**

三个消融批次(homefilter、timeratio、combo)已于 18:24 整体归档:outputs/run_top500-emix-* 连同各自的 vis 与索引树一起进了 archive/,live data/ 只剩 data/lerobot 十个 84x84 数据集。随后把「可视化」这一环补成了硬性保证:编排脚本必须做到**建完 EmixIndex(或发现索引已存在)立刻出对应的可视化**,且渲染必须限定最大绘制点数,否则耗时不可接受。

具体改动:run.py 的 vis 阶段现在显式传 `--max-points`(配置键 `visualization.max_points`,默认 10000)与 `--skip-click-images`(`visualization.skip_click_images`,默认 true),五个 runbook 配置都写入了这两个键;单跑 `--phase index` 结束后自动接 vis 阶段;completed_vis 对无 assets 目录的渲染也能正确复用(跳过点击图时不产生 assets)。dataflow.sh 本来就带这两个参数,未动。另新建 scripts/backfill_vis.py(无锁、串行、CPU-only,支持多 --config 与 --index-root/--output-root/--max-points 覆盖),删掉了 scripts/temp/ 里引用已不存在的 scripts.flow 的坏旧版。

对当前数据的可视化已全部生成:三个索引变体 × 10 任务共 30 组 HTML+PNG,从归档索引**只读**渲染(索引与数据本就一一对应,不重建),输出在 outputs/intersections(homefilter 构建,即原 data/emix_index)、outputs/intersections_timeratio、outputs/intersections_combo。单任务实测约 56 秒(旧渲染含点击图预计算要 9-17 分钟),性能取舍是新 HTML 不再带点击查看原图面板,三张分析图(3D 交点、2D 投影、簇直方图)与 PNG 不受影响。

## 0911
**三个消融批次合并成一个套件自动跑**

前两组(homefilter、timeratio)的可视化检查通过后,把第三组也加了进来,三个批次用一个固定脚本按顺序自动跑完:检查数据是否齐全(缺了才重新采集)→建索引→出可视化→训练→评测→自动写结果表。每批跑完两卡再跑下一批,预计全部结束需要 4-6 天。

**第三组叫 combo,把前两组的删除规则叠加**:既删起点 2cm 内的配对,又删时间跨度超过平均轨迹长度 20% 的配对。想回答的问题是:空载等待的样本和进度差太远的样本,一起删掉是不是比单独删更好。这里是控制变量设计:两个过滤的参数与单变量组完全一致(0.02m 同 homefilter,0.2 同 timeratio),本组唯一的新变量是「两者同时生效」;最初把时间比例写成 0.15,如果那样跑,结果变好也说不清是叠加的功劳还是参数更紧的功劳,已改回 0.2。另外 timeratio 本身也有过一次改动:比例最初设 60%,后收紧到 20% 并重建了索引,现在跑的是 20% 版本。

删掉的比例在各任务上差别很大,这一点值得留意:0.2 比例下单罐放篮子删了 81%(它恰好是上一批 EMIX 提升最大的任务),叠方块删了约 65%,而有的任务只删 1-2%。如果这批结果变化集中在删除比例大的任务上,就说明时间跨度确实抓到了东西;如果到处都差不多,这个维度可能没那么重要。

目录规范上走了一段弯路:输出一度放到顶层 `runs/` 目录,与"实验输出必须在 outputs/ 下"的既有规范冲突,已改回 `outputs/run_<批次名>/{vis,train,eval,summary}/`,路径校验也恢复为只认 outputs/。已渲染的可视化随之迁移,无损失。

## 0911
**为什么做 homefilter 和 timeratio 这两组实验**

现在有一个猜想：**EMIX 可能比较擅长一次抓放，但任务要连续抓放几次时，它可能分不清“已经做完什么、接下来该做什么”，把前面得到的好处丢掉了。** 这个猜想来自 top500-emixfix 这批结果（commit `3d3c51e`，run `flow-20260907-222045`），还没有得到证实。

比如，任务刚开始时，手臂在起点；放完第一块面包后，手臂也可能回到差不多的位置。但这两张图里的面包数量、位置和任务进度不同。把它们叠起来训练，可能会让模型搞不清下一步。即使第一轮抓得更好了，第二轮出错，整个任务还是失败。

因此做下面两组实验。两组都使用原来 top500-emixfix 的数据，只重新训练 EMIX，普通 DP（baseline）沿用归档结果。每组只改一种配对筛选规则，方便看出是哪种改动有用。

**第一组：homefilter，去掉公共起点附近的配对。**

这组想检查：问题是否主要出在“手臂位置差不多，但任务进度不同”的起点附近。具体做法是删掉距离公共起点小于 2cm 的配对。之前观察到，很多配对集中在手臂空载、返回或等待的位置，真正抓取和放置的位置反而少。

批次名是 `top500-emix-homefilter`。数据从归档复制回 `data/lerobot`，不重新采集。在已检查的三个 500 条轨迹任务里，每条轨迹的起点都一样，约为 `(-0.001,-0.395,0.953)`，标准差为 0；起点 2cm 内的帧占全部帧的 2–6%。所以先把过滤半径设为 `0.02m`，与 `xyz_epsilon` 相同，也可以用 `--exclude_start_radius` 修改。

已实现 `exclude_start_radius`，包括配置、命令行和两种索引构建方式；索引记录中会保存 `start_point_xyz` 和 `start_filter_removed`。计划评测 100 个场景，seed 为 0。与旧 baseline 比较时，要核对实际场景清单；配置里都写 seed 0，不代表两次最终选中的场景一定相同。

**第二组：timeratio，去掉时间相差太远的配对。**

这组想检查：问题是否来自把“刚开始做”和“快做完了”的画面混在一起。之前的统计发现，38–59% 的配对中，两点在各自轨迹里的进度相差超过 30%。例如，一条轨迹刚出发的画面，可能与另一条轨迹快结束的画面配在一起，因为手臂都在起点附近。

批次名是 `top500-emix-timeratio`。规则是：两点在各自轨迹里的时间位置之差，不能超过本批所有轨迹平均长度的 60%。参数为 `max_pair_time_ratio=0.6`，可用 `--max_pair_time_ratio` 修改。这组不加起点过滤。homefilter 查的是“配对发生在哪里”，timeratio 查的是“两个画面在任务过程里相隔多远”。

本次记录时已完成：功能提交 `f8e555a`、10 个任务的索引重建（暂存 `/tmp/tr_index`），以及 SUMMARY 的初始内容。可视化排在 homefilter 后面，两组共用 flow 锁，按顺序运行。输出放在 `outputs/intersections_timeratio/`，避免和另一批混在一起。

两组过滤都会改变模型看到哪些样本、各类样本出现多少次。**如果成功率提高，也不能马上认定是“进度判断”被修好了，还要看改善发生在第几轮、什么动作上。**

**后面还可以试：两张图本身是否真的有足够大的差别。**

现在找到的配对满足“手臂姿态接近、接下来动作不同”，但两张图可能看起来几乎一样。如果图里没有足够的信息解释动作为什么不同，强行拿来训练，未必能帮助模型学会看图。后面可以考虑少用这类配对，或者把它们筛掉。

先把配对按图像差异从小到大分组，看看原图、叠起来的图和对应动作。重点看：变的是物体位置、剩余物体、任务进度，还是只有背景、光照、机械臂外观变了。也要比较 homefilter 和 timeratio 是否已经改变了这些配对的比例。

**不是图像差得越大越好。** 有时只有一个小物体移动了，但它正好决定下一步；有时整张图亮了很多，却与动作无关。如果看图后觉得这个方向有希望，再单独试图像差异筛选。还要和“随机删掉同样数量的配对”比较，确认好处来自筛选规则，而不只是配对变少了。这是后续想法，目前还没做，也不放进正在进行的两组实验里。

**哪些结果支持这个猜想，哪些结果不支持**

这里一次完整的“抓起来再放下”算一轮，不把接近、抓取、搬运、释放各算一轮。下面的差值都是 EMIX 减去 baseline，单位为百分点。

| 抓放轮数 | 任务及差值 | 任务数 | 平均差值（每个任务权重相同） |
|---|---|---:|---:|
| 1 轮 | 单罐放篮子 +15；面包放平底锅 +5 | 2 | +10 |
| 2 轮 | 双鞋放置 +11；双罐放盒子 +5；汉堡薯条放置 +1；叠碗 0；叠方块 −4；面包放篮子 −8 | 6 | +0.83 |
| 3 轮 | 按大小排方块 0；按颜色排方块 −2 | 2 | −1 |

三个回退任务都要连续处理多个物体，而且轮数越多，这批任务的平均收益越小。这是支持猜想的线索。但双鞋也要做两轮，却提升了 11 点，说明**多轮任务不一定变差，也不一定只能得到很小的提升。**

下一步要在相同评测场景里，分别看第一轮、第二轮、第三轮。最想确认的是：有没有“第一轮比 baseline 好，第二轮反而差”的情况。第二轮成功率要在第一轮已经成功的样本中算，不能把第一轮就失败的样本都当成第二轮失败。还要记录第一次出错发生在哪里：重复抓已经处理的物体、漏掉剩余物体、做完一轮后停住，还是知道该抓谁但没抓准、没放准。

如果第一轮就明显变差，或者主要是抓放不准，就不支持“模型搞不清任务进度”这个解释。两条轨迹动作不同，也不代表它们一定处于不同阶段，更不能直接说其中一个动作是错的。

目前每组只有 2、6、2 个任务，任务难度、相机视角和放置精度要求也不同。按大小排序的任务，两种方法都只有 3%，几乎都没学会，也很难据此比较。还没有换多个训练随机种子重复实验，所以这些平均数只能提出问题，不能证明原因。

抓放轮数来自归档专家数据里夹爪开闭的统计，成功率来自原始 metrics。详细数据见[当前实验检查](results/20260911-top500-emix-regression-audit.md)和[历史结果对比](results/20260911-emix-historical-comparison.md)。

## 0911
**500 条轨迹、俯视相机这批实验已经归档**

批次名为 `top500-emixfix`。归档位置是 `archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911`，commit 为 `3d3c51e`，run 为 `flow-20260907-222045`。

baseline 总成功率为 43.8%，EMIX 为 46.1%，相差 **+2.3 个百分点**。之前 200 条轨迹那批是 −2.9 点，这次变成正数。说明扩大工作区后 EMIX 并不是在所有情况下都没用，但还不知道这次改善分别有多少来自数据增加、配对规则变化和索引构建修复。

按实际差值统计是 5 个任务提高、3 个降低、2 个持平。旧记录写成了“5 胜 2 负 3 平”，这里更正。单罐放篮子 +15、双鞋 +11 值得关注；面包放篮子仍低了 8 点，之前 200 条轨迹时低了 11 点。按大小排序双方都是 3%，任务本身仍没有学好。

旧记录用“超过 ±5 点”判断结果是否可信，这个办法没有统计依据。现在只能把这些当作本批观察到的差异，还不能说已经证明稳定提升或回退。和 200 条轨迹相比，baseline 平均提高 2.5 点、EMIX 提高 7.7 点，两者差距改善了 5.2 点；但同时改了多个条件，不能把它全部算到某一个改动上。旧构建在 500 条轨迹规模下无法完成，所以也没有“500 条轨迹＋旧构建”的结果可比较。

每个任务评测 100 次，配置 seed 为 0。数值为成功率百分比，括号里是成功次数：

| 任务 | baseline | EMIX | 相差多少个百分点 |
|---|---|---|---|
| blocks_ranking_rgb | 28 (28/100) | 26 (26/100) | −2 |
| blocks_ranking_size | 3 (3/100) | 3 (3/100) | 0 |
| place_bread_basket | 45 (45/100) | 37 (37/100) | −8 |
| place_bread_skillet | 87 (87/100) | 92 (92/100) | +5 |
| place_burger_fries | 36 (36/100) | 37 (37/100) | +1 |
| place_can_basket | 49 (49/100) | 64 (64/100) | +15 |
| place_cans_plasticbox | 20 (20/100) | 25 (25/100) | +5 |
| place_dual_shoes | 33 (33/100) | 44 (44/100) | +11 |
| stack_blocks_two | 49 (49/100) | 45 (45/100) | −4 |
| stack_bowls_two | 88 (88/100) | 88 (88/100) | 0 |
| **全部任务** | **43.8** (438/1000) | **46.1** (461/1000) | **+2.3** |

归档时已经清空 `data/{robotwin,lerobot,emix_index}` 和 `outputs/{train,eval,intersections}`。模型检查点已删除，释放约 351G；每个评测 run 只保留前 10 段视频。

## 0907
**启动 500 条轨迹的俯视相机实验**

这次用修复后的索引构建方式重新跑完整流程。新方式分批处理候选配对，避免一次占用太多内存；每个任务最多保留“轨迹数 × 1000”个候选对，所以 500 条轨迹最多保留 50 万对。这是整个任务的总上限，不是每条轨迹都分到 1000 对。上限和随机种子会写进索引记录。

本批参数为：位置差小于 `0.02m`、姿态差小于 `0.26rad`，并去掉每条轨迹最后 `0.1s` 的配对候选。旧记录说“与 200 条轨迹那批的配对阈值不变”，后来的[历史核对](results/20260911-emix-historical-comparison.md)发现不对：旧索引实际使用 `0.01rad`。因此两批不只是数据量和构建方式不同。

RGB 和 size 排序直接复用 0813–0906 已有的 500 条轨迹和修复后的索引，每个任务 50 万组，已检查可视化并通过新旧构建结果的一致性测试。其余 8 个任务由 `assets.py` 排队重新采集，每个任务 500 条，当时预计需要 2–3 天。

采集之后由 `run.py` 继续：建立索引（当时预计每任务约 8 分钟）、生成可视化（`outputs/intersections`，HTML 和 PNG）、训练和评测。普通 DP 与 EMIX 各训练 30,000 步，各评测 100 次，seed 为 0。

启动时的记录：

- 主仓库 commit `3d3c51e`，子模块 commit `61798c9`，`episode_num=500`。
- tmux 会话为 `top500emixfix`。
- 命令为 `assets.py --variant cen_arm_top_cam --gpus 0 1 --episodes 500 && run.py --config configs/scripts/run_cen_arm_top_cam_500ep.yaml`。
- 采集日志在 tmux 窗口和 `logs/top500emixfix-tmux.log`，通过 pipe-pane 保存。启动时 flow run ID 尚未生成，需等采集结束后查看 `logs/flow-*`；最终 ID 见 0911 归档记录。
- 详细配置在 `run_cen_arm_top_cam_500ep.yaml` 和各任务的 `train_config_snapshot.json`：`steps=30000`、`batch=64`、`lr=1e-4`、`resnet18`、`horizon=16`、`n_obs=1`、`n_act=8`、`seed=0`。
- EMIX 使用 `alpha=beta(2)`，本体状态取第一个配对样本（anchor），损失权重 `1.0`；选择 EMIX 批次的概率按 logarithmic scheduler 从 `0.6` 降到 `0`。评测保存视频。
- 旧对照归档为 `archive/DP-10tasks-200ep-emix-baseline-cen-arm-top-cam-20260904`，当时 EMIX 平均低 2.9 个百分点。

## 0904
**检查为什么扩大工作区后，EMIX 的好处变少了**

当时检查了 right_wide 和 top 两批的配对索引，以及机械臂末端的位姿，发现三个值得追查的问题。下面保留当时的数字，但它们只是线索，还没有证明回退原因。

1. **手腕方向要求可能太严。** 两点位置要相差不到 2cm，手腕方向还要相差不到 `0.01rad`，约 0.57°。当时统计的位置候选通过率有 94–99%，但加上方向要求后只剩 17–66%。保留下来的方向差有 95% 不超过 `0.0099rad`，已经很接近门限。扩大工作区后，手臂从不同方向接近物体，方向更难相同，可能因此更容易选中起点附近，而不是实际操作的位置。
2. **配对可能把不同进度混在一起。** 38–59% 的配对中，两点在各自轨迹里的进度相差超过 30%。例如一条轨迹刚开始，另一条快结束，但手臂碰巧都在起点附近。叠图后同时监督“出发抓东西”和“回去等待”，可能影响模型判断。旧记录直接写成“教出了不动的模式”，现在还没有足够证据确认这一点。
3. **真正抓放的位置选得少。** 当时配对触及的原始帧只有 2–16%，而且集中在没有拿物体的时段。训练初期约 60% 的批次使用 EMIX，这可能让模型花很多时间学这些位置，实际抓取和放置反而练得不够。这里的 60% 是初期概率，不是整个训练的平均比例。

当时的背景是：旧批中面包放平底锅达到 32→95，单罐放篮子达到 44→66；换成居中手臂、扩大工作区和 200 条轨迹后，三个机位批次的平均差值分别为 +1、+0.6、−2.9 点。旧批被记录为约 500 条轨迹、较窄的物体分布，但归档缺少对应数据，不能独立核实所有条件。

旧记录认为“配对看的是机械臂末端位置，所以相机没有影响”，这句话不成立。配对选择可以不依赖相机，但混合出来的图像仍然受视角影响。另外，也不能直接把 ±5 点内的差值都当作没有效果，仍需重复训练和公平评测。

当天做了两个改动，不需要重新采集数据：

- 把 `rpy_epsilon` 从 `0.01` 放宽到 `0.26`，约从 0.57° 放宽到 15°，代码在 `embodied_mixup/configs/config.py`。当时测得方向筛选的通过率：bread_skillet 从 33.4% 到 96.1%，can_basket 从 17.1% 到 72.0%，stack_bowls 从 66.3% 到 98.2%。通过率提高不代表这些配对一定更有用。
- 新增 `exclude_tail_seconds`，默认 `0.1s`，按当时 `dataset_fps=17` 约为末尾 2 帧，让这些帧不参与配对。改动覆盖 config、`candidates/pairs.py` 两种构建方式和 CLI 的 `--exclude_tail_seconds`、`--dataset_fps`；索引的 `config_snapshot` 也记录新参数。这个范围很小，不能保证去掉所有开始与结束的配对；当时考虑必要时再试 1–2 秒。

位置门限仍是 2cm。改动的想法是让实际抓放位置更容易配上对，同时少混入收尾画面。两个参数都可以从命令行修改，旧索引需要重建。当天计划先用归档的 200 条轨迹重建索引，比较配对数量、涉及哪些帧、跨进度配对的比例，再决定是否重训。

## 0820
**修复图像问题后，成功率上升，但部分任务仍然回退**

修复 OIDN 错误后，成功率明显上升。第一版结果使用 seed 0，每个任务评测 100 次。以下均为 baseline→EMIX：

- 提高：place_bread_skillet 32→95，place_can_basket 44→66，stack_blocks_two 65→70。
- 降低：blocks_ranking_rgb 38→27，place_bread_basket 70→66，place_burger_fries 73→71。
- 持平：blocks_ranking_size 4→4。
- 写这条记录时，place_cans_plasticbox 和 stack_bowls_two 的 EMIX 还没评完，flow 正在 5060 上跑 stack_bowls_two。

准备对三个回退任务补测 seed 1 和 seed 2，各 100 次，每个任务共补 200 次。这里只换评测随机种子，不重新训练，而且只评 EMIX。baseline 沿用旧结果只能作为参考，不能当成相同场景下的新对照。

新增脚本 `scripts/eval_reseed.sh`，复用 flow 生成的 `emix_eval.yaml`，从命令行覆盖 episodes、seed 和 output-dir。输出放在 `outputs/eval_reseed/<task>_dp_emix_seed{1,2}/`。seed 1 在 4070 的 `emix-twin` 环境立即运行；seed 2 在 5060 的 `emix-twin-5060` 环境排队，等 flow 主进程结束再运行。两个环境都去掉了 `LD_LIBRARY_PATH` 中的 `/usr/local/cuda`。当时说的“卡 0”是 `nvidia-smi 0`，对应 5060，但运行时要用 `CUDA_VISIBLE_DEVICES=1`。

看视频时还发现：EMIX 有时第一次没夹住，会再试一次，而不是一直停住。这可能说明它更能处理执行中的失误，但目前只是视频观察，还没统计重试次数和恢复成功率。

## 0810
**数据多了以后，建立配对索引时内存不够**

帧数增加后，需要查找的近邻和候选对太多，导致内存耗尽（OOM）。优化后，单个数据集已经能成功建立 EmixIndex。之后数据更多时，可能还要限制搜索范围，例如只在一部分轨迹里找配对，避免所有数据一起搜索。这还只是后续想法。

此前已完成 10 个任务的单臂宽视角版本。窄视角几乎看不到实际操作区域，所以先换成宽视角；如果重要物体根本不在画面里，要求模型更依赖视觉未必有帮助。HangingMug 当时还没有做好，需要继续调试采集代码。

RoboTwin 默认有 click bell 单臂任务，但 baseline 和 EMIX 都很容易成功，不适合用来区分两种方法，因此继续做其他抓放任务。评估代码默认只接受左臂动作，而新任务可能随机使用左右臂，所以增加任务包装，把相关随机化和操作限制在左侧。

看 200 条轨迹的可视化时，发现能用于配对的交点只有约一百个，因此考虑多采集数据。但改造任务后，专家完成任务的成功率下降，采集 500 条轨迹会很慢。采集失败也可能导致某些位置的数据缺失，让配对更难找到。

增加到 500 条轨迹后，又发现交点大多集中在轨迹末尾。后来调整任务，让场景更多样，但专家成功率也更低，配对情况没有明显改善。怀疑这与任务流程和 RoboTwin 的实现有关，还没有确认。

继续用这批数据训练，baseline 成功率为 88%，EMIX 为 87%。只差 1 点，可能来自随机波动，目前不足以说明两者有稳定差别。


## 2026-10-30 05:12 +0800 — sin-v0-seg monitor

- seg_mix task place_bread_skillet_left_oppo_cam: flow attempt 1 failed at 03:56:14 during train config validation because its output directory was under /root/autodl-tmp/sin-v0-seg/formal/seg_mix/outputs, outside the required /root/autodl-tmp/outputs root. Flow attempt 2 started at 03:56:42 and recovered without monitor intervention; its 30,000-step checkpoint completed at 04:36:38.
- Evaluation is active for the same task (100 episodes, seed 0); flow log reports episode 24/100 at 05:12. Episode videos 0000–0023 are present, with episode 0023 still being written. seg_patch, seg_mix_overlap033, and seg_patch_overlap033 have not started.
- No completed metrics files are present for the four arms yet (0/40); this is in-progress status, not an experiment result. GPU 0 at 05:11:34: 17% utilization, 5,770 MiB / 32,607 MiB. No settings or processes changed.

## 2026-10-30 07:12 +0800 — sin-v0-seg monitor

- `seg_mix` task 1 (`place_bread_skillet_left_oppo_cam`) completed training and 100-episode evaluation; its metrics file has 100 records. The earlier 03:56:14 launch failed the output-root guard, and the 03:56:42 attempt recovered and completed this task without a monitor-side change.
- The chain advanced to task 2 (`place_can_basket_left_oppo_cam`), training at 5,817/30,000 steps, about 11.4 steps/s in the latest log. One of 40 expected arm/task metrics files is complete; all other arms remain queued.
- GPU 0: 60% utilization, 5,875/32,607 MiB. The 16 dataloader workers are active; no training settings, processes, or outputs were changed.

## 2026-10-30 08:12 +0800 — sin-v0-seg monitor

- `seg_mix` task 2 (`place_can_basket_left_oppo_cam`) training has completed and its 100-episode evaluation is active. The flow log records episode 10/100, at 495/700 environment steps in the latest sample; video `episode_0010.mp4` is being written. Task 1 has one complete 100-record metrics file; batch total is 1/40 metrics files. Other arms have not started.
- GPU 0 sampled at 31% utilization, 5,770 MiB / 32,607 MiB; evaluator CPU use was about 211%, with ffmpeg writing the current video. The log continues advancing; no settings or processes changed.

## 2026-10-30 13:12 +0800 — sin-v0-seg monitor

- `seg_mix` advanced to task 3 (`stack_blocks_two_left_oppo_cam`); training is complete and its 100-episode evaluation is active at episode 9/100, environment step 622/800. Current evaluation process is a child of the live flow; no competing experiment task was found.
- Two of 40 expected arm/task metrics files are complete and each contains 100 episode records (`seg_mix` tasks 1–2). The three later arms have not started. GPU 0 sampled at 61% utilization, 3,276 MiB / 32,607 MiB; evaluator CPU use was about 180%.
- No failures or protocol changes observed; processes and settings left untouched.

## 2026-10-30 - seg performance inspection

- Inspected the active seg_mix evaluator (PID 45878), third task stack_blocks_two; log reached episode 50/100. No experiment restart or installation interruption.
- RTX 5090 has 32 GiB VRAM; observed evaluator use about 2 GiB. GPU is NUMA node 0; CPU quota is 25 cores although 208 logical CPUs are visible. Evaluation performs expert seed validation and serial simulation before/between policy calls.
- Training already uses cuDNN benchmark and CUDA pinned-memory loading; evaluation already uses torch.inference_mode. No batch, seed, diffusion-step, precision, worker-count, data, or policy changes made.
- Tried reversible evaluator thread affinity to GPU-local CPUs 0-51,104-155. Initial 20-second windows had 219 steps (default) versus 65 steps plus an episode boundary (local); subsequent 12-second alternating windows were within initialization with no rollout steps. These samples do not establish a speedup. All original thread affinity masks restored; no tuning retained.
- Low utilization alone is not evidence that a numerical optimization is equivalent. No measured acceleration claimed.


## 2026-10-30 17:12 +0800 — sin-v0-seg monitor

- seg_mix advanced to task 4, stack_bowls_two_left_oppo_cam: 30,000-step training finished; evaluation has completed 16/100 episodes and is continuing. The wrapper and flow remain active with a single evaluator.
- Verified 3/40 complete metrics files: seg_mix place_bread_skillet, place_can_basket, and stack_blocks_two, each with episodes=100, 100 unique episode records, and consistent success counts/rates. All ten task slots checked for each of the four arms; later arms have no completed metrics yet.
- GPU 0 sampled at 25% utilization, 5,642/32,607 MiB; evaluator CPU about 225%, with ffmpeg writing episode video. Rollout log advanced from step 149 to 294/900 and then episode 16 completion during inspection. No recovery or protocol changes needed; processes left untouched.


## 2026-10-31 00:26 +0800 — sin-v0-seg monitor

- Batch stopped on seg_mix task 4, stack_bowls_two_left_oppo_cam. Wrapper log last modified 2026-10-30 21:19:23 +0800: episode 96/100 completed, then pybind11 inc_ref / PyGILState_Check failures for seeds 100648–100665, evaluator segmentation fault (exit 139), and flow-train-eval-gpu-0 failure. No active wrapper, flow, train, eval, or installation process found; no default tmux server found. GPU 0: 0% utilization, 0/32607 MiB. Data disk has 131 GiB available.
- Checked all ten configured tasks in each of four arms: three complete metrics files in seg_mix (place_bread_skillet, place_can_basket, stack_blocks_two), each with episodes=100, 100 unique episode IDs and seeds, and consistent success counts/rates. Other arms have zero metrics. Completed-task model weights were pruned by the existing flow cleanup; its completion guard accepts retained checkpoint metadata plus matching metrics. Batch is not complete.
- Failed task retains its 30,000-step model (1043982652 bytes) and 96 episode videos, but no metrics.json. evaluation/evaluate.py writes records only after all episodes; scripts/run.py deletes an incomplete evaluation directory before retrying. Native crash recovery is ambiguous; no code/config repair or restart applied, and all processes, settings, logs, and outputs left untouched. Recommend diagnosing native simulator/GIL lifecycle handling and preserving the partial evaluation before a protocol-preserving retry through the existing wrapper. No seed skipping, dependency changes, retraining, or altered evaluation settings applied.

## 2026-10-31 02:03 +0800 — sin-v0-seg monitor

- The existing wrapper resumed after the earlier `seg_mix` task 4 evaluator crash. Flow `20261031-005234-558930-cuda-0` reused the completed training and evaluations for tasks 1–3 and reused task 4 training; it removed the incomplete evaluation and launched a fresh evaluation for `stack_bowls_two_left_oppo_cam`.
- Current task 4 evaluation is active at episode 23/100 and environment step 376/900. The evaluator logged infeasible-layout seed skips (100159, 100162); it remains CPU-active. All three complete `seg_mix` metrics files contain 100 records each; total batch completion is 3/40. No other arm has started, and no result is claimed for task 4.
- GPU 0 sampled at 0% utilization, 2,094/32,607 MiB; evaluator CPU was 238%. This is the CPU-heavy evaluation phase, with no training dataloader active. No process, code, configuration, protocol setting, or output was changed during this inspection; logs and tmux session were left intact.

## 2026-10-31 19:59 +0800 — sin-v0-seg monitor

- `seg_mix` task 4 (`stack_bowls_two_left_oppo_cam`) remains in evaluation after training completed at 900/900 steps. Current evaluator PID 550366 is alive, but its log has not advanced past episode 86/100 (last log write 15:15) and no episode video newer than episode 85 (15:11); the episode-86 ffmpeg child has been blocked reading its pipe for over five hours. This is a stalled evaluation, not a completed result.
- Earlier retries of this same task logged repeated pybind11 `PyGILState_Check()` failures and ended with evaluator segmentation faults (exit 139). Current retry has not exited; it logged infeasible layout skips through seed 100650. GPU 0 sampled at 0% utilization with 5,492/32,607 MiB allocated; evaluator cumulative CPU time is 10:53:21, but its current wait channel is `futex_wait_queue_me`. No training dataloader is active during this eval.
- Only three valid 100-episode metric files exist, all in `seg_mix` (tasks 1–3); the other three arms have none. No complete result is claimed. Left all processes, outputs, logs, and experiment settings untouched because the evaluator is still active and safe recovery from the repeated native crash/stall is ambiguous. Resume review after the active evaluator exits; do not change evaluation seeds or skip episodes.

## 2026-10-31 23:11 CST — GIL/hang recovery and result validity

- Evaluator 550366 had made no log progress since 15:15 at 86/100, last rejection seed 100650. Native stack attachment via isolated py-spy was denied by the container; exact C++ cause remains unconfirmed. CPU percentages previously reported were lifetime averages, not interval measurements.
- Source inspection: Base_Task.close_env delegates to inherited gym.Env.close without explicit resource teardown. Repeated scene construction and native lifecycle issues are suspected; this is not proof of the particular GIL fault.
- Preserved failed output in /root/autodl-tmp/sin-v0-seg/formal/diagnostics/hung-eval-550366, terminated only the hung evaluator, and allowed the existing supervisor to restart the queue.
- Added evaluation seed/episode progress state with configuration identity, atomic saves and reuse; ten seed attempts per process followed by exec to discard native resources. Each expert validation and rollout has a 1200-second faulthandler timeout that dumps Python stacks and exits. Runtime failures now abort for retry of the same seed; only SceneInitRejectError is classified as a skippable exception. Existing false expert-success checks remain unchanged.
- No old partial records were fabricated: legacy run did not persist episode seed records before completion. New recovery begins a fresh evaluation using the retained final checkpoint. Lifecycle changes may alter nondeterministic planner behavior; do not claim bitwise equivalence.
- Audit of original completed seg_mix task logs: no GIL exception in the first three completed tasks. place_can_basket skipped ten unstable-object errors and one target_pose=None error; stack_blocks_two skipped one target_pose=None error. These results are provisional for comparisons until actual accepted seed sets and selection semantics are audited against baseline. Do not reinterpret runtime errors as policy failures or silently skip them.
- Validation so far: Python compilation passed for evaluate.py and run.py; new evaluator PID 275520 started. End-to-end completion and recycling validation pending.

## 2026-10-31 — non-attach GIL diagnostics

Current evaluation remains active (8/100 at inspection). No experiment process was interrupted for this diagnosis.

- kernel.core_pattern pipes to /usr/share/apport/apport, which is absent inside this container. No accessible core artifacts found at standard apport locations; a shell "core dumped" message is not evidence of a retained dump. Host-side capture cannot be assessed from this container.
- py-spy attachment was denied. ptrace_scope=1, seccomp filtering active, effective capabilities lack SYS_PTRACE. These observations do not independently identify which mechanism denied attachment.
- Scanned all loaded shared objects for the exact PyGILState_Check failure text. Matches: mplib/pymp, manifold3d, scipy spatial distance and pocketfft, triton/libtriton. SAPIEN does not contain this exact text. This narrows potential emitters, not necessarily the underlying cause.
- Robot.set_planner constructs MplibPlanner/SapienPlanningWorld for both arms when need_topp is true, even with planner=curobo. MPLib is therefore on the active initialization path.
- Versions: sapien 3.0.0b1, mplib 0.2.1, torch 2.7.1+cu128, gymnasium 0.29.1; installed pybind11 package 3.1.0 does not establish which pybind11 headers each binary was compiled against.
- Built diagnostics/gil_capture.so: LD_PRELOAD interposer on __cxa_throw logs native library paths, symbol names and module-relative offsets only for std::runtime_error mentioning pybind11 or PyGILState. It forwards the original exception. Tested with a clearly labelled synthetic exception and native imports; no genuine GIL reproduction yet.
- evaluate.py opt-in startup hook is limited to seg config names and diagnostics/gil_capture.enabled sentinel. The next natural interpreter recycle loads it. Output goes to existing evaluator/flow logs as [GIL-NATIVE]. No pointer attachment or system core configuration change required.
- Source retrieval from GitHub timed out; no upstream defect claim made.

## 2026-10-31 23:59 CST — sin-v0-seg monitor

- `seg_mix` task 4 (`stack_bowls_two_left_oppo_cam`) remains in evaluation, recovery attempt 4, after training reached 900/900. The active log records episode 11/100; the evaluator is alive and CPU-active (236% over the sampled interval), with no newer episode record during this check. GPU 0 sampled at 1% utilization and 2,848/32,607 MiB.
- The current retry follows the prior task-4 assertion failure (`target_pose=None`) and earlier native evaluator failures. Evaluation is making CPU-side progress, so processes, code, configuration, outputs, protocol settings, and tmux/log state were left untouched. Three valid 100-record metrics files exist for `seg_mix`; no other arm has started and no overall result is claimed.

## 2026-11-01 00:54 CST — sin-v0-seg monitor

- Current task: `seg_mix` / `stack_bowls_two_left_oppo_cam`, evaluation retry 6, progressing through 26/100 episodes. Previous attempt failed at seed 100156 (`target_pose=None`); supervisor relaunched through its existing recovery path. Retry output includes videos through episode 25. GPU 0: 1%, 2,596/32,607 MiB; evaluator process alive and CPU-active.
- No code/config/protocol change, restart, or termination by monitor. Three earlier `seg_mix` metrics complete; task 4 and remaining arms are incomplete. No result claimed.

## 2026-11-01 01:55 CST — sin-v0-seg monitor

- The prior `seg_mix` / `stack_bowls_two_left_oppo_cam` evaluation failed during expert validation at seed 100288 after 35 completed episodes, with `target_pose cannot be None for move action`; the flow exited with status 1. The existing supervisor launched recovery attempt 7 at 01:24 through the wrapper. The resumed evaluator has advanced to seed 100348 with 46 completed episodes; no final metrics exist for this task.
- GPU 0 sampled at 0% and 2,546/32,607 MiB while the evaluator was CPU-active (222% lifetime average); this is evaluation, with no training dataloader active. Three metrics files exist in `seg_mix` and none in the other arms. No monitor restart, repair, termination, or protocol change was made; the active chain was left running.


## 2026-11-01 02:55 +0800 — sin-v0-seg monitor

- The chain is stopped at `seg_mix` / `stack_bowls_two_left_oppo_cam`: the evaluator failed at seed 100357 during expert validation with `target_pose cannot be None for move action`, after episode 48/100. Its atomic resume file retains 48 records and seed 100357; no final task metrics exist. No live wrapper, flow, train, evaluator, or installer process was found. GPU 0 sampled at 0% and 0/32607 MiB.
- Checked all ten configured tasks in all four arms: three valid 100-record metrics files in `seg_mix`; the remaining 37 are incomplete. Earlier completed metrics remain provisional for comparisons because of the documented seed-selection audit.
- Restart is unsafe without resolving the changed wrapper: it invokes a task-first script limited to the first five tasks and rebuilt environments, rather than the requested four-arm, ten-task order. The training environment and validation marker are absent; the rebuild log reports `file:///root` is not a Python project, and validation says installation did not complete. No restart, repair, protocol change, or process termination was performed. Preserve the 48-record resume state; diagnose the environment and restore the requested full arm order before a validated, protocol-preserving restart through the wrapper.
