## 2026-10-25 13:33 CST：恢复 burger-fries EMIX 并监控五分钟

- 前一轮于 2026-10-24 23:51:24 失败。正式日志显示训练进度条为恢复后 22,397/25,000，约全局 step 27,397/30,000；DataLoader 在 `multiprocessing.resource_sharer` 接收共享 tensor 文件描述符时抛出 `RuntimeError: received 0 items of ancdata`。GPU/训练进程退出，评估未开始。
- 故障现场未见 OOM、磁盘耗尽或 `/dev/shm` 用满：重启前 data disk 可用 141 GB，`/dev/shm` 可用 30 GB，内核日志没有 OOM 记录。训练配置使用 16 个 DataLoader workers；原 shell `RLIMIT_NOFILE` 软/硬限额为 1024/1048576。此类 ancdata EOF 与 DataLoader worker/FD 传递通道中断相符；具体 worker 为何退出没有日志证据，根因尚未确定。
- 确认 `checkpoints/last` 指向完整 025000 checkpoint，模型、optimizer、RNG 和 step 元数据均存在；旧 optimizer 文件按最新点保留策略已清理。于 2026-10-25 13:25:52 通过 tmux 重启原 `train-eval` 批次，从 025000 恢复；将该运行 shell 的 `ulimit -n` 提高到 65535，未改训练配置或 16-worker 参数。
- 连续监控五分钟：训练成功到达 30000/30000，保存最终 checkpoint 并正常转入 burger-fries EMIX 评估；截至 13:32:19 评估已写入 Episode 1/100，截至 13:33:21 runner 仍为 `FORMAL_RUNNING_TRAIN-EVAL`，GPU 有活动、磁盘约 140 GB 可用，未出现新异常。此结果支持 FD/worker IPC 资源压力为一个可疑因素，但 5000-step 恢复尾段未复现故障不能证明该原因。
- 最终评估 metrics 尚未生成；最终 checkpoint 的模型权重和 optimizer 状态保持到评估成功，runner 成功写出匹配完整 metrics 后按既定规则删除具体 tensor 文件，保留 checkpoint 目录和元数据。

## 2026-10-24 23:38 CST：Checkpoint 状态保留标准与恢复批次处理

- 用户确认：所有已完成评估的策略都不保留任何 checkpoint 模型权重和优化器状态；只删除具体 tensor 文件，保留 checkpoint 目录及配置、训练步数、scheduler/RNG 等元数据。评估必须有与最终 checkpoint 匹配且 episode 数完整的 metrics 才算完成。
- 对本批已完成评估的 16 个策略清理了全部编号 checkpoint 下的 `model.safetensors` 与 `optimizer_state.safetensors`：本轮移除 79 个文件、131,537,837,200 bytes；此前同次操作中已移除的两个文件约 3,131,863,340 bytes，合计约 134.67 GB（125.4 GiB）。未删除目录或训练数据。数据盘清理后 142 GB 可用。
- 固定后续训练默认：每 5000 steps 仍保留各编号 checkpoint 的权重和元数据；优化器状态只保留 `checkpoints/last` 指向的最新完整点。每次新 checkpoint 写完并更新 `last` 后删除其它编号点的优化器状态。恢复只能使用 `last` 指向的最新完整点；旧点、缺失模型/优化器/RNG/step 状态的 checkpoint 均拒绝作为恢复点。最终评估成功后，该策略所有编号点的模型权重和优化器状态都删除。此决定已写入 `docs/EXP.md`。
- 已在 `embodied_mixup/policies/train.py` 加入 latest-only optimizer-state pruning、最新完整点恢复校验，以及恢复时复用既有 output directory；三种训练模式都在 checkpoint 保存成功后执行清理。当前运行训练进程在代码改动前已加载旧版本，因此另启临时 `sin-v0-optimizer-prune-watch` 监视当前 formal 批次：burger-fries EMIX 的 010000 点完成并更新 `last` 后，已删除 005000 的具体 optimizer state，保留 010000 恢复状态；仅监视当前批次，批次结束自动退出。
- 已恢复的批次继续从 005000 起步，目前最新完整 checkpoint 为 010000，训练仍在运行；`scripts/run.py` 当前进程已加载的评估后清理 hook 会在每个策略评估成功后清除全部编号 checkpoint 的模型权重和优化器状态。当前 GPU 训练活动，数据盘约 142 GB 可用。
- 没有删除任何 checkpoint 文件夹；本次未运行测试。

## 2026-10-23 15:03 CST: disk-full recovery and checkpoint retention

The corrected sin-v0-patch formal run stopped at 10:29 while saving the 20,000-step optimizer state for stack_blocks_two_left_oppo_cam Patch: ENOSPC on the 200 GiB data disk. Eight policy arms already had final 30,000-step models and 100-episode metrics. Verified those artifacts, removed their 40 intermediate checkpoints, and removed the failed arm's 5,000/10,000-step and incomplete 20,000-step checkpoints. Data-disk free space rose from 1.9 MiB to 125 GiB. Preserved the complete 15,000-step state under the hidden resume-source directory. Started a single tmux trainer at 15:01:26 from that checkpoint; its output snapshot records start_step=15000 and total_steps=30000. The recovery wrapper will then invoke formal-run.sh train-eval, which reuses completed arms. Updated PATROL.md to clean verified intermediate checkpoints when free space falls below 25 GiB.

## 2026-10-22 12:34 CST：第二次 Patch BUG 运行停止与清理

发现 `labels != 0` 把场景表面与机器人像素也拼贴，样本帧的掩码覆盖整个画面。停止 `sin-v0-patch-oppo-run-fixed` 及其训练进程，并关闭旧的半小时 Codex 巡检。本批完整评估仅有首任务 vanilla/mix/错误 Patch 85/95/94 与第二任务 vanilla 50/100；第二任务 mix 在约 19,355/30,000 步停止。以上均作废，简要结果见 `docs/results/20261022-sin-v0-patch-scene-background-bug-invalid-run.md`。

检查提交 `5fc08ec` 的新实例名称筛选：十任务样本帧保留 A 图背景，掩码面积约 0.7%–6.6%；`test_patch_training.py` 的 9 项测试通过。按用户要求仅删除旧 checkpoint、评估产物与运行日志，不修改训练数据、标注或索引；随后以同一实验版本和协议从头运行，开启新版巡检。

## 2026-10-20 暂停正式采集并准备视角预览
- 用户要求暂停正式采集，为当前 blocks_ranking 任务的全部视角变体各提供一个 1 ep MP4 示例，供其选择正式视角。
- 12:50：冻结 runbook、prepare worker 和当前转换器，防止转换结束后自动进入正式采集；转换当时约 216/300，未完成，保留进程及部分产物。正式采集在用户选择视角前保持暂停。

- 12:53：已启动 8 视角预览任务，单回合收集与 MP4 输出写入 `/root/autodl-tmp/sin-v0-patch-view-previews/`，不覆盖正式数据。
- 为兼顾转换成功目标并确保不自动开始正式收集，恢复当前 84×84 转换子进程，保持 run.py worker 与 runbook 冻结；转换结束后它们不会接续下一任务。
- 首次转换失败原因记录：TorchCodec 缺少 FFmpeg `libavutil.so.*` 等共享库；当前重试在冻结前已到约 216/300。后续以完整数据集元信息和全部 300 条转换产物验证成功。

- 2026-10-20T13:00:00+08:00 inspection: converter=98545, annotations=292/300, view_videos=8/8, preview_process=none, GPU=0 %, 2 MiB, 7.30 W, data_free=175G system_free=8.2G.

- 2026-10-20T13:00:00+08:00 view previews complete: one episode MP4 for each of the 8 viewpoints.

- 2026-10-20T13:05:00+08:00 inspection: converter=98545, annotations=292/300, view_videos=8/8, preview_process=none, GPU=0 %, 2 MiB, 8.78 W, data_free=175G system_free=8.2G.

- 2026-10-20T13:05:00+08:00 view previews complete: one episode MP4 for each of the 8 viewpoints.

- 13:05：用户询问视频位置，已在 outputs/view_previews/blocks_ranking 建立入口软链接，指向数据盘预览目录；8 个视频与 README 清单可从该项目路径访问。转换巡检仍在进行。

- 2026-10-20T13:10:00+08:00 inspection: converter=none, annotations=300/300, view_videos=8/8, preview_process=none, GPU=0 %, 2 MiB, 8.53 W, data_free=175G system_free=8.2G.

- 2026-10-20T13:10:00+08:00 conversion success: blocks_ranking_rgb_left_cen_arm_top_cam has 300 episodes, 300 instance annotations, and data/video artifacts. The formal queue remains paused.

- 13:13：确认 right_wide 相机横向偏移为 +0.30 m（相机 x=0.268 m，默认 x=-0.032 m），这不是相机到机械臂基座的欧氏距离。按用户要求启动 +0.15 m 偏移的独立单 ep 预览；临时相机位姿在采集后自动恢复，EXP.md 正式视角保持待选。

- 2026-10-20T13:15:00+08:00：+0.15 m right_wide 单 ep 视频已生成并完整解码验证：outputs/view_previews/blocks_ranking/videos/blocks_ranking_right_wide_015_ep0.mp4。


## Opposite wide preview corrected to XZ mirror (2026-10-20)

- Corrected `left_oppo_cam` after user clarification: mirror the central-wide viewing direction across XZ (Y direction reversed), align camera X with the active arm base at `x=0`, and aim its XOY projection at the base area `(0, -0.45)`. Pose: position `(0.0, 0.45, 1.35)`, forward `(0.0, -0.894427, -0.447214)`, left `(-1.0, 0.0, 0.0)`, D435-W.
- One-episode preview generated and encoded successfully (747 frames): `outputs/view_previews/blocks_ranking/videos/blocks_ranking_opposite_mirror_ep0.mp4`.
- Isolated raw preview: `/root/autodl-tmp/sin-v0-patch-view-previews/collections_opposite_mirror_xoz/blocks_ranking_rgb_left_oppo_cam/blocks_ranking_rgb_left_oppo_cam`.
- The earlier YOZ-mirrored preview was superseded and removed from the published preview manifest/README. The formal collection remains paused. Task config was restored to `episode_num: 500` after the preview run.


## Opposite preview: 180-degree image rotation and steeper pitch (2026-10-20)

- Updated `left_oppo_cam` to roll the image 180 degrees by reversing the camera left axis and increased downward pitch from about 26.6 to 35 degrees. Position remains `(0.0, 0.45, 1.35)` with optical axis aimed at `(0.0, -0.45, 0.72)`; D435-W retained.
- Regenerated the one-episode MP4 at `outputs/view_previews/blocks_ranking/videos/blocks_ranking_opposite_mirror_ep0.mp4` (750 frames). The previous video at that path has been replaced.
- Isolated raw preview: `/root/autodl-tmp/sin-v0-patch-view-previews/collections_opposite_roll180_pitch35/blocks_ranking_rgb_left_oppo_cam/blocks_ranking_rgb_left_oppo_cam`. Formal collection remains paused; task config restored to `episode_num: 500`.


## sin-v0-patch data cleanup and view protocol update (2026-10-20)

- Removed prior collected data from `/root/autodl-tmp/sin-v0-patch/data` and `/root/autodl-tmp/sin-v0-patch/native`, including the previous top-camera dataset and native captures for all camera variants.
- Removed raw temporary preview collections under `/root/autodl-tmp/sin-v0-patch-view-previews/collections*`; retained preview MP4s, experiment logs, outputs directory, and code/configuration.
- Updated the sin-v0-patch section of `docs/EXP.md` to use `left_oppo_cam` (D435-W, 40-degree downward pitch, 180-degree image rotation) and 300 successful episodes per task. Formal collection remains paused.

## sin-v0-patch-oppo 执行记录（2026-10-20，Asia/Shanghai）

- 依据 `docs/EXP.md` 和 SOP-05，废弃旧 top-camera run/data 作为本实验输入；保留其日志与检查点但不复用。新实验使用十个 `left_oppo_cam` 任务、每任务 300 个成功 episode、实例标注，并按 prepare → index → visualization → train/eval 屏障顺序执行。GPU 任务保持单队列，CPU index/可视化按 SOP 串行。
- 运行配置隔离为 `configs/scripts/run_sin_v0_patch_oppo.yaml`（正式数据根 `/root/autodl-tmp/sin-v0-patch-oppo/formal/`）和 pilot 配置；十个实验任务配置均已核对存在。加入 EXP 所要求的 DP、DDPM、优化器、EMIX debug/visual 配置字段。
- `dependencies/_install_twin.sh --check` 成功：RTX 5090 D、CUDA Torch、SAPIEN RT renderer、OIDN 功能探针通过。
- 为 pilot 转换增加 `--task <task>` 作为 SOP converter 文档定义的 instruction sidecar 缺失回退；`scripts/run.py` 的语法检查通过。变更前备份位于 `/root/autodl-tmp/sin-v0-patch-oppo/run.py.before-instruction-fallback`。
- Pilot `blocks_ranking_rgb_left_oppo_cam` 单 episode 采集完成（日志 state：`PILOT_COLLECTION_DONE`）。HDF5 有 740 帧；RGB 与 `instance_id` 均为 740 帧，掩码为 `uint32`、240×320；实际可见 ID 均可映射，`patch_target` 为 ID 33，`unmapped=[]`。本任务本次采集未出现大于 255 的实例 ID，故不能据此宣称已验证 >255；需用显式合成高 ID 用例验证保存/变换链路。
- Pilot conversion 第一次因缺 instruction sidecar 失败；已补 `--task` 回退后重试，第二次因 `emix` 缺 TorchCodec 所需的 FFmpeg 动态库而失败（libavutil/libavdevice 等，主机现有旧版 FFmpeg 不兼容）。记录状态：`PILOT_CONVERT_RC=1`。没有创建系统库软链接。
- 已下载 micromamba 到数据盘并运行 conda-forge `ffmpeg=8` dry-run。解析器显示 Python 3.10 pin 且求解完成，当前仍在等待 transaction package plan/下载元数据输出；正式安装前先审阅变更范围。GPU 当前空闲（0%，2 MiB），正式训练尚未开始，符合 pilot gate 未通过时不启动正式任务的约束。
- 启动 `sin-v0-patch-oppo-inspect` tmux 巡检器，每 1800 秒记录状态、GPU、磁盘、进程和日志年龄；状态进入正式运行后，worker 意外消失时才按 runbook 恢复。巡检脚本和日志在 `/root/autodl-tmp/sin-v0-patch-oppo/inspect-30m.sh`、`inspection.log`。
- 15:05 检查点：尝试在数据盘独立前缀解析 FFmpeg 8（不修改 `emix` 环境），micromamba 两次均成功读 shard 索引并求解，但 transaction 包计划阶段长时间无输出；已中断，确认没有完成安装。ORCHE.md 已记录这一状态及后续方案。巡检状态正常，旧暂停进程未被自动重启。
- 15:07 修订半小时巡检脚本：过滤 T 状态暂停进程，避免旧 top-camera 进程掩盖正式 worker 退出；formal training 状态若 GPU 显示 0% 则写入异常记录，缺 worker 时仍只在正式 runbook tmux 不存在时恢复。`bash -n` 检查通过，巡检 tmux 已重启。
- 15:08 用 micromamba 在数据盘隔离前缀 `/root/autodl-tmp/sin-v0-patch-oppo/ffmpeg-libs` 开始安装 FFmpeg 8，不修改 `emix` Conda 环境；已审阅事务计划为 138 个包、约 163 MB。安装 tmux `sin-v0-patch-oppo-ffmpeg`，日志 `ffmpeg-install.log`。完成后仅为 pilot 转换临时设置 `LD_LIBRARY_PATH`。
- 15:10 FFmpeg 8 已在隔离前缀安装完成（138 包，约 661 MB）。用该库重启 pilot `prepare` 阶段后，flow 返回 RC=0，并确认复用已转换完成的数据集（标注 sidecar 存在），所以不再重新采集。LeRobot sidecar `annotations/episode_0.instances.h5` 含 739 个 frame_ids 和 `(739,84,84)` 的 uint8 标签图；视频为 84×84、17 fps、739 帧；源 HDF5 有 740 个 RGB/uint32 instance_id 行，转换按相邻状态-动作对生成 739 行，长度关系合理。
- 目前确认 >255 的源实例 ID 在实际采集样本中不存在（max 可见 ID 35）；转换标签是按 patch_target 语义生成的二值/类别 mask，uint8 标签类型本身不代表源 ID 保留证明。仍须对 `InstanceAnnotations` 的原始 ID 路径做 ID 513 合成往返检查，完成后才算 pilot gate 全部通过。
- pilot 的独立 `prepare` 成功只验证已有 artifacts 可被 flow 复用；正式十任务尚未启动，也未跳过 >255 门槛。GPU 仍空闲，因当前阶段是转换/验收而非训练。
- 15:21 合成 >255 门槛检查首次因测试断言选错 resize 后像素坐标失败；按最近邻 half-pixel 坐标修正测试后通过：原始 uint32 ID 513 经 HDF5 保存/读回一致、renderer map 映射到 patch_target，84×84 最近邻变换后类别标签仍为 1。检查脚本/日志在数据盘 `verify-high-id.py`、`high-id-check.log`。
- 15:30 正式 runbook 首次启动即因 tmux 非登录 shell 的 PATH 不含 conda 失败（RC=127），未进入 prepare。已将调用修正为 `/root/miniconda3/bin/conda`，并更新巡检器按 `formal.pid` 检查正式进程，避免旧暂停进程误报。准备立即重启。
- 15:33 已修复正式 runbook 的 conda 绝对路径，并修订巡检器用 formal.pid 检查正式进程；两脚本 bash -n 检查通过，巡检 tmux 重启。
- 15:34 正式十任务队列已启动，state=FORMAL_RUNNING_PREPARE，run session sin-v0-patch-oppo-run，日志在数据盘 formal-run.log。首任务 blocks_ranking_rgb_left_oppo_cam 正按成功 episode 目标重试失败 seed（seed 0–3 当前失败，流程继续尝试）；RTX 利用率约 77%、显存约 2329 MiB、功耗约 121 W。采集/渲染按 SOP 保持单 GPU 顺序队列。
- 15:37 GPU/CPU巡检：首任务约 20 个成功 episode，失败 seed 自动重试；collect_data.py 单核约 99%，GPU 读数波动 0–34%（启动时 77%），显存约 2331 MiB、功耗约 121–124 W。仿真仍在运行，按 SOP 不并发第二个 GPU 任务。
- 15:44 checkpoint: formal prepare is active on task 1; 56 successful episodes reached (latest seed 149 failed and will retry). GPU 72%, 2331 MiB, 127 W. The fixed single-task queue is delivering strong GPU use during this sample; no extra GPU job is queued.
- 2026-10-20 15:巡检间隔已按用户要求从 1800 秒调整为 3600 秒；巡检器已重启，正式实验 tmux 队列未改动。
2026-10-20 17:10 手动巡检：formal 状态为 FORMAL_FAILED_RC=1；首任务实际已完成 300/300 集并写出 .complete，但 RoboTwin 固定将数据写到仓库 data/robotwin，flow 配置的 collection_root 位于数据盘，转换器因此找不到文件。GPU 0%，显存 2 MiB；失败后无正式 worker。根卷 86%（4.3 GiB 可用），数据盘约 176 GiB 可用。
2026-10-20 17:15 修复 scripts/run.py：collect 阶段成功后，将 RoboTwin 默认输出 ROOT/data/robotwin/{task}/{task} 移入配置 collection_root；检测目标已存在时拒绝覆盖。python3 -m py_compile 通过。已把 blocks_ranking_rgb_left_oppo_cam 的 300 集完整目录（data/instance_objects/instructions/video/_traj_data 各 300 项，含 .complete）迁至数据盘正式 collection 路径，源路径已移除；数据未删除。
2026-10-20 17:18 正式 runbook 续跑，state=FORMAL_RUNNING_PREPARE；flow 确认复用完整 collection 并开始 convert-84x84，300 集、84x84、17fps、实例标注开启。根卷降至 76%（7.5 GiB 可用），数据盘约 173 GiB 可用。GPU 0% 是当前视频转换阶段的采样，不另起 GPU 任务，维持 SOP 单任务阶段队列。
2026-10-20 17:36 人工巡检：远端短暂 SSH 超时后重连成功。formal 状态仍为 FORMAL_RUNNING_PREPARE；首任务转换进程持续运行，日志显示 RoboTwin episodes 251/300 后完成 732 条样本映射，正在执行 libsvtav1 AV1 编码；进程约 110% CPU。GPU 0%/2 MiB，符合当前 CPU 视频编码阶段。SOP 要求 prepare 全任务屏障后才开始 index/训练，不并发绕过屏障；本次保持队列不变。巡检脚本 sleep=3600。
2026-10-20 18:50 阶段进展巡检：formal 队列已完成 task 1/10 blocks_ranking_rgb_left_oppo_cam 的 300 集采集及 84×84 标注转换，prepare 正在 task 2/10 place_burger_fries_left_oppo_cam 的采集。当前 episode 约 143/300（日志最新出现 episode 143 retries），collector 子进程运行约 63 分钟、约 133% CPU；GPU 67%、2645 MiB、137 W、47°C，负载良好。state 文件仍显示 17:18 FORMAL_RUNNING_PREPARE（该字段只在阶段切换更新，非心跳）；活动 worker 和收集进程确认仍在运行。根卷 76%（7.4 GiB 可用），数据盘 173 GiB 可用。

## sin-v0-patch 数据准备 CPU 优化（2026-10-20）

- 针对 patch 数据从原始采集到可训练评估前的 CPU 瓶颈做等价优化。已观测到首个 300 episode 任务的转换流程耗时约 1751 秒；图像统计重算 300 个 episode 耗时约 8 分 15 秒。
- `data_processing/convert_to_lerobot.py` 不再复制随后未使用的原始 RGB 帧；生成实例标注时跳过不会被保留的 RGB 变换，只处理标签图。输入输出样本与标注语义保持一致。
- `data_processing/instance_annotations.py` 对范围较小的整数 renderer ID 使用稠密查找表构造标注，避免 `np.unique(..., return_inverse=True)` 的大逆向索引数组；超出适用范围时保留原实现回退。`transform_pair` 支持仅变换标签图。
- `data_processing/fix_stats.py` 按 episode 一次批量解码图像帧，减少逐帧查询/解码开销；统计仍由相同的帧张量计算。
- `embodied_mixup/candidates/emix_index_utils.py` 将数据集 SHA256 改为分块读取，并复用规范 JSON 编码器，降低峰值内存和重复初始化开销；文件哈希语义保持不变。
- 本次未运行完整数据重建或训练评估；优化后的端到端耗时尚待后续任务实测。
2026-10-20 19:43 对提交 cd60cce6215e0819b284af97d10aacc08802fa21 做性能影响核对（提交时间 19:09:18）：改动涉及 convert_to_lerobot、fix_stats、instance_annotations、emix_index_utils。静态分析显示：实例 ID 映射对 max ID≤1,000,000 使用 dense lookup/bincount，避免 np.unique inverse 的大 int64 数组；标注转换支持 RGB=None，并去除原尺寸 RGB 帧额外拷贝；stats 改为按 episode 时间戳批量读视频帧；索引指纹 SHA256 改为 8 MiB 分块，降低整文件 read_bytes 的峰值内存。当前活跃的是 place_burger_fries_left_oppo_cam 的 RoboTwin 采集，进程运行约 1:55、133% CPU，GPU 48%、2645 MiB、127.5 W；这些改动不作用于当前采集阶段，待后续转换/index 才生效。当前无同一数据集/同配置的新旧基准结果，不能据运行吞吐确认提速；活动正式实验期间不额外并发 CPU benchmark。
2026-10-20 21:06 按 EXP/SOP 进度巡检发现正式采集 episode_num 偏差：task 2 模板 place_burger_fries_left_oppo_cam.yml 默认 500，scripts/run.py 未把 pipeline.episodes_per_task=300 传给 data_collection/robotwin.py；此前 seed 清单已记录 482 个成功 pre-motion 轨迹。已修复 run.py 增加 --episodes 300，并 py_compile 通过。安全停止旧采集进程后保留数据：保留 episode0..299 与对应 300 个 seeds；额外 episode300..481（182 个 pkl）及完整 seed 清单移到数据盘 overcollection/place_burger_fries_left_oppo_cam，不删除。重启 formal runbook；确认 wrapper 命令含 --episodes 300，任务配置已重写 episode_num -> 300，collect_data.py 正在用保留的 raw 轨迹执行后续处理。启动后 CPU 117%、GPU 63%、显存 6693 MiB、功耗 192 W；队列仍在 prepare，task 2。
2026-10-20 23:13 完成全项目采集入口与默认集数核对。data_collection/robotwin.py 的 Python API 和 --episodes CLI 默认均为 300；scripts/assets.py 默认 300；scripts/dataflow.sh 默认 NUM=300 并显式传 --episodes "$NUM"；scripts/run.py 每次采集显式传 pipeline.episodes_per_task，当前正式 config 为 300，启动校验也要求任务模板 episode_num=300。脚本调用点全部补齐 episode 参数：SOP oppo 示例 300；10ep 临时工具显式 10；专项 run_oppo_cam 工具显式 500；serial_deliver 默认 300 且可通过 EPISODES 覆盖。SOP 表格和 5060 用法示例已标注。py_compile 和涉及 shell 脚本的 bash -n 均通过。当前正式 task 3/10 `place_cans_plasticbox_left_oppo_cam` 采集进程命令显式含 --episodes 300，运行约 1:13，GPU 55%、显存 2667 MiB、功耗 131 W；未打断运行任务。
2026-10-20 23:15 将本实验十个 `left_oppo_cam` RoboTwin task YAML 的 `episode_num` 默认统一为 300，避免未显式覆盖时回退到 500；当前采集中任务 3/10 仍在运行且命令含 --episodes 300。仅保留独立 10ep/500ep 历史 pipeline 的显式目标值。

2026-10-21 00:03 巡检机制核对：当前 sin-v0-patch-oppo-inspect 远端脚本每 3600 秒记录 formal state、GPU、磁盘、活动进程和日志年龄；异常恢复仅覆盖 formal worker 丢失。它是自动化 shell 巡检，不是模型每小时自行唤醒，因此不能替代我在会话中进行的 SOP/性能判断或主动通知。发现旧 sin-v0-patch-monitor 指向历史 sin-v0-patch-ops 且状态陈旧，已停止，避免误认为其在监控当前实验。当前 task 3/10 place_cans_plasticbox_left_oppo_cam 采集 episode 约 122，GPU 54%、5708/32607 MiB、204 W；collector 约 125% CPU。GPU 未满，但当前是单仿真采集阶段；先遵守 SOP 不并发第二个仿真器，保留 GPU 余量，后续巡检评估阶段衔接/CPU 瓶颈后再优化。ORCHE 已更正为小时巡检并注明自动脚本与人工判断的边界。

2026-10-21 00:16 人工补巡检并核实 Codex CLI 定时器：本机 00:08 启动的首次 Codex run 因远端 SSH 端点短时超时，三次连接尝试未完成巡检，未写远端日志且未改动任务。00:16 SSH 恢复后人工确认 formal prepare 的 task 3/10 place_cans_plasticbox_left_oppo_cam 已推进至 episode 204；collector 活跃约 2:16，CPU 约 121%；GPU 三次读数 73%/5452 MiB/207 W、58%/5452 MiB/190 W、51%/5708 MiB/180 W；单仿真运行，按 SOP 暂不并发第二模拟器。根卷剩余 5.3 GiB（83%），数据盘剩余 171 GiB（15%）。Codex CLI 每小时 cron 已安装；因 CLI 对模型报告失败仍返回进程码 0，正在给调度器加正文状态校验，避免将连接失败视为成功。

2026-10-21 00:20 Codex CLI 定时巡检已安装在本机 cron：每小时整点启动独立 codex exec，单轮锁、45 分钟硬超时，SSH 失败报告未完成；结果保存在本机 /home/rogers/.codex/automation/emix-hourly-inspection.log。首次试跑于 00:08 遇远端 SSH 超时，未操作远端；00:16 人工补巡检确认 task 3/10 episode 204，00:19 又确认推进至 episode 220，GPU 56%、5452 MiB、200 W，根卷 83%（5.4 GiB 可用）、数据盘 15%（171 GiB 可用），collector 仍运行。AI 自动巡检于 00:18-00:19 做到三次 GPU 采样（61/60/36%）但因多次 SSH 超时未完成文档复核及远端 LOG 写入，runner 已将报告状态标为 incomplete (rc=2)，未修改实验进程/数据。调整后 prompt 精简为相关 SOP 分段读取，继续保留远端轻量巡检。
2026-10-21 09:11:11 手动进度检查：formal prepare 正在 task 9/10 place_dual_shoes_left_oppo_cam 采集，日志 episode 已到 279（正在保存该 episode 帧），collector 活跃；GPU 70%、6558 MiB、181 W。formal/lerobot 下前 8 个任务均已有 .complete，task 9 与 task 10 尚未完成。当前无训练进程，仍处于采集/数据准备阶段。任务状态文件时间戳仍为 2026-10-20 21:06，属阶段开始时间，不能当作心跳。

2026-10-21 11:22:17 手动故障巡检：发现 formal 阶段于 10:29 因 emix 环境缺少 hdbscan 在第 7 个索引任务失败，GPU 归零；远端旧巡检脚本仅在 FORMAL_RUNNING 下做 worker 检查，FORMAL_FAILED 时 action=none，小时 Codex 本轮又遇 SSH 超时，未能自动诊断。

2026-10-21 11:23:48 Manual recovery: formal index phase had failed at 10:29 with ModuleNotFoundError for hdbscan. The previous hourly monitor only checked missing workers while state was FORMAL_RUNNING, so it recorded action=none for FORMAL_FAILED; the 11:00 Codex check could not reconnect over SSH. Installed hdbscan 0.8.44 in emix, pinned it in dependencies/requirements.txt, and added a valid post-install import check to dependencies/_install.sh. Updated formal-run.sh for phase-resume and the remote monitor for 5-minute checks and verified hdbscan failure recovery.

2026-10-21 11:24:54 手动复核恢复状态：remote state=FORMAL_RUNNING_INDEX；formal-run.log 已推进到 place_bread_basket_left_oppo_cam 索引，读取 133414 个 EEF pose 并开始 KD-tree 查询，上一任务 stack_blocks_two 已进入 HDBSCAN。GPU 单次采样 0%、2 MiB、7.17 W；当前为 CPU 索引计算，日志持续推进，非 worker 停摆。远端 5 分钟监控已于 11:21 重启并记录自动续跑 action；下一轮监控将在约 11:26。当前不并发训练，遵守 SOP 阶段屏障。

### 2026-10-21 Hourly inspection: repair and resume

- Asia/Shanghai time: 2026-10-21 12:44:31
- Formal state: FORMAL_RUNNING_INDEX 2026-10-21T12:16:22+08:00. Latest formal log progress: [gpu 0 | 000-0000.log] Filtering neighbors:  27%|██▋       | 57659/217402 [03:07<09:53, 269.20it/s]
- Config verified: 10 left_oppo_cam tasks, 300 episodes/task, 17 FPS, 84x84 RGB, per-frame instance annotations enabled; baseline/mix/patch share data and index.
- GPU 0 samples: 12:22:01=0%,2/32607 MiB,7.99W; 12:22:11=0%,2/32607 MiB,7.94W; 12:22:21=0%,2/32607 MiB,7.57W. The samples were taken during CPU neighbor filtering; GPU idle is expected in this index stage. At 12:25 the log mtime and size advanced and neighbor filtering advanced from 33% to 35%.
- Disk at write time: Filesystem      Size  Used Avail Use% Mounted on ; overlay          30G   24G  6.5G  79% / ; /dev/nvme1n1    200G   50G  151G  25% /root/autodl-tmp
- Root cause: visualization HTML generation succeeded, but PNG rendering had no Chrome. After adding Playwright Chromium, launch exposed missing Ubuntu libasound.so.2 (exit 127). Earlier hdbscan failure was already repaired. The old watcher matched that stale historical error and could trigger a false recovery.
- Repair: pin playwright==1.64.0 in dependencies/requirements.txt; installer now installs Chromium with system dependencies and validates launch/page rendering; render_vis_png.py locates Playwright-managed Chromium; installed Playwright/Chromium and Ubuntu libasound2 on emix. A real 1280x960 PNG smoke render passed. inspect-30m.sh now scopes diagnosis to the latest failure context and requires a PNG smoke before browser-triggered resume.
- Resume: watcher started index at 12:16:22 after the stale hdbscan match. Task 2/10 HTML+PNG completed; task 3/10 neighbor filtering progressed. No second GPU job was launched and no active experiment config was changed.
- Validation: git diff --check, bash -n, Python compilation, and browser PNG smoke passed. Index/vis log and artifacts progressed.

2026-10-21 13:22 Manual inspection: formal had advanced through index and visualization but failed at train-eval before GPU initialization because the configured data-disk output path was rejected by the project outputs guard. Added an opt-in EMIX_ADDITIONAL_OUTPUT_ROOTS whitelist and exported the exact formal outputs directory from formal-run.sh. Verified the path is accepted only with the explicit allowlist. Resumed at train-eval at 13:18; GPU samples were 93 percent, 89 percent, and 94 percent with 5860-5876 MiB and 442-469 W. Training is active; no schedule overlap is needed while the single GPU job uses about 90 percent.


2026-10-21 13:28:57 Asia/Shanghai hourly formal inspection follow-up:
- State: FORMAL_RUNNING_TRAIN-EVAL (runbook phase started 2026-10-21T13:18:10+08:00). Index and visualization are complete/reused for all 10 tasks. First training task is place_bread_skillet_left_oppo_cam baseline, 12016/30000 steps at 13:26, progressing around 26 steps/s; 16 data-loader workers are active.
- Formal config/protocol: 10 left_oppo_cam tasks, 300 successful episodes each, 17 FPS, 84x84 RGB, per-frame instance annotations, baseline/mix/patch shared source data and index. Single GPU 0 queue remains serial; no parallel simulation or training.
- GPU samples at 13:21:03/13:21:13/13:21:50: 96%/91%/9%, each 5876 MiB of 32607 MiB; power 461.68/461.38/108.60 W. The brief low sample coincided with active training and the log continued advancing to 12016 steps by 13:26. Disk at 13:26: root 6.4 GiB free (79%); /root/autodl-tmp 145 GiB free (28%).
- Failure cause: initial train-eval rejected the configured data-disk output path; formal-run.sh explicitly exports EMIX_ADDITIONAL_OUTPUT_ROOTS for the formal outputs root, and the emix resolver import check accepted the expected baseline path. A later automatic resume hit a second-resolution flow log directory collision (FileExistsError).
- Repair: scripts/run.py now includes microseconds in flow log directory names. inspect-30m.sh recognizes only the latest matching flow-collision/output-root failure context and resumes at train-eval only when the formal runbook is absent. The active runbook is already training and was not interrupted.
- Validation: python3 -m py_compile scripts/run.py; bash -n inspect-30m.sh formal-run.sh; git diff --check; explicit emix resolve_project_output_path check passed. Monitor PID 894105 is alive; latest inspection action=none while the formal worker is active. Orchestration order and GPU concurrency are unchanged.


2026-10-21 13:29:52 Asia/Shanghai hourly formal inspection follow-up:
- State: FORMAL_RUNNING_TRAIN-EVAL (runbook phase started 2026-10-21T13:18:10+08:00). Index and visualization are complete/reused for all 10 tasks. First training task is place_bread_skillet_left_oppo_cam baseline, 12016/30000 steps at 13:26, progressing around 26 steps/s; 16 data-loader workers are active.
- Formal config/protocol: 10 left_oppo_cam tasks, 300 successful episodes each, 17 FPS, 84x84 RGB, per-frame instance annotations, baseline/mix/patch shared source data and index. Single GPU 0 queue remains serial; no parallel simulation or training.
- GPU samples at 13:21:03/13:21:13/13:21:50: 96%/91%/9%, each 5876 MiB of 32607 MiB; power 461.68/461.38/108.60 W. The brief low sample coincided with active training and the log continued advancing to 12016 steps by 13:26. Disk at 13:26: root 6.4 GiB free (79%); /root/autodl-tmp 145 GiB free (28%).
- Failure cause: initial train-eval rejected the configured data-disk output path; formal-run.sh explicitly exports EMIX_ADDITIONAL_OUTPUT_ROOTS for the formal outputs root, and the emix resolver import check accepted the expected baseline path. A later automatic resume hit a second-resolution flow log directory collision (FileExistsError).
- Repair: scripts/run.py now includes microseconds in flow log directory names. inspect-30m.sh recognizes only the latest matching flow-collision/output-root failure context and resumes at train-eval only when the formal runbook is absent. The active runbook is already training and was not interrupted.
- Validation: python3 -m py_compile scripts/run.py; bash -n inspect-30m.sh formal-run.sh; git diff --check; explicit emix resolve_project_output_path check passed. Monitor PID 894105 is alive; latest inspection action=none while the formal worker is active. Orchestration order and GPU concurrency are unchanged.

2026-10-21 13:33:44 progress follow-up: state remains FORMAL_RUNNING_TRAIN-EVAL; place_bread_skillet_left_oppo_cam baseline advanced to 22170/30000 steps (15:07 elapsed, about 25 steps/s). Single-GPU train-eval remains active; no orchestration changes. See preceding entry for three GPU samples and disk evidence.

2026-10-21 13:34:46 monitor reload follow-up: restarted sin-v0-patch-oppo-inspect so the edited FORMAL_FAILED recovery rule is loaded (PID 926479). Fresh inspection reports action=none with the formal runbook active; baseline training advanced to 23325/30000.

### 2026-10-21 13:58 Asia/Shanghai Codex 巡检：修复评估依赖并续跑

- State: FORMAL_RUNNING_TRAIN-EVAL 2026-10-21T13:55:24+08:00. At 13:38:57 the formal run failed in task 1 baseline-eval with ModuleNotFoundError: No module named embodied_mixup in emix-twin. The repository package already existed; the twin install path omitted it.
- Repair: added editable local package -e .. to dependencies/requirements-emix-twin.txt; _install_twin.sh now checks evaluator helper imports in read-only --check and at normal install completion. Installed with conda run -n emix-twin python -m pip install --no-deps --editable /root/EMIX.
- Validation: embodied_mixup.datasets and embodied_mixup.configs.config imports passed from /root/EMIX; evaluation/evaluate.py --help imported successfully; bash -n dependencies/_install_twin.sh passed; _install_twin.sh --check passed including RTX 5090, SAPIEN 3.0.0b1 render, OIDN render, Curobo, and the new package check.
- Resume: restarted /root/autodl-tmp/sin-v0-patch-oppo/formal-run.sh train-eval at 13:55:24. scripts/run.py acquired the exclusive GPU flow lock, skipped completed baseline training because its 30000-step model.safetensors exists, and entered baseline-eval. At 13:57:17 the evaluator log reached rollout step 177/500 and recorded episode 1/100 success; process PID 936442 remained active at about 136% CPU, so low GPU samples during evaluation startup were not treated as a stall.
- GPU at 13:57:18/13:57:28/13:57:38: 2/3/5%, 2066/3318/2808 MiB of 32607 MiB, 122.71/106.65/101.00 W. Disk: root 30G, 24G used, 6.4G free (79%); /root/autodl-tmp 200G, 67G used, 134G free (34%).
- Diagnosis/orchestration: prepare, conversion, index, and visualization artifacts are complete/reused; no second simulator or trainer was launched. The old remote 5-minute inspector is absent/stopped; only the local half-hour Codex patrol is responsible. Single-GPU serial orchestration is unchanged.
- Live recovery evidence: state remained FORMAL_RUNNING_TRAIN-EVAL; process chain was formal-run.sh PID 936021, scripts/run.py PID 936040/936041, evaluator PID 936442. Formal log mtime advanced from 13:55:34 to 13:57:17 and included episode progress. No active remote monitor process was found.

### 2026-10-21 14:00 Asia/Shanghai progress follow-up

- Formal state remains FORMAL_RUNNING_TRAIN-EVAL 2026-10-21T13:55:24+08:00; task 1 place_bread_skillet_left_oppo_cam baseline-eval advanced from rollout step 177 to 500/500 and logged episode 2/100 (episode 1 had success=True, episode 2 success=False). The evaluation PID 936442 remained running at about 130% CPU and formal log mtime advanced to 13:59:25.
- A point sample at 13:59:47 showed GPU 10%, 2872/32607 MiB, 96.65 W. The three 10-second GPU samples and disk readings from 13:57 are in the preceding entry. This confirms active simulation/evaluation; orchestration remains single-GPU and serial.

2026-10-21 14:01 Asia/Shanghai progress: formal state FORMAL_RUNNING_TRAIN-EVAL 2026-10-21T13:55:24+08:00; baseline evaluation log mtime advanced to 14:00:29 and reached rollout 359/500, episode 3/100 success=True; evaluator remained CPU-active. No schedule change.

### 2026-10-21 14:37 负责人交接与实地巡检

- 责任沿革（按用户现场说明）：14:00 起负责人曾转交本机 tmux `moni` agent；远端五分钟 shell inspector 与旧 hourly shell monitor 已停用；本机 cron 已改为每半小时向 `moni` 发送巡检提醒。随后用户正式交接给当前 Codex，由当前 Codex 唯一负责实验推进、修复、续跑和巡检。
- 14:32 起 SSH 恢复并从 `/root/EMIX` 检查现场。state 为 `FORMAL_RUNNING_TRAIN-EVAL 2026-10-21T13:55:24+08:00`；formal-run PID 936021、baseline evaluator PID 936442 均活动。沿用首任务已有 30000-step checkpoint，没有重训。
- rollout/episode 证据：formal log 在 14:34:05 的样本为 rollout 319/500、episode 39/100 success；14:36:31 样本为 rollout 151/500；14:37:00 样本为 rollout 163/500、episode 40/100 success。episode 内 rollout step 重置，episode 成功计数前进，评估持续推进。
- evaluator CPU 约 128–129%。GPU 样本：14:35:13 为 0%、2686 MiB、109.84 W；14:36:31 为 24%、5869 MiB、271.72 W；14:37:00 为 0%、2256 MiB、94.81 W。GPU 瞬时偏低时，CPU 占用与 rollout/episode 增长确认 simulator/eval 正在工作；没有并发第二个仿真器或训练。
- 磁盘未变：根卷 79%，可用 6.4 GiB；数据盘 34%，可用 134 GiB。没有发现需要重启、改活跃配置或重训的异常，保持 train-eval 串行队列。

### 2026-10-21 14:38 follow-up rollout/GPU inspection

- 14:38:48 Asia/Shanghai: state remains `FORMAL_RUNNING_TRAIN-EVAL`; formal-run PID 936021 and evaluator PID 936442 remain active. Evaluation log has advanced into the next rollout at step 121/500 after the prior episode 40 success; evaluator CPU is about 129%.
- GPU sample is 100%, 5869 MiB, 309.33 W. Together with the earlier 0–24% samples, this shows utilization varies through simulator/evaluation work; this sample confirms the active workload can fully use the GPU. Do not add a concurrent simulator or training job to chase higher average utilization.
- No restart, retraining, or active-config change. Continue the current baseline evaluation and serial queue.

### 2026-10-21 14:40 EXP/SOP cross-check

- Read the applicable experiment, training, evaluation, and batch orchestration sections in `docs/EXP.md`, SOP-03, SOP-04, and SOP-05. The live work remains baseline evaluation in train-eval; single-GPU serial execution and the phase ordering remain in force. No active configuration, process, or schedule was changed.
- Rechecked the 14:00 moni/monitoring handoff, 14:37 and 14:38 live inspection records, and corrected 14:35 GPU values in `docs/ORCHE.md`; all required entries are present.

### 2026-10-21 15:35 Asia/Shanghai Codex inspection

- Patrol start was 2026-10-21 14:58:04 CST in local cwd /home/rogers/Projects/emix (uid 1000, gid 1000, supplementary group sudo); remote checkout access was verified only through ssh emix at /root/EMIX as uid 0. No local project files were inspected or changed.
- Read the remote AGENTS.md, root autodl.md and en.md, applicable current EXP.md sections, ORCHE.md, recent LOG.md entries, and all six SOP files. SOP phase barriers require serial use of the single GPU; evaluation runs in emix-twin and must use the prepared checkpoint/configuration.
- SSH was intermittently unavailable during document reads, reporting both Timeout, server connect.bjb1.seetacloud.com not responding and TCP timeout to 198.18.0.8:49857 (SSH exit 255). Each failed connection was followed by a 20-second wait and retry. SSH recovered and the live inspection completed at 2026-10-21 15:34:49 CST.
- Live state file: FORMAL_RUNNING_TRAIN-EVAL 2026-10-21T13:55:24+08:00. Prepare, annotation conversion, all ten indexes, and visualization remain complete. The active task is place_bread_skillet_left_oppo_cam baseline evaluation; training reused the existing 30000-step checkpoint. Formal log recorded episode 94/100 success=True at rollout step 133/500. Evaluation process PID 936442 was active at about 128% CPU; recent video artifacts advanced through episode_0093.mp4.
- GPU 0 (RTX 5090 D) sample: 1% utilization, 2872/32607 MiB, 104.24 W. CPU-heavy rollout was active and episode progress advanced, so low instantaneous GPU utilization is not evidence of a stall. Disk: root 79% used with 6.4 GiB free; /root/autodl-tmp 34% used with 134 GiB free. No second simulator/trainer was running in the process snapshot.
- No fault requiring repair was found. No experiment process, configuration, or schedule was changed. Keep the one-GPU train-eval queue serial; next verify metrics when this evaluation completes and allow the existing runbook to advance without starting a duplicate task.
- Updated docs/EXP.md current execution status and docs/ORCHE.md current status and inspection table with this verified snapshot.

### 2026-10-21 16:04:03 CST half-hour Codex patrol

- Local start: 2026-10-21 16:00:17 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), group sudo. Remote /root/EMIX reached as uid=0(root).
- SSH path: direct connection initially succeeded, then two consecutive reads timed out at 198.18.0.8:49857 (exit 255); each was followed by a 20-second wait. ProxyCommand through 127.0.0.1:7892 succeeded and was used for subsequent SSH commands.
- Scheduler migration verified locally: crontab runs /home/rogers/.local/bin/emix-codex-patrol every 30 minutes; that script invokes codex exec with --dangerously-bypass-approvals-and-sandbox. This replaces the earlier tmux-message patrol route. This migration is recorded once here.
- Remote state at 16:03:23: FORMAL_RUNNING_TRAIN-EVAL 2026-10-21T13:55:24+08:00. First task place_bread_skillet_left_oppo_cam baseline evaluation completed: metrics.json reports 89 successes / 100 episodes (success_rate 0.89). The same task's EMIX training process PID 4238 and DataLoader workers were active; formal log advanced from 28052 to 29651/30000 steps between 16:02:17 and 16:03:23. No second simulator or trainer was observed.
- GPU 0 at 16:03:23: 94% utilization, 5992/32607 MiB, 472.34 W. At 16:03:02: 90%, 5992 MiB, 477.55 W. CPU load averages 6.63/8.62/8.78 at 16:03:02, with active training/DataLoader workers. Root disk 79% used, 6.4 GiB free; /root/autodl-tmp 41% used, 119 GiB free.
- No failure or resource conflict found; no experiment process, code, config, or queue was changed. Maintain the single-GPU serial train-eval runbook. Next verify EMIX final checkpoint and evaluation metrics after the active training completes, then let the runbook advance.

2026-10-21 16:04:39 CST patrol follow-up: place_bread_skillet_left_oppo_cam EMIX training finished; final 030000 model.safetensors exists. The same formal runbook (PID 936021, scripts/run.py PID 936041) advanced without restart to EMIX evaluation (PID 7715, 136% CPU) in emix-twin. GPU at 16:04:25 was 8%, 2697/32607 MiB, 107.21 W during evaluation startup. No duplicate job launched. Evaluation metrics are pending; continue serial queue.

2026-10-21 16:05:48 CST evaluation progress check: same EMIX evaluation remains active; formal log advanced to rollout step 183/500. GPU 0 sampled at 100%, 5869/32607 MiB, 275.77 W. The earlier low startup sample did not indicate a stall. No orchestration change.

### 2026-10-21 16:36:11 CST half-hour Codex patrol

- Local patrol start: 2026-10-21 16:30:16 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), supplementary group sudo; umask 0002. Remote checkout /root/EMIX was reached as root via ssh emix.
- SSH path: direct connection initially succeeded; the next two direct commands failed with exit 255 and `Connection to 198.18.0.8 port 49857 timed out`. Each was followed by a 20-second retry delay. ProxyCommand via 127.0.0.1:7892 succeeded and was used for the remaining patrol. Two later proxy sessions failed with `Connection timed out during banner exchange` and `Connection to UNKNOWN port 65535 timed out`; patrol continued after retry.
- Read remote AGENTS.md, root Markdown guidance, EXP.md, ORCHE.md, recent LOG.md, and all six SOPs. Existing cron-to-codex-exec migration was already recorded at 16:04 and was not repeated as a new event.
- Live at 16:35:29 CST: state FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. Formal runbook PID 936021, scripts/run.py PID 936041, and EMIX evaluator PID 7715 were alive; evaluator used about 127% CPU. First task place_bread_skillet_left_oppo_cam baseline metrics reports 89/100 success. Its baseline and EMIX final 030000 model.safetensors exist. EMIX evaluation log advanced from episode 25/100 at 16:32 to episode 28/100 at 16:34; its metrics.json is pending. All ten converted dataset .complete files and ten emix_index.json files are present.
- GPU 0 RTX 5090 D sampled 18% / 2180 MiB / 103.90 W at 16:32:09, 3% / 2106 MiB / 105.17 W near 16:35, and 15% / 5869 MiB / 246.35 W at 16:35:29. CPU load was 1.73/1.55/2.50 at 16:32:09. Root disk 79% used, 6.4 GiB free; /root/autodl-tmp 43% used, 116 GiB free. Variable GPU utilization accompanies active CPU-heavy simulator rollout, not a demonstrated stall.
- No fault or GPU conflict found. No experiment process, code, configuration, or schedule was changed. Keep the serial single-GPU queue. Next verify EMIX 100-episode metrics and allow the existing runbook to advance to patch training/evaluation without a duplicate launch.

### 2026-10-21 17:04:33 CST half-hour Codex patrol

- Local start: 2026-10-21 17:00:17 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), supplementary group sudo. Remote checkout /root/EMIX was accessed via ssh emix.
- SSH path: direct access first succeeded; two subsequent direct reads failed with exit 255 and exact error: Connection to 198.18.0.8 port 49857 timed out. Each failure was followed by a 20-second delay. ProxyCommand via 127.0.0.1:7892 then succeeded; subsequent patrol commands used that route. No local project checkout was used.
- Read remote AGENTS.md, root Markdown guidance, current EXP.md, ORCHE.md, recent LOG.md, and six SOPs. Scheduler migration was already recorded at 16:04; this is an ordinary subsequent patrol.
- Live at 17:04:01 CST: state FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. Formal runbook PID 936021, scripts/run.py PID 936041, and EMIX evaluator PID 7715 are alive. The place_bread_skillet_left_oppo_cam EMIX evaluation advanced from episode 53/100 near 17:02 to episode 54/100 at 17:03:35; rollout steps were still advancing. EMIX metrics.json remains pending; baseline metrics previously verified at 89/100. Ten converted datasets and ten indexes were observed earlier in this patrol.
- Evaluator used about 128% CPU. GPU 0 RTX 5090 D sampled 2% / 2686 MiB / 114.78 W at 17:02:18 and 63% / 3318 MiB / 120.45 W at 17:04:01. Root disk: 6.4 GiB free (79% used); data disk: 116 GiB free (43% used). No second simulator or trainer was found.
- No failure, conflicting GPU job, or stalled rollout was found. No experiment process, code, configuration, or schedule was changed. Keep the single-GPU serial train-eval queue; next verify EMIX 100-episode metrics and the automatic transition to patch training/evaluation.

### 2026-10-21 17:34:42 CST half-hour Codex patrol

- Local start: 2026-10-21 17:30:16 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), supplementary group sudo; umask 0002. Remote /root/EMIX reached as uid=0(root).
- SSH path: direct access initially succeeded. One later direct read failed with exit 255 and `Timeout, server connect.bjb1.seetacloud.com not responding.`; retry succeeded after 20 seconds. Two consecutive later direct reads failed with the same error and exit 255, each followed by a 20-second delay. ProxyCommand via 127.0.0.1:7892 succeeded and was used for the remaining patrol.
- Read remote AGENTS.md, root Markdown guidance, EXP.md, ORCHE.md, recent LOG.md, and all six numbered SOPs. The cron-to-codex-exec migration was already recorded at 16:04 and is not a new event.
- Live at 17:33:50 CST: state FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. Formal runbook PID 936021, scripts/run.py PID 936041, and EMIX evaluator PID 7715 were alive. First task place_bread_skillet_left_oppo_cam baseline metrics.json reports 89/100 success (0.89). Baseline and EMIX final 030000 model.safetensors exist. EMIX evaluation reached episode 82/100 success=True in the formal log; its metrics.json was absent. Ten converted dataset .complete files and ten emix_index.json files were present.
- Evaluator PID 7715 used about 128% CPU. GPU 0 RTX 5090 D samples: 20% / 2170 of 32607 MiB / 104.26 W at 17:33:40, and 29% / 2684 MiB / 104.63 W at 17:34:05. Root disk 79% used, 6.4 GiB free; data disk 43% used, 116 GiB free. No second simulator or trainer appeared in the process and GPU snapshots.
- No fault, stall, or conflicting GPU job was found. No process, code, configuration, or schedule was changed. Keep the single-GPU serial train-eval queue. Next verify EMIX 100-episode metrics and automatic transition to patch train/eval, then remaining tasks; do not start a duplicate job.

### 2026-10-21 18:03:54 CST half-hour Codex patrol

- Local start: 2026-10-21 18:00:16 CST; cwd /home/rogers/Projects/emix; uid 1000 (rogers), gid 1000, group sudo, umask 0002. Remote checkout /root/EMIX was accessed as root through ssh emix.
- SSH path: the first direct command succeeded; the next two direct commands failed with exit 255 and exact error `Timeout, server connect.bjb1.seetacloud.com not responding.` Each failure was followed by a 20-second delay. The 127.0.0.1:7892 ProxyCommand fallback succeeded and was used for the remaining patrol.
- Read remote AGENTS.md, root Markdown guidance, EXP.md, ORCHE.md, recent LOG.md, and all six numbered SOPs. The cron-to-codex-exec migration was recorded at 16:04 and was not repeated as a new event.
- Live state: FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. Formal runbook PID 936021 and scripts/run.py PID 936041 remained alive. First task place_bread_skillet_left_oppo_cam baseline metrics are 89/100; EMIX evaluation completed with metrics 84/100 at 17:53:17. The same runbook advanced to patch training without a restart. Patch trainer PID 74039 and its DataLoader workers were active; formal log advanced from step 10671/30000 at 18:03:03 to 11340/30000 at 18:03:34. Patch checkpoints 005000 and 010000 exist. Ten converted dataset .complete files and ten emix_index.json files remain present.
- GPU 0 RTX 5090 D at 18:03:11: 94%, 5992/32607 MiB, 474.45 W; at 18:03:34: 92%, 5992 MiB, 485.81 W. One GPU compute app (trainer PID 74039) was listed. CPU load average at 18:03:11 was 13.01/13.17/8.03 with active training workers. Root disk has 6.4 GiB free (79% used); data disk has 110 GiB free (46% used).
- No fault, resource conflict, or blocked GPU stage was found. No process, code, configuration, or schedule was changed. Keep the serial single-GPU queue. Next verify patch 30000-step checkpoint and 100-episode metrics, then the runbook transition to the remaining tasks.


### 2026-10-21 18:37:36 CST half-hour Codex patrol

- Local start: 2026-10-21 18:30:17 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), group sudo; cwd writable. All project inspection and changes used ssh emix at /root/EMIX as uid=0(root).
- SSH path: first direct connection to 198.18.0.8:49857 timed out (exit 255); a retry after 20 seconds succeeded. Later direct commands intermittently failed with the same TCP timeout and with "Timeout, server connect.bjb1.seetacloud.com not responding." (exit 255). After two consecutive direct failures and 20-second retry delays, ProxyCommand through 127.0.0.1:7892 succeeded and was used for the rest of this patrol.
- Read remote AGENTS.md, en.md, autodl.md, EXP.md, ORCHE.md, recent LOG.md, and all SOP files in bounded chunks. The cron-to-codex-exec migration was previously recorded at 16:04; this patrol did not record it as a new migration.
- Live state at 18:36:58: FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. Formal runbook PID 936021 and scripts/run.py PID 936041 remained alive. The first task place_bread_skillet_left_oppo_cam patch training produced the 030000 model.safetensors at 18:17:15 and the same runbook entered patch evaluation. Evaluator PID 78260 was active at about 131% CPU; formal log progressed from episode 17/100 at 18:34 to episode 20/100 at 18:36:51. Patch metrics.json was still absent. Baseline and EMIX metrics contain 100 records each and report 89/100 and 84/100 successes. Ten converted dataset .complete files and ten emix_index.json files were present.
- GPU 0 RTX 5090 D samples at 18:36:48 and 18:36:58 were 100% / 5869 MiB / 322.09 W and 71% / 2768 MiB / 94.00 W of 32607 MiB. Only the active evaluator appeared as a compute app in the earlier snapshot. CPU load at 18:34:34 was 1.10/1.61/3.99. Root disk remained 79% used with 6.4 GiB free.
- Capacity finding: three first-task training outputs occupied about 18 GiB per arm, with each 5k checkpoint directory about 3.0 GiB; the 200 GiB data disk had only 99 GiB free before cleanup. Verified baseline and EMIX 100-episode metrics, their final 030000 checkpoints, and run.py resume conditions. Removed only the ten 005000-025000 intermediate checkpoint directories for these two completed/evaluated arms. Preserved both final checkpoints, metrics, configurations, logs, and all patch checkpoints while patch evaluation remains active. Data disk free space increased to 128 GiB (37% used).
- No process, active configuration, GPU queue, source data, or formal evaluation was changed. Running: first-task patch evaluation; queued: remaining tasks under the same serial single-GPU runbook. No current fault or stage blocker. Next verify patch 100-episode metrics and automatic queue transition; after each completed arm, verify final checkpoint and metrics before considering intermediate-checkpoint cleanup to keep data-disk headroom.

### 2026-10-21 19:04:08 CST half-hour Codex patrol

- Local start: 2026-10-21 19:00:13 CST; cwd /home/rogers/Projects/emix confirmed; uid=1000(rogers), gid=1000(rogers), supplementary group sudo. Remote /root/EMIX reached as uid=0(root).
- SSH path: two consecutive direct connections to 198.18.0.8:49857 timed out (exit 255); each was followed by a 20-second delay. ProxyCommand through 127.0.0.1:7892 succeeded and was used for the remainder of this patrol.
- Read remote AGENTS.md, root Markdown guidance, current EXP.md, ORCHE.md, recent LOG.md, and all six numbered SOPs in bounded chunks. The cron-to-codex-exec migration was recorded at 16:04 and was not repeated as a new event.
- Live state: FORMAL_RUNNING_TRAIN-EVAL since 2026-10-21T13:55:24+08:00. Formal runbook PID 936021, scripts/run.py PID 936041, and patch evaluator PID 78260 remained active. First task place_bread_skillet_left_oppo_cam baseline and EMIX metrics have 100 records each, with 89/100 and 84/100 successes. Patch 030000 model.safetensors exists; patch evaluation had reached episode 44/100 by 19:02:15, and the following rollout advanced to step 31/500 at 19:03:37. Patch metrics.json remains absent. Ten converted dataset .complete markers and ten emix_index.json files were present.
- Evaluator used about 131% CPU. GPU 0 RTX 5090 D samples were 2% / 3104 of 32607 MiB / 85.85 W at 19:03:19 and 21% / 5869 MiB / 227.10 W at 19:03:37; only the evaluator appeared as a GPU compute app in the earlier snapshot. CPU load average at 19:02:38 was 1.19/1.36/1.84. Root disk: 6.4 GiB free, 79% used; /root/autodl-tmp: 128 GiB free, 37% used.
- No fault, stalled rollout, resource conflict, or waiting GPU compute stage was found. No experiment process, code, configuration, or queue was changed. Keep the single-GPU serial runbook. Next verify patch 100-episode metrics and automatic transition to the remaining tasks; after a completed arm, verify final checkpoint and metrics before considering intermediate-checkpoint cleanup.

### 2026-10-21 19:34:46 CST half-hour Codex patrol

- Local start: 2026-10-21 19:30:13 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), group sudo. Remote /root/EMIX reached as uid=0(root). All project file reads and writes used ssh emix.
- SSH path: initial direct command succeeded. Later direct commands intermittently failed with exit 255 and exact errors `Connection to 198.18.0.8 port 49857 timed out` and `Timeout, server connect.bjb1.seetacloud.com not responding.` Two consecutive direct failures were each followed by 20 seconds; proxy SSH via 127.0.0.1:7892 then succeeded and was used for subsequent commands.
- Read remote AGENTS.md and root Markdown guidance, current EXP.md/ORCHE.md/recent LOG.md, and applicable SOP training/evaluation/batch orchestration sections. Prior cron-to-codex-exec migration is already recorded at 16:04; no new migration event was added.
- State: FORMAL_RUNNING_TRAIN-EVAL since 2026-10-21T13:55:24+08:00. Formal runbook PID 936021, scripts/run.py PID 936041, and patch evaluator PID 78260 are active. First task place_bread_skillet_left_oppo_cam patch evaluation progressed from episode 44/100 at 19:02 to episode 74/100 at 19:34:19; patch metrics.json is pending. Baseline and EMIX previously verified 100-episode metrics report 89/100 and 84/100. All three 030000 model.safetensors exist. Ten emix_index.json files and 20 .complete markers were counted in the formal tree.
- GPU 0 RTX 5090 D at 19:34:19: 45% utilization, 2256/32607 MiB, 93.49 W; evaluator used about 130% CPU. Earlier sample at 19:33:34 was 16%, 2178 MiB, 83.54 W; another sample reached 33%, 5869 MiB, 296.38 W. Root disk 79% used with 6.4 GiB free; data disk 37% used with 128 GiB free. Advancing online rollout explains variable GPU utilization. No second simulator or trainer or waiting GPU compute stage was found.
- No fault or resource conflict requiring repair. Kept the existing single-GPU serial runbook; no process, code, configuration, or queue change. Next verify patch 100-episode metrics and automatic transition to remaining tasks. Only consider its intermediate-checkpoint cleanup after final checkpoint and metrics are both verified.
- 2026-10-21 19:35:52 CST follow-up: read all eight SOP-related Markdown files and the remaining EXP.md historical sections in separate <=120-line SSH reads. No additional active-stage constraint or fault was found.

### 2026-10-21 20:04:47 CST half-hour Codex patrol

- Local start: 2026-10-21 20:00:17 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), group sudo. Remote /root/EMIX reached as root only through ssh emix.
- SSH: initial direct AGENTS.md read succeeded. Two later direct reads failed before remote output; the second returned exit 255 with Connection to 198.18.0.8 port 49857 timed out. After 20 seconds, ProxyCommand via 127.0.0.1:7892 succeeded and was used for remaining commands. Scheduler migration was recorded at 16:04 and not repeated.
- Read remote AGENTS.md, root Markdown, EXP.md, ORCHE.md, recent LOG.md, and all six numbered SOPs in bounded SSH reads. State file: FORMAL_RUNNING_TRAIN-EVAL since 13:55:24; ten indexes and 20 .complete markers exist.
- First task place_bread_skillet_left_oppo_cam completed baseline/mix/patch evaluations: 89/100, 84/100, 87/100 successes, each with 100 records. Patch metrics written at 20:00:19; referenced 030000 final checkpoint exists. Original runbook PIDs 936021/936041 automatically advanced to place_can_basket_left_oppo_cam baseline training, PID 145528. At 20:03:46 log reached 4939/30000 steps, loss 0.014. No second trainer or simulator observed.
- GPU 0 RTX 5090 D at 20:03:46: 93%, 5868/32607 MiB, 461.87 W. Trainer CPU about 96%, DataLoader workers active. Root disk free 6.4 GiB.
- Verified patch final checkpoint, metrics, and run.py reuse conditions. Removed only five first-task patch intermediate checkpoints 005000 through 025000; preserved 030000, metrics, logs, configs, data, and videos. Data disk free space rose from 128 to 142 GiB (30% used). No active process, source data, configuration, or GPU queue changed.
- No active fault, GPU conflict, or waiting GPU compute stage. Keep single-GPU serial runbook. Next verify second-task baseline checkpoint/evaluation and subsequent mix/patch transition; monitor disk capacity.

### 2026-10-21 20:35:23 CST half-hour Codex patrol

- Local start: 2026-10-21 20:30:14 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), group sudo, umask 0002. Remote checkout /root/EMIX reached as root. All project inspection and writes used ssh emix.
- SSH: first direct command to 198.18.0.8:49857 timed out (exit 255); retry after 20 seconds succeeded. Two subsequent direct reads consecutively timed out (exit 255), each followed by 20 seconds. ProxyCommand through 127.0.0.1:7892 succeeded and was used for the remaining patrol.
- Read remote AGENTS.md, README.md, en.md, autodl.md, EXP.md, ORCHE.md, recent LOG.md, and all eight SOP-related Markdown files in separate reads of at most 120 lines. The cron-to-codex-exec migration was already recorded at 16:04 and was not repeated.
- State file remained FORMAL_RUNNING_TRAIN-EVAL since 2026-10-21T13:55:24+08:00. Formal runbook PID 936021 and scripts/run.py PID 936041 stayed active. Second task place_can_basket_left_oppo_cam baseline training produced its 030000 model.safetensors at 20:20:56; the same runbook automatically moved to baseline online evaluation, evaluator PID 149566 (about 127% CPU). Its formal log advanced from episode 6/100 at 20:32 to episode 7/100 at 20:34:33. Baseline metrics.json is still absent, so no success rate is reported.
- First task place_bread_skillet_left_oppo_cam baseline/mix/patch metrics were rechecked: 100 records each, success rates 0.89, 0.84, 0.87; their final 030000 checkpoints remain present. Ten emix_index.json files and 20 .complete markers remain present.
- GPU 0 RTX 5090 D at 20:34:33: 78% utilization, 3322/32607 MiB, 144.16 W; another sample during the rollout was 100%, 5869 MiB, 232.13 W. Root disk has 6.4 GiB free (79% used); /root/autodl-tmp has 125 GiB free (38% used). Evaluator rollout and episode progress confirm active GPU/CPU work. No second trainer or simulator, waiting GPU stage, fault, or resource conflict found.
- No process, source data, code, configuration, or queue change. Keep the existing serial single-GPU runbook. Next verify second-task baseline 100-episode metrics, then automatic mix/patch transition. Review intermediate checkpoint cleanup only after the corresponding final checkpoint and metrics are both verified; continue watching data-disk capacity.

### 2026-10-21 21:04:12 CST half-hour Codex patrol

- Local start: 2026-10-21 21:00:18 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), supplementary group sudo. Remote checkout /root/EMIX reached as root.
- SSH: two direct attempts failed with exit 255 and exact error `Connection to 198.18.0.8 port 49857 timed out`; each was followed by 20 seconds. ProxyCommand via 127.0.0.1:7892 succeeded (remote exit 0) and carried subsequent commands. The cron-to-codex-exec migration was already recorded at 16:04 and was not repeated.
- Checked remote AGENTS.md, root Markdown guidance, EXP.md, ORCHE.md, recent LOG.md, and active train/eval and batch SOP constraints. State remains FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. The original formal-run PID 936021 and scripts/run.py PID 936041 remain alive. Second task place_can_basket_left_oppo_cam baseline evaluator PID 149566 is active at about 125% CPU. At 21:03:18 the formal log reached Episode 25/100; the 030000 model.safetensors exists (written 20:20:56), while its metrics.json is still absent. Ten emix_index.json files and 20 .complete markers were counted.
- GPU 0 RTX 5090 D sampled 100% / 5869 of 32607 MiB / 274.08 W at 21:02:48, then 1% / 3346 MiB / 107.51 W at 21:03:47 during online simulation. Root disk free 6.4 GiB (79% used); data disk free 125 GiB (38% used); CPU load average was 1.15/1.29/1.63 at 21:02:48. Active evaluator and advancing episode log explain the variable GPU use. No second trainer/simulator, waiting GPU compute stage, resource conflict, or fault was found.
- No process, source data, code, configuration, or GPU schedule was changed. Keep the single-GPU serial train-eval runbook. Next verify baseline 100-episode metrics and automatic mix/patch transition. Only consider intermediate-checkpoint cleanup after final checkpoint and metrics are both verified; monitor data-disk capacity.

- 2026-10-21 21:05:37 CST follow-up: read all eight SOP-related Markdown files in separate at-most-120-line SSH commands. The active EXP protocol takes precedence over stale historical examples in SOPs. Baseline evaluator PID 149566 remained active at about 124% CPU; formal log advanced to Episode 26/100 at 21:05:24. GPU sample was 86% / 5869 MiB / 297.51 W. No new fault or scheduling change.


### 2026-10-21 21:35:49 CST half-hour Codex patrol

- Local start: 2026-10-21 21:30:17 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), supplementary group sudo. Remote checkout /root/EMIX reached as uid=0(root). All project file operations used SSH.
- SSH path: initial direct connection timed out at 198.18.0.8:49857 (exit 255), retry after 20 seconds succeeded. Later direct reads failed with exit 255 and errors `Connection to 198.18.0.8 port 49857 timed out` and `Timeout, server connect.bjb1.seetacloud.com not responding.`; each failed attempt was followed by 20 seconds. After two consecutive direct failures, ProxyCommand via 127.0.0.1:7892 succeeded and was used for the remaining patrol.
- Read remote AGENTS.md, root Markdown guidance, current EXP.md, ORCHE.md, recent LOG.md, and all eight SOP-related Markdown files in separate reads of at most 120 lines. The cron-to-codex-exec migration was already recorded at 16:04 and was not repeated.
- Live state at 21:34:57 CST: FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. Original formal-run PID 936021 and scripts/run.py PID 936041 remain alive. Second task place_can_basket_left_oppo_cam baseline evaluator PID 149566 is active at about 124% CPU; formal log reached Episode 43/100 at 21:34:48, up from Episode 26/100 at 21:05:24. A later invalid seed 100167 was skipped by the existing evaluator; no fatal error observed. Baseline final 030000 model.safetensors exists, while its metrics.json remains pending. Ten indexes and 20 .complete markers were counted.
- GPU 0 RTX 5090 D sampled 4% / 2792 of 32607 MiB / 96.02 W at 21:34:57, then 0% / 2262 MiB / 103.24 W at 21:35:19 during online rollout. The evaluator is the only GPU compute process observed. CPU load was 1.22/1.61/1.67. Root disk had 6.4 GiB free (79% used); data disk had 125 GiB free (38% used). Advancing evaluation confirms the low instantaneous GPU utilization is not a stalled or waiting GPU stage.
- No active fault, resource conflict, or waiting GPU computation stage was found. No experiment process, source data, code, configuration, or GPU queue was changed. Keep the single-GPU serial runbook. Next verify baseline 100-episode metrics and automatic transition to mix/patch; consider intermediate-checkpoint cleanup only after final checkpoint and metrics are both verified. Continue watching data-disk capacity.

### 2026-10-21 22:06:18 CST half-hour Codex patrol

- Local patrol start: 2026-10-21 22:00:19 CST; cwd /home/rogers/Projects/emix; local uid=1000(rogers), gid=1000(rogers), group sudo. Remote /root/EMIX accessed as uid=0(root); all project reads and writes used SSH.
- SSH path: initial direct command succeeded. Two subsequent direct commands failed with exit 255: `Connection to 198.18.0.8 port 49857 timed out` and `Timeout, server connect.bjb1.seetacloud.com not responding.` Each failure was followed by a 20-second wait. ProxyCommand via 127.0.0.1:7892 then succeeded and carried the remaining patrol. The cron-to-codex-exec migration was already recorded at 16:04 and was not repeated.
- Read remote AGENTS.md, root Markdown guidance, current EXP.md, ORCHE.md, recent LOG.md, and all six numbered SOPs in bounded SSH reads. Current state remains FORMAL_RUNNING_TRAIN-EVAL since 13:55:24; original runbook PIDs 936021/936041 and baseline evaluator PID 149566 are active. The evaluator used about 124% CPU. Second task place_can_basket_left_oppo_cam baseline evaluation reached Episode 61/100 by 22:04:47, up from Episode 43/100 at the previous patrol. Its final 030000 model.safetensors exists (20:20:56); metrics.json remains absent. First task place_bread_skillet_left_oppo_cam three metrics files were rechecked at 100 records and success rates 0.89/0.84/0.87. Formal assets contain 20 .complete markers and 10 emix_index.json files.
- GPU 0 RTX 5090 D sampled 75% / 3318 of 32607 MiB / 138.86 W at 22:04:47 and 84% / 3342 MiB / 132.48 W at 22:05:36. Only the active evaluator appeared as a GPU compute process. Root disk has 6.4 GiB free (79% used); data disk 125 GiB free (38% used); CPU load was 1.08/1.27/1.45. No second simulator/trainer or waiting GPU compute stage was found.
- No active fault, resource conflict, or stalled rollout was found. No experiment process, source data, code, configuration, or GPU queue was changed. Keep the serial single-GPU runbook. Next verify second-task baseline 100-episode metrics and automatic mix/patch transition. Consider its intermediate-checkpoint cleanup only after final checkpoint and metrics both exist; continue monitoring disk capacity.

### 2026-10-21 22:34:05 CST half-hour Codex patrol

- Local start: 2026-10-21 22:30:16 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), group sudo, umask 0002. Local PATH lacked date/id; absolute paths verified identity and time. Remote /root/EMIX reached as root; all project reads and writes used SSH.
- SSH: initial direct AGENTS.md read succeeded. Next two direct reads failed with exit 255: "Connection to 198.18.0.8 port 49857 timed out" and "Timeout, server connect.bjb1.seetacloud.com not responding." Each was followed by a 20-second wait. ProxyCommand via 127.0.0.1:7892 succeeded and carried later commands. Cron-to-codex-exec migration was already recorded at 16:04.
- Read remote AGENTS.md, root autodl.md and en.md, EXP.md, ORCHE.md, recent LOG.md, and all six numbered SOPs in bounded SSH reads. State remained FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. Formal-run PID 936021, scripts/run.py PID 936041, and place_can_basket_left_oppo_cam baseline evaluator PID 149566 remained alive; evaluator about 124% CPU. Formal log reached Episode 76/100 at 22:32:34, up from 61/100 at prior patrol. Final 030000 model.safetensors exists; metrics.json pending. Ten indexes and 20 .complete markers present.
- GPU 0 RTX 5090 D at 22:32:59: 81% utilization, 3342/32607 MiB, 140.51 W. Only evaluator PID 149566 was a GPU compute process. CPU load 1.77/1.57/1.51; root disk 6.4 GiB free (79% used); data disk 125 GiB free (38% used). No second simulator/trainer or waiting GPU computation stage was found.
- No failure, resource conflict, or stalled rollout. No process, source data, code, configuration, or queue change. Keep serial train-eval. Next verify second-task baseline 100-episode metrics and automatic mix/patch transition; consider removing intermediate checkpoints only after final checkpoint and metrics both exist. Monitor disk capacity.


### 2026-10-21 23:04:57 CST half-hour Codex patrol

- Local start: 2026-10-21 23:00:16 CST; cwd /home/rogers/Projects/emix confirmed; local uid=1000(rogers), gid=1000(rogers), group sudo. Remote checkout /root/EMIX accessed as root only through ssh emix.
- SSH: initial direct AGENTS.md read succeeded. Later direct reads intermittently timed out with exit 255 and errors "Connection to 198.18.0.8 port 49857 timed out" and "Timeout, server connect.bjb1.seetacloud.com not responding."; failed attempts were followed by 20-second waits. ProxyCommand via 127.0.0.1:7892 succeeded and carried the remainder of this patrol. The cron-to-codex-exec migration was already recorded at 16:04 and was not repeated.
- Read remote AGENTS.md, en.md, autodl.md, EXP.md, ORCHE.md, recent LOG.md, and all eight SOP-related Markdown files through bounded SSH reads. State file remains FORMAL_RUNNING_TRAIN-EVAL since 13:55:24. Original formal-run PID 936021, scripts/run.py PID 936041, and second-task place_can_basket_left_oppo_cam baseline evaluator PID 149566 remain alive. At 23:04:29 the formal log reached Episode 94/100, up from 76/100 at the preceding patrol. Final 030000 checkpoint exists; baseline metrics.json is still absent. Ten emix_index.json files and 20 .complete markers remain present.
- Evaluator used about 124% CPU. GPU 0 RTX 5090 D sampled 100%, 5869/32607 MiB, 258.86 W at 23:03:11, and 5%, 2848 MiB, 85.11 W at 23:04:29 during online rollout. The evaluator was the only GPU compute application observed. CPU load at 23:03:11 was 1.72/1.50/1.41. Root disk had 6.4 GiB free (79% used); data disk 125 GiB free (38% used). No second simulator/trainer or waiting GPU computation stage was found.
- No failure, resource conflict, or stalled rollout. No experiment process, source data, code, configuration, or queue change. Keep the serial train-eval runbook. Next verify second-task baseline 100-episode metrics and automatic transition to mix/patch; review intermediate checkpoint cleanup only after final checkpoint and metrics both exist. Monitor disk capacity.

### 2026-10-21 23:45:51 CST half-hour Codex patrol

- Local start: 2026-10-21 23:30:14 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), supplementary group sudo, umask 0002. Remote /root/EMIX reached as uid=0(root). All project reads and changes used SSH.
- SSH path: first direct command succeeded. Two later direct reads failed consecutively with exit 255 and exact error `Timeout, server connect.bjb1.seetacloud.com not responding.`; each was followed by a 20-second wait. ProxyCommand via 127.0.0.1:7892 succeeded (remote exit 0) and carried the remainder of the patrol. Cron-to-codex-exec migration was already recorded at 16:04 and was not repeated.
- Read remote AGENTS.md, root Markdown, EXP.md, ORCHE.md, recent LOG.md, and all eight SOP-related Markdown files in bounded SSH reads. State initially FORMAL_RUNNING_TRAIN-EVAL. Second task place_can_basket_left_oppo_cam baseline evaluation wrote metrics.json at 23:16:48 with 100 records, 56 successes (0.56). Its 030000 checkpoint was present. The original runbook had automatically entered the same task's mix training; at 23:33:24 trainer PID 242795 was active, with GPU 92%, 5900/32607 MiB, 474.12 W. Ten emix_index.json files and 20 .complete markers remained present.
- Verified baseline metrics and final checkpoint, then removed only its five 005000-025000 intermediate checkpoint directories (about 15 GiB). Final 030000, metrics, logs, configurations, and data were preserved; data-disk free space rose from 113 to 128 GiB at the immediate before/after samples.
- At 23:34:06 the formal runbook failed with RC=1 because mix DataLoader worker PID 243449 died by segmentation fault near step 20526/30000. Cgroup memory.events showed oom=0 and oom_kill=0; the underlying segfault cause is unconfirmed. The 020000 checkpoint contained model, optimizer, scheduler, RNG, and step state. No second GPU job was active.
- Resumed the mix arm from its 020000 checkpoint in the separate data-disk output place_can_basket_left_oppo_cam_dp_emix_resume_20261021, preserving batch size 64, num_workers 16, seed 0, and total steps 30000. The recovery process exited 0 at 23:43:14 after 10000 additional steps; its training_step.json reports 30000 and model.safetensors is 1043982652 bytes. Moved only the verified 030000 checkpoint into the canonical train output, updated the canonical last symlink, and wrote resume_provenance.json. The separate recovery output and mix-recovery.log retain the restart evidence. DataLoader iterator/sample order after restart may differ from an uninterrupted run; no equivalence claim is made.
- Relaunched the existing formal-run.sh from train-eval at 23:43:44 in tmux session sin-v0-patch-oppo-run-recover. It reused the completed first task's three arms and second-task baseline train/eval and mix training, then entered second-task mix online evaluation. At 23:45:06 evaluator PID 246362 was active at about 136% CPU and had reached simulator step 7/700; GPU was 72%, 5869/32607 MiB, 248.00 W. Formal state was FORMAL_RUNNING_TRAIN-EVAL. Root disk had 6.4 GiB free; data disk had 122 GiB free; CPU load 1.99/6.23/7.96 at 23:44:23. Only the evaluator used GPU compute. Mix metrics.json remains pending.
- Running: second-task mix online evaluation. Queued: its patch train/eval and the remaining tasks in the existing single-GPU runbook. No concurrent simulator or trainer was started. Next verify mix episode progress and 100-episode metrics, then automatic patch transition; continue disk monitoring and only clean intermediate checkpoints after final checkpoint and metrics are verified.

### 2026-10-22 00:05 CST half-hour Codex patrol

- Local patrol began near 00:00 CST in /home/rogers/Projects/emix; the first exact local timestamp captured was 00:00:43 CST because the initial PATH lacked date/id. Local uid=1000(rogers), gid=1000, group sudo, umask 0002; remote checkout /root/EMIX was accessed as uid=0(root). All project reads and writes used ssh emix.
- SSH: the initial invocation failed locally before connecting (exit 127, /usr/bin/timeout: failed to run command 'ssh': No such file or directory). After a 20-second wait, /usr/bin/ssh reached the remote host with exit 0. Two later direct reads failed with exit 255: "Timeout, server connect.bjb1.seetacloud.com not responding." and "Connection to 198.18.0.8 port 49857 timed out"; each was followed by a 20-second delay. The 127.0.0.1:7892 ProxyCommand fallback succeeded with remote exit 0 and carried subsequent commands.
- Read remote AGENTS.md, root autodl.md/en.md, current EXP.md, ORCHE.md, recent LOG.md, and all six numbered SOPs in bounded SSH reads. The cron-to-codex-exec migration was already recorded on 2026-10-21 at 16:04 and is not a new event.
- At 00:03:47 CST, state was FORMAL_RUNNING_TRAIN-EVAL since 2026-10-21T23:43:44+08:00. Recovery runbook PID 245944, scripts/run.py PID 245961, and second-task place_can_basket_left_oppo_cam mix evaluator PID 246362 remained active. The evaluator was at about 125% CPU. Formal log reached Episode 10/100 by 00:03; mix metrics.json was still absent. Baseline metrics previously verified at 56/100. Ten emix_index.json files and 20 .complete markers remained present.
- GPU 0 RTX 5090 D samples at 00:03:12/00:03:47: 4%/25%, 2112/3318 of 32607 MiB, 98.33/113.95 W. The evaluator was the only listed GPU compute application. CPU load at 00:03:12 was 1.06/1.47/3.40. Root disk had 6.4 GiB free (79% used); data disk had 122 GiB free (40% used). The active evaluator and episode progress explain variable GPU use; no second simulator/trainer or waiting GPU compute stage was found.
- No new failure, resource conflict, or stalled rollout was found. No experiment process, code, configuration, or queue was changed. Keep the single-GPU serial runbook. Next verify mix 100-episode metrics and automatic transition to patch train/eval, then monitor disk and review intermediate checkpoint cleanup only after final checkpoint and metrics both exist.

### 2026-10-22 00:34:38 CST half-hour Codex patrol

- Local start: 2026-10-22 00:30:12 CST; cwd /home/rogers/Projects/emix; uid=1000(rogers), gid=1000(rogers), supplementary group sudo, umask 0002. Local PATH lacked date/id; absolute paths verified time and identity. Remote /root/EMIX reached as uid=0(root); all project reads and writes used ssh emix.
- SSH: the first direct command succeeded; later direct reads intermittently failed with exit 255 and exact errors `Connection to 198.18.0.8 port 49857 timed out` and `Timeout, server connect.bjb1.seetacloud.com not responding.`. Failed attempts were followed by 20-second delays. After consecutive direct failures, ProxyCommand through 127.0.0.1:7892 succeeded and carried the remainder of this patrol. The cron-to-codex-exec migration was already recorded on 2026-10-21 at 16:04 and was not repeated.
- Read remote AGENTS.md, root Markdown guidance, EXP.md, ORCHE.md, recent LOG.md, and all eight SOP-related Markdown files in separate reads of at most 120 lines.
- Live state at 00:34 CST: FORMAL_RUNNING_TRAIN-EVAL since 2026-10-21T23:43:44+08:00. Recovery runbook PID 245944 and scripts/run.py PID 245961 remained active. Second-task place_can_basket_left_oppo_cam mix evaluator PID 246362 used about 123% CPU; formal log advanced from Episode 10/100 at previous patrol to Episode 28/100. Mix metrics.json remained absent, so no mix success rate is reported. Baseline metrics remain 56/100; baseline and mix final 030000 model.safetensors files are present (1043982652 bytes each). Formal assets contain ten emix_index.json files and 20 .complete markers.
- GPU 0 RTX 5090 D samples during the patrol ranged 4%-79% utilization, 2294-5871 of 32607 MiB, and 104.49-234.53 W; only the active evaluator appeared as a GPU compute process. CPU load at 00:33:11 was 1.46/1.53/1.73. Root disk had 6.4 GiB free (79% used); data disk had 122 GiB free (40% used).
- No new failure, stalled rollout, resource conflict, or waiting GPU computation stage was found. The active online evaluator includes CPU-heavy simulator work and must retain the single-GPU barrier. No process, code, configuration, or queue change was made. Running: second-task mix evaluation; queued: its patch train/eval and remaining tasks under the existing serial runbook. Next verify mix 100-episode metrics and automatic patch transition, then monitor disk capacity and consider intermediate-checkpoint cleanup only after final checkpoint and metrics are both verified.

### 2026-10-22 01:03:44 CST half-hour Codex patrol

- Local start 2026-10-22 01:00:15 CST; cwd /home/rogers/Projects/emix confirmed. Local uid=1000(rogers), gid=1000(rogers), sudo group; remote checkout /root/EMIX accessed as uid=0(root). All project operations used SSH.
- SSH: initial direct command succeeded. Two consecutive later direct reads failed with exit 255: `Connection to 198.18.0.8 port 49857 timed out` and `Timeout, server connect.bjb1.seetacloud.com not responding.`; each was followed by a 20-second wait. ProxyCommand via 127.0.0.1:7892 succeeded and carried the remaining commands. The cron-to-codex-exec migration was already recorded at 2026-10-21 16:04 and was not repeated.
- Reviewed remote AGENTS.md, root Markdown guidance, current EXP.md, ORCHE.md, recent LOG.md, and relevant SOP-00/01/02/03/04/05 constraints in bounded SSH reads. State file remains FORMAL_RUNNING_TRAIN-EVAL since 2026-10-21T23:43:44+08:00. Recovery runbook PIDs 245944/245961 and place_can_basket_left_oppo_cam mix evaluator PID 246362 were active; evaluator used about 123% CPU. Formal log reached Episode 43/100 by 01:02:23 CST, up from 28/100 at the previous patrol. Mix metrics.json is pending. Baseline metrics.json has 100 records and 56/100 successes. Ten emix_index.json files and 20 .complete markers were counted.
- GPU 0 RTX 5090 D sample at 01:02:35 CST: 65% utilization, 3322/32607 MiB, 143.14 W; the active evaluator was the only listed GPU compute process. CPU load was 1.37/1.39/1.52. Root disk had 6.4 GiB free (79% used); data disk had 122 GiB free (40% used). Rollout advanced during the patrol; no second simulator/trainer, waiting GPU computation stage, or resource conflict was found.
- No new fault observed and no experiment process, source data, code, configuration, or queue change was made. Keep the existing serial single-GPU runbook. Next verify mix 100-episode metrics and automatic transition to patch train/eval; monitor disk capacity. Consider intermediate checkpoint cleanup only after the final checkpoint and metrics for this arm are verified.
- Follow-up 2026-10-22 01:04:39 CST: completed bounded reads of all eight Markdown files under docs/sop (six numbered SOPs and two supporting notes), with at most 120 lines per SSH command. Rechecked formal log: mix evaluation advanced to Episode 44/100 by 01:04:31; runbook PIDs 245944/245961 and evaluator PID 246362 remained active. GPU sampled 65%, 3322/32607 MiB, 141.64 W. No intervention.

### 2026-10-22 01:29 CST path-layout correction

- Confirmed the active formal YAML had absolute `/root/autodl-tmp/sin-v0-patch-oppo/formal/...` roots and `formal-run.sh` set `EMIX_ADDITIONAL_OUTPUT_ROOTS`, bypassing the project output-root rule. Before the change, project `data/` contained only empty task directories, `outputs/` only contained the view-preview link, and `logs/` held 284 KiB of small logs; matching generic data-disk roots did not exist.
- Moved project `data/`, `outputs/`, and `logs/` to `/root/autodl-tmp/data`, `/root/autodl-tmp/outputs`, and `/root/autodl-tmp/logs`, then made the project paths symlinks. Existing view-preview and log contents were preserved. Added project-visible links to the active run tree.
- Moved the formal output directory within the same data-disk filesystem to `/root/autodl-tmp/outputs/sin-v0-patch-oppo/formal`; kept `/root/autodl-tmp/sin-v0-patch-oppo/formal/outputs` as a compatibility symlink. The project `outputs/` resolves to `/root/autodl-tmp/outputs`, so both old absolute paths and new project-relative paths resolve under the approved output root.
- Changed the formal YAML roots to project-relative paths and removed `EMIX_ADDITIONAL_OUTPUT_ROOTS` from the next-run script. Restored strict `resolve_project_output_path()` enforcement. Documented the mapping in `autodl.md`.
- At 01:29:59, the existing formal run remained active without restart: state `FORMAL_RUNNING_TRAIN-EVAL`; evaluator PID 246362 was at Episode 56/100 for `place_can_basket_left_oppo_cam` EMIX. Four completed evaluation metrics and five 30000-step checkpoints remained visible through the new project paths. Root disk had 6.4 GiB free; data disk had 122 GiB free at 01:27.


### 2026-10-22 01:33:22 CST half-hour Codex patrol

- Local start: 2026-10-22 01:30:34 CST; cwd `/home/rogers/Projects/emix`; local uid=1000(rogers), gid=1000(rogers), groups sudo; remote checkout `/root/EMIX` accessed as uid=0(root). No project files were read or changed locally.
- SSH: the initial local invocation failed before connecting because `ssh` was not found through PATH (exit 127); using `/usr/bin/ssh` connected directly at 01:30:54 and read remote `AGENTS.md`. A subsequent direct read failed twice with exit 255 and `Connection to 198.18.0.8 port 49857 timed out`; each failure was followed by a 20-second wait. ProxyCommand through `127.0.0.1:7892` then succeeded (remote exit 0) and carried the remaining patrol. This proxy path was used only after consecutive direct failures. The cron-to-codex-exec migration was already recorded and was not repeated.
- Reviewed remote root Markdown guidance, current `docs/EXP.md`, `docs/ORCHE.md`, the recent `docs/LOG.md` history, and the applicable training, evaluation, and batch-run SOP sections. Experiment protocol continues to require one-GPU serialized train/eval; active online evaluation runs in `emix-twin`.
- At 2026-10-22 01:32:24, formal runbook `/root/autodl-tmp/sin-v0-patch-oppo/formal-run.sh train-eval` and its `scripts/run.py` parent remained active (PIDs 245943-245961). Evaluator PID 246362 was processing `place_can_basket_left_oppo_cam` EMIX; the formal log advanced from Episode 56/100 at 01:29 to Episode 58/100 at 01:32. The 100-episode metrics file is still pending. Across the formal output tree, 10 final `model.safetensors`, 4 completed evaluation `metrics.json`, 10 EMIX indexes, and 20 `.complete` markers were found.
- GPU 0 RTX 5090 D sampled at 01:32:13: 29% utilization, 5869/32607 MiB, 311.67 W. The active evaluator was the only GPU compute process listed. It used about 123% CPU; system load was 1.49/1.56/1.44. Root filesystem had 6.4 GiB free; `/root/autodl-tmp` had 122 GiB free. The latest formal log reached Episode 58/100 and a video for episode 57 was still being written; evaluation is progressing.
- No stalled stage, second simulator/trainer, waiting GPU computation stage, or resource conflict was found. No process, data, code, configuration, or queue changes were made. Keep the existing serial runbook. Next verify completion and metrics for the current EMIX evaluation, then observe its automatic transition to patch train/eval and continue watching disk capacity.

### 2026-10-22 02:09:43 CST half-hour Codex patrol

- Local patrol began before 02:05 CST; the exact start time was not captured because the initial PATH lookup for `date` failed. At 02:08:20 CST, absolute local utilities confirmed cwd `/home/rogers/Projects/emix`, uid 1000 (`rogers`), gid 1000, supplementary group `sudo`, and umask `0002`. The scheduler is configured for `codex exec --dangerously-bypass-approvals-and-sandbox`; SSH reached `/root/EMIX` as uid 0. No local project copy was used or modified.
- SSH: the first direct connection succeeded and read remote `AGENTS.md`. A later direct read timed out with `Connection to 198.18.0.8 port 49857 timed out`; after 20 seconds a direct retry succeeded. Later two direct reads failed consecutively, first with `Timeout, server connect.bjb1.seetacloud.com not responding.` and then `Connection to 198.18.0.8 port 49857 timed out`; the failed reads were separated by a 20-second wait. ProxyCommand through `127.0.0.1:7892` then succeeded with remote exit 0 and was used for subsequent SSH commands.
- Read remote `AGENTS.md`, all root Markdown files, all of `docs/EXP.md`, `docs/ORCHE.md`, the latest 200 lines of `docs/LOG.md` in bounded chunks, and all eight Markdown files under `docs/sop/` in reads no larger than 120 lines. The cron-to-Codex-exec migration was already recorded at 2026-10-21 16:04 and was not repeated as a new migration.
- At 02:09:22 CST, state remained `FORMAL_RUNNING_TRAIN-EVAL` since 2026-10-21T23:43:44+08:00. Runbook tmux session `sin-v0-patch-oppo-run-recover`, runbook PID 245944, `scripts/run.py` PID 245961, and evaluator PID 246362 were active. The second task `place_can_basket_left_oppo_cam` EMIX evaluation reached Episode 79/100; its `metrics.json` is pending. The baseline metrics file is present; the previously verified baseline result is 56/100. Formal data tree contains 10 completed LeRobot datasets, 10 completed RoboTwin collections, and 10 EMIX indexes. The output tree contains four completed evaluation metrics files and five 030000 final checkpoints.
- The active evaluator used about 125% CPU. GPU 0 RTX 5090 D sampled 74% utilization, 3338/32607 MiB, 138.32 W at 02:09:22; a preceding sample at 02:08:31 was 0%, 2476 MiB, 106.18 W. PID 246362 was the only GPU compute process. Episode progress and the later GPU sample confirm variable use during online simulator rollout, not a demonstrated stall. CPU load was 1.21/1.30/1.68; root disk had 6.2 GiB free (80% used), and `/root/autodl-tmp` had 122 GiB free (40% used).
- Project `data/`, `outputs/`, and `logs/` resolve under `/root/autodl-tmp/`; the formal data and index artifacts are exposed through `/root/autodl-tmp/data/sin-v0-patch-oppo/formal/`, while formal outputs resolve under `/root/autodl-tmp/outputs/sin-v0-patch-oppo/formal/`. Counts above were checked against these actual formal paths.
- No fault, resource conflict, or waiting GPU computation stage was found. No process, data, code, configuration, or queue was changed. Keep the existing single-GPU serial runbook. Running: second-task EMIX evaluation; queued: its patch train/eval and the remaining tasks. Next verify the 100-episode EMIX metrics and automatic patch transition, then continue monitoring disk capacity. Do not start a duplicate evaluator or trainer.
