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
