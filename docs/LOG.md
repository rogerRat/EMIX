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
