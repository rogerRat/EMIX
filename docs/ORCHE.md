# sin-v0-patch 执行编排记录

更新时间：2026-10-20（远端时区 Asia/Shanghai）。实验协议以 [EXP.md](EXP.md) 为准，执行细节遵循全部 SOP；本文只记录资源排程、状态与巡检决策。

## 执行约束

- 当前协议为 10 个 `left_oppo_cam` 任务 × `vanilla/mix/patch` 三臂；每任务采集 300 个成功 episode，17 FPS，84×84 RGB；三臂共享状态/动作数据与 EMIX 索引。Patch 训练必须使用逐帧无损实例标注及经验证的逻辑对象角色映射。
- 采集/仿真与评估运行于 `emix-twin`；转换、索引、可视化和训练运行于 `emix`。训练矩阵、处理顺序与归档依 EXP/SOP 执行。
- GPU 0 为 RTX 5090。单卡任务串行使用 GPU，CPU 预处理与日志/配置核对尽量排在 GPU 阶段之前或可安全重叠的间隙。不得并发启动两个仿真器或两个训练任务争用同一 GPU。
- 每 30 分钟检查状态、GPU 利用率/显存、磁盘、当前日志更新时间及阶段产物；GPU 空闲且前置条件满足时，立即推进队列下一项或调整批次衔接。不能通过违反协议依赖来填充 GPU。

## 阶段排程

1. 环境/仓库/磁盘和现存进程盘点；冻结并隔离不符合当前 `left_oppo_cam` 协议的旧运行。
2. 检查十个 `left_oppo_cam` task config 与目标对象角色表；做一个 episode 的真实 SAPIEN actor-ID/RGB/annotation 同步验收，覆盖大于 255 的 ID、边界帧及无损保存。
3. 验收通过后，正式采集十任务各 300 条成功轨迹；每任务立即做成功数、HDF5、RGB 与 annotation 对齐巡检，按 SOP 转换到 84×84。
4. 完成共用 EMIX index，跑索引完整性与交叉 episode 配对核验；生成并检查交集可视化。
5. 对十任务三臂生成解析配置并差异审计；小批量验收 vanilla/mix/patch 读数、拼贴边界/非目标像素和 `L_A + L_B` 语义。
6. 按资源屏障逐任务运行 30 个 30k-step 训练和最终 checkpoint 的 100 episode 评估；当前单 GPU 下依 SOP 串行训练/评估，准备下一任务的 CPU 配置/校验材料以缩短 GPU 空档。
7. 汇总结果、保留指标/配置/日志/摘要及 5% 评估视频，移除归档 checkpoint。

## 当前状态 / 2026-10-20

- 仓库 `/root/EMIX` 存在大量用户已有未提交修改；不回滚这些更改。GPU 0 空闲；根卷约 8.5 GiB 可用，`/root/autodl-tmp` 约 175 GiB 可用（以每轮巡检实测为准）。
- `sin-v0-patch-run` 的旧 `scripts/run.py` 作业及其子进程处于 SIGSTOP 状态，日志显示处理旧 `blocks_ranking_rgb_left_cen_arm_top_cam`；`/root/autodl-tmp/sin-v0-patch/data` 当前不存在。该旧视角与现行协议不同，不能把旧运行的输出作为本实验数据或继续进入训练。
- `sin-v0-patch-monitor` 旧监控按整点每小时运行，状态文件停留在 `PREVIEW_WITH_015_READY`；它不满足当前每 30 分钟巡检要求。新实验需使用独立 state/log/config 和 30 分钟监控，避免触发旧脚本的错误恢复。
- 视角预览已留存在 `outputs/view_previews/blocks_ranking/videos/`，仅作为相机预览，不替代正式标注数据。
- 当前开跑门槛尚未满足：尚需确认十个正式 task config 可用、十任务角色映射完成，以及至少一个任务真实实例 ID 与逐帧 RGB 对齐。没有通过门槛前不得启动正式批采集后的训练。

## 巡检记录

| 时间（Asia/Shanghai） | 阶段/进程 | GPU | 磁盘 | 发现与动作 |
|---|---|---|---|---|
| 2026-10-20 14:20 | 旧 run 冻结；转换日志在 top-camera 任务 | 0% / 2 MiB | 根卷 8.5 GiB；数据盘约 175 GiB | 确认旧运行不符合现协议，保留日志/检查点状态，停止将其作为续训来源；检查新协议采集配置后另建运行配置 |

### 2026-10-20 14:40 更新

- `dependencies/_install_twin.sh --check` 通过：RTX 5090 D、CUDA 12.8 Torch、SAPIEN 3.0.0b1、RT renderer 和 OIDN denoiser 可用；只有已知的 pkg_resources/Vulkan ICD warning，功能探针成功。
- 10 个 `left_oppo_cam` 任务配置都存在。正式配置 `configs/scripts/run_sin_v0_patch_oppo.yaml` 将三臂、300 episodes、标注开启及隔离数据根指向 `/root/autodl-tmp/sin-v0-patch-oppo/formal/`；pilot 配置单独指向 `/pilot/`。
- 真实验收 pilot 已启动：`blocks_ranking_rgb_left_oppo_cam` 单 episode，临时 task config，`--instance-annotations` 开启。不会更改正式 300-episode 配置。session: `sin-v0-patch-oppo-pilot`，状态文件 `/root/autodl-tmp/sin-v0-patch-oppo/state`。
- 新增 `sin-v0-patch-oppo-inspect` tmux 巡检器，每 1800 秒记录状态、GPU/磁盘/进程/日志年龄；formal 状态若 worker 消失则用同一 resumable runbook 恢复。巡检日志：`/root/autodl-tmp/sin-v0-patch-oppo/inspection.log`。
- pilot 启动时 GPU 利用率为 77%、显存 2329 MiB；RTX 5090 正在采集。根卷约 8.3 GiB 可用，数据盘约 178 GiB 可用。
- 下一步：检查 pilot 的实例标注对象表与逐帧 RGB/掩码一致性（含 >255 ID）；用 pilot 完成 84×84 标注转换并核验保存格式。通过后立即启动正式十任务 prepare 阶段；随后按强制阶段屏障做 index/vis/train-eval。
