# Experiment Results

正式实验结果汇总索引。每个结果页记录协议、数据来源、逐任务 metrics 和结论边界。

| 实验 | 日期 | 状态 / 结论 | 详情 |
|---|---|---|---|
| sin-v0-patch，left_oppo_cam | 2026-10-30 | 有效十任务批次；BaselineDP 38.7%，Mix DP 38.3%，Patch DP 38.3%。单 seed 描述性结果。 | [Patch 批次结果](results/20261030-sin-v0-patch-results.md) |
| sin-v0-patch，场景背景被误当作 Patch 前景 | 2026-10-22 | 无效批次，已作废，不纳入比较。 | [错误批次记录](results/20261022-sin-v0-patch-scene-background-bug-invalid-run.md) |
| sin-v0-patch，早期掩码错误运行 | 2026-10-22 | 无效批次，已作废，不纳入比较。 | [早期错误记录](results/20261022-sin-v0-patch-bug-invalid-run.md) |
| 10-task 500ep EMIX 回归审计 | 2026-09-11 | 完成；结果限制和逐任务分析见报告。 | [回归审计](results/20260911-top500-emix-regression-audit.md) |
| EMIX 历史结果比较 | 2026-09-11 | 历史跨批次比较，受配置差异限制。 | [历史比较](results/20260911-emix-historical-comparison.md) |
| 消融任务级结果 | 2026-09-16 | 完成；描述性结果。 | [消融结果](results/20260916-ablation-task-level-results.md) |
| 视角实验汇总 | 历史 | 汇总见报告。 | [视角实验](results/viewpoint_experiment_summary.md) |
| TorchCodec / FFmpeg 运行环境 | 2026-10-29 | 环境修复记录，非实验结果。 | [运行环境记录](results/20261029-torchcodec-ffmpeg-runtime.md) |
