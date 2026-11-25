# 实验入口与结果管理

BAAI training-only protocol pi05-baai-baseline uses configs/scripts/pi05_baai_baseline.yaml; see the 2026-11-25 entry in docs/EXP.md.

唯一编排入口：`python scripts/experiment.py PROTOCOL --mode MODE`。目前支持 `multi-v0`、`multi-v1-scratch`、`v0-pi05-paramsearch` 和 `v0-pi05-100k`。使用对应实验环境的 Python，先检查 GPU、数据、磁盘及是否已有同一实验运行。

- `--mode prepare`：验证数据并生成训练和评估配置及索引，可能覆盖已有生成文件。
- `--mode mock`：验证已生成配置及真实样本；先执行 prepare。
- `--mode run`：准备、验证后依次训练和评估；会使用既有检查点恢复或跳过已完成任务。启动前检查输出目录与配置是否匹配，勿在同一路径并发运行。

人工配置分别为 `configs/scripts/multi_v0.yaml`、`configs/scripts/multi_v1_scratch.yaml`、`configs/scripts/v0_pi05_paramsearch.yaml` 和 `configs/scripts/v0_pi05_100k.yaml`。种子、任务、数据源、训练步数及评估参数均在此修改；`configs/scripts/<协议>/` 是生成物。当前内部实现固定输出路径；如需独立新实验，先扩展入口和协议实现，避免复用旧输出。底层训练和评估脚本是实现细节，不作为 Agent 启动入口。

完成后保存协议配置、生成配置、代码版本、运行日志、训练指标、评估 JSON、图片、视频及归档清单；保留最终评估所引用的模型和需要续训的检查点。清理前核对运行进程、评估配置与恢复路径，逐项记录删除清单；禁止清理正在运行的实验。归档后验证可读取并核对文件清单，再考虑清理原始文件。未经用户明确要求，不使用任何哈希。
