# sin-v0-patch 错误运行记录

最后修改：2026-10-22 CST

状态：**错误结果（BUG），整次正式运行作废；不得用于方法比较或汇总。** 实验版本 `sin-v0-patch`、协议时间戳 `261020` 不变。

原因：Patch 代码仅复制 B 图中标为 `patch_target` 的单个物体像素；本实验要求复制 B 图中所有非背景物体像素。旧 Patch 臂与协议不符，故本次运行结果整体标记为错误。

停止时已有完整 100-episode 评估：`place_bread_skillet_left_oppo_cam` 的 vanilla/mix/错误 Patch 分别为 89/100、84/100、87/100；`place_can_basket_left_oppo_cam` 的 vanilla 为 56/100。该任务 mix 评估尚未生成完整 metrics；其余条件尚无完整评估。这些数值仅用于识别已作废运行，不作为有效实验结果。

2026-10-22 约 02:18 CST 停止旧训练与评估，并关闭旧巡检。旧运行的 checkpoint 与评估产物按要求删除；训练数据和 EMIX 索引保留。修正后的正式运行从头开始，沿用同一版本与协议。
