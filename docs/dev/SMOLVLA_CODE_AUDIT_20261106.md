# SmolVLA multi-v0 低成功率代码审计（2026-11-06）

范围：演示数据、LeRobot 转换、训练预处理、400k 模型和 RoboTwin 评估。正式 multi-v0-searchparam 评估运行中，没有改动其代码或配置。区分已经确认的差异与尚未测定的成功率影响。

## 已确认的两处训练评估错配

1. 场景指令。scripts/run_multi_v0.py 的 make_eval_config 为每项任务写入一条固定 task_prompt，来自第一条训练数据。evaluation/evaluate.py 调用 task.play_once() 后不使用返回的场景信息，此任务所有评估 seed 复用该 prompt。RoboTwin 子仓库 main 分支的 scripts/eval_policy_xpolicylab.py 同样在 play_once() 后通过 build_instruction 和 generate_episode_descriptions 生成本场景指令，再 set_instruction（main 中该逻辑约在 632-662、809-846 行；当前 emix 分支的 script/eval_policy.py 也有相同模式）。固定指令可能说错物体：place_cans_plasticbox 的固定文字提及 Coca-Cola can（071_can/base3）；抽查该任务 300 条训练场景，只有 46 条含这个资产，254 条没有。训练场景计数不能当成评估场景错配率，但证明同任务确有不同物体配置。place_burger_fries 固定文字对应的 base2 托盘仅在 300 条训练场景中出现 63 次。

2. 动作时间。专家数据通常每 15 个仿真步存一帧，仿真器为 250 Hz，所以连续帧约 0.06 秒。转换器以当前帧为观察、下一帧左臂 qpos 为 action；SmolVLA chunk_size=50、n_action_steps=50。评估却对每个模型 action 调用 task.take_action(qpos)，RoboTwin 为每个目标重新执行 TOPP 完整轨迹，每个 action 消耗的仿真步数不固定。对 blocks_ranking_rgb、place_burger_fries 各 100 对相邻专家帧按同一规划器测量，步数中位数为 67、70，各有 99/100 超过 15 步，90 分位数约 136、132。TOPP 返回 0 时的 50 步回退分别为 39/100、26/100。另用 400k checkpoint 在一个真实 place_cans_plasticbox 帧上生成 50-action chunk，同一规划器逐动作测量：中位数 101 步，50 个动作合计 5053 步=20.212 秒；训练采样语义约为 750 步=3 秒。这是一个场景的直接测量，不能当作全部 rollout 平均数。该 chunk 没有 TOPP 异常或 0 步回退，模型与专家单步最大关节变化中位数接近（约 0.046、0.047 弧度）。

## 已核查、未发现明显错误

- 左臂关节映射：RoboTwin qpos 顺序是左 6 关节、左夹爪、右 6 关节、右夹爪。转换与评估均用前 7 维；三个任务抽查原始 HDF5 与 LeRobot 对应帧，state/action 最大误差为 0。
- 视觉格式：原始 JPEG 与 LeRobot 84×84 视频帧平均绝对差约 0.005，符合有损压缩；交换红蓝通道后误差更大。评估也作相同 resize、除以 255。
- 预处理：运行项目 bootstrap_lerobot_runtime 后，真实样本通过训练和评估处理器，图像、state、语言 token 和 mask 完全相等。10 任务 action 归一化再反归一化最大误差约 4.07e-8；提示词均未达到 48 token 截断上限。
- 语言和图像确实进入模型：固定随机数同一输入输出一致；仅换图像时动作平均绝对差约 0.384，仅换文字约 0.273。这证明通道有效，不证明理解正确。
- 训练权重：基础权重和 400k checkpoint 的 500 个 tensor 名称与形状相同；冻结的视觉、文本主干未变，动作专家和 state projection 已变化。抽查动作均有限且在合理关节范围内。
- 评估显式错误：100k/200k/300k 共 3000 个 episode 未记录显式异常 note 或右臂运动告警；不能排除 TOPP 静默回退。

## 解释结果时的其他变量

100k 与 200k 共用 813/1000 个 seed，200k 与 300k 共用 805/1000；共有 seed 的成功率变化方向与总量一致，seed 变化不是低成功率的唯一原因。400k 学习率约 2.5e-6；视觉骨干冻结，这是可能限制效果的设置，尚非代码 bug。

## 下一步

保留正在运行的正式评估。随后固定同一 seed、场景清单、checkpoint，分别诊断：按 episode_info 提供场景指令；对齐动作执行时间；两项一起修改。逐步记录 TOPP 执行步数、失败与回退、仿真时间和成功条件。受控对照前，无法确定这两处差异各自降低了多少成功率。改评估协议后的成功率不能与旧协议数字直接等同比较。
