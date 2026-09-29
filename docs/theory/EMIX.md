# Embodied Mixup: Pose-Consistent Multimodal Mixup for Breaking Proprioceptive Shortcut

## Abstract

视觉-本体感机器人策略看起来天然合理，但近期研究表明，它们常常会在训练中形成对本体感信号的过度依赖，进而压制视觉特征学习，并在需要视觉消歧的关键阶段出现明显退化。本文提出 **Embodied Mixup (EMIX)**，一种面向具身操作策略的训练时增强方法。其核心思想是：在训练集中寻找末端位姿近似一致、但动作可能不同的样本对，在这些本体感歧义点上构造视觉混合输入，并用反映多峰动作结构的监督目标来训练策略。这样做的目标不是让模型输出“折中”的平均动作，而是迫使模型意识到同一位姿下可能存在多个合理动作，从而把决策依据从单纯的本体感捷径重新拉回到视觉上下文。

本文当前草稿面向现有连续动作策略实现，理论上覆盖 `ACT`、`Diffusion Policy` 与 `SmolVLA` 三类系统；但在实验规划上，我们先做一个最小验证实验：基于 LeRobot 实现的 `Diffusion Policy` 和 `SmolVLA`，在同一个 ManiSkill 仿真数据集上采用统一训练步数协议，先确认 EMIX 是否具有最基本的增益。`ACT` 暂时跳过，离散 token 自回归版本则作为后续扩展方向保留。本文的目标是给出一个清晰、可讨论、可落地的论文草稿框架，后续实验结果、消融与定量分析将在进一步讨论后补充。

## 1. Introduction

机器人操作中的视觉-本体感融合一直被认为是合理且必要的设计。视觉提供环境与目标的空间线索，本体感提供机器人自身状态与控制反馈，两者结合似乎能够同时兼顾“看清楚”和“动得准”。然而，最近的分析指出，这种融合并不总是带来真正的互补收益。相反，模型在训练过程中很容易形成一种稳定但脆弱的偏置：它优先利用更容易优化的本体感信号，而不是视觉信息。这个现象在运动一致阶段通常看不出严重问题，但在动作模式需要切换、位姿本身无法决定下一步行为的运动转换阶段，会迅速暴露为性能退化 [1]。

我们把机器人轨迹大致分成两类阶段。

1. 运动一致阶段：机器人执行持续、稳定的运动，例如匀速接近、持续夹取或平滑推送。在这类阶段中，本体感往往足以支撑闭环控制。
2. 运动转换阶段：机器人需要从一种运动模式切换到另一种模式，例如接近到定位、定位到抓取、抓取到装配，或者需要姿态重整。此时单靠本体感通常不够，视觉上下文成为决定动作的关键依据。

问题就在于，标准行为克隆式训练并不会自动区分这两类阶段。对优化器来说，低维、稳定、梯度更直接的本体感信号往往更“好学”，于是模型会先把损失降下来，再逐渐把视觉分支边缘化。换句话说，模型不是不知道视觉重要，而是在训练中被更省力的本体感捷径带偏了。近期工作从梯度竞争的角度也支持了这一点：在关键阶段，本体感通路往往提供更陡的优化路径，从而压制视觉特征的学习 [1]。

Embodied Mixup 的出发点很直接：如果模型总是在本体感足够相似的地方偷懒，那么就应该在这些地方人为制造“仅靠本体感无法唯一决定动作”的样本。EMIX 不是简单地对图像做插值，而是专门在位姿相近、动作可能分叉的位置上构造混合输入，并让模型面对一个更接近真实世界的训练信号：**同一个状态，可能有不止一个合理动作**。这会把原本容易被压成单峰的监督分布，重新拉成多峰结构，迫使策略必须真正读懂视觉。

本文的主要贡献可以概括为：

1. 提出一种面向机器人操作的位姿感知混合策略，用于显式打破本体感主导的 shortcut learning。
2. 将“本体感歧义节点”作为训练重点，在这些位置上构造带有多峰结构的监督信号。
3. 给出面向当前工程栈的连续动作版本，其中初步实验优先适配 LeRobot 上的 `Diffusion Policy` 和 `SmolVLA`，`ACT` 暂不纳入首轮验证。
4. 保留离散动作 token 版本作为未来扩展，使该方法可以自然延伸到更多具身基础模型。

## 2. Background and Related Work

### 2.1 Vision-Proprioception Policies and Modality Bias

视觉-本体感策略的直觉很强，但它们也最容易产生模态偏置。原因并不复杂：本体感信号通常维度更低、结构更稳定、与动作之间的局部对应更直接，因此在标准损失下更容易获得较快下降。模型一旦发现“靠本体感就能把训练损失压下去”，就会在优化中逐渐形成依赖。这种依赖在训练集上未必有明显代价，但在需要视觉消歧的关键节点上会变成灾难 [1]。

从更一般的角度看，这类问题本质上属于 shortcut learning。模型并不是学习了任务真正需要的因果线索，而是抓住了一个更简单的相关性代理。对具身任务而言，这个代理常常就是本体感。EMIX 的作用，就是人为打断这种简单相关性，让模型无法再把本体感当成唯一答案。

### 2.2 Mixup and Vicinal Risk Minimization

经典 Mixup [4] 的核心思想是将输入与标签同时做凸组合，从而把经验风险最小化扩展为邻域风险最小化。它的效果并不依赖复杂的结构设计，而是依赖一个简单但重要的假设：如果两个样本之间存在连续过渡，那么模型也应该在这个过渡区域上保持平滑。

EMIX 继承了这个思想，但做了关键改造。机器人操作不是普通图像分类，动作标签也不是天然的单峰类别。更重要的是，机器人状态空间存在物理语义，不能随意对任意两个样本做混合。于是 EMIX 不再随机配对，而是只在位姿相近的样本之间构造混合输入。这样得到的样本不是“纯粹的视觉拼接”，而是带有明确具身语义的训练邻域。

### 2.3 Multimodal Action Modeling

在具身学习中，动作并不总是单值函数。尤其在运动转换阶段，同一位姿附近可能对应多个合理动作：继续靠近、暂停观察、轻微调整、开始抓取等都可能合理。标准行为克隆常常把这种多解性压缩成一个确定性标签，结果就是训练信号本身被过度简化。EMIX 的一个关键目标，就是把这种多解性重新显式化。

这也是为什么本文不把动作目标简单写成单一回归值，而是强调多峰监督。对于离散 token 模型，可以直接构造软标签分布；对于当前仓库里的连续动作模型，则通过双路加权损失来表达同一个混合条件下的多解结构。

## 3. Method

### 3.1 Problem Formulation

我们考虑一个视觉-本体感策略

$$
\pi_\theta(a \mid o, s),
$$

其中 $o$ 表示视觉观测，$s$ 表示本体感状态，$a$ 表示动作。训练数据集为

$$
\mathcal{D} = \{(o_i, a_i, s_i)\}_{i=1}^{N}.
$$

这里的核心不是把动作看作单一确定值，而是承认同一类本体感状态附近可能存在动作歧义。若存在两个样本满足

$$
\|s_i - s_j\| < \epsilon, \quad a_i \neq a_j,
$$

则它们构成一个本体感歧义区域。在这个区域内，单靠本体感不足以唯一确定正确动作，视觉上下文必须参与决策。

### 3.2 Proprioceptive Ambiguity as Training Target

EMIX 的训练重点不是所有样本，而是那些“本体感相似但动作可能分叉”的节点。我们把这些节点称为本体感歧义节点。直观地说，它们就是机器人轨迹里的分叉口：从外部看，机器人差不多站在同一个位姿上，但下一步到底该怎么动，必须看环境。

这类节点通常对应运动转换阶段。例如：

1. 从接近目标切换到定位目标。
2. 从定位切换到抓取。
3. 从抓取切换到抬起或放置。
4. 从稳定运动切换到重新调整姿态。

如果训练集里这类节点被统一压成单一标签，模型就会倾向于记住“在这个位姿下最常见的动作”，而不是理解当前场景到底发生了什么。EMIX 要做的事情，就是在这些节点上强迫模型看到“多个答案都合理”，从而削弱这种单峰压缩。

### 3.3 Pose-Consistent Image Interpolation

给定锚点样本 $(o, a, s)$，我们首先检索另一个满足位姿一致性约束的样本 $(o', a', s')$：

$$
\|s - s'\| < \epsilon.
$$

其中 $\epsilon$ 是位姿相似度阈值。满足这个条件后，再构造混合观测

$$
o^{\text{mix}} = \alpha \cdot o + (1-\alpha)\cdot o',
$$

其中 $\alpha \in (0,1)$，通常可由 Beta 分布采样得到。

这个操作的关键不在于“把两张图像平均一下”，而在于“只对物理上允许的样本平均”。因为两张图像对应的机器人位姿近似一致，所以混合后不会把机械臂严重叠影成不可解释的形状；相反，背景、物体位置、颜色等环境信息会被平滑地插值出来。这样一来，视觉输入会保留足够的具身语义，同时又变得更加难以仅靠局部纹理猜答案。

从直觉上讲，这一步相当于在训练集中人为补出一些“中间地带”样本。模型如果只想依赖本体感，就会发现这些样本在视觉上是模糊的；如果想真正做对，就必须读懂视觉上下文。

#### 3.3.1 Current Point-Observation Contract

当前 EMIX-LeRobot 实验强制采用点观测，而不是历史轨迹观测：

$$
o^{\text{input}} = o_t, \qquad s^{\text{input}} = s_t.
$$

`n_obs_steps` 必须严格等于 1。EMIX index 中的 member 仍然是一个
`source_frame_index` 点；loader 可以继续读取后续帧来构造未来动作监督窗口，但这些
后续帧不得进入 `observation.*`。这样可以避免模型通过当前点之后的图像/state 变化
直接识别未来运动，从而保持“本体当前观测不能唯一决定未来动作”的实验语义。

当前实现对 EMIX loader、训练 batch 和 policy forward/select_action 都有单帧断言。
如果需要研究过去小轨迹，应作为单独的 temporal-EMIX 协议重新定义 pair、输入和评估，
不能与本文当前点观测实验混合。

对应到当前实现：

1. 当前帧图像使用一个 `alpha` 混合。
2. 当前 state 使用 anchor member 的 state，不做 state 插值。
3. 两个 member 的未来 action 窗口保留在 `emix_actions`，使用同一个 `alpha` 做监督。
4. action 窗口中的 padding 由 `action_is_pad` 或对应 mask 屏蔽，不进入损失。

### 3.4 Action Supervision for the Current Continuous-Action Stack

当前仓库已经接入的策略包括 `ACT`、`Diffusion Policy` 和 `SmolVLA`。它们的共同点是：动作输出仍然是连续空间中的回归、扩散或 flow matching 形式，而不是纯离散 token 自回归形式。就初步实验而言，我们优先在 LeRobot 的 `Diffusion Policy` 和 `SmolVLA` 上验证 EMIX，`ACT` 暂时保留为理论对照和后续扩展目标，因此本文当前可落地的版本以这两类隐式分布建模策略为主。

设混合输入对应两路原始动作目标 $a$ 与 $a'$，则总损失写成

$$
\mathcal{L} = \alpha \mathcal{L}_A + (1-\alpha)\mathcal{L}_B.
$$

这里的含义很简单：

1. 在同一个 mixed input / condition 上，分别计算两路目标的损失。
2. 用混合系数 $\alpha$ 做加权聚合。
3. 对总损失只做一次反向传播。

这个设计避免了把混合样本强行压成一个“平均动作”。原因在于，损失函数约束的是模型在同一条件下对两种动作解释的响应能力，而不是要求它输出一个居中的折中值。换句话说，EMIX 不是让模型学会“中间动作”，而是让模型学会“这个状态下可能有两类合理动作”。

当前工程实现里，这个原则是按“整段窗口共享同一个 $\alpha$”落地的，而不是按帧、按步重新加权。具体来说：

1. `Diffusion Policy` 先对两路动作序列分别加噪并前向，再用同一个 `emix_alpha` 混合两路 score matching loss，最后用动作有效掩码归一化。
2. `SmolVLA` 先对两路动作序列分别计算 flow matching loss，再用同一个 `emix_alpha` 对整段时间轴做加权，并用动作 pad mask 屏蔽无效步。
3. `ACT` 先预测单一路径动作，再计算两路目标与预测动作的 L1 loss，最后用同一个 `emix_alpha` 在整段动作窗口上加权。

#### 3.4.1 Temporal Action-Window Supervision

在具体实现中，输出同样不一定是单步动作，而可能是未来窗口内的动作序列，例如

$$
\{a_t, a_{t+1}, \dots, a_{t+H}\}.
$$

对于这种序列输出，训练时必须保证未来窗口内的动作标签完全可访问，也就是在构造 EMIX batch 时，监督端不能只看到当前时刻的单个动作，而要能够读取整段未来动作序列。随后，对窗口内每一个时间步都使用同一个混合系数 $\alpha$ 做一致加权的动作混合损失。

更具体地说，如果两路原始动作序列分别记为

$$
\mathbf{a} = \{a_t, a_{t+1}, \dots, a_{t+H}\}, \quad \mathbf{a}' = \{a'_t, a'_{t+1}, \dots, a'_{t+H}\},
$$

那么对应的监督应该在整个时间窗口上保持同一组混合权重，而不是对不同时间步分别采样不同的权重。这样可以确保动作序列的监督结构与输入侧的时间一致性对齐，避免把一个连续的未来窗口拆成彼此不一致的局部目标。

这条约束和输入侧的时间窗口一致性是对应的：输入侧要求多帧图像共用一个 $\alpha$，输出侧要求多步动作共用一个 $\alpha$，两者共同保证 EMIX 在时间维度上是一个一致的混合，而不是逐帧、逐步随机拼接出来的局部噪声。

#### 3.4.2 Why This Does Not Collapse to the Mean

对于连续动作策略，最危险的情况就是模型把两个动作的监督压成一个平均输出。但 EMIX 的目标不是做数值平均，而是保留多峰结构。

对于 `Diffusion Policy`，训练目标本质上是 score matching 或噪声回归。在这种设置下，模型学习的是不同动作邻域上的去噪方向。只要两个动作模式在状态空间中对应不同的吸引盆，去噪过程就会自然把样本分流到不同的汇点，而不是停留在中间均值。

对于 `SmolVLA` 这类连续动作回归或 flow matching 结构，双路加权损失的作用更像是把单点回归变成局部多解约束。模型会被迫在同一视觉条件下同时解释两种动作来源，最终学习到的是更平滑、但不单峰化的表示。

从工程上看，当前版本的关键不是“显式建一个概率分布”，而是“在训练信号上保留分叉”。这是当前连续动作栈中最直接、最稳定的实现方式。

#### 3.4.3 ACT Cannot Be Debiased by Loss Alone

这里需要单独强调一个结构性限制：`ACT` 的核心输出是**确定值**，而不是显式动作分布。它本质上仍然是一个确定性回归器，即便训练时对损失做再多加工，只要输出头不变，优化目标仍然会回到标准的 $L_2$ 回归范式。对于多峰数据而言，确定性回归的最优解通常仍然是条件平均，因此从原理上看，`ACT` 很容易发生均值坍塌。

这意味着，对于 `ACT` 来说，仅仅修改 loss 并不能严格避免坍塌。原因不是 loss 设计得不够好，而是**模型架构本身决定了它学到的是一个单点解**。如果训练目标背后存在 MultiModal Distribution (MMD)，那么想要真正建模多峰结构，必须修改 `ACT` 的输出形式，而不是只调整监督权重。

更严格地说，`ACT` 如果要表达多峰分布，至少需要引入以下一类架构改造：

1. 输出多个正态分布的均值和方差，用混合密度的方式显式表示多峰动作分布。
2. 直接输出多个候选动作解，再通过排序、选择或采样机制决定最终执行动作。
3. 将输出头改造为 MLP 加离散 token 预测表，使模型先预测动作离散分布，再从分布中采样或解码。

因此，`ACT` 的问题不是“能不能通过更聪明的 loss 处理多峰”，而是“它当前的单值输出头本身就不适合严格表达多模态分布”。在不修改模型架构的前提下，loss 最多只能起到软正则化作用，不能从根本上消除均值坍塌的倾向。

### 3.5 Discrete Token Extension

虽然当前仓库没有实现纯离散动作 token 自回归版本，但为了便于后续扩展，这里保留其理论接口。

如果动作被离散化为 token，那么 EMIX 可以自然退化为软标签监督。设两路原始动作 token 分别为 $a_1$ 与 $a_2$，则目标分布可写为

$$
P(a_{\text{target}}) = \alpha \cdot \text{OneHot}(a_1) + (1-\alpha) \cdot \text{OneHot}(a_2).
$$

对应的交叉熵损失为

$$
\mathcal{L}_{\text{CE}} = -\sum_k P(a_{\text{target},k}) \log P_\theta(a_k \mid x_{\text{mix}}).
$$

这一版本的直觉更直接：模型不再被要求在两个动作之间“选一个唯一答案”，而是被允许在多个合理 token 上分配概率质量。这样可以更自然地表达歧义状态下的多峰动作分布。

### 3.6 Architectural Compatibility

EMIX 的一个重要特点是它并不依赖单一模型架构，而是依赖一种更普适的几何关系：**输入空间中的位姿一致性，和动作空间中的多峰结构。**

对于离散 token 架构，模型天然输出概率分布，因此可以直接在多个 token 上分配概率质量。对于连续动作架构，模型以回归、扩散或 flow matching 的形式学习连续轨迹，此时多峰结构会通过局部引导和训练分流体现出来。但这里必须区分不同架构的能力边界：`Diffusion Policy` 和 `SmolVLA` 这类隐式分布建模方法，确实可以通过训练目标去塑造多峰行为；而 `ACT` 这种确定性回归头，即使配合多路 loss，也只能提供有限的正则化效果，不能仅靠 loss 就严格实现 MMD。

两类架构的形式不同，但目标一致：都不要让模型把一个本来多解的问题压缩成单解。只是对于 `ACT` 来说，这一目标需要先改架构，再谈 loss。

因此，EMIX 不只是一个具体增强技巧，更像是一种训练信号设计原则：当任务存在本体感歧义时，应该主动让监督分布变得多峰，而不是继续假装它是单峰的。

## 4. Experimental Design

本节记录当前 EMIX 实验设计、已观察到的问题和后续修正计划。当前阶段的首要目标已经从“直接比较 `Baseline` 和 `Ours`”调整为“先建立一个有非零且稳定成功率的 `Baseline`”。如果 `Baseline` 本身接近失败，则 `Ours` 的提升或退化都无法被客观解释。

### 4.1 Goal

最小实验只验证基本效果，不追求完整消融。核心假设是：在视觉-本体感混合策略中，如果训练过程在本体感相近但动作存在分叉的样本上显式引入 EMIX，那么模型在后续训练中应该更依赖视觉上下文，并在验证集或任务评测中表现出更好的泛化。

### 4.2 Models

首轮只使用两个基于 LeRobot 的连续动作策略：

1. `Diffusion Policy`
2. `SmolVLA`

`ACT` 不纳入这一轮实验，以避免把模型架构差异和 EMIX 机制本身的效果混在一起。

### 4.3 Dataset and Environment

初始实验只在 `StackCube` 任务上训练。数据集、预处理、训练 batch 采样方式和评估协议在所有对比组之间保持一致，尽量把变量压缩到是否启用 EMIX 这一项。

这里使用的数据输入必须先经过 `data_processing/convert_to_lerobot_v2.py` 转成标准 `LeRobotDataset`，并且只能使用这种标准结构；其他本地数据布局即使能被读取，也不属于当前实验定义。

当前已经训练并评估的 `Diffusion Policy` 配置包括：

1. `configs/train/baseline/DP_stack-cube.yaml`
2. `configs/train/baseline/DP_stack-cube-10w.yaml`
3. `configs/train/ours/DP_stack-cube.yaml`

这些配置共同使用 `data/lerobot/maniskill/stack_cube_rgb_b_500_84x84_nonoop`。该数据集的有效策略输入是 `observation.images.base_camera` 和 `observation.state`。其中 `observation.state` 是机器人关节和夹爪状态，动作监督是 `pd_ee_delta_pose` 下的 7 维末端增量动作：

```text
action = [dx, dy, dz, dax, day, daz, gripper]
```

需要特别注意：虽然数据集中存在 `observation.eef_pose`，但当前 LeRobot `Diffusion Policy` 实现只把字面键 `observation.state` 作为 robot state 条件输入。单独保留的 `observation.eef_pose` 不会进入 diffusion conditioning。因此，当前模型实际不是使用 `base_camera + qpos + eef_pose`，而是使用：

```text
policy_input = base_camera_rgb_84x84 + robot_qpos
policy_output = eef_delta_pose_action
```

### 4.4 Protocol

实验采用固定总步数的混合训练协议，而不是旧版本中的 warmup 后分叉协议。

1. `Baseline` 从第 1 step 到最后 1 step 都只使用普通 LeRobot 数据集训练，不读取 EMIX batch。
2. `Ours` 与 `Baseline` 使用相同的总训练步数、batch size、优化器、学习率调度、随机种子和源数据集。
3. `Ours` 在训练过程中按固定概率插入 EMIX batch，例如 `emix.training.scheduler.batch_prob = 0.10` 表示每一步有 10% 概率从物化 EMIX 数据集中采样一个 mixed batch，其余 90% 仍从普通 LeRobot 数据集中采样。
4. `Baseline` 与 `Ours` 的对比变量只允许是 batch 来源调度：`Baseline` 的 EMIX batch 概率为 0，`Ours` 的 EMIX batch 概率为预设比例。
5. 若需要从同一个 checkpoint 恢复训练，两组仍必须保持相同的 `start_step` 和 `total_steps`，并且不要让 `Ours` 获得额外训练步数。

为保证公平性，`Baseline` 和 `Ours` 必须共享同一个训练预算定义。旧版本中“前 4w step warmup、再分叉 2w step”的定义已经废弃；当前实现强调的是“相同总步数下，`Ours` 用少量 EMIX batch 替换普通 batch”。

### 4.5 Current Status

当前结果显示，直接增加训练步数不能解决问题。`Baseline` 在训练到 10w step 后，`SR` 仍然约为 `0.02` 到 `0.00`，并且 `Return` 和平均 `Reward` 没有随训练步数增加而稳定提升，反而出现下降。这说明当前问题不应继续被解释为训练步数不足。

当前更可能的问题包括：

1. `StackCube` 对观测要求较高，而当前策略只使用单个 `84x84` base camera 和 qpos。缺少 hand camera、深度、物体状态或真正进入模型的 eef pose，可能导致任务关键阶段不可观测或弱可观测。
2. `observation.eef_pose` 虽然存在于数据集中，但没有真正进入 `Diffusion Policy` 的 robot state 条件向量。若需要使用 eef pose，应把它显式合并到 `observation.state`，而不是作为单独字段保留。
3. 当前数据集使用 `nonoop` 预处理。该处理可能删除夹爪闭合、接触保持、对齐等待等对模仿学习有用的低速或静止片段，从而改变训练分布和评估分布。
4. 当前 EMIX 的 `action_filter` 对 7 维 `eef_delta_pose` 动作直接计算整体 L2 距离。这把平移、旋转和夹爪信号混在同一个尺度下，可能让 gripper 或旋转维度主导筛选结果。该逻辑可以作为粗筛，但不适合作为严格的末端动作分叉判据。

因此，当前 `Baseline` 和 `Ours` 的直接比较暂时无效。必须先得到一个稳定、可解释、非零成功率的 `Baseline`，再恢复 EMIX 对比。

### 4.6 Baselines

最小实验中的 baseline 只有一个：

1. `Baseline`: standard non-EMIX training.

它表示在相同初始化、相同源数据、相同总步数预算下，训练过程完全不引入 EMIX batch。与之对应，`Ours` 表示相同总步数预算下显式设置 `emix.training.training_mode: mixed`，并通过固定比例或调度器插入 EMIX batch。默认最小验证建议使用固定比例，例如 10% EMIX batch。

在当前修正阶段，`Baseline` 不再只是 EMIX 对比组，而是一个诊断门槛。只有当 `Baseline` 达到可比较的成功率后，才继续进行 `Ours` 对比。经验上：

1. 如果 `Baseline SR < 0.1`，优先诊断数据、观测、动作和评估链路，不做 EMIX 结论。
2. 如果 `Baseline SR` 进入可观察区间，例如 `0.3` 到 `0.7`，可以开始比较 `Ours` 是否提升。
3. 如果 `Baseline SR` 接近饱和，则需要改用更低数据量、更难初始状态或 sample efficiency 指标。

### 4.7 Metrics

最小实验只看一个主指标：

1. 主指标：`SR`。

在当前诊断阶段，还需要同时记录：

1. `Return`
2. 平均 `Reward`
3. `success_once`
4. `success_at_end`
5. 评估视频中的失败模式

这些指标不是最终论文主指标，但用于判断模型是在随机失败、稳定执行错误子行为，还是只差最后堆叠成功条件。

### 4.8 Expected Comparison

这一轮实验只关心相同总训练步数下 `Ours` 与 `Baseline` 的相对差异。

1. 如果 EMIX 有效，`Ours` 应在相同总 step 下取得更高 `SR`。
2. 如果两者差异不明显，则说明在当前 `StackCube`、阈值设置、EMIX batch 比例和总训练预算下，EMIX 还没有表现出最基本的收益。
3. 如果 EMIX 明显退化，则说明混合策略、位姿阈值或动作监督形式需要重新校准。

该比较只有在 `Baseline` 已经具备合理成功率后才成立。当前结果中，`Baseline` 和 `Ours` 都接近失败，因此不能据此判断 EMIX 是否有效。

### 4.9 Next Experiments

后续实验按诊断优先级推进，不优先更换模型。

第一阶段，保留 `StackCube` 和 `Diffusion Policy`，修正观测和数据处理：

1. 训练 `full data + base camera + hand camera + qpos` 的 `Baseline`。
2. 训练 `full data + base camera + hand camera + qpos + eef_pose` 的 `Baseline`，其中 eef pose 必须合并进 `observation.state`。
3. 对比 `full data` 和 `nonoop data`，确认 no-op 删除是否损害策略学习。
4. 暂停 EMIX batch，直到普通 `Baseline` 达到稳定非零成功率。

第二阶段，做低维 sanity check：

1. 使用包含 qpos、eef pose 和 object pose 的低维 oracle 版本训练 `Baseline`。
2. 如果低维 oracle 仍然失败，则优先检查动作标签、控制模式、数据重放、评估 wrapper 和训练代码。
3. 如果低维 oracle 成功，而视觉版本失败，则说明主要瓶颈是视觉可观测性、相机配置或分辨率。

第三阶段，修正 EMIX 候选筛选：

1. 将 `action_filter` 从整体 7 维 L2 改为显式分解的末端动作判据。
2. 分别计算 translation delta、rotation delta 和 gripper transition。
3. 保留 future eef xyz direction divergence 作为轨迹分叉信号。
4. 在修正后的候选集上重新构建 `emix_index.json`。

建议的动作筛选逻辑是：

```text
valid =
  future_xyz_angle >= threshold
  and (
    translation_delta >= trans_threshold
    or rotation_delta >= rot_threshold
    or gripper_changed
  )
```

第四阶段，如果 `StackCube` 仍然无法得到有效 `Baseline`，则切换到更简单任务做方法验证：

1. `PickCube`
2. `PushCube`
3. `PullCube`

其中 `PickCube` 和 `PushCube` 更适合作为首个可比较任务。等 EMIX 在较简单任务上完成闭环后，再回到 `StackCube`。

### 4.10 Scope

这一轮实验不包含：

1. `ACT` 对比。
2. 更复杂的阈值搜索。
3. 多数据集泛化。
4. 额外的结构消融。

这些内容留到最小实验跑通之后再补充。

## 5. Discussion

EMIX 适合解决的，不是所有形式的多模态学习问题，而是一个很具体的场景：视觉-本体感策略在关键阶段因为本体感过强而学坏了。它的优势在于实现简单、目标明确、和现有连续动作策略兼容性高。就当前计划而言，我们会先在 LeRobot 上用 `Diffusion Policy` 和 `SmolVLA` 做最小验证实验，在同一个 ManiSkill 仿真数据集上采用相同总步数的混合训练协议，先确认 EMIX 是否至少带来最基本的收益；`ACT` 暂时不纳入首轮验证，因为它的确定性回归头决定了它需要额外的架构改造才能严格表达多峰分布。

但它也有边界。第一，位姿一致性阈值 $\epsilon$ 的设定很重要，太松会引入不物理的混合，太紧则难以获得足够多的配对样本。第二，混合图像虽然能增强视觉歧义，但如果场景过于复杂，仍可能引入额外噪声。第三，EMIX 对不同架构的作用并不对称：`Diffusion Policy` 和 `SmolVLA` 这类隐式分布建模方法更适合通过 EMIX 的 loss 设计来表达多峰结构，而 `ACT` 作为确定性回归器，若不改输出头或整体架构，就无法严格避免均值坍塌。第四，EMIX 更擅长缓解“本体感主导”的问题，但不一定单独解决所有形式的环境偏置或数据碎片化问题。

因此，EMIX 更合理的定位不是一个万能方法，而是一个针对具身 shortcut learning 的数据层训练干预。它可以与梯度调节、推理时去偏、数据集重组等方法结合，形成更完整的训练方案 [1][2][3]。当前阶段的实验重点会先放在 LeRobot 的 `Diffusion Policy` 与 `SmolVLA` 上，先完成最小实验，再讨论是否为 `ACT` 增加架构级改造。

## 6. Conclusion

本文草稿提出了 Embodied Mixup (EMIX)：一种专门针对视觉-本体感机器人策略的位姿感知混合方法。EMIX 通过在本体感歧义节点上构造混合视觉输入，并使用多峰式动作监督，打断模型对本体感的单一路径依赖，使其在关键动作切换阶段重新依赖视觉上下文。

当前版本已经明确面向连续动作策略落地，并可作为 `ACT`、`Diffusion Policy` 和 `SmolVLA` 的统一训练框架草稿使用。需要注意的是，`ACT` 在这里更适合作为理论讨论对象或后续扩展基线；首轮实验我们将优先在 LeRobot 的 `Diffusion Policy` 和 `SmolVLA` 上，基于 ManiSkill 仿真数据集和相同总步数的混合训练协议验证 EMIX。若要严格建模 MMD，`ACT` 仍然必须先修改其输出头或整体架构。后续工作将围绕实验设计、定量分析、消融验证以及离散 token 版本扩展继续推进。

## References

[1] Lu J, Xia W, Wu Y, Lu Z, Hu D. *When would Vision-Proprioception Policies Fail in Robotic Manipulation?* arXiv preprint arXiv:2602.12032, 2026. Accepted by ICLR 2026.

[2] Wu S, Luo X, Zhang J, Xie J, Song J, Shen H T, Gao L. *Policy Contrastive Decoding for Robotic Foundation Models.* arXiv preprint arXiv:2505.13255, 2025.

[3] Xing Y, Luo X, Xie J, Gao L, Shen H, Song J. *Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation.* arXiv preprint arXiv:2508.06426, 2025.

[4] Zhang H, Cisse M, Dauphin Y N, Lopez-Paz D. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018.
