# 不同相机视角下 EMIX 的实验总结

## 摘要

本报告总结 `archive` 中 `place_object_scale_left` 在不同相机视角下的 Diffusion Policy 与 EMIX 对比实验。结果表明，EMIX 的效果具有明显的视角依赖性：在能够同时观察操作物体、目标区域和机械臂全局关系的 Wide 视角下，退火式 EMIX 通常明显优于 Base；在机械臂严重遮挡任务物体的视角下，EMIX 基本没有收益，部分配置反而下降。

本文的主要解释是：EMIX 的收益取决于动作分歧是否具有可见、可辨认的视觉原因。EMIX 并不是无条件地提高视觉使用率，而是在本体状态近似相同、动作发生分叉的样本上提高视觉条件的训练压力。当相机保留了导致动作分叉的任务信息时，这种压力可以削弱 proprioception shortcut；当任务信息被遮挡时，EMIX 会把不可观测性转化为冲突监督。

## 1. 实验设置

实验任务为 `place_object_scale_left`，策略为 Diffusion Policy。EMIX 使用当前帧视觉输入，匹配本体状态近似相同但未来动作存在分歧的两个样本。当前实现采用 anchor state，不对 state 做插值；图像使用同一个混合系数 `alpha` 混合，两个 member 的未来动作窗口使用对应的加权监督。该协议与 [EMIX 理论说明](./EMIX.md) 中的 point-observation 和双路动作监督定义一致。

主要配置如下：

| 项目 | 设置 |
|---|---|
| 任务 | `place_object_scale_left` |
| 观测 | 当前单帧 `head_camera` + robot state |
| 相机视角 | Central、普通 Left、Wide、Left-oppo |
| 输入分辨率 | 源 LeRobot 图像为 `84 × 84 × 3`；部分 Wide 实验使用名称含 `224` 的策略配置，需注意这通常表示网络侧 resize，而不是新增原始视觉信息 |
| 训练步数 | 30,000 steps |
| 评估规模 | 100 episodes/配置 |
| EMIX pair | 末端位姿近似一致、未来动作分歧 |
| 混合系数 | 主要为 `Beta(2, 2)`；Central 另有 `Beta(0.4)` 和 `Beta(1.6)` |
| 混合调度 | 固定概率、线性退火、log 退火和分段退火 |

需要强调，本文中的“EMIX 效果”不能只理解为一个单一数字。固定比例和退火比例代表不同的训练分布；在 Wide 视角下，二者甚至产生相反的结果。

## 2. 主要结果

### 2.1 各视角的成功率

| 数据/视角 | Base | EMIX 配置 | 成功率 | 相对 Base |
|---|---:|---|---:|---:|
| Central / 机械臂遮挡明显 | 15% | Constant，`Beta(0.4)` | 16% | +1 个百分点 |
| Central / 机械臂遮挡明显 | 15% | Constant，`Beta(1.6)` | 14% | −1 个百分点 |
| Central / 机械臂遮挡明显 | 15% | Linear，50%→0% | 9% | −6 个百分点 |
| 普通 Left | 29% | Constant，`Beta(2)` | 35% | +6 个百分点 |
| 普通 Left | 29% | Linear，50%→0% | 43% | +14 个百分点 |
| Wide | 6% | Constant，30% | 1% | −5 个百分点 |
| Wide | 6% | Linear，50%→0% | 24% | +18 个百分点 |
| Wide / 后续配置 | 0% | Linear，50%→0% | 20% | +20 个百分点 |
| Wide / 后续配置 | 0% | Log，50%→0% | 24% | +24 个百分点 |
| Left-oppo | 44% | Linear | 35% | −9 个百分点 |
| Left-oppo | 44% | Log | 37% | −7 个百分点 |
| Left-oppo | 44% | Step first-half | 37% | −7 个百分点 |
| Left-oppo | 44% | 其他 Emix 变体 | 21%–31% | −13 至 −23 个百分点 |

因此，结论不是“Wide 上所有 EMIX 都超过 Base”，而是：

1. Wide 上的退火式 EMIX 有稳定的正向信号；
2. 固定比例 EMIX 可能破坏后期真实观测分布上的校准；
3. 机械臂遮挡严重或视角导致任务信息不可见时，EMIX 的收益消失，甚至反转；
4. 普通 Left 视角提供了一个中间情形：视觉仍足够可用，因此 EMIX 可以改善 Base。

### 2.2 成对 seed 的结果

评估 episode 的 seed 大部分重合，因此可以进一步比较同一 seed 下两个策略的成败，而不只比较独立成功率。

| 对比 | 共同 seed | 仅 Base 成功 | 仅 EMIX 成功 | 两者均成功 |
|---|---:|---:|---:|---:|
| Wide Base vs Linear EMIX | 98 | 0 | 18 | 6 |
| Central Base vs Linear EMIX | 99 | 12 | 6 | 3 |
| Left-oppo Base vs Linear EMIX | 99 | 21 | 12 | 23 |

Wide 的 `0 / 18` 不对称结果支持其改善并非只由少数偶然 episode 造成；Central 和 Left-oppo 则显示 EMIX 无法稳定修复视角本身造成的观测缺陷。

## 3. 机制解释

### 3.1 EMIX 的隐含前提：动作分歧必须可由视觉解释

设两个样本满足：

\[
 s_1 \approx s_2, \qquad a_1 \ne a_2.
\]

EMIX 希望模型发现，两个动作分支的差异来自视觉中的任务变量，例如物体位置、目标位置、物体朝向、接触状态或遮挡关系，而不是来自本体状态本身。

这要求在选定的 pair 上，视觉对动作仍有条件信息：

\[
I(A;V\mid S) > 0.
\]

如果机械臂遮挡了真正决定动作的物体或目标，则可能出现：

\[
 V_1 \approx V_2, \qquad A_1 \ne A_2.
\]

此时模型面对的是近似相同的输入和不同的动作监督。Diffusion Policy 虽然能够表示多峰动作分布，但在当前 rollout 中没有足够证据选择正确的峰；EMIX 不能凭空恢复被遮挡的状态信息。

因此，EMIX 的作用更准确地说是“放大视觉条件依赖”，而不是“创造视觉可观测性”。

### 3.2 Central/遮挡视角中的失败模式

Central 的 debug 图像显示，机械臂在画面中占据主要区域，任务物体只在边缘或机械臂下方露出少量像素。该视角会同时造成以下问题：

1. **任务变量不可见。** 不同物体或目标配置在图像中可能几乎无法区分。
2. **像素差异被机械臂差异主导。** 即使 EEF 位姿接近，不同关节构型或遮挡边界也会产生显著像素变化。
3. **线性混合产生不真实图像。** 被机械臂挡住的物体可能在 mixed image 中以半透明形式出现，形成评估时不存在的视觉模式。
4. **动作分歧变成冲突监督。** 当两条轨迹的动作分叉原因被挡住时，模型只能学习一个更高熵、更难校准的条件动作分布。

这可以解释 Central 上 EMIX 仅达到与 Base 接近的水平，而 Linear 配置甚至下降到 9%。这里的下降不应简单归因于优化失败，更可能是 pair 的视觉可辨识性不足以及 mixed image 域差异共同造成的。

### 3.3 Wide 视角中的正向机制

Wide 视角的优势主要有三点。

首先，它通常同时保留机械臂、操作物体和目标区域，模型能够利用物体—目标—末端执行器之间的全局几何关系。动作分叉在视觉中具有可解释原因，因此 EMIX pair 更接近有意义的视觉反事实。

其次，相同的三维 pose 误差在远距离相机中对应更小的像素位移。相同的位姿阈值在 Wide 下更容易得到机器人区域对齐良好的 pair，混合后不容易出现严重重影。

最后，Wide 中物体虽然可见，但在 `84 × 84` 输入中可能只占较少像素。普通 Base 容易依赖更强的 proprioception shortcut；EMIX 则通过 state-matched、action-divergent pair 增强任务相关小区域的梯度，促使模型真正使用视觉。因此，Wide 上出现“Base 很低、退火式 EMIX 明显提升”并不矛盾：Wide 提供了较高的潜在可观测性，但需要 EMIX 帮助模型利用它。

### 3.4 为什么退火优于固定比例

Wide 实验中固定 30% EMIX 只有 1%，而 Linear 和 Log 退火分别达到 24% 和 24%。这说明混合数据的时序比简单的平均混合量更重要。

一种合理解释是 curriculum effect：

1. 训练早期使用较高 EMIX 比例，打破模型对 proprioception 的 shortcut；
2. 训练后期逐步降低 EMIX 比例，使策略回到真实图像分布上校准动作细节；
3. 避免在训练结束时仍然大量暴露于透明叠加或非物理 mixed image。

固定 30% 可能在训练后期持续引入域外观测，使 diffusion score 对真实 observation 的精细拟合受到干扰。该现象也说明 EMIX 更适合作为视觉表示和 shortcut-breaking 的训练阶段，而未必适合作为全程固定的数据分布。

## 4. 对当前结论的限定

当前实验支持视角与 EMIX 存在交互效应，但还不能将所有差异归因于相机本身。

### 4.1 Base 成功率过低

Wide Base 为 6% 或 0%，低于一个适合直接比较方法增益的稳定区间。此时 EMIX 从 0% 提升到 20%–24%，可以说明它可能改善了训练或视觉利用，但不能直接等价为“EMIX 在一个强 Base 上带来 24 个百分点的泛化增益”。应先验证 Base 的观测、数据、动作和评估链路。

### 4.2 各视角数据并非严格同一数据集

各数据集都约为 500 episodes，但总帧数、轨迹内容和 EMIX index 数量不同。Central 的 source dataset 命名也与其他视角不完全一致。因此，除相机外还可能存在示范轨迹、初始状态和 pair 分布差异。

### 4.3 位姿阈值没有做视角归一化

当前 pair 主要使用三维 pose 距离约束。相同的米制阈值在不同相机内参、距离和焦距下对应不同的像素误差，Central 与 Wide 的 pair 视觉质量可能因此并不相同。

### 4.4 训练 seed 数量有限

当前 archive 主要反映单个训练 seed，而 evaluation 只有 100 episodes。成对 evaluation seed 可以降低 rollout 方差，但不能代替多训练 seed 的统计结论。

### 4.5 策略实现需要做零 EMIX 等价性检查

Base 使用 `diffusion`，EMIX 使用 `emix_diffusion`。应确认在 `batch_prob=0` 时两者的 forward、归一化、动作采样和 checkpoint 初始化完全等价，排除 wrapper 或训练路径差异。

## 5. 后续实验建议

优先级最高的是建立“可见性—EMIX 增益”的直接证据，而不是继续盲目搜索混合比例。

### 5.1 同一状态跨相机重渲染

使用完全相同的 simulator state、action trajectory 和 pair index，仅改变 camera pose，生成 Central、Left 和 Wide 的图像。训练时保持 batch、优化器、总步数和随机种子一致。这样可以将相机视角从数据集差异中隔离出来。

### 5.2 记录 pair 级可见性指标

对每个 pair 记录：

- 任务物体和目标的可见面积比例；
- 机器人 mask IoU 或投影关键点误差；
- task ROI 内的 RGB/LPIPS 差异；
- mixed image 与真实图像的距离；
- 未来动作 divergence 和任务阶段。

然后按可见性分桶，比较每一桶的 EMIX uplift。预期结果是：任务相关视觉可见率越高、机器人投影越一致，EMIX 增益越大。

### 5.3 图像干预实验

至少比较以下输入：

1. 原始 Wide 图像；
2. mask 掉任务物体和目标的 Wide 图像；
3. mask 或 inpaint 机械臂的图像；
4. 遮挡视角增加 hand/辅助相机后的图像。

如果 Wide 的收益在移除任务物体后消失，而遮挡视角在增加辅助相机后恢复，则可直接支持条件可观测性解释。

### 5.4 训练控制变量

建议加入三类 control：

- **Same-action mix：** pair 的未来动作相近，只做视觉混合，用于区分普通数据增强和动作分歧机制；
- **Alpha shuffle：** 保持 mixed image 不变，打乱图像混合系数与动作损失权重之间的对应关系；
- **Visibility-gated EMIX：** 过滤任务物体可见率低或机器人投影不一致的 pair。

如果只有正确的 action-divergent pair、正确的 alpha 对齐和高可见性 pair 才能提升 Wide，则可以较强地确认本文机制。

### 5.5 调度和评估

在每个视角上使用相同的 EMIX batch 数量，比较固定 5%、10%、20%、30% 与 Linear/Log 退火，而不是直接比较不同实验中不一样的混合预算。每个配置至少使用 3–5 个训练 seed，并报告：

- 成功率均值和置信区间；
- 成对 seed 的胜负统计；
- approach、grasp、transport、place 分阶段成功率；
- 失败视频中的遮挡、抓取、对齐和放置失败比例。

## 6. 总结

当前实验最支持的结论是：

> **EMIX 的有效性由任务相关视觉的条件可观测性调制。Wide 视角保留了能够解释动作分歧的全局视觉关系，因此退火式 EMIX 能够削弱 proprioception shortcut 并提升策略；机械臂严重遮挡的视角无法提供这些解释线索，EMIX 反而会将不可观测性放大为冲突监督。**

因此，未来的 EMIX pair 筛选不应只检查三维 EEF pose 是否接近，还应检查视觉层面的任务可见性和机器人投影一致性。一个更完整的筛选原则应接近：

\[
\text{valid pair}=
\text{pose-consistent}
\land \text{action-divergent}
\land \text{task-visible}
\land \text{render-consistent}.
\]

这也意味着 EMIX 不是完全视角无关的方法。它更适合被理解为一种依赖视觉可观测性的数据构造和 shortcut-breaking 机制，而不是对任意相机配置都保证正向收益的通用增强方法。

## 7. Left-oppo-cam 的专项分析

### 7.1 Left-oppo 并非从头到尾都不可见

`left-oppo-cam` 不完全等同于 Central 的“全程严重遮挡”。从评估视频看，Left-oppo 在初始阶段通常能够看到待操作物体和电子秤；但随着机械臂接近物体，腕部和末端执行器从相机方向进入前景，物体在接触阶段逐渐被遮挡。

因此，Left-oppo 更适合描述为一种**阶段性可观测性不足**：

- 初始阶段：物体和目标通常可见；
- approach 阶段：物体可见率开始下降；
- grasp/contact 阶段：物体被机械臂严重遮挡；
- place 阶段：目标区域可能仍可见，但物体与夹爪的相对关系不完整。

这也解释了为什么 Left-oppo 的 Base 成功率为 44%，反而高于 Central 的 15%。Left-oppo 的原始图像仍足以让 Base 利用初始视觉、robot state 和自然轨迹先验完成一部分任务；但它并不保证在 EMIX 最需要视觉区分动作分支的时刻仍然提供足够信息。

### 7.2 EMIX 将部分可观测性变成条件动作冲突

Left-oppo 的 EMIX index 并没有表现出明显的数量级异常：pair 数量约为 16,204，pose xyz 距离均值约为 7.6 mm，action divergence 均值约为 0.090。这说明退化不太可能简单归因于“pair 太少”或“动作分歧太小”。

更可能的情况是：pair 在三维本体空间中满足约束，但在图像空间中不满足有效混合的要求。尤其在抓取阶段，可能出现：

\[
V_1 \approx V_2, \qquad A_1 \ne A_2.
\]

两条轨迹对应不同的物体位置、接触关系或抓取方向，但这些差异已经被机械臂挡住。由于当前实验采用单帧输入，模型没有历史帧可以回看，也没有辅助相机补充被遮挡信息。此时 mixed image 近似相同，而动作监督仍然要求模型解释两套不同的未来动作。

这会使 EMIX 产生比原始 Base 更强的条件歧义：Base 只需学习自然数据中的视觉—动作相关性和轨迹先验，而 EMIX 主动把不同动作分支压到相近的视觉条件附近。Diffusion Policy 可以表达多峰动作分布，却不能从当前输入中判断 rollout 应选择哪一个峰，因此可能出现动作分布变宽、动作分支错误或接触阶段不稳定。

### 7.3 与 Central 和 Wide 的区别

Left-oppo 与 Central 的差异在于：Central 从一开始就被机械臂占据大量视野，Base 自身已经很弱；Left-oppo 则在早期保留了更多可用视觉，因此 Base 可以取得较高成功率。Left-oppo 与 Wide 的差异在于：Wide 通常能在 approach、grasp、transport 和 place 阶段保留物体—目标—末端的全局关系，而 Left-oppo 恰好在 grasp/contact 阶段失去其中最关键的物体和接触信息。

因此，Left-oppo 的问题不是“视觉完全不存在”，而是：

> **动作分歧发生的关键时刻，决定动作分支的视觉原因不稳定或不可见。**

这是一种比全程不可见更容易误判的情况。模型在训练和评估初始阶段可能表现正常，但在接触和精确放置阶段会暴露出 EMIX 的冲突监督问题。

### 7.4 调度强度是放大因素，但不是根本原因

Left-oppo 的部分实验从较高混合强度开始：Linear 和 Log 配置均为 `1.0 → 0.0`；较温和的 Step first-half 配置则是在前 50% 训练中使用 0.5 的混合比例。结果为：

| 配置 | 成功率 |
|---|---:|
| Base | 44% |
| Constant 30% | 25% |
| Linear，1.0→0 | 35% |
| Log，1.0→0 | 37% |
| Step，前半程 0.5 | 37% |
| Step，前半程 0.8 | 28% |

这些结果表明，过高的早期混合比例或不合适的调度会加重退化；但即使使用相对温和的 Step/Log 调度，结果仍低于 Base。因此，调度强度更像是放大因素，根本问题仍然是 Left-oppo 在关键阶段的视觉可辨识性不足。

### 7.5 最可能的失败链条

Left-oppo 的失败过程可以概括为：

```text
物体初始可见
    ↓
Base 学到稳定的视觉与 proprioception 轨迹先验
    ↓
EMIX 匹配本体状态相近、未来动作不同的轨迹
    ↓
进入接近/抓取阶段后，机械臂遮挡动作分歧的视觉原因
    ↓
mixed image 无法唯一对应两套动作监督
    ↓
条件动作分布变宽或选择错误分支
    ↓
抓取和放置成功率下降
```

### 7.6 针对 Left-oppo 的验证实验

建议首先进行以下验证：

1. **只在物体可见阶段使用 EMIX。** 按任务阶段或物体可见率筛选 pair，删除接触阶段严重遮挡的样本，比较 all-stage EMIX 和 visible-stage-only EMIX。
2. **增加 pair-level visibility gate。** 除 pose distance 和 action divergence 外，加入物体可见面积、目标可见面积、机器人 mask IoU、task ROI feature distance 和 mixed-image realism 指标。
3. **加入历史帧或辅助相机。** 比较单帧 Left-oppo、短历史窗口、Left-oppo 加 hand camera，以及 Left-oppo 加 Wide camera，检验补充时序或视角信息后 EMIX 是否恢复正收益。
4. **按失败阶段统计成功率。** 分别记录 approach、grasp、lift、transport 和 place 的成功率，重点检查 EMIX 是否主要在 grasp/contact 阶段退化。
5. **加入 same-action 和 alpha-shuffle control。** 如果只有 action-divergent、alpha 正确对应且任务物体可见的 pair 才有收益，就能更直接地支持条件可观测性解释。

综上，Left-oppo 的结果不应简单概括为“左侧相机不好”。更准确的结论是：**Left-oppo 的原始视觉足以支持一个较强的 Base，但不适合未经可见性筛选的 EMIX；它在动作分歧最关键的接近和抓取阶段遮挡任务变量，EMIX 因而把自然数据中可利用的轨迹先验转化成了显式的条件动作冲突。**
