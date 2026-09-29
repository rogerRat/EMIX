# Embodied Mixup: 基于位姿感知的模态混合打破本体感受主导

## 1. Introduction

融合视觉与本体感受输入的视觉-本体感受策略已成为机器人操作领域的主流范式。然而，如近期分析 [1] 所揭示，这一看似合理的模态融合隐藏着关键的失败模式：模型会发展出对本体感受的病态依赖，严重抑制视觉特征学习，并恰好在视觉信息不可或缺时导致显著的性能退化。

**核心观察。** 机器人操作轨迹可分解为两个截然不同的阶段：

| 阶段类型 | 核心特征 | 主导模态 |
|---|---|---|
| **运动一致阶段**（Motion-Consistent） | 持续、稳定的运动（如匀速接近、持续夹取），运动模式无切换 | 本体感受（提供精准的闭环运动控制） |
| **运动转换阶段**（Motion-Transition） | 运动模式发生切换（如接近 $\to$ 定位、抓取、装配、姿态调整），需要精准的空间定位 | 视觉（提供目标位置与环境线索；本体感受不包含目标相关信息） |

![](https://raw.githubusercontent.com/Koorye/Images/master/20260427200048761.png)

实验发现令人瞩目：

- **运动一致阶段：** 视觉-本体感受策略优于纯视觉策略。
- **运动转换阶段：** 视觉-本体感受策略相比纯视觉策略出现显著的性能退化。

**根因分析。** 在运动转换阶段，视觉线索往往极其细微，仅体现为像素级差异，本质上难以学习。相反，本体感受信号是低维且简洁的（如夹爪开合度、位姿增量），为损失下降提供了更为直接的途径。因此，模型倾向于优先依赖"更易用"的本体感受模态，使其主导整个优化过程，严重抑制视觉特征学习。这种抑制在运动一致阶段无害（本体感受足以支撑），但在运动转换阶段则成为灾难——此时视觉是唯一可靠的信息来源。近期研究 [1] 进一步从梯度层面证实：本体感受分支提供了更陡峭的损失下降路径，在训练中系统性压制了视觉模态的梯度贡献。数据集组织层面的分析 [3] 也揭示了同源的捷径依赖现象：大规模数据集的碎片化结构使得模型将任务无关的环境特征误作判据。

本文提出的Embodied Mixup方法将原本的单个峰值分布转换为多个峰值分布，从而惩罚只关注部分信息而只能输出有偏的单峰值分布的模型。即，构建的第二峰相当于是设计的一个 modality shortcut learning regularization. 

**本文贡献。** 我们提出 **Embodied Mixup**，一种训练时增强策略，显式识别本体感受歧义节点——即不同轨迹中相同机器人位姿产生不同动作的交叉点——并在这些关键节点执行模态感知的混合。通过在歧义状态建立显式的多峰动作分布，Embodied Mixup 打破本体感受主导，迫使策略利用视觉特征进行消歧。

## 2. Preliminaries

### 2.1 Problem Formulation

考虑一个机器人操作策略 $\pi_\theta(a \mid o, s)$，将视觉观测 $o$ 与本体感受状态 $s$ 映射为动作 $a$。训练数据集 $\mathcal{D} = \{(o_i, a_i, s_i)\}_{i=1}^{N}$ 由专家演示轨迹构成，其中 $o_i$ 表示视觉观测（RGB 图像），$s_i$ 表示末端执行器 6D 位姿，$a_i$ 表示动作指令。

### 2.2 Proprioceptive Ambiguity

我们定义**本体感受歧义节点**为一个状态 $s^*$，在该状态下两条或多条轨迹尽管共享（近似）相同的本体感受状态，却产生不同的动作：

$$\exists\, (o_1, a_1, s_1),\, (o_2, a_2, s_2) \in \mathcal{D}: \quad \|s_1 - s_2\| < \epsilon \;\wedge\; a_1 \neq a_2$$

在此类节点上，仅凭本体感受无法确定正确动作——需要视觉上下文进行消歧。这些节点通常与运动转换阶段重合，此时机器人需要根据环境线索切换运动模式。

### 2.3 Modality Greedy Bias

标准训练目标（如 MSE 损失、行为克隆损失）在优化过程中缺乏模态感知。当视觉与本体感受通路同时可用时，来自低维本体感受分支的梯度信号占据主导，形成对本体感受依赖的**贪心偏置**。正如 [1] 从梯度层面所揭示的：在运动转换阶段，视觉信号的损失下降曲线远比本体感受平缓，优化器自然倾向于沿本体感受梯度的方向快速下降，从而在参数更新中系统性压制视觉梯度的贡献。该偏置表现为：

1. 训练过程中视觉特征提取被抑制。
2. 推理时视觉注意力权重接近零。
3. 在本体感受歧义节点上发生灾难性失败。

## 3. Related Works

### 3.1 Modality Bias and Gradient Dominance

视觉-本体感受策略中的模态偏倚问题近期受到广泛关注。GAP [1] 通过时序控制实验系统地揭示了这一现象：在运动一致性阶段临时引入本体感受分支对策略性能几乎无影响，但在运动转换阶段进行同样的切换反而导致性能退化，表明视觉-本体感受策略在需要视觉指引的关键阶段已形成对本体感受的惯性依赖。GAP 进一步从梯度层面指出，本体感受分支的梯度信号提供了更陡峭的损失下降路径，在训练中自然占据优化主导地位，从而持续压制视觉模态的学习。为解决这一问题，GAP 提出了带阶段引导的梯度调节算法（Gradient Adjustment with Phase-guidance），利用本体感受信号估计每个时间步属于运动转换阶段的概率，并据此自适应地缩减本体感受梯度的幅度，以实现动态的模态协作。该方法在梯度层面直接介入优化过程，与本文从数据增强层面构建多峰监督信号形成互补。

### 3.2 Inference-Time Debiasing

与训练时干预不同，Policy Contrastive Decoding（PCD）[2] 提出了一种推理阶段的插件式去偏方案。其核心思路是：对同一策略输入原始图像与经目标对象遮蔽（inpainting）后的反事实图像，通过对比二者输出的动作概率分布，显式抑制策略对环境捷径特征的依赖：

$$\pi^{*}_\theta(a_i \mid l, o_i) = \frac{1}{C} \pi_\theta(a_i \mid l, o_i) \left(\frac{\pi_\theta(a_i \mid l, o_i)}{\pi_\theta(a_i \mid l, \hat{o}_i)}\right)^{\alpha}$$

其中 $\hat{o}_i$ 为目标对象被遮蔽后的反事实观测，$\alpha \geq 0$ 控制偏离强度。PCD 处理的是**视觉模态内部**的捷径学习——策略错误地依赖背景纹理、视角偏差等环境共现特征，而非任务相关对象。Embodied Mixup 处理的则是**模态之间**的捷径学习——策略错误地依赖低维本体感受信号而忽视视觉观测。二者作用层级不同，原则上可以叠加使用。然而 PCD 存在固有局限：当目标完全离开相机视野时，其 Track2Mask 管线无法生成有效掩码，方法自动退回基线策略；在近距离操作中目标占据视野比例过大时，过度遮蔽可能导致对比失效。

### 3.3 Dataset-Level Shortcut Learning

Shortcut Learning [3] 从数据组织层面揭示了捷径学习的另一重根因：当大规模机器人数据集（如 Open X-Embodiment）由多个独立采集的子数据集拼接而成时，子数据集内部多样性不足与子数据集之间分布差异过大共同导致模型将任务无关的环境特征误认为决策依据。该工作的核心洞察在于数据集缺乏足够的联通性（connectivity）与支持集重叠（support overlap）——数据中缺少仅改变任务核心因素而保持其他条件不变的对照样本，任务核心变量的变化总是伴随场景、视角、背景、光照等大量上下文变量的同步变化，使得核心因素与环境因素在训练分布中高度耦合。从统计学习角度看，训练数据无法区分究竟是 $\text{Core}_{[\cdot]} \to a_{[\cdot]}$ 还是 $(\text{Ltn}, \text{Scne}, \text{Vpnt}, \dots) \to a_{[\cdot]}$ 决定了动作。问题不在于模型能力不足，而在于训练分布本身缺乏足够的支持集覆盖，导致因果变量与环境变量无法被有效解耦。从几何角度看，不同子数据集在条件空间中形成若干彼此孤立的数据岛屿（data islands），缺少连接这些岛屿的过渡样本，模型无法沿连续变化路径观察"环境改变而任务不变"或"任务改变而环境不变"的情况，此即数据集变量空间的不联通（disconnected variable space）。

为解决这一问题，该工作提出视角增强（ZeroNVS 生成对立视角）和对象增强（Grounded-SAM2 分割交换）等数据层面干预手段。这些方法本质上是在构造原本缺失的反事实样本（counterfactual samples）和跨域桥梁（bridges），使模型能够观察到 $(\text{Core}, \text{Context}_A)$ 与 $(\text{Core}, \text{Context}_B)$ 同时对应相同动作的情况，从而学习到环境因素并非决定动作的关键依据。这些方法与 Embodied Mixup 在操作形式上具有相似性（均在视觉空间进行内容重组），但目标与约束截然不同：Shortcut Learning 的增强方法旨在提高数据集的联通性与因果可识别性，其核心问题是"模型应关注哪些变量进行决策"（What should matter?）；而 Embodied Mixup 的目标是在本体感受歧义节点附近构造合理动作的多峰监督信号，其核心问题是"同一状态下有多少种合理动作"（How many actions are reasonable?）。前者解决任务核心因素与环境因素的解耦问题，后者解决决策空间的多解性（multi-modality）问题；前者关注因果归因，后者关注行为分布建模，二者属于不同层面的泛化问题。

### 3.4 Data Augmentation and Mixup

经典 Mixup [4] 将经验风险最小化（ERM）扩展为邻域风险最小化（Vicinity Risk Minimization, VRM），其隐含假设为：训练样本凸组合所对应的标签亦应满足凸组合关系，从而在样本间空白区域施加低曲率的线性先验，约束模型函数空间复杂度。Mixup 的核心贡献在于揭示了训练样本间的线性插值可以同时正则化模型的记忆化倾向（CIFAR-10 上 80% 噪声标签下错误率仅 24.0%）、对抗脆弱性（白盒 FGSM 攻击下 Top-1 错误率从 90.7% 降至 75.2%）以及标签噪声敏感性，且几乎零额外实现开销。消融实验进一步表明，跨类别混合比同类别混合对泛化能力的提升更为关键，且与 Dropout 的组合效果最优。

Embodied Mixup 继承了 VRM 的核心思想——通过构造人工邻域迫使模型学习样本间平滑的决策边界——但将其从无结构的图像分类场景迁移至具有物理语义的机器人操作场景，并引入了本质性的改造：将随机配对替换为位姿一致性约束下的结构化配对，将均匀混合替换为保留本体感受稳定性的模态感知混合。

## 4. Methodology

### 4.0 Current Engineering Scope

当前仓库当前真正落地的 Embodied Mixup 训练版本, 是面向现有连续动作策略的工程实现.

已接入的三类 policy 分别是:

- `ACT`: transformer backbone + continuous action decoder
- `Diffusion Policy`: continuous action diffusion model
- `SmolVLA`: VLM backbone + continuous action flow-matching expert head

因此, 当前代码统一采用**加权双路 continuous-action loss**:

$$
\mathcal{L} = \alpha \mathcal{L}_A + (1-\alpha)\mathcal{L}_B
$$

即:

1. 对同一个 mixed input / condition 做一次训练前向或共享条件下的双路 loss 计算
2. 分别以 branch A 和 branch B 的连续动作目标计算两项 loss
3. 按混合系数 $\alpha$ 做加权聚合
4. 对总 loss 做一次 backward

必须明确说明:

- 当前仓库**暂时没有纯离散动作 token 自回归解码模型**
- 因此, 本文后面关于离散自回归 / soft-label token distribution 的内容, 当前仍是**方法草稿与未来扩展方向**
- 当前代码**没有实现**该离散版本
- 当前实现是面向 `ACT / Diffusion / SmolVLA` 现有代码形态的**连续动作空间版本**

### 4.1 Overview

Embodied Mixup 的核心思想是：在训练数据中识别本体感受歧义节点，并在这些关键点执行模态感知的混合。通过构建保留机器人位姿、同时插值环境特征的混合观测，并提供反映底层动作分布多峰性质的软动作标签，迫使策略通过视觉推理而非本体感受捷径来消歧。

方法由两个互补步骤组成：

1. **基于位姿一致性的图像拼贴：** 从共享相似本体感受状态的轨迹对中构建视觉混合观测。
2. **动作概率预测：** 用编码歧义节点处多峰动作结构的软标签分布替代确定性动作目标。

### 4.2 Step 1: Pose-Consistency Image Interpolation

给定训练数据集 $\mathcal{D}$，首先基于末端执行器 6D 位姿构建位姿索引表 $\mathcal{T}$。对于锚点样本 $(o, a, s)$，检索满足位姿一致性准则的近邻样本 $(o', a', s')$：

$$\|s - s'\| < \epsilon$$

其中 $\epsilon$ 为位姿相似度阈值。混合观测通过像素级线性插值获得：

$$o^{\text{mix}} = \alpha \cdot o + (1 - \alpha) \cdot o'$$

其中 $\alpha \sim \text{Beta}(\lambda, \lambda)$ 或 $\alpha \in (0, 1)$ 为混合系数。

**与经典 Mixup [4] 的理论关联。** 如第 3.4 节所述，经典 Mixup 将 ERM 扩展为邻域风险最小化（VRM）。Embodied Mixup 继承了 VRM 的核心思想——通过构造人工邻域迫使模型学习样本间平滑的决策边界——但将其从无结构的图像分类场景迁移至具有物理语义的机器人操作场景，并引入了本质性的改造：

1. **结构化邻域取代随机凸组合。** 经典 Mixup [4] 的配对是随机采样，而 Embodied Mixup 以位姿一致性为约束构建邻域，确保混合样本在机器人状态空间中具有物理意义——仅当两条轨迹的末端执行器位姿近似重合时才执行混合。
2. **模态感知取代均匀混合。** 经典 Mixup [4] 对所有输入维度施加同一混合比，而 Embodied Mixup 利用位姿约束天然地实现了模态感知的混合：机器人手臂因位姿一致而几乎不产生重影伪影（本体感受维度保持稳定），环境特征则经历平滑插值（视觉维度获得丰富增强）。

**核心优势。** 由于位姿一致性约束保证了机器人手臂在两张图像中占据几乎相同的位置，叠加后的 $o^{\text{mix}}$ 不会出现严重的手臂重影伪影。同时，环境特征——背景外观、物体颜色、物体位置——经历平滑插值，创造出丰富的增强视觉上下文，迫使策略必须学会解析这些视觉线索。

### 4.3 Step 2: Action Probability Prediction

动作目标的设计取决于策略架构。

#### 4.3.1 Autoregressive / Discrete-Token Policies

本节描述的是一种**尚未在当前仓库实现**的扩展版本.

原因是:

- 当前仓库接入的 `ACT / Diffusion / SmolVLA` 都不是纯离散动作 token 自回归解码器
- 它们的动作输出头仍然是连续动作回归、扩散或 flow matching
- 因此, 当前工程实现没有落地这里描述的双峰 soft-label / token-distribution 监督

对于将动作离散化为 Token 的策略（如 RT-2、OpenVLA），Embodied Mixup 自然转化为**软标签监督**：

- **输入：** 线性插值图像 $x_{\text{mix}} = \alpha x_1 + (1 - \alpha) x_2$；语言指令进行拼接或重写。
- **目标分布：** 不再要求模型预测单一动作 Token，而是构建两个原始动作对应 Token 上的混合分布：

$$P(a_{\text{target}}) = \alpha \cdot \text{OneHot}(a_1) + (1 - \alpha) \cdot \text{OneHot}(a_2)$$

- **损失函数：** 预测分布与混合目标分布之间的交叉熵，同时起到标签平滑的作用：

$$\mathcal{L}_{\text{CE}} = -\sum_{k} P(a_{\text{target},k}) \log P_\theta(a_k \mid x_{\text{mix}})$$

**为何不会坍缩为中间均值。** 自回归架构的归纳偏置在于：模型并非直接输出动作的确定性数值，而是输出动作空间上的**概率分布**（Possibility Map）。在 Embodied Mixup 的语境下，混合目标分布 $P(a_{\text{target}})$ 在动作空间中呈现为两个高密度概率球体——分别对应 $a_1$ 与 $a_2$ 的邻域，两者之间则是概率近于零的语义荒漠。推理时通过从该分布中采样抽取单一动作，天然跳过了中间无语义的区域，从而避免了模式坍缩。换言之，**概率分布的离散采样机制是防止均值退化的第一道屏障**。需要注意的是，这里的概率分布需要是在标准动作分布的较近邻域。即，球体的密度分布应该尽可能集中，否则在空间中可能对两个动作的输出数值平均分配一定的概率可能性，导致有可能采样输出无意义的平均动作。实际上，离散Tokens对应的概率是动作空间里概率密度的积分，因此甚至可设置一个极端的情况为“直接将标准答案所属的bin 设置为概率为 $\alpha \times 100\% $ 或者 $(1 - \alpha) \times 100\%$”

![image.png](https://raw.githubusercontent.com/Koorye/Images/master/20260413210801421.png)

#### 4.3.2 Diffusion Policies (Continuous Action Space)

这部分已经在当前仓库的训练代码中以工程形式落地:

- 对同一个 mixed observation / condition
- 共享同一组 diffusion timestep
- 分别以 A / B 两路连续动作目标计算两项 diffusion loss
- 再按 $\alpha$ 做加权聚合

它是 training-time weighted dual-branch diffusion loss, 不是对推理过程做两次 eval.

对于预测分数场的 Diffusion 策略，直接对预测动作取平均将退化为单峰均值。我们提出两种策略：

**策略一：加权分数匹配。** 在去噪过程中，针对同一混合输入，计算相对于两个目标动作的分数匹配损失的加权组合：

$$\mathcal{L}_{\text{diff}} = \alpha \|\epsilon_1 - \epsilon_\theta(a_{k,1}, k, x_{\text{mix}})\|^2 + (1 - \alpha) \|\epsilon_2 - \epsilon_\theta(a_{k,2}, k, x_{\text{mix}})\|^2$$

其中 $\epsilon_1, \epsilon_2$ 为两个动作的噪声目标，$k$ 为扩散时间步，$\epsilon_\theta$ 为模型的噪声预测。

**为何加权分数匹配不会退化为中间均值。** 尽管上述损失在形式上与 MSE 类似，但它**绝不会**导致模式坍缩到 $\alpha a_1 + (1-\alpha)a_2$ 的中间均值。原因有两层：

1. **去噪量是当前状态的函数，而非常数。** 去噪函数 $\epsilon_\theta$ 学习的并非"输出一个固定的去噪向量"，而是**从当前未去噪动作到去噪改变量的映射** $\epsilon_\theta(\cdot, k, x_{\text{mix}})$。该映射的标准答案由两个汇点（$a_1$ 与 $a_2$）各自向外发散的高斯噪声场构成。因此，动作空间中不同位置的粒子将接收到**不同的去噪矢量**——靠近 $a_1$ 的粒子被推向 $a_1$，靠近 $a_2$ 的粒子被推向 $a_2$。这本质上是在连续空间中雕刻出一个**双井势能面**，其中 $a_1$ 与 $a_2$ 是两个稳定汇点（吸引盆），而两者之间的中间位置是不稳定的马鞍点。
2. **去噪是动态时间演化的多步过程。** Diffusion 的最终输出并非去噪函数的单步结果，而是由多步马尔可夫链迭代演化而来。每一小步的去噪量虽然微小，但方向的差异会在迭代中被放大。因此，即使初始噪声只有极微弱的偏向，经过 $T$ 步演化后，动作粒子必然滚入两侧汇点之一，而不会停留在无语义的中间地带。

换言之，加权分数匹配的 Loss 在形式上是两个 MSE 项的线性组合，但**语义上**它约束的是去噪矢量场在每个局部位置的梯度方向，而非强迫模型输出一个折中动作。正是这种"局部梯度约束 + 多步演化"的机制，使得 Diffusion 策略能够自然地在连续动作空间中实现多峰消歧。

**策略二：多峰轨迹生成。** 推理时，由于模型在训练中已见过混合特征，Diffusion 过程会自适应地偏向两个动作模式之一。这使得模型在处理歧义输入时，能够从多条有效轨迹中采样选择其一，而非机械地坍缩为中间均值动作。

![image.png](https://raw.githubusercontent.com/Koorye/Images/master/20260413210751402.png)

### 4.4 Architectural Inductive Bias and Multimodal Representation Compatibility

从当前仓库代码实现角度, 还需要补充一个现实约束:

- `ACT` 在当前代码里是 continuous action chunk reconstruction
- `Diffusion Policy` 在当前代码里是 continuous action score matching
- `SmolVLA` 在当前代码里是 continuous action flow / velocity matching

因此, 当前版本的 Embodied Mixup 多峰监督, 不是通过离散 token 概率分布显式表达, 而是通过**同一 mixed condition 下的双路连续动作加权 loss**来注入多解结构.

Embodied Mixup 的有效性不仅源于其数据增强策略本身，更在于**架构的表征空间天然兼容多峰监督信号**。不同架构对动作空间施加了截然不同的先验偏置，决定了多峰消歧的实现路径：

| 维度 | 自回归 / 离散架构（VLA） | Diffusion 架构 |
| --- | --- | --- |
| **动作空间形态** | 不连续的离散网格。先验地认为世界由孤立的符号（Token）构成。 | 无限维连续空间。先验地认为世界由连续的物理流体（SDE/ODE）构成。 |
| **空间几何拓扑** | **双孤岛超球体。** 仅在 $a_1$ 与 $a_2$ 极其狭窄的邻域（超球体）内具有高概率值，球体之外是一片概率为 0 的语义荒漠。 | **双井势能面。** 动作空间不存在生硬边界，整个空间是由去噪函数统治的连续矢量场。$a_1$ 与 $a_2$ 是两个稳定汇点（吸引盆），两者之间的中间位置是不稳定的马鞍点。 |
| **多峰实现机制** | **静态空间算积分。** 直接在动作空间的多个离散位置（格子）上同时点亮概率质量，形成离散多峰分布。推理时从该分布中采样，天然跳过中间无语义区域。 | **动态时间走流场。** 不显式计算概率绝对值，而是通过多步马尔可夫链去噪演化，让动作粒子在连续空间中受去噪矢量场的"合力"驱动，自发分流至不同汇点。 |
| **消歧物理图像** | 在离散分布中进行二选一采样，跳过中间无语义的真空带。 | 动作空间被雕刻为双井势能面。正中间是极不稳定的马鞍点（山脊），粒子受随机噪声或微弱视觉拨动，必然滚入某一侧的汇点（盆地）。 |

**核心论点。** 两种架构之所以都不会退化为"中间均值坍缩"，根源在于各自的归纳偏置天然排斥单峰折中：

- **自回归架构** 的防御机制是**概率分布 + 离散采样**。模型输出的不是确定性数值，而是可能性地图（Possibility Map），推理时的采样步骤在结构上禁止输出分布之外的动作值，因此中间均值在采样空间中根本不存在。
- **Diffusion 架构** 的防御机制是**状态依赖的去噪矢量场 + 多步时间演化**。去噪函数学习的是"从当前位置到标准答案汇点的演化方向"，而非一个固定输出。两个汇点各自向外发散的噪声场在空间中形成了引力流场，多步迭代使微小的方向差异被放大，最终将粒子分流至某一侧汇点。

Embodied Mixup 的精妙之处正在于：它确保了**监督信号的多峰结构**与**架构表征空间的几何特性**之间形成了完美兼容——无论模型以何种方式编码动作空间，都能在其天然拓扑中找到对应的多峰锚点，从而在运动转换阶段完成视觉消歧。

## 5. Experiments

### 5.1 Experimental Setup

**仿真环境。** 我们在 ManiSkill2 的操作任务上评估 Embodied Mixup，使用 Franka Emika Panda 机械臂和 RGB 相机观测。

**数据输入。** 实验里的源数据必须先用 `data_processing/convert_to_lerobot_v2.py` 转成标准 `LeRobotDataset`，并且只接受这种标准结构，不接受其他本地布局变体。

**基线模型。** 对比以下标准训练的模型：
- Diffusion Policy（ResNet18 视觉骨干网络，ImageNet 预训练）
- ACT（Action Chunking with Transformers）
- SmolVLA

**评估指标。** 成功率、回合回报，以及分阶段成功率（运动一致阶段 vs. 运动转换阶段）。

### 5.2 Results

*（待实验完成后补充。）*

## 参考文献

[1] Lu J, Xia W, Wu Y, Lu Z, Hu D. When would Vision-Proprioception Policies Fail in Robotic Manipulation? [J]. arXiv preprint arXiv:2602.12032, 2026. (Accepted by ICLR 2026)

[2] Wu S, Luo X, Zhang J, Xie J, Song J, Shen H T, Gao L. Policy Contrastive Decoding for Robotic Foundation Models [J]. arXiv preprint arXiv:2505.13255, 2025.

[3] Xing Y, Luo X, Xie J, Gao L, Shen H, Song J. Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation [J]. arXiv preprint arXiv:2508.06426, 2025.

[4] Zhang H, Cisse M, Dauphin Y N, Lopez-Paz D. mixup: Beyond Empirical Risk Minimization [J]. arXiv preprint arXiv:1710.09412, 2017. (Published at ICLR 2018)
