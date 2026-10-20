# EMIX 的文献定位与改进路线

**核心判断**

EMIX 以缓解视觉—本体感模态偏倚为出发点，问题选择有充分的文献依据。需要修正的是中间机制：**让动作分布具有多个模式，并不充分保证策略根据视觉选择正确模式。** 更值得保留和发展的资产，是“机器人状态相近、后续行为不同”的配对发现能力；它可以用于直接学习视觉与动作的对应关系，也可以用于检测模型是否走了本体感捷径。

建议将工作定位为：**利用状态匹配的示范对，改善模仿策略的条件视觉依赖，使其在本体状态不足以决定动作时，依据可见任务状态选择正确行为。** 图像混合成为这一目标下待验证的训练手段。其价值应由真实图像上的分支选择、闭环成功率和扰动稳健性共同证明。

下一版最优先的实验是：保留普通 BC，在真实配对图像上增加动作对应关系辅助监督；再单独检验像素混合的增量收益。GAP 应成为直接的模态偏倚基线。基于扩散去噪误差的条件排序值得试验，但不应一开始替代 BC，也不应把所有动作分歧都当成负样本。

本文覆盖截至 **2026-09-11** 可核验的一手文献，重点分析 16 篇工作，另附[可筛选的文献对照表](20260911-emix-literature-matrix.csv)。正式会议论文、预印本、工作坊短文在来源表中分别标注。结论分为文献结果、仓库事实与待验证建议；没有新增训练结果，也不把外部基准的提升幅度迁移到本项目。

本地结果对应归档 `archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911`、提交 `3d3c51e`、运行 `flow-20260907-222045`。原检查点已按归档规范删除，本文中的条件能量、视觉干预和新损失均是后续验证方案，不是对原模型已完成的测量。

**一、当前实现与原始论证之间的距离**

当前 DP 的实际计算为：对两个端点图像进行像素级线性混合，保留第一个端点的本体状态，以同一个视觉—状态条件分别监督两个原始动作块。两次去噪使用同一份噪声和时间步，但输入的带噪动作不同。损失是两个分支去噪 MSE 的加权和；代码中没有独立的“正确图像应比错误图像更支持当前动作”的约束。[本地实现 L1–L2]

设图像为 $V_i,V_j$，本体状态为 $q_i,q_j$，动作块为 $A_i,A_j$，则可概括为：

$$
V_\alpha=\alpha V_i+(1-\alpha)V_j,\qquad c_\alpha=(V_\alpha,q_i),
$$
$$
\mathcal L_{\rm mix}=\alpha\ell_\theta(A_i;c_\alpha)+(1-\alpha)\ell_\theta(A_j;c_\alpha).
$$

这里“两个动作都接受监督”是代码事实；“最终采样分布必然呈现两个清晰峰”则还需要测量。更强的“为了得到两个峰，模型必然关注视觉”不能由这条损失推出。Diffusion Policy 本来就通过条件扩散建模多模态动作分布，表达多种动作的能力与有效利用条件信息是两个不同问题。[^1]

以下是针对当前设计的分析推导，而非某篇论文已经对 EMIX 作出的结论。假设固定 $q$ 下两种动作等概率出现，而所有训练样本都使用同一张混合图像。一个完全忽略视觉、只学习 $p(A\mid q)$ 的扩散模型，仍可学到双峰分布。对于带噪混合密度，其 score 满足：

$$
\nabla_x\log\sum_k w_kp_{k,t}(x)
=\sum_k r_k(x,t)\nabla_x\log p_{k,t}(x),\quad
r_k=\frac{w_kp_{k,t}(x)}{\sum_lw_lp_{l,t}(x)}.
$$

分支责任权重 $r_k$ 可以由带噪动作 $x$ 决定，不必来自图像。当前实现中，即使两个分支共享噪声，$x_t^i-x_t^j=\sqrt{\bar\alpha_t}(A_i-A_j)$ 依然存在。因此，“忽略视觉只能输出单峰”不是适用于 DP 的一般前提。

这个反例只否定必然性，不证明 EMIX 实际没有使用视觉。不同图像、不同混合比例以及原始 BC 样本，仍可能使视觉有助于降低训练误差。需要测量的是：**在未混合的真实图像上，视觉是否提高了正确动作分支的条件概率。** 用信息量表述，增加 $H(A\mid q)$ 本身不保证增加有用的 $I(A;V\mid q)=H(A\mid q)-H(A\mid V,q)$；也不能把这条恒等式直接当成当前损失的优化保证。

还要区分策略架构。当前 ACT 实现对同一次输出计算两个加权 L1，其逐坐标最优解具有加权中位数性质；它本身不要求输出两个模式。因此，如果后续论文同时覆盖 ACT 和 DP，需要分别描述监督机制，不能共用“强制双峰”的理论解释。[本地实现 L3]

**二、最相关文献如何改变工作定位**

|文献方向|原文提供的证据或方法|对 EMIX 的实际价值|适用边界|
|---|---|---|---|
|GAP，ICLR 2026[^2]|分析视觉—本体感竞争，按运动阶段调制本体分支梯度；包含 UNet-DP 实验|最直接的问题定位与方法基线|不能直接证明当前回退就是梯度竞争|
|OGM-GE，CVPR 2022[^3]|在线估计模态贡献，调制主导分支梯度|解释“低维易学信号压制另一模态”的优化路径|主要是多模态分类，贡献估计不能原样搬到 DP|
|Causal Confusion，NeurIPS 2019[^4]|模仿学习可能利用非因果关联；通过干预识别有效因素|约束因果表述，指导受控验证|近邻配对不是物理干预|
|CLASS，2025 预印本[^5]|用动作序列 DTW 相似性构造软对比视觉表征|最容易借用的视觉—行为监督路线|动作相似与任务等价并非总是一致|
|PCD，ICLR 2026[^6]|对比原图与对象遮挡图的动作概率，进行推理时重加权|直接研究“视觉条件对动作分布贡献多少”|扩散策略采用采样与 KDE，有代价及近似|
|ImMimic，CoRL 2025[^7]|映射对齐人类与机器人轨迹，再做跨域插值|机器人领域与 MixUp 接近的重要前作|混合条件和动作，与当前双端点分布监督不同|
|Crossway Diffusion，2023/2024[^8]|从去噪中间表征重建观测，联合辅助损失与 DP|辅助视觉学习必须比较的直接 DP 路线|重建背景也可降低损失，不等于动作相关性|
|Shortcut Learning，CoRL 2025[^9]|研究数据子集内部多样性不足与跨子集碎片化|提醒评估条件组合覆盖，而非只增加 episode|通用机器人数据集与单任务 DP 的证据层级不同|
|What Matters, When?，2026 工作坊短文[^10]|视觉目标随 pick/place 阶段及任务状态变化|强化“看什么、何时看”的条件依赖目标|近期初步证据，以 ACT 为主|
|Cross-View Action Consistency，2026 预印本[^11]|同物理状态的跨视角动作流一致性|为相机变化建立正确的不变性对照|flow-VLA 的结果尚需在本项目 DP 验证|
|Diffusion Classifier，ICCV 2023[^12]|用去噪误差估计不同条件的相对支持程度|构建无需完整采样的条件匹配诊断|去噪能量是代理量，不是精确动作似然|
|Set-Supervised DP，2026 预印本[^13]|利用纠正数据学习可接受动作集合；报告成对偏好损失局限|处理合法多解、错误负样本和纠正数据|不能把示范间分歧直接解释成人类否定反馈|
|Action Lookup Table，ICLR 2026[^14]|提出并实证分析稀疏示范下 DP 的动作记忆机制|检查收益是否来自更好的视觉检索与分支选择|是特定数据稀疏条件下的机制假说|
|BYOVLA，2024 预印本[^15]|测量并抑制对无关区域的敏感性|为“视觉更敏感”增加反向控制组|研究对象是 VLA 与测试时观测处理|
|Causal-ACT，2025 预印本[^16]|以结构因果建模处理观测中的因果混淆|帮助区分关联约束与因果可识别性|其假设与理论不自动适用于 EMIX|

GAP 和 CLASS 是下一轮最值得直接复现或适配的两条路线。GAP 的关键是按阶段调节学习，而不是让视觉和本体感始终同等重要；其附录已经覆盖 UNet 扩散头，相关性高于一般多模态分类工作。CLASS 则使视觉表征接受行为关系监督；EMIX 已有的相近状态、分歧动作索引，可以为这种监督提供较难的候选对。[^2][^5]

PCD 与 BYOVLA 提供互补的诊断视角：前者利用任务对象信息带来的动作分布变化，后者压制对无关区域的反应。因而不能仅用“遮住图像后性能下降更多”来证明缓解了模态偏倚；更有价值的结果是，对决定任务的视觉变化作出正确响应，同时保持对无关背景的稳健性。[^6][^15]

ImMimic 说明，机器人 MixUp 的研究叙事已有邻近工作。可以借鉴其先建立对应关系、再插值的顺序，但不能直接将不同正确动作平均为新的控制目标。EMIX 更可能成立的差异点是配对选择和条件行为约束；“首次把 MixUp 用于机器人”或“插值天然物理有效”均不应作为当前贡献。[分析判断][^7]

最新的阶段视觉依赖和跨视角一致性工作，使“在什么条件下应该变、什么条件下应该不变”成为更清晰的设计框架。但前者是非归档工作坊短文，后者是 2026 年 8 月预印本；它们适合启发实验，不能取代成熟基线或支撑过强的普适结论。[^10][^11]

**三、重新解释当前回退：四种可区分的机制**

历史分析已经排除了过于简单的单因素叙事。Bread Basket 从右广角 200ep 到俯视 200ep，baseline 从 26% 升至 42%，EMIX 从 33% 变为 31%；RGB 三个 200ep 机位的 home 配对占比都约 96%，却同时出现正负收益。当前 Can Basket 跨阶段配对占比很高，仍得到正差。因此，home 集中、跨阶段配对和预算替换是风险指标，不能单独解释任务收益符号。[本地证据 L5–L6]

|机制猜想|如何与模态偏倚目标联系|现有证据的边界|最有区分力的结果|
|H1：学到多分支，但视觉没有决定分支|输出多样性提高，条件选择没有同步改善|当前损失允许该情形；尚无 checkpoint 诊断证明其发生|EMIX 多样性上升，而真实图像配对匹配率、对象干预正确率没有上升|
|H2：混合降低了原本可用的视觉信息|模型被要求用视觉，但图像中关键状态变模糊|Bread/RGB 的视角翻转与此相容，尚未隔离相机和数据|固定轨迹与配对，真实端点监督优于强混合；视角×混合出现稳定交互|
|H3：分歧其实不能由当前单帧解释|任务阶段、接触状态或历史未观测；视觉也无法消歧|当前只用单帧；EEF 相近不等于完整状态相同|增加短历史或真实阶段状态显著改善，单帧视觉强化无效|
|H4：视觉改善存在，但控制曝光或精度损失更大|缓解一种偏倚，并不保证净收益为正|调度回放约替换 25% 普通批次；若干 pick 窗口稀释|保持普通 BC 曝光后收益恢复，或条件选择改善而抓取/放置精度下降|

这四种机制可以同时存在，且任务权重不同。例如，俯视使 Bread 的对象与完成状态更清晰，可能提高真实视觉条件的价值，也提高破坏它的代价；这是 H2 的任务化假说。Can 的净提升也可能主要来自有用的正则化或分支覆盖，即使 H1/H4 的某些风险同时存在。不能为了得到统一解释，把正向任务当作异常剔除。

当前三项负差为 Bread 45→37、RGB 28→26、Stack Blocks 49→45；十任务总成功数为 438→461。它们应被称为当前观察，而非已经跨训练 seed 复现的效应。历史机位数据分别采集、姿态阈值曾由 0.01 放宽到 0.26 rad，且各方法评测场景清单不完全一致；这些混杂因素要求新实验固定数据、训练 seed 与场景状态。[L5–L6]

**四、先把“相同状态、不同动作”分成不同问题**

建议将现有候选索引升级为带关系类型和可信度的配对表，而不是只存“距离够近、动作够远”。下表是方法设计建议。

|配对类型|例子|应该学习的性质|监督选择|
|本体相近，视觉任务状态不同，正确动作不同|手臂回到相近位置，但剩余面包位置不同|视觉敏感性：根据对象/进度选分支|真实图像动作对应、条件排序|
|完整场景相同，存在多个可接受动作|从目标两侧绕行都能成功|保留合法多模态|多正样本/集合监督；不强迫互斥|
|单帧相似，隐藏历史不同|接触状态或之前的操作决定后续行为|补足可观测性|短历史或记忆；不靠更强单帧惩罚|
|物理状态和动作相同，仅相机/无关外观不同|同一仿真状态用两台相机渲染|对无关变化保持动作一致|跨视角/背景一致性|

当前 EEF 距离阈值只能提供近似控制变量。尤其动作目标为绝对关节位置：EEF 相近时，完整关节状态仍可能不同，把 $A_j$ 放到 $q_i$ 下未必保持同样的执行意义。应增加完整活动关节差、夹爪状态、动作块有效长度，以及当前视觉是否能判断分歧原因的检查。[L1–L2、L4]

不建议直接删除所有 home 或跨阶段样本。回到相似姿态后，根据已经完成的操作选择下一对象，恰好可能是最有价值的视觉消歧场景。应该剔除或降权的是无法解释、无法执行、或者标签可信度低的配对，而不是仅凭“阶段不同”判断有害。

**五、改进 A：保留真实图像，用行为关系直接训练视觉表征**

这是首选的低耦合改动，借鉴 CLASS 的行为监督思想，但以下具体损失和状态匹配组合是待验证的本项目建议。[^5]

保留标准 DP 的原始图像 BC。对配对端点额外构造图像 $V_i$ 与候选动作块 $\{A_k\}$ 的匹配任务，使用共享视觉编码器 $f_\theta$ 和小型训练期动作投影头 $g_\phi$：

$$
s_{ik}=\operatorname{sim}(f_\theta(V_i),g_\phi(\psi(A_k)))/\tau,
\qquad
\mathcal L_{\rm align}
=-\sum_i\sum_{k\in P_i}\widetilde w_{ik}
\log\frac{\exp s_{ik}}{\sum_{l\in C_i}\exp s_{il}}.
$$

$C_i$ 是候选集合，$P_i$ 是一个或多个正样本，正权重归一化。初版可以采用高置信的一正一负；存在合法多解时保留多正样本。动作表示 $\psi$ 可先比较“活动关节未来变化＋夹爪事件”与“未来 EEF 位移轨迹”，不要让冻结右臂、填充帧或不同起始关节位置主导距离。DTW 适合容忍不同演示速度，但夹爪开闭时机需要单独处理，不能被自由时间对齐抹掉。

这个辅助头不读取 $q$、带噪动作、episode 编号或端点序号；候选顺序随机化。它必须借助图像与动作的对应关系降低损失，不能仅根据低维本体输入完成任务。推理仍只使用原有 DP，训练期动作投影头可以移除。

目标为 $\mathcal L=\mathcal L_{\rm BC}+\lambda_a\mathcal L_{\rm align}$。先不加混合损失，确定真实配对是否有用，再加 $\lambda_m\mathcal L_{\rm mix}$ 测独立增益。这样能够区分“索引挖掘带来的价值”和“像素混合带来的价值”。

该方案仍有两项局限：视觉特征可能记住背景与演示身份；DP 也可能忽略辅助头学到的特征。因此必须按 episode 划分验证集，并检查实际策略的条件动作匹配和闭环结果。辅助分类准确率本身不足以宣布解决模态偏倚。

**六、改进 B：保持动作不变，直接比较正确与错配视觉条件**

这一路线借鉴 Diffusion Classifier 用去噪误差比较条件支持程度的思路，并受 PCD 的输出分布对比启发。下面的训练损失是新建议，不是两篇论文已验证的机器人算法。[^12][^6]

对动作 $A_i$ 固定同一 $q_i,t,\epsilon,x_t^i$，只替换图像条件。定义：

$$
E_i(V)=\mathbb E_{t,\epsilon}\left[w(t)
\left\|\epsilon_\theta(x_t^i,t;V,q_i)-\epsilon\right\|_{M}^{2}\right],
\qquad x_t^i=\sqrt{\bar\alpha_t}A_i+\sqrt{1-\bar\alpha_t}\epsilon,
$$
$$
\mathcal L_{\rm rank}
=\tfrac12\left([m+E_i(V_i)-E_i(V_j)]_+
 +[m+E_j(V_j)-E_j(V_i)]_+\right).
$$

第二项使用固定的 $A_j,q_j,x_t^j$。$M$ 只统计有效时间步和预先规定的活动动作维度，并按维度数归一化；该诊断掩码不要求同时更改原始 BC。$m$ 按验证集能量尺度确定，达到间隔后 hinge 项不再推动错配误差无限增大。总目标仍为 $\mathcal L_{\rm BC}+\lambda_r\mathcal L_{\rm rank}$，先用弱辅助权重。

与当前混合损失的关键区别是：**同一次比较中的带噪动作完全不变**。一个完全忽略图像的模型会得到相同的两个能量，无法满足正间隔；不能通过识别带噪动作属于哪个端点来完成这项比较。但模型仍可能学到图像身份或其他伪相关，所以这只是更直接的视觉依赖约束，不是因果视觉理解的保证。

建议先将它实现为离线诊断，再决定是否进入训练。固定随机噪声进行配对比较，并分低、中、高 SNR 汇总结果；不能简单认为“噪声越大越迫使模型看图”。在极高噪声极限，$x_t\approx\epsilon$，epsilon 预测也可能容易忽略条件。未经推导加权的去噪 MSE 应称为条件支持代理能量，不能报告成精确对数似然。

负样本合法性是最大风险：$V_j,q_i$ 是跨示范拼接条件，可能暴露机械臂姿态矛盾。初版只用完整本体状态非常接近、视觉分歧明确的配对；更强版本应在仿真中固定机器人状态、改变任务对象并重新获得专家动作。随机把另一条示范动作视为“错误动作”不够可靠。

Set-Supervised DP 的实验报告了 DP-DPO 在部分动作块任务上退化，说明相对排序并非天然优于 BC；该工作也考虑从离线示范构造辅助负动作。借鉴它时，应重视动作集合与标签可信度，而不是只复制“使用负样本”这一形式。[^13]

**七、改进 C：适配 GAP，验证优化竞争是否确实存在**

GAP 针对运动转换阶段减弱本体分支的梯度影响，为视觉学习留出机会。对本项目，它首先是机制基线：如果在不使用 EMIX 配对时也能稳定改善条件视觉依赖，说明优化竞争是一条有效路径；若只提升配对模型，则可能与数据关系监督互补。[^2]

当前安装的 LeRobot DP 将原始状态向量与视觉特征直接拼接，没有独立可训练的本体编码器。[L7] 因此，对原始 $q$ 做 `detach()` 或仅缩放其输入梯度，通常不会改变任何上游可学习本体参数，因为这条上游路径根本不存在。把这种操作标为 GAP 复现会产生误导。

可行适配是引入一个小型状态编码器，或明确分离状态与视觉进入条件调制层的投影，然后对可学习的状态分支实施阶段调制。必须同时训练“同架构、无调制”对照，隔离新增参数和归纳偏置的影响；记录实际参数更新及视觉指标，不仅记录被缩放后的梯度范数。

阶段信号初版可来自专家轨迹中的运动转换、夹爪事件与停顿；作为训练期监督即可。不要全时段删除本体感，也不要预设两种模态梯度应相等。精确关节控制仍需要本体信息，目标是防止它在需要对象定位或进度判断时替代视觉。

**八、改进 D：建立任务相关敏感性与无关变化一致性**

跨视角动作一致性的关键前提是两个图像对应同一物理状态及同一动作目标。其 flow-VLA 工作使用同一流坐标比较预测速度；迁移到本项目 DP，可在相同 $A,q,t,\epsilon,x_t$ 下，比较两个有效视角的去噪预测。[^11]

$$
\mathcal L_{\rm view}
=\mathbb E\left\|\epsilon_\theta(x_t,t;V^{(1)},q)
-\epsilon_\theta(x_t,t;V^{(2)},q)\right\|_M^2.
$$

两张图都保留原始 BC。优先使用同时采集的相机，或同一仿真状态重新渲染；历史右广角与俯视数据分别采集，不能直接当作逐帧一致性配对。也不要对不同目标位置或不同完成状态使用该损失，否则会压制本来希望学习的视觉响应。

阶段与对象约束可作为第二阶段扩展：pick 更关注当前目标，place 更关注容器/落点，下一轮选择关注剩余物体和完成状态。近期条件视觉依赖工作提供了相关诊断，但其注意力正则不能原样加到当前 CNN-UNet 上；需要针对空间特征、关键点或额外目标定位头设计适配。仿真对象 mask 可优先用于评估，避免初版同时引入额外分割模型和标注依赖。[^10]

PCD 可用作训练后诊断或附加基线，但不建议作为第一个改动。其扩散实现通过多次动作采样和逐维 KDE 近似分布，包含动作维度独立假设；对多步关节动作块，推理成本和联合动作一致性都需要重新评估。不能把简单的两次 denoiser 相减直接称作 PCD 复现。[^6]

**九、保留 EMIX 时，怎样修改混合本身**

如果真实配对监督验证有效，混合可以继续作为分布平滑或正则项。应首先保留真实端点锚点，比较弱混合与当前 Beta(2) 混合，并固定配对集合。中间比例图像是否同时保留对象身份、空间关系及进度信息，需要在不同机位上测量，而不是从 EEF 接近推断。

配对权重可由完整本体匹配度、动作分歧、视觉可辨性和操作窗口覆盖共同决定，但初版不要把模型当前损失高直接当作样本价值。高损失可能来自标签冲突和不可观测性。可先在人工核验的小规模配对集建立可信度标准，再测试自动打分；权重引入后同时报告有效样本量，防止训练收缩到少数 episode。

基于对象 mask 的局部混合、基于几何的图像对齐或特征空间混合，均可作为后续消融。它们分别增加标注、几何或表征假设；特征插值也不天然具有物理可实现性。更换混合空间之前，先完成“原始端点配对是否已经足够”的关键实验。

如果多帧历史能解决 H3，则需要同步改造当前单帧数据接口和配对定义。历史窗口必须在时间上对齐，并包含决定当前任务状态的信号。随意混合两条不同进度的历史，可能重新制造不可解释的监督冲突。

**十、实验优先级与判定规则**

第一阶段选择 Bread Basket、RGB 和 Can Basket：前两项包含历史收益翻转，后一项作为当前正向对照。Stack Blocks 在第二阶段加入，用于检查精确放置与操作窗口曝光。先用固定训练 seed 排查机制和实现，再对入选方法及基线补足至少三个训练 seed；单 seed 筛选结果不能作为最终论文结论。

|优先级|实验|相对对照只改变什么|回答的问题|
|P0|固定评测场景清单；生成高置信配对验证集|评估基础设施|小差值是否可复现；候选是否确实可由视觉区分|
|P1|普通 DP；当前 EMIX；只重采样真实端点的 DP|配对采样与像素混合|收益来自样本选择还是混合|
|P1|DP＋真实配对对齐损失 A|加入行为关系辅助项|不制造混合图像能否提升视觉条件利用|
|P1|上一项＋原始 EMIX 混合项|仅添加混合损失|混合是否还有独立价值|
|P2|DP＋条件排序 B；GAP 适配及同架构对照|分别改变输出约束、优化路径|哪条机制更有效，是否需要组合|
|P2|同轨迹双视角的 baseline/EMIX 2×2|相机与混合|隔离历史 Bread/RGB 翻转中的视觉交互|
|P3|跨视角一致性、短历史、可信度加权|逐项增加模块|解决已由诊断证实的剩余瓶颈|

要区分两种公平性。**固定普通 BC 曝光**用于解释机制：所有方法看到相同数量的原始样本，辅助项产生的额外计算单列。**固定总训练计算**用于比较实际效率：按 GPU 小时或明确的前向/反向预算比较，并报告原始与配对曝光。当前一次 EMIX 批次有两个 denoiser 分支，固定优化步数不等于固定计算；“30k 对 30k”不能同时解决这两个问题。

关键指标应连成证据链：

1. **条件匹配。** 在按 episode 留出的、可视觉消歧的配对上，测正确图像能量低于错配图像的比例、间隔及 SNR 分层；再测固定采样噪声下动作分支随图像改变是否正确。
2. **相关与无关干预。** 固定机器人状态，改变目标位置/剩余对象并重新查询专家，检查动作变化方向；另改变背景、光照或无关物体，检查不必要变化。普通跨示范图像置换仅称敏感性测试。
3. **闭环阶段成功。** 报告首次 pick、lift、place、下一对象选择及最终任务成功；明确条件概率的分母，避免把前段失败算作后段独立失败。
4. **控制精度与覆盖。** 测抓取位置、释放位置、夹爪时机及接触末段误差，同时报告各阶段的普通/配对训练曝光。
5. **泛化与统计。** 固定场景状态及有效 seed 清单，记录多训练 seed 均值与离散度、成对评测区间；对大量任务与消融中的选择性正结果保持同一报告规则。

图像遮挡、显著图和梯度大小可以作为辅助可视化，但均不能单独建立因果结论。一个模型可能对整张图高度敏感，却主要依赖背景；也可能视觉表征变好了，而控制头没有使用它。最有说服力的是“视觉条件匹配改善—相关干预响应正确—闭环瓶颈改善”同时发生。

若 A 有效而附加混合无效，应保留状态匹配与行为监督作为主方法，不必为了保留 EMIX 名称继续强调混合。若 B 只改善匹配指标但降低闭环成功，应检查负样本与动作分布破坏。若只有保持 BC 曝光才恢复性能，则预算稀释是重要机制；若短历史独立解决回退，问题定位应扩展为可观测性而非纯模态竞争。

**十一、建议的论文主张与贡献边界**

可以采用如下问题表述：在示范中，本体状态与任务进度常存在强相关，策略可能不充分利用视觉中决定下一步行为的因素。状态匹配而行为不同的示范对，为发现这类条件决策提供训练与评估材料。方法通过真实观测上的行为对应关系监督，提升策略在相近本体状态下区分任务状态的能力，并用受控视觉变化验证其适用边界。

与 GAP 的差异应是**数据关系与监督目标**，与 CLASS 的差异应是**状态条件化的候选构造及策略输出验证**，与 PCD 的差异应是**训练期学习及部署开销**，与 ImMimic 的差异应是**解决的问题和配对标签语义**。这些差异只是可争取的贡献空间；现有检索不能替代完整的新颖性审查，更不能保证组合本身足够发表。

应撤回或降级的主张包括：“双峰必然迫使模型看图”“EEF 相近保证图像混合物理合理”“模态权重越平衡越好”“所有动作分歧都是视觉消歧样本”“个别历史正差证明稳定泛化”。替代它们的是可测量的条件匹配提升、受控敏感性、操作阶段成功与预算公平性。

最小可行版本不需要同时堆叠四类改进：**先做原始 BC＋真实配对动作对齐，保留当前索引作为候选来源，补充可信度筛选与条件匹配诊断；随后以有无像素混合的消融决定 EMIX 的最终形式。** 这条路线最直接回应原始模态偏倚动机，也最容易解释成功或失败。

**本地证据与代码定位**

- L1：[DP 双分支去噪损失](../../embodied_mixup/policies/DP/emix_diffusion.py)，`_compute_emix_loss`，约 29–73 行。
- L2：[端点加载、像素混合与 anchor state](../../embodied_mixup/policies/utils/emix_loader_registry_utils.py)，约 470–550 行。`emix_source_images`、`emix_states` 已存在，可作为新辅助项的数据入口；仍需确认训练预处理对嵌套端点的归一化和设备搬运。
- L3：[ACT 加权 L1](../../embodied_mixup/policies/ACT/emix_act.py)，约 30–52 行。
- L4：[候选构造](../../embodied_mixup/candidates/pairs.py)与[动作过滤](../../embodied_mixup/candidates/action_filter.py)；阈值和有效索引以归档 metadata 为准。
- L5：[当前十任务回退审计](../results/20260911-top500-emix-regression-audit.md)，包含采样、事件窗口、评测匹配及输出数据路径。
- L6：[历史对比审计](../results/20260911-emix-historical-comparison.md)，包含真实 metrics、机位、阈值变化和统计限制。
- L7：本机安装文件 `/home/rogerspyke/miniforge3/envs/emix-twin/lib/python3.10/site-packages/lerobot/policies/diffusion/modeling_diffusion.py`，`_prepare_global_conditioning`，约 246–282 行。本条指向已检查的本机依赖，不代表所有 LeRobot 版本。
- 背景材料：[原方法说明](method.md)、[原理论说明](EMIX.md)、[旧文献综述](emixLiteratureSurvey.md)。当前代码实际为像素混合与双动作端点监督，阅读旧综述时需避免沿用“特征混合、动作线性插值”的泛化描述。

**Sources**

[^1]: Cheng Chi et al. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137). 首发 2023-03-07，RSS 2023；使用其条件扩散与多模态动作建模机制。
[^2]: Jingxian Lu, Wenke Xia, Yuxuan Wu, Zhiwu Lu, Di Hu. [When would Vision-Proprioception Policies Fail in Robotic Manipulation?](https://arxiv.org/html/2602.12032v1). 2026-02-12，ICLR 2026；重点 §4、§10.2。会议与代码入口见[作者项目页](https://gewu-lab.github.io/GAP/)。
[^3]: Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, Di Hu. [Balanced Multimodal Learning via On-the-fly Gradient Modulation](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf). CVPR 2022；多模态优化背景。
[^4]: Pim de Haan, Dinesh Jayaraman, Sergey Levine. [Causal Confusion in Imitation Learning](https://papers.neurips.cc/paper_files/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html). NeurIPS 2019；因果混淆与干预边界。
[^5]: Sung-Wook Lee, Xuhui Kang, Brandon Yang, Yen-Ling Kuo. [CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation](https://arxiv.org/html/2508.01600v1). 2025-08-03，预印本；重点 §4.1–4.3。本文不据 arXiv 页面推断其后续录用状态。
[^6]: Shihan Wu, Xu Luo, Ji Zhang, Junlin Xie, Jingkuan Song, Heng Tao Shen, Lianli Gao. [Policy Contrastive Decoding for Robotic Foundation Models](https://arxiv.org/html/2505.13255v6). 首发 2025-05-19，所用 v6 为 2026-04-24，ICLR 2026；重点 §3.2 的条件分布比与 KDE。
[^7]: Yangcen Liu, Woo Chul Shin, Yunhai Han, Zhenyang Chen, Harish Ravichandar, Danfei Xu. [ImMimic: Cross-Domain Imitation from Human Videos via Mapping and Interpolation](https://arxiv.org/abs/2509.10952). 2025-09-13，CoRL 2025；[作者实验室论文页](https://star-lab.cc.gatech.edu/papers/liu-shen-ImMimic/)。
[^8]: Xiang Li, Varun Belagali, Jinghuan Shang, Michael S. Ryoo. [Crossway Diffusion: Improving Diffusion-based Visuomotor Policy via Self-supervised Learning](https://arxiv.org/html/2307.01849v3). 首发 2023-07-04，所用 v3 为 2024-01-11；重点 §III，观测重建辅助监督。
[^9]: Youguang Xing, Xu Luo, Junlin Xie, Lianli Gao, Heng Tao Shen, Jingkuan Song. [Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation](https://proceedings.mlr.press/v305/xing25a.html). CoRL 2025，PMLR 305:3239–3266。
[^10]: Vivek Chavan, Pengtao Xie, Yahuan Shi, Oliver Heimann, Kevin Haninger, Jörg Krüger. [What Matters, When? Diagnosing and Improving Conditional Visual Grounding in Visuomotor Imitation Policies](https://arxiv.org/abs/2609.05376). 2026-09-04；arXiv metadata 标注 ECCV 2026 DexHAND 工作坊非归档扩展摘要，作为初步证据。
[^11]: Bingqi Huang, Bingchuan Wei, Xuan Wang, Yingkai Cai, Zhaokui Wang. [Cross-View Action Consistency for Camera-Robust Vision-Language-Action Policies](https://arxiv.org/html/2608.06965v1). 2026-08-07，预印本；同状态跨视角配对与流预测一致性。
[^12]: Alexander C. Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, Deepak Pathak. [Your Diffusion Model is Secretly a Zero-Shot Classifier](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf). ICCV 2023；条件密度代理与去噪误差比较。
[^13]: Zhaoting Li, Gang Chen, Javier Alonso-Mora, Cosimo Della Santina, Jens Kober. [Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion through Corrections](https://arxiv.org/html/2606.01865v1). 2026-06-01，预印本；重点 §III–V，集合监督、离线构造及 DP-DPO 对照。
[^14]: Chengyang He, Xu Liu, Gadiel Sznaier Camps, Joseph Bruno, Guillaume Sartoretti, Mac Schwager. [Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative](https://proceedings.iclr.cc/paper_files/paper/2026/hash/122ea6470232ee5e79a2649243348005-Abstract-Conference.html). ICLR 2026；作为稀疏数据下动作记忆机制的边界证据。
[^15]: Asher J. Hancock, Allen Z. Ren, Anirudha Majumdar. [Run-time Observation Interventions Make Vision-Language-Action Models More Visually Robust](https://aasherh.github.io/byovla/). 2024，arXiv:2410.01971；使用作者项目页对无关区域敏感性及测试时处理的说明。
[^16]: Yifei Chen, Yuzhe Zhang, Giovanni D’urso, Nicholas Lawrance, Brendan Tidd. [Improving Generalization Ability of Robotic Imitation Learning by Resolving Causal Confusion in Observations](https://arxiv.org/html/2507.22380v1). 2025-07-30，预印本；结构因果建模假设与 Causal-ACT。
