**500ep top-camera EMIX 任务差异审计（2026-09-11）**

补充：后续[历史对比审计](20260911-emix-historical-comparison.md)发现，旧top-200ep索引的实际姿态阈值为0.01 rad，当前为0.26 rad，并非归档SUMMARY所说的阈值不变。历史RGB在几乎相同home/阶段分布下出现正负收益，说明下文H1/H2是共同风险线索，不能单独解释历史收益翻转；视角与图像混合的交互需要优先验证。

对象：`archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911`，父仓库 commit `3d3c51e`，run `flow-20260907-222045-cuda-0-1`。本次核对时，当前代码相对该 commit 只有两份文档变化，没有训练/索引实现差异。本报告不修改训练或归档数据，也没有重训模型。

**结论：有可验证的解释线索，但没有一个指标能单独解释任务收益。** 最强的直接证据是：①实际评测场景没有完全配对；②EMIX 对 home 附近和不同操作阶段的交汇点强烈重采样，显著稀释抓取/释放训练窗口；③存在真实的“第一次操作完成后 ↔ 任务刚开始”的混合训练样本。最值得验证的机制是“任务瓶颈与增广覆盖错位 + 任务进度信息混合 + 真实数据曝光减少”。can_basket 是必须保留的反例：其跨阶段配对最多，共同 seed 上仍有正向差值，因此不能用“跨阶段多就一定更差”概括结果。

**1. 结果可信度：先修正比较口径**

原始结果为 baseline 438/1000、EMIX 461/1000，差 +2.3 个百分点；逐任务 5 个正差、3 个负差、2 个相等。归档 SUMMARY 的“5 胜 2 负 3 平”不是按差值正负直接计数，其“±5 点噪声界限”“唯一实质落后”等措辞也没有统计检验支持。

重新读取全部 `eval/*/*/metrics.json`，按记录中的 `seed` 取交集：

|任务|原始 baseline→EMIX|共同 seed 数|共同 seed 成功数 baseline→EMIX|EMIX 独胜 / baseline 独胜|共同集精确 McNemar p|
|---|---:|---:|---:|---:|---:|
|blocks_ranking_rgb|28→26|99|28→26|18 / 20|0.871|
|blocks_ranking_size|3→3|97|3→3|3 / 3|1.000|
|place_bread_basket|45→37|99|44→36|11 / 19|0.200|
|place_bread_skillet|87→92|96|84→88|7 / 3|0.344|
|place_burger_fries|36→37|98|34→36|18 / 16|0.864|
|place_can_basket|49→64|50|23→32|18 / 9|0.122|
|place_cans_plasticbox|20→25|89|19→21|14 / 12|0.845|
|place_dual_shoes|33→44|82|29→37|17 / 9|0.169|
|stack_blocks_two|49→45|100|49→45|23 / 27|0.672|
|stack_bowls_two|88→88|73|64→65|8 / 7|1.000|

机制来自 [evaluation/evaluate.py](../../evaluation/evaluate.py)：每个模型各自从 seed 100000 开始运行专家，只有 `plan_success and check_success()` 的场景才进入策略评测。相同配置 seed 不保证两个 run 接受相同场景。为什么专家接受集合不一致，本次没有进一步隔离；不能直接断定是 GPU、物理仿真还是规划随机性导致。

共同集上 can_basket 仍为 46%→64%，dual_shoes 为 35.4%→45.1%，因此正向信号并没有因取交集消失；但共同集更小，且是条件选择后的子集。各任务的检验均未达到未校正的 0.05，不能宣称已证明提升/退化，也不能反向宣称没有效果。这些检验只描述本次两个 checkpoint 在共同 seed 上的差异，不覆盖训练 seed 方差；相同 seed 也不自动证明仿真轨迹逐位一致。

bread_basket 的 −8 点在 99 个共同 seed 上仍保留；同机位 200ep 批为 −11 点，是优先研究的重复方向，但两批共享实验因素，不是独立多训练 seed 复现。

**2. 本次实际执行的数据分析**

读取 10 个任务的全部归档 parquet、每任务 500,000 个索引组；仅统计 `state.training=true` 的组（每任务实际约 47.2–49.8 万组）。复用项目的 [extract_pick_place_points](../../embodied_mixup/visualization/plot_actions.py)，从显式 `observation.left_gripper` 检测单调闭合/张开，得到 pick/place 中点。共检测到 10,000 次 pick、10,000 次 place，符合 RGB/size 各三次、bread_skillet/can_basket 各一次、其余任务各两次、每任务 500 episodes 的预期。

统计定义：

- 原始分布按全数据帧均匀计算；EMIX 分布按可训练组均匀、两个端点各一半权重计算。由于 alpha~Beta(2,2) 对称，这也是两个监督分支的期望权重。实际图像包含两端点，而 observation.state 始终来自 member_0。
- 阶段为当前帧前已发生的夹爪事件数。两个对象任务中，0=首次 pick 前，1=首次 pick 到首次 place，2=首次 place 到第二次 pick，3=第二次 pick 到第二次 place，4=最后 place 后。它是动作进度代理，不是物体身份/接触真值；“已发生”以趋势中点划界。
- home 指距归档初始 EEF 参考点 `[-0.00141631,-0.39504284,0.95340055]` 小于 3 cm，使用数据集坐标；不是机器人全关节 home 判定。
- 抓取/释放窗口表示从当前帧开始的 16 帧监督窗口包含相应事件中点，按 episode 边界及双分支共同有效长度裁剪。17 Hz 下约为未来 0.9 秒。它衡量事件曝光，不代表所有接触相关动作都已被该指标覆盖。

|任务|原始→EMIX home占比|跨阶段配对|原始→EMIX抓取窗口|原始→EMIX释放窗口|
|---|---:|---:|---:|---:|
|blocks_ranking_rgb|6.26%→73.23%|65.2%|10.63%→0.49%|10.63%→0.69%|
|blocks_ranking_size|6.21%→73.22%|65.6%|10.60%→0.47%|10.60%→0.68%|
|place_bread_basket|7.07%→68.85%|49.2%|12.48%→0.77%|12.48%→1.02%|
|place_bread_skillet|5.76%→51.66%|19.1%|10.95%→2.79%|10.95%→0.71%|
|place_burger_fries|7.08%→72.36%|48.0%|12.26%→0.34%|12.26%→0.44%|
|place_can_basket|14.36%→65.82%|82.3%|7.69%→0.04%|7.69%→0.00%|
|place_cans_plasticbox|4.73%→78.48%|46.1%|7.93%→0.04%|7.93%→0.01%|
|place_dual_shoes|6.18%→44.70%|47.6%|12.12%→11.31%|12.12%→0.30%|
|stack_blocks_two|2.27%→34.37%|20.8%|14.82%→2.56%|14.82%→5.20%|
|stack_bowls_two|2.40%→13.90%|22.3%|15.28%→2.93%|15.28%→5.93%|

![原始和EMIX采样覆盖对比](../../outputs/analysis/top500_20260911/coverage_summary.png)

不能把“50 万组”理解为“均匀增加了 50 万种有效操作状态”。bread_basket 的 49.5 万训练组只触及 39.2% 原始帧；原始帧中按端点次数排序最密集的 1% 承载 29.6% 端点权重。RGB/size 约 73% 端点挤在 home 邻域。这里反映的是样本分布集中，不能据此把相关样本换算成独立统计样本数。

**3. 实现为什么会产生这种分布**

[pairs.py](../../embodied_mixup/candidates/pairs.py) 先选跨 episode、XYZ <2 cm、RPY 差 <0.26 rad 的邻居；[action_filter.py](../../embodied_mixup/candidates/action_filter.py) 再要求当前动作向量差至少 0.05、8 帧后两条轨迹各移动至少 2 mm、运动夹角至少 30°。这是主动选择分叉，不是动作相似性过滤。

共同 home 附近存在大量几何近邻，又有出发/返回和不同轮次的运动分叉，容易满足条件；夹爪开合时 EEF 经常近乎静止，未来位移下界与方向夹角条件可能排除这些片段。统计证实接触事件覆盖很低，但每个门限的贡献仍需对原始候选池逐门消融才能确定。

`max_candidate_pairs_per_episode=1000` 的实现是 **总 episode 数×1000 的全局随机配对上限**，不是每个 episode 独立获得 1000 对，也不是按空间或阶段均衡。随后聚类去掉噪声组；训练 [train.py](../../embodied_mixup/policies/train.py) 使用 `shuffle=True` 均匀遍历组，没有 cluster-balanced 或阶段均衡采样。全局 cap 可解决内存问题，但不会自动校正采样密度。

`gripper_action_filter_enabled=true` 容易被误读：当前代码计算未来开合方向是否不同的 flag，但最终 `keep` 只由动作差与未来方向条件决定，该 flag 没有作为拒绝条件。邻居阶段的夹爪一致性又使用 `np.sign(gripper_signal)`，并非夹爪数值近似约束。应先明确这是否符合设计意图，不能仅凭名字认定是 bug。

[emix_loader_registry_utils.py](../../embodied_mixup/policies/utils/emix_loader_registry_utils.py) 构造 `I_mix=alpha*I0+(1-alpha)*I1`、`state=q0`。动作是两条原始关节目标序列，**没有把动作线性平均**。[emix_diffusion.py](../../embodied_mixup/policies/DP/emix_diffusion.py) 在同一个混合条件下分别对两条动作加噪、预测，以 alpha 加权两个 score-matching loss。多模态本身是方法要建模的目标，不能把“两条动作不同”直接称为监督错误；真正需要验证的是这种人工条件下学到的分布能否迁移回真实图像和真实状态。

**4. 几个可行猜想，按建议优先级排列**

**H1：EMIX 的空间/阶段重采样与任务瓶颈错位，并替换了真实操作训练。证据较强，因果待测。**

重放实际 logarithmic scheduler（0.6→0，30k steps，Python RNG seed=0）得到 7,518 个 EMIX batch、22,482 个普通 batch，即 25.06% 的优化步骤替换为 EMIX。固定 30k 总步数时，普通数据更新比 baseline 少约四分之一。不是“额外加了一批增广样本，原数据曝光不变”。

bread_basket 的抓取窗口在普通帧中占 12.48%，EMIX 端点仅 0.77%。按 batch 数和 alpha 期望粗略加权，整体抓取事件窗口曝光约 9.55%，比 baseline 少约 23.5%。这是监督窗口质量的代理，不是实际梯度贡献的精确等式。RGB/size 与 plasticbox 也有类似错位。

成功任务的一个区别是 dual_shoes：EMIX 抓取窗口占 11.31%，非常接近普通数据的 12.12%；其 home 集中程度低于 bread_basket。这提供了“更有效地覆盖操作相关状态”的正向线索，但它不能单独解释所有提升：skillet/can 的抓取曝光也很低仍有正差。

最小检验：①同样选取索引端点，但完全不混图、不做双分支训练的重采样对照，区分采样分布与混合目标；②保持原有 30k 普通 batch 曝光，插入当前 7,518 个 EMIX batch，另加等总步数的纯 baseline 对照；③在总步数相同条件下控制 home/操作阶段采样比例。若只补足普通曝光即可恢复 bread_basket，H1 获支持；若重采样但不混合也回退，分布偏置本身就值得优先修正。

**H2：重复操作在 home 处的进度信息被混合，导致阶段选择/停止行为混淆。bread_basket 有直接样本支持，但不是普适解释。**

bread_basket 49.2% 配对跨阶段，约 42.0% 全部配对是阶段 0↔2，即“还没做第一轮”和“已做完第一轮、尚未抓第二块”。RGB/size 则约 65% 跨阶段，主要是 0/2/4 三个轮次在共同区域相遇。只给一帧观测，状态又保留第一分支，区分任务进度依赖当下场景中的物体配置；叠图可能削弱这一信息。

可定位的实际训练例子：bread_basket `debug/step_00010004_sample_00025.png`，group `emix_group_f405aced30a9`，alpha=0.4899；member_0 是 episode 66 / 全局帧 17219 / 局部帧 128 / 阶段2，member_1 是 episode170 / 全局帧43716 / 局部帧4 / 阶段0。调试图显示不同篮子与面包布局叠加，后续 EEF 运动方向相反。这不是仅由总体统计推出来的假想样本。

评测 seed100011（两模型都是 episode2）：baseline 成功；EMIX 视频在后半段显示一块面包在篮中、另一块仍在桌上，机器人在 home 附近持续停留到超时。它与进度混淆相容，但也可能是之前误抓/物体位移后缺少恢复，不能靠这个个例确诊。[视频抽帧](../../outputs/analysis/top500_20260911/video_place_bread_basket_2.jpg)。seed100001 的 EMIX 失败则出现较早的抓取/物体扰动现象，提示失败并非全是“第二轮停住”。

关键反例：can_basket **82.3%** 配对是阶段0↔2（未开始↔已释放），比 bread_basket 更高，共同 seed 上仍有正差。它只有一个对象，完成后的停止不同于中途必须继续下一轮，因而可能对进度混合更宽容；这只是可测解释，不能当作已证实机制。

最小检验：在 bread_basket/RGB 中只保留同事件阶段配对，控制组数、home 占比、alpha 与训练曝光；另外区分出发/回程，因“同一阶段2”仍包含第一轮回程和第二轮出发。评测记录第一块放置成功率、第二次抓取启动率、完成一块后 idle 时长、全部物体完成率。若只改善第二轮启动而非首次抓取，H2 的预测更具体地成立。增加时序观测是后续设计选择，当前实现断言 `n_obs_steps=1`，不能直接改配置就称为公平对照。

**H3：整图混合对简单任务的视觉正则化收益，和对多对象/精密关系辨认的损害同时存在。中等证据。**

在84×84图像下，混合两个不同位置/外观的篮子、面包或积木会产生半透明重影。实际 bread_basket 调试样本已经观察到这种叠加。单次 bread_skillet 的结构更简单，200ep/500ep 都是约 +5～6 点；bread_basket 要连续处理两个面包，200ep/500ep 都是负差；RGB/size 必须保留对象属性和当前排列进度。这与“对任务条件信息的依赖不同”相容。

但不能笼统说“多对象一定有害”或“抓取难就受益”：dual_shoes 同样两轮操作却提升；其 seed100005 视频中 baseline 实际已搬运两只鞋但最终失败，EMIX 成功，说明至少该个例差别更接近最终摆放条件，而非单纯不会抓取。stack_blocks_two 的成功判定要求约2.5cm横向、1.2cm高度精度，而 stack_bowls_two 原始成功率已达88%，底层技能和剩余误差容限也不同。这里只能提出任务条件敏感性，不能从任务名字推断失败部位。

最小检验：固定同一组和相同真实数据曝光，对比现有 Beta(2,2)、alpha 向0/1偏移的分布、纯端点单分支；必要时再做保留机器人/交互对象的局部增广。可先用训练日志/录像统计“选错对象/提前停止/抓取失败/放置偏差”，避免只看总体成功率。更清晰的 alpha 若对 bread/RGB 特别有帮助、对 skillet 收益小，才支持该机制。

**H4：当前位姿接近不保证关节状态和整个动作窗口兼容。实现上存在风险，当前没有任务间因果证据。**

输入保留 q0，却让动作1也在 q0 条件下被监督；实际 action 是14维绝对关节目标，不是笛卡尔增量。bread_basket 两分支左臂6维关节状态差 L2 的中位数约0.110、P90约0.215；can_basket P90约0.336。2cm/0.26rad的EEF门限没有显式限制关节空间或未来夹爪时序。这里的L2是关节弧度的合成量，不是某一个关节的旋转，也不是空间位置误差。

但 can 的状态差更大仍正向，故无法用它单独解释 bread 回退。适合小范围控制试验：随机交换 anchor（当前 member0总是较早全局帧，存在分支非对称）；增加关节状态近邻约束；按整个16帧的夹爪开合兼容性限制配对。各试验要匹配采样阶段分布，否则无法区分“更兼容”和“换了一批更容易的样本”。不要直接改为动作平均，因为这会改变 EMIX 的多模态目标。

**5. 最小验证路线与当前不能下的结论**

先固定一次生成的评测 seed/场景清单，每个模型在完全相同清单上运行，不再为每个 checkpoint 独立筛专家；保存场景初始化信息与随机状态，并分离策略扩散采样种子。检查点已按归档规范删除，本次不能直接用这两个原模型重评。

第一轮优先用 bread_basket 作为回退任务、dual_shoes 作为正向任务（也可加单次 skillet）。用同一数据版本和3个训练 seed，对照 baseline、现有EMIX、索引端点重采样但不混合、同阶段EMIX；同步记录上述分阶段指标。若预算紧，应先验证采样/阶段因素，再扩展alpha或关节门限试验。真实曝光补足实验需配相同额外步数的baseline，避免把额外训练收益当作EMIX改善。

目前不支持：把总体提升归因于构建修复；把widen门限认定为无害或有害；仅凭50万组就认为样本覆盖充分；认为HDBSCAN自动保证语义阶段一致；把size双3%归因于EMIX；以某一个视频或空间相关性宣称因果机制。200→500ep同时改变数据规模、数据采集内容、姿态门限（0.01→0.26 rad）、去尾与候选构建/封顶，未做因子分离。

**产物与复现**

工作目录根目录下运行：

```bash
/home/rogerspyke/miniforge3/envs/emix-twin/bin/python outputs/analysis/top500_20260911/audit.py
/home/rogerspyke/miniforge3/envs/emix-twin/bin/python outputs/analysis/top500_20260911/supplement.py
/home/rogerspyke/miniforge3/envs/emix-twin/bin/python outputs/analysis/top500_20260911/video_contacts.py
```

[audit.py](../../outputs/analysis/top500_20260911/audit.py) / [supplement.py](../../outputs/analysis/top500_20260911/supplement.py)：只读归档，输出分析文件；[audit.json](../../outputs/analysis/top500_20260911/audit.json) 为全量统计；[window_audit.json](../../outputs/analysis/top500_20260911/window_audit.json) 为按真实共同有效窗口计算的曝光及阶段配对统计；各任务都有XY/YZ/pick/place/阶段对照PNG及用于复核的NPZ。输出目录位于仓库忽略目录，需要长期保存时应连同本报告一起保留。

视频只抽查了5个成败差异的共同seed例子，每个两条视频，并非对1000次评测的失败类型统计；抽帧按各视频时长等比例采样，横向列不是严格同步时间，图上标有实际时间。所有假设都应以完整分阶段复评检验。
