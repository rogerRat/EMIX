# 2026-06-10: PlugCharger-v1 Replay 大量 Episode 失败的调研报告

## 概述

在执行 `data_processing/replay.py` 将 PlugCharger-v1 轨迹从 `pd_joint_pos` 控制模式转换为 `pd_ee_delta_pose` 时，500 个 episode 中仅有 22 个（4.4%）成功 replay，其余均输出 "Episode is not replayed successfully. Skipping"。

本报告记录：问题现象、根因分析、实验验证、解决方案及当前行动。

---

## 1. 问题现象

### 1.1 原始命令

```bash
python data_processing/replay.py \
  --traj-path demos/PlugCharger-v1/motionplanning/20260610_173934.h5 \
  -c pd_ee_delta_pose \
  -o rgb \
  --save-traj
```

### 1.2 输出

```
Episode 1 is not replayed successfully. Skipping
Episode 2 is not replayed successfully. Skipping
Episode 8 is not replayed successfully. Skipping
...
（共 478 个 episode 失败）
```

### 1.3 输出数据集情况

- 输出文件: `20260610_173934.rgb.pd_ee_delta_pose.physx_cpu.h5`
- 输出 JSON 记录仅 22 个 episode（原始数据集有 500 个）
- **所有失败 episode 均未写入输出数据集**：代码中只有 `success=True` 时才调用 `env.flush_trajectory()` 保存
- 失败 episode 的轨迹数据**完全不存在于转换后的 h5 文件中**

### 1.4 关键数据对比

| 指标 | 原始数据集 | Replay 输出 |
|------|-----------|-------------|
| Episode 数 | 500 | 22 |
| 控制模式 | pd_joint_pos | pd_ee_delta_pose |
| max_episode_steps（元数据声明） | 200 | 200 |
| Episode 步数范围 | 143-247 | 148-792 |

---

## 2. 根因分析

### 2.1 主因：控制模式转换的精度损失

`pd_joint_pos -> pd_ee_delta_pose` 的转换是**近似映射**，存在固有精度损失：

1. **原始动作精度高**：`pd_joint_pos` 直接指定 8 个关节的目标角度，控制器内部 PD 控制器精确追踪
2. **转换后精度降低**：`pd_ee_delta_pose` 使用 7 维动作（3D 位置增量 + 3D 姿态增量 + 1D 抓取），由逆运动学（Pinocchio）从关节目标计算而来
3. **裁剪（Clipping）问题**：转换后的 delta pose 动作需归一化到 [-1, 1] 范围，超出范围的值被裁剪，导致末端执行器无法完全到达目标位置
4. **累积误差**：每步的小偏差在轨迹执行过程中逐步累积

**定量证据**：

| 条件 | 成功率 |
|------|--------|
| 默认参数 (max_retry=0)，全部500 ep | 22/500 = 4.4% |
| 前30个 episode，默认参数 | 19/30 = 63.3% |
| 前30个 episode，max_steps=500（更正：与默认200几乎无差异） | ~63% |
| 前30个 episode，max_retry=5 | 更多 episode 在重试后成功 |

**末端执行器误差追踪**：

| Episode | 状态 | 最大 EE 偏差 | 最终 EE 偏差 |
|---------|------|-------------|-------------|
| Ep 0 (seed=0) | 成功 | 0.254m | 0.134m |
| Ep 1 (seed=1) | 失败 | 0.225m | 0.091m |

偏差值本身并非决定因素——PlugCharger 任务的容错窗口极小（需将充电器精确插入插座），即使较小的末端偏差也可能导致任务失败。

### 2.2 补充说明：max_episode_steps 的影响（更正）

原始数据收集时，motion planning solver **忽略了 `truncated` 信号**，继续执行直到任务成功：

```python
# data_collection/panda/run.py 中 solve() 函数
# 运动规划器不检查 truncated，持续执行直到 success=True
res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
```

这导致原始 500 个 episode 中，**148 个的 elapsed_steps > 200**（最大 247 步）。

**重要更正**：经过进一步的实验验证，`max_episode_steps=200` **不是导致 replay 失败的显著原因**。原因如下：

1. ManiSkill 的 `TimeLimitWrapper` 只在 step >= 200 时设 `truncated=True`，**不会阻止继续调用 `env.step()`**
2. `replay.py` 的 `from_pd_joint_pos` 转换函数会执行完**所有**原始 actions，不受 truncated 影响
3. 之前成功 replay 的数据中，seed=3 的 episode 从 146 步 replay 成 457 步，seed=18 从 171 步 replay 成 792 步——远超 200 步限制，但仍然成功
4. 实验中用 `max_episode_steps=500` 重新测试失败的 episode，成功率几乎没有提升

**步数分布数据**（仅供参考，不构成主因）：

| 步数范围 | Episode 数 | 占比 |
|-----------|-----------|------|
| 125-149 | 26 | 5.2% |
| 150-174 | 179 | 35.8% |
| 175-199 | 147 | 29.4% |
| 200-224 | 92 | 18.4% |
| 225-247 | 56 | 11.2% |

**结论**：`max_episode_steps=200` 仅在使用 `--discard-timeout` 参数时才会导致问题（将 truncated 的 episode 判定为失败）。默认不加 `--discard-timeout` 时，truncated 的 episode 只要 `success=True` 也会被保存。真正导致大量失败的原因是**控制模式转换精度不足**（见 2.1 节）。

### 2.3 ~~次因：物理仿真非确定性~~（已更正：确定性状态泄漏）

> **重大更正**：经过深入调查，PhysX CPU 仿真实际上是**确定性的**。
> 之前认为"重试有时能成功"的结论是错误的，原因如下：

#### 原始实验的错误结论

之前用 `--max-retry 5` 测试时，每次重试创建了**新的 gym 环境**（`gym.make()`），
这等同于每次重试都是全新的环境，当然能得到相同的结果。
但 `replay_cpu_sim` 实际上**复用同一个 env 对象**跨重试。

#### 真实机制：env 复用导致确定性状态泄漏

| 条件 | 结果 |
|------|------|
| 每次重试创建**新** env 对象 | **始终成功**（seed=2，10/10 次 True） |
| **复用**同一个 env 对象（如 replay_cpu_sim） | **仅第一次成功**（seed=2，第1次 True，第2-6次 False） |
| 仅复用 target_env，新 ori_env | 第1次 True，后续 False |
| 仅复用 ori_env，新 target_env | **始终成功** |

**结论**：状态泄漏来自 `target_env`（`pd_ee_delta_pose` 环境），而非 `ori_env`。

#### 根因分析

`target_env` 复用时，`env.reset()` **无法完全恢复** PhysX 仿真状态：
- 关节位置 (`qpos`) 正确重置 ✓（已验证：重置后数值完全相同）
- 但 SAPIEN/PhysX 内部状态（接触历史、求解器缓存等）**未完全重置** ✗
- 第一次 `from_pd_joint_pos_to_ee` 调用 `env.step()` 后，PhysX 内部累积了
  仿真状态（接触力、摩擦等），这些状态在 `env.reset()` 后残存

这导致：
1. 第一次转换尝试：PhysX 状态干净，转换成功率高
2. 第二次及后续尝试：PhysX 状态被"污染"，转换成功率降为 0
3. 此行为是**确定性的**，不是随机的——同样的操作序列总是给出同样的结果

#### 对 `--max-retry` 的影响

`--max-retry` 参数在 `replay_cpu_sim` 中复用同一个 env 对象，
因此重试机制**无法提高成功率**。第一次失败的 episode，重试也会失败。

如果需要重试机制生效，必须在每次重试时**重新创建 env 对象**，但这不是当前代码的行为。

---

## 3. 关键代码逻辑分析

### 3.1 Replay 成功判定（replay.py:380-396）

```python
success = info.get("success", False)
if args.discard_timeout:
    success = success and (not truncated)

if success or args.allow_failure:
    successful_replays += 1
    if args.save_traj:
        env.flush_trajectory()      # <-- 只有成功才保存
    if args.save_video:
        env.flush_video(ignore_empty_transition=False)
    break                            # <-- 成功后跳出重试循环
else:
    if args.verbose:
        print("info", info)
else:
    # for 循环的 else: 所有重试均失败
    env.flush_video(save=False)
    tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")
```

关键点：
- `success` 由环境评估函数 `evaluate()` 返回，PlugCharger 任务中只有充电器完全插入才返回 True
- `--allow-failure` 可强制保存失败 episode（但动作质量低）
- `--max-retry N` 可重试 N+1 次

### 3.2 [严重 BUG] RecordEpisode 缓冲区累积导致轨迹数据污染

**这是本次调研中发现的最重要的 Bug。**

`replay_cpu_sim` 函数在处理 episode 序列时，**复用同一个 RecordEpisode wrapper**。当 episode 失败（`success=False`）时，代码**不调用** `env.flush_trajectory()`，导致 RecordEpisode 的内部缓冲区**保留了失败 episode 的所有步骤数据**。

当后续某个 episode 成功并调用 `env.flush_trajectory()` 时，它保存的是**整个缓冲区的内容**——包括前面所有失败 episode 的残留数据。

#### Bug 机制详解

```
Episode 0 (seed=0): SUCCESS -> flush_trajectory() 保存 156 步, 缓冲区清空
Episode 1 (seed=1): FAIL    -> 不 flush, 缓冲区累积 152 步
Episode 2 (seed=2): FAIL    -> 不 flush, 缓冲区累积 152+159=311 步
Episode 3 (seed=3): SUCCESS -> flush_trajectory() 保存 311+146=457 步！
```

#### 实验验证

| 成功 Episode | 前置失败 Episode 数 | 预测缓冲区大小 | 实际保存动作数 | 匹配 |
|-------------|-------------------|-------------|------------|------|
| ep0 (seed=0) | 0 | 157-1=156 | 156 | YES |
| ep3 (seed=3) | 2 (ep1+ep2) | 460-3=457 | 457 | YES |
| ep11 (seed=18) | 3 (ep8+ep9+ep10) | 796-4=792 | 792 | YES |
| ep15 (seed=28) | 2 (ep13+ep14) | 595-3=592 | 592 | YES |
| ep18 (seed=31) | 2 (ep16+ep17) | 620-3=617 | 617 | YES |

（预测公式：累计步数 + 重置次数 - 重置次数 = 累计步数；减去重置次数是因为 RecordEpisode.reset() 添加的 dummy step 不计入实际动作数）

#### 影响范围

1. **轨迹数据被污染**：成功 episode 保存的轨迹中混入了来自其他完全不同 episode 的动作和观测数据
2. **elapsed_steps 虚增**：从 146 步膨胀到 457 步，实际只有最后 146 步属于该 episode
3. **训练数据质量严重受损**：如果直接使用这些 replay 后的数据进行训练，模型会学习到跨 episode 的噪声动作序列
4. **StackCube 未受影响**：因为 StackCube 几乎所有 episode 都成功（接近 100%），缓冲区不会被失败 episode 污染

#### 修复方案

在 `replay_cpu_sim` 中，**每次 episode 失败后必须清空缓冲区**：

```python
# 当前代码 (replay.py:394-396):
else:
    env.flush_video(save=False)
    tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

# 修复方案: 在 else 分支中添加缓冲区清空
else:
    env.flush_video(save=False)
    env.flush_trajectory(save=False)  # <-- 新增: 清空缓冲区，防止数据泄漏
    tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")
```

`flush_trajectory(save=False)` 会清空缓冲区但不写入磁盘，从而确保下一次 episode 的缓冲区是干净的。

### 3.3 控制模式转换流程（from_pd_joint_pos_to_ee）

1. 在原始环境（`pd_joint_pos`）中逐步执行原始动作
2. 每步读取 arm controller 的 `_target_qpos`（目标关节角度）
3. 通过 Pinocchio 正运动学计算目标末端位姿
4. 计算 delta pose（目标位姿 - 当前位姿）
5. 归一化到 [-1, 1] 动作空间
6. **裁剪检查**：如果超出范围，最多重试 4 次当前步（逐步逼近目标）
7. 在目标环境（`pd_ee_delta_pose`）中执行转换后的动作

---

## 4. 实验验证

### 4.1 实验 1：默认参数 vs max_episode_steps=500

| Episode | seed | 默认(200步) | 500步 | 说明 |
|---------|------|------------|-------|------|
| 0 | 0 | 成功 | 成功 | 短 episode，不受截断影响 |
| 1 | 1 | 失败 | 失败 | 转换精度问题，非截断导致 |
| 2 | 2 | 失败 | 失败 | 转换精度问题，非截断导致 |
| 8 | 12 | 失败 | **成功** | 非确定性（实际是 env 复用状态泄漏，见4.4节） |
| 13 | 25 | 失败 | **成功** | 同上 |
| 14 | 27 | 失败 | 失败 | 转换精度问题 |
| 17 | 30 | 失败 | **成功** | 同上 |
| 20 | 35 | 失败 | 失败 | 转换精度问题 |
| 22 | 41 | 失败 | **成功** | 同上 |

**更正后的结论**：提高 max_episode_steps 对成功率几乎没有影响，因为 ManiSkill 的 TimeLimitWrapper 不会真正阻止继续执行。实验中 max_episode_steps=500 与默认 200 的成功率差异极小，进一步证明失败的主因是转换精度而非步数截断。

> 注：之前标记为"非确定性"的成功案例，实际是测试时创建了新 env 对象导致的。
> 在复用 env 对象的条件下，这些 episode 的成功/失败模式是确定性的。

### 4.2 ~~实验 2：max_retry=5 重试机制~~（已更正：重试无效）

> **重大更正**：之前的重试测试是在每次重试时创建新 env 对象的条件下进行的，
> 这与 `replay_cpu_sim` 的实际行为不同。

**更正后的实验**：

| 条件 | 结果 |
|------|------|
| 每次重试创建新 env（之前测试的方法） | seed=2 第5次成功，seed=17 第1次成功 |
| 复用同一个 env（replay_cpu_sim 实际行为） | **仅第1次成功，后续全部失败** |
| 完全新建 env 每次尝试 | **每次都成功**（seed=2，5/5 True） |

**更正后的结论**：`--max-retry` 在当前实现中**完全无效**，因为 `replay_cpu_sim` 复用同一个 env 对象。

### 4.3 实验 3：use_first_env_state

使用 `--use-first-env-state` 设置与原始轨迹相同的初始状态，测试结果：

| Episode | 默认 | use_first_env_state | 说明 |
|---------|------|-------------------|------|
| 0 | 成功 | 成功 | |
| 1 | 失败 | 失败 | 初始状态一致也无法挽救 |
| 2 | 失败 | 失败 | |
| 8 | 失败 | 失败 | |

**结论**：初始状态差异不是失败原因。

---

## 5. StackCube 对比参照

StackCube-v1 的 replay 数据验证了以上分析：

| 任务 | 原始大小 | Replay 大小 | 成功率推测 |
|------|---------|------------|-----------|
| StackCube-v1 | ~23.1 GB | ~23.3 GB | 接近 100% |
| PlugCharger-v1 | ~28.8 GB | ~1.74 GB | 4.4% |

StackCube 任务（叠方块）对末端精度要求较低，即使有转换误差也能成功完成。PlugCharger 任务（插充电器）需要极高的定位精度，转换误差足以导致任务失败。

---

## 6. 解决方案

### 6.1 [已修复] RecordEpisode 缓冲区累积 Bug

**修改文件**：`data_processing/replay.py` 第 394-398 行

**修改内容**：

```python
# 修改前（第 394-396 行）：
else:
    env.flush_video(save=False)
    tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

# 修改后：
else:
    env.flush_video(save=False)
    if args.save_traj:
        env.flush_trajectory(save=False)
    tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")
```

**修复原理**：当 episode 失败时，`env.flush_trajectory(save=False)` 会清空 RecordEpisode 的内部缓冲区但不写入磁盘，防止失败 episode 的残留数据污染后续成功 episode 的轨迹。

**验证结果**：端到端测试通过，修复后 replay ep3（seed=3）的轨迹仅包含 146 个动作，与原始数据一致，不再混入 ep1/ep2 的数据。

### 6.2 ~~已实施：增大重试次数~~（已更正：max-retry 无效）

> **更正**：经深入调查，`--max-retry` 参数在当前 `replay_cpu_sim` 实现中**无法提高成功率**。
> 原因是 `replay_cpu_sim` 复用同一个 env 对象跨重试，而 PhysX 状态泄漏导致
> 第一次失败后后续重试必然失败。只有每次重试重新创建 env 对象才能让重试生效。

### 6.3 ~~建议：修改代码支持 max_episode_steps参数~~（已更正）

> **更正**：经实验验证，`max_episode_steps=200` 不是导致 replay 失败的原因（ManiSkill 的 TimeLimitWrapper 不阻止继续执行）。此修改对提升成功率帮助极小，不再作为优先建议。

### 6.4 替代方案：保留失败 episode

```bash
python data_processing/replay.py \
  --traj-path demos/PlugCharger-v1/motionplanning/20260610_173934.h5 \
  -c pd_ee_delta_pose \
  -o rgb \
  --save-traj \
  --allow-failure
```

**注意**：`--allow-failure` 会保存所有 episode，但失败 episode 的动作质量不可靠，需在后续训练中过滤。

### 6.5 根本方案：在目标控制模式下重新采集数据

如果需要高质量的 `pd_ee_delta_pose` 轨迹数据，最根本的方案是直接在该控制模式下运行 motion planning solver 重新采集。这需要为 `pd_ee_delta_pose` 控制器重新编写求解器。

---

## 7. 下一步操作

修复 Bug 后需重新运行 replay：

```bash
python data_processing/replay.py \
  --traj-path demos/PlugCharger-v1/motionplanning/20260610_173934.h5 \
  -c pd_ee_delta_pose \
  -o rgb \
  --save-traj \
  --max-retry 5
```

原始失败的输出已备份为：

```
demos/PlugCharger-v1/motionplanning/20260610_173934.rgb.pd_ee_delta_pose.physx_cpu.h5.bak_20260610_225801
demos/PlugCharger-v1/motionplanning/20260610_173934.rgb.pd_ee_delta_pose.physx_cpu.json.bak_20260610_225801
```

---

## 8. 总结

| 因素 | 影响程度 | 修复状态 |
|------|---------|---------|
| pd_joint_pos -> pd_ee_delta_pose 转换精度损失 | 高（约30%永久失败） | 无法修复（固有精度问题） |
| **RecordEpisode 缓冲区累积 Bug** | **严重（数据污染）** | **已修复（replay.py:394-398）** |
| ~~max_episode_steps=200 截断~~ | ~~中~~ | ~~已更正：实际不影响~~ |
| ~~物理仿真非确定性~~ | ~~低~~ | ~~已更正：实为确定性状态泄漏~~ |
| PhysX 状态泄漏（env 复用） | 高（重试完全无效） | 需要修改代码：每次重试重新创建 env |
| PlugCharger 任务精度要求极高 | 根本原因 | 无法修复（任务特性） |