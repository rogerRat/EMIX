# 2026-06-08: ManiSkill API 使用深度调研文档

## 概述

本文档记录 `evaluation/evaluate.py` 对 ManiSkill 包的完整依赖关系：在哪些位置、以什么程度、使用了 ManiSkill 的什么功能、做了什么。同时包含本对话中的所有调研结论：gymnasium 版本兼容性、ManiSkillVectorEnv 不可替代性分析、reconfiguration_freq 语义差异、以及实机验证结果。

---

## 1. evaluate.py 对 ManiSkill 的调用点总览

| 调用点 | 文件:行号 | ManiSkill 组件 | 功能描述 |
|--------|-----------|----------------|----------|
| `gym.make("StackCube-v1", ...)` | evaluate.py:95 | ManiSkill 环境注册系统 | 通过 gymnasium 注册表创建 ManiSkill 环境 |
| `RecordEpisode(env, ...)` | evaluate.py:107 | `mani_skill.utils.wrappers.record` | 录制评估视频和轨迹 |
| `env.unwrapped` | evaluate.py:133 | `gym.Wrapper.unwrapped` → `BaseEnv` | 获取底层 BaseEnv 以访问 `device`/`num_envs` |
| `base_env.device` | evaluate.py:160 | `BaseEnv.device` | 获取 GPU/CPU 设备，用于创建 per-env 追踪张量 |
| `base_env.num_envs` | evaluate.py:160 | `BaseEnv.num_envs` | 获取并行环境数量 |
| `env.reset(seed=...)` | evaluate.py:154 | `BaseEnv.reset()` (经 Wrapper 链) | 全量 reset |
| `env.step(action)` | evaluate.py:168 | `BaseEnv.step()` (经 Wrapper 链) | 推进仿真一步 |
| `env.reset(options={"env_idx": ...})` | evaluate.py:206-208 | `BaseEnv.reset()` partial reset | 仅重置已完成的环境 |
| `info["success"]` | evaluate.py:173,190 | `BaseEnv.get_info()` → `evaluate()` | 任务定义的成功判断 |
| `env.render_images` | evaluate.py:211 | `RecordEpisode.render_images` | 获取累积的视频帧 |
| `env.flush_video()` | evaluate.py:212 | `RecordEpisode.flush_video()` | 将帧写入视频文件 |

**注意**：`evaluation/policies/lerobot.py` 和 `evaluation/policies/utils.py` **不依赖 ManiSkill**，它们只使用 `torch`、`numpy`、`cv2` 和 `lerobot`。

---

## 2. 每个调用点的详细分析

### 2.1 `gym.make(...)` — 环境创建

```python
base_env = gym.make(
    cfg.env_id,               # "StackCube-v1"
    num_envs=cfg.num_envs,    # 5
    max_episode_steps=cfg.max_episode_steps,  # 200
    reconfiguration_freq=0,
    obs_mode=cfg.obs_mode,    # "rgb"
    control_mode=cfg.control_mode,  # "pd_joint_pos"
    render_mode="rgb_array",
    sensor_configs=dict(shader_pack="default", width=512, height=512),
    sim_backend="auto",
)
```

**ManiSkill 注册机制**：ManiSkill 通过 `@register_env("StackCube-v1", ...)` 将环境注册到 gymnasium。`gym.make` 调用时，gymnasium 通过注册表找到 `mani_skill.utils.registration.make()`，后者实例化 `StackCubeEnv(BaseEnv)` 并用 Wrapper 链包裹。

**返回的 Wrapper 链**（由内到外）：

```
StackCubeEnv (BaseEnv)
  ← MSTimeLimit (ManiSkill 自定义 TimeLimit)
    ← gymnasium.wrappers.TimeLimit
```

**关键参数语义**：

| 参数 | 当前值 | 语义 |
|------|--------|------|
| `num_envs=5` | 5 | 创建 5 个并行环境。`sim_backend="auto"` 时 `num_envs>1` 自动切换到 GPU 并行仿真 |
| `reconfiguration_freq=0` | 0 | 不在 reset 时重新配置场景。设为 1 时每次 full reset 会销毁并重建整个 PhysX 场景，**但不能与 partial reset 同时使用** |
| `sim_backend="auto"` | auto | `num_envs>1` 时自动选择 `"physx_cuda"`，否则选择 `"physx_cpu"` |
| `obs_mode="rgb"` | rgb | 观测模式为 RGB 图像 |
| `control_mode="pd_joint_pos"` | pd_joint_pos | 控制模式，决定动作空间维度 |
| `render_mode="rgb_array"` | rgb_array | 支持视频录制 |

**`sim_backend="auto"` 的 GPU 仿真**：

当 `num_envs=5` + `sim_backend="auto"` 时，解析为 `"physx_cuda"`。ManiSkill：
1. 调用 `physx.enable_gpu()` 初始化 PhysX GPU 系统
2. 创建 N 个子场景（sub-scene），排列在方形网格上
3. 所有并行仿真在同一 GPU 上同时执行
4. `step()` 和 `reset()` 操作以批量 tensor 形式传入/传出

### 2.2 `RecordEpisode` — 视频录制包裹器

```python
base_env = RecordEpisode(
    base_env,
    output_dir=os.path.join(output_dir, "videos"),
    save_video=True,
    save_trajectory=cfg.save_trajectory,  # False
    video_fps=cfg.video_fps,               # 30
    max_steps_per_video=cfg.max_episode_steps,  # 200
)
```

**`RecordEpisode` 是 ManiSkill 提供的 `gym.Wrapper` 子类**，不是 gymnasium 标准库的一部分。

**关键行为**：

| 功能 | 实现 |
|------|------|
| 帧捕获 | 每次 `step()` 后调用 `self.capture_image()` → `self.env.render()` → `tile_images()` 拼接所有并行环境的帧 |
| 帧存储 | `self.render_images`：Python `list`，每元素是一个 numpy 数组 |
| 视频写入 | `self.flush_video()`：将 `render_images` 中的帧通过 `images_to_video()` 写入 MP4 |
| 自动分割 | `self._video_steps` 计数器，达到 `max_steps_per_video` 时自动 `flush_video()` |
| 轨迹记录 | `self._trajectory_buffer`：按 step 存储 obs/action/reward/terminated/truncated/success/fail |
| 轨迹写入 | `self.flush_trajectory()`：写入 HDF5 文件 |
| partial reset 处理 | 在 `reset(options={"env_idx": ...})` 中正确处理部分环境的轨迹边界 |

**类型约束**：`RecordEpisode.__init__` 通过 `super().__init__(env)` → `gym.Wrapper.__init__`，而 `gym.Wrapper` 在 gymnasium 中有 `isinstance(env, gym.Env)` 断言。这意味着 **`RecordEpisode` 只能包裹 `gymnasium.Env`，不能包裹 `gymnasium.VectorEnv`**。

### 2.3 `base_env = eval_envs.unwrapped` — 获取 BaseEnv

```python
base_env = eval_envs.unwrapped
```

`gym.Wrapper.unwrapped` 沿 Wrapper 链递归到最内层，返回 `StackCubeEnv` 实例（`BaseEnv` 子类）。evaluate.py 使用它来获取：

| 属性 | 类型 | 用途 |
|------|------|------|
| `base_env.device` | `torch.device` | 创建 per-env 追踪张量的设备（`'cuda'` 或 `'cpu'`） |
| `base_env.num_envs` | `int` | 创建追踪张量的维度 |

### 2.4 `env.step(action)` — 仿真推进

```python
obs, rew, terminated, truncated, info = eval_envs.step(action)
```

**调用链**：
```
evaluate.py: eval_envs.step(action)
  → RecordEpisode.step(action)
    → RecordEpisode 内部:
      - 可选: 捕获初始帧
      - super().step(action) → gym.TimeLimit.step(action)
        → MSTimeLimit.step(action) → BaseEnv.step(action)
          → BaseEnv._step_action(action)   # 执行动作
          → BaseEnv._elapsed_steps += 1     # 递增步数
          → BaseEnv.get_info()              # 计算 info
          → BaseEnv.get_obs(info)           # 计算观测
          → BaseEnv.get_reward(obs, action, info)  # 计算奖励
          → 计算 terminated (基于 success/fail)
          → 返回 truncated = zeros           # BaseEnv 不计算 truncation!
        → MSTimeLimit: 用 elapsed_steps >= max_episode_steps 覆盖 truncated
      → gym.TimeLimit: 也用 _elapsed_steps >= max_episode_steps 覆盖 truncated
    → RecordEpisode 内部:
      - 追加帧到 render_images
      - 追加 step 数据到 _trajectory_buffer
      - 检查 _video_steps >= max_steps_per_video → 自动 flush_video
      - 返回 (obs, rew, terminated, truncated, info)
```

**`BaseEnv.step()` 返回值的关键特性**：

| 返回值 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `obs` | `dict` 或 `torch.Tensor` | 依 `obs_mode` 而定 | 观测 |
| `rew` | `torch.Tensor` | `(num_envs,)` | 每个环境的奖励 |
| `terminated` | `torch.Tensor` | `(num_envs,)` | 每个 env 的终止标志 |
| `truncated` | `torch.Tensor` | `(num_envs,)` | **BaseEnv 始终返回全 False** |
| `info` | `dict` | - | 至少包含 `elapsed_steps` 和 `success` |

**重要**：`BaseEnv.step()` 返回的 `truncated` 始终为全零。截断（时间限制）由 `MSTimeLimit` Wrapper 根据 `_elapsed_steps >= max_episode_steps` 计算。这意味着 **truncation 逻辑在 Wrapper 层而非环境层**。

### 2.5 `info["success"]` — 任务成功判断

```python
if "success" in info:
    success_once = success_once | info["success"]
```

`info["success"]` 由 `BaseEnv.get_info()` → `evaluate()` 方法填充。对于 `StackCube-v1`，这是一个 `torch.Tensor` 形状 `(num_envs,)`，dtype `bool`，表示每个并行环境是否满足成功条件。

部分任务还定义了 `info["fail"]`（如物体掉落），evaluate.py 未使用但 `ManiSkillVectorEnv` 的 `record_metrics` 会追踪。

### 2.6 `env.reset(options={"env_idx": ...})` — Partial Reset

```python
obs, info = eval_envs.reset(options={"env_idx": done_np})
```

**这是 evaluate.py 中最关键的 ManiSkill 特有 API 之一**。

**`BaseEnv.reset()` 的 partial reset 流程**：

1. 解析 `options["env_idx"]` 为需要重置的环境索引
2. 创建 `scene._reset_mask`：`True` 对应重置的 env，`False` 对其他 env
3. 只对重置的 env 调用 `_set_episode_rng()` 重置随机种子
4. `self._elapsed_steps[env_idx] = 0` — 只重置指定 env 的步数计数
5. `_clear_sim_state()` — 清零速度等物理状态
6. `_initialize_episode(env_idx, options)` — 调用任务自定义的初始化（如随机化物体位置）
7. `_reset_mask` 恢复为全 True
8. GPU 仿真：`_gpu_apply_all()` → `gpu_update_articulation_kinematics()` → `_gpu_fetch_all()`

**关键约束**：`reconfiguration_freq != 0` 与 partial reset **互斥**。ManiSkill 的 `BaseEnv.reset()` 中有断言：

```python
if len(env_idx) != self.num_envs and reconfigure:
    raise RuntimeError("Cannot do a partial reset and reconfigure the environment.")
```

这是 evaluate.py 将 `reconfiguration_freq` 从 1 改为 0 的原因。

### 2.7 `env.render_images` 和 `env.flush_video()` — 视频输出

```python
if eval_cfg.save_video and len(eval_envs.render_images) > 1:
    eval_envs.flush_video()
```

| API | 来源 | 类型 | 功能 |
|-----|------|------|------|
| `render_images` | `RecordEpisode` | `list[numpy.ndarray]` | 累积的渲染帧列表 |
| `flush_video()` | `RecordEpisode` | 方法 | 将帧写入 MP4 文件，清空 `render_images` |

---

## 3. Wrapper 链完整结构

evaluate.py 中完整的环境 Wrapper 链：

```
RecordEpisode                     ← 最外层，视频/轨迹录制
  └─ gymnasium.wrappers.TimeLimit ← gymnasium 标准 TimeLimit
       └─ MSTimeLimit              ← ManiSkill 自定义 TimeLimit
            └─ StackCubeEnv       ← ManiSkill BaseEnv 子类（GPU 并行仿真）
```

**每个 Wrapper 的职责**：

| Wrapper | 职责 | 是否 ManiSkill 提供 |
|---------|------|---------------------|
| `StackCubeEnv` | GPU 并行物理仿真、任务定义 | 是 |
| `MSTimeLimit` | 批量化 truncation 判断 | 是 |
| `gym.TimeLimit` | 标准 truncation 判断 | 否（gymnasium） |
| `RecordEpisode` | 视频/轨迹录制 | 是 |

---

## 4. ManiSkillVectorEnv 不可替代性分析

### 4.1 ManiSkillVectorEnv 提供的功能

| 功能 | 实现方式 | evaluate.py 是否需要 | evaluate.py 如何替代 |
|------|----------|---------------------|---------------------|
| Auto-reset | `step()` 中检测 done，调用 `reset(options={"env_idx": ...})` | 需要 | 手动检测 `done.any()` 并调用 `reset` |
| `record_metrics` | 追踪 `returns`, `success_once`, `fail_once`, `episode_len` | 部分（不需要 `fail_once`） | 手动创建 per-env 张量追踪 |
| `final_info`/`final_observation` 注入 | 在 auto-reset 前保存 info/obs，注入到 reset 后的返回值 | 不需要 | 直接在 done 检测时记录指标 |
| `ignore_terminations` | 将 `terminated` 强制设为 False | 可选（见 4.2） | 可加一行 `terminated[:] = False` |
| VectorEnv API | `num_envs`, `single_observation_space` 等 | 不需要 | 直接用 BaseEnv 的属性 |

**结论：ManiSkillVectorEnv 的所有功能都可以在 evaluate.py 中手动实现，且已经实现。没有不可替代的功能。**

### 4.2 `ignore_terminations` 语义差异

这是 dev/rgs 与 main 分支唯一的行为差异：

| | main 分支 | dev/rgs 分支 |
|---|-----------|-------------|
| `ignore_terminations` | `True` | 未设置（等效 `False`） |
| 成功后的行为 | 不提前终止，继续到 max_episode_steps | 立即标记 terminated=True |
| 所有 env 结束方式 | 同步超时（truncated.all()） | 异步 per-env done |
| 评估语义 | "固定步数内能否成功" | "首次成功时间" |

**影响**：对 `success_once` 指标无影响（两种方式都正确记录首次成功），但改变了 episode 长度和 return 的含义。如果需要恢复 `ignore_terminations=True` 语义，只需在 done 检测前加：

```python
if ignore_terminations:
    terminated[:] = False
```

### 4.3 `reconfiguration_freq` 语义分析

| 值 | 行为 | 与 partial reset 的关系 |
|----|------|------------------------|
| `0`（默认） | 仅在首次创建时配置场景，后续 reset 仅调用 `_initialize_episode` | 兼容 partial reset |
| `1` | 每次 full reset 重新配置场景（销毁+重建 PhysX 场景） | **不兼容** partial reset |
| `N>1` | 每 N 次 full reset 重新配置一次 | 仅在全量 reset 时触发 |

**实测结果**（gymnasium 0.29.1 + mani_skill 3.0.1）：

- `reconfiguration_freq=0` + partial reset：`_initialize_episode()` 仍然被调用，仍然会随机化初始状态（如 cube 位置）
- `reconfiguration_freq=0` 不等于"固定初始状态"，它只是不销毁重建 PhysX 场景。初始状态的随机化由 `_initialize_episode()` 完成，与 `reconfiguration_freq` 无关

**结论**：`reconfiguration_freq=0` 对评估多样性没有负面影响，同时兼容 partial reset。

---

## 5. gymnasium 版本兼容性验证

### 5.1 实测环境

| 组件 | 版本 |
|------|------|
| gymnasium | 0.29.1 |
| mani_skill | 3.0.1 |
| sapien | 3.0.3 |

### 5.2 gymnasium 0.29.1 验证结果

| 测试项 | 结果 |
|--------|------|
| `gym.make("StackCube-v1", num_envs=2, sim_backend="auto")` | ✅ |
| `env.unwrapped` 获取 BaseEnv | ✅ |
| `base_env.device` / `base_env.num_envs` | ✅ |
| `env.step(action)` 返回 tensor | ✅ |
| `env.reset(options={"env_idx": ...})` partial reset | ✅ |
| `RecordEpisode(base_env, ...)` 包裹 | ✅ |
| `RecordEpisode + partial reset` 循环 | ✅ |
| `env.render_images` / `env.flush_video()` | ✅ |
| TimeLimit 截断行为（不 reset 则 truncated 持续为 True） | ✅ 已确认 |

### 5.3 gymnasium 1.3.x 兼容性

用户已验证 dev/rgs 的 evaluate.py 在 gymnasium 1.3.x 上正确运行。

**不兼容的只有 `ManiSkillVectorEnv`**：

```python
# gymnasium >= 1.0 中 VectorEnv 不再接受位置参数
class VectorEnv(Generic[ObsType, ActType, ArrayType]):
    # 没有自定义 __init__，super().__init__(num_envs, obs_space, act_space) 会报 TypeError
```

```python
# RecordEpisode 的类型断言
class RecordEpisode(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env, Env), ...  # ManiSkillVectorEnv 是 VectorEnv，不是 Env
```

**evaluate.py 的手动 auto-reset 方案在 gymnasium 0.29.1 和 1.3.x 上都能正确运行**，因为它完全不使用 `VectorEnv` API。

### 5.4 ManiSkillVectorEnv 在 gymnasium 0.29.1 上也可工作

实测 `ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)` 在 gymnasium 0.29.1 上可以初始化和运行。原因是 `IS_GYMNASIUM_1` 标志（基于 `gym.__version__ > "1.0.0"` 字符串比较）在 0.29.1 上为 `False`，走的是旧版 `super().__init__(num_envs, obs_space, act_space)` 分支。

**但** `RecordEpisode(ManiSkillVectorEnv(...))` 仍然失败，因为 `gym.Wrapper` 的 `isinstance(env, Env)` 断言与 gymnasium 版本无关。

---

## 6. 核心问题回答

### Q1: dev/rgs 的 evaluate.py 是否在 gymnasium 0.29.1 和 1.3.x 上都能正确运行？

**是的**。实测验证（gymnasium 0.29.1）和用户验证（gymnasium 1.3.x）均通过。根本原因是：evaluate.py 的所有操作都基于 `gymnasium.Env` 接口（`gym.make`、`env.step`、`env.reset`、`env.unwrapped`），不涉及 `VectorEnv` API。`gymnasium.Env` 接口在 0.29.x 和 1.x 之间没有破坏性变化。

### Q2: ManiSkillVectorEnv 是否有 evaluate.py 没有实现或无法实现的不可替代功能？

**没有**。所有功能都可以手动实现且已经实现：

1. **Auto-reset** → 手动 `done.any()` + `env.reset(options={"env_idx": ...})`
2. **record_metrics** → 手动 per-env 张量追踪（`returns`, `success_once`, `episode_lens`）
3. **final_info 注入** → 直接在 done 时记录指标，不需要注入到返回值
4. **ignore_terminations** → 可选，一行代码实现

此外，`ManiSkillVectorEnv` 本身与 `RecordEpisode` 不兼容（类型断言问题），所以即使在 gymnasium 0.29.1 上也无法同时使用两者。main 分支的代码在 `save_video=True` 时从未正确运行过。

---

## 7. 调用链总结图

```
evaluate.py
│
├── gym.make("StackCube-v1", num_envs=5, ...)
│   └── ManiSkill 注册系统 → StackCubeEnv(BaseEnv)
│       └── GPU 并行仿真 (PhysX CUDA)
│
├── RecordEpisode(base_env, ...)
│   └── ManiSkill wrapper: 视频录制 + 帧拼接
│
├── env.step(action) 调用链:
│   RecordEpisode.step
│   → gym.TimeLimit.step
│   → MSTimeLimit.step
│   → BaseEnv.step
│     → _step_action(action)
│     → _elapsed_steps += 1
│     → get_info() → evaluate() → info["success"]
│     → get_obs(info)
│     → get_reward(obs, action, info)
│     → terminated = success | fail
│     → truncated = zeros
│   → MSTimeLimit: truncated = (elapsed_steps >= max_steps)
│   → RecordEpisode: render_images.append(...), trajectory buffer
│
├── env.reset(options={"env_idx": ...}) 调用链:
│   RecordEpisode.reset
│   → gym.TimeLimit.reset
│   → MSTimeLimit.reset
│   → BaseEnv.reset
│     → scene._reset_mask[env_idx] = True
│     → _elapsed_steps[env_idx] = 0
│     → _initialize_episode(env_idx)  # 随机化初始状态
│     → GPU sim sync
│
├── base_env.device → torch.device("cuda")
├── base_env.num_envs → 5
├── info["success"] → torch.Tensor (bool), shape (5,)
├── env.render_images → list[numpy.ndarray]
└── env.flush_video() → images_to_video() → MP4
```

---

## 8. 版本与兼容性矩阵

| 组件 | 版本 | 说明 |
|------|------|------|
| gymnasium | 0.29.1 | 当前安装版本，`VectorEnv` 有 `__init__` |
| gymnasium | 1.3.x | 之前安装版本，`VectorEnv` 无 `__init__` |
| mani_skill | 3.0.1 (已安装) / 3.0.0b22 (requirements.txt) | ManiSkillVectorEnv 通过 `IS_GYMNASIUM_1` 适配 |
| sapien | 3.0.3 | PhysX 仿真引擎 |
| `IS_GYMNASIUM_1` | `gym.__version__ > "1.0.0"` | 0.29.1 → False, 1.3.x → True |

| 方案 | gym 0.29.1 | gym 1.3.x | RecordEpisode 兼容 |
|------|-----------|-----------|---------------------|
| ManiSkillVectorEnv | ✅ 可初始化 | ❌ TypeError | ❌ VectorEnv 非 Env |
| dev/rgs 手动方案 | ✅ 实测通过 | ✅ 用户已验证 | ✅ 直接包裹 Env |