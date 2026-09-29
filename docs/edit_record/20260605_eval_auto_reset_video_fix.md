# 2026-06-05: 评估脚本 Auto-Reset 逻辑与视频保存修复

## 概述

修复 `evaluation/evaluate.py` 中因缺少 auto-reset 逻辑导致视频保存异常（除第一个视频外，所有视频仅含 2 帧）的问题，以及 partial reset 与 `reconfiguration_freq=1` 冲突导致 `RuntimeError` 的问题。

## 根因分析

### 问题现象

运行评估后，`videos/` 目录中生成了数百个视频文件，但只有 `0.mp4` 有正常帧数（201 帧 / 6.7 秒），其余所有视频均仅含 2 帧（0.067 秒）。

### 根因

之前修复 `ManiSkillVectorEnv` 兼容性问题时，我们移除了 `ManiSkillVectorEnv` 包裹器。但 `ManiSkillVectorEnv` 除了类型转换外，还提供了 **auto-reset 机制**——当某个并行环境的 episode 结束（terminated 或 truncated）时，自动对其执行 partial reset 并在 `info` 中注入 `final_info`/`final_observation`。

移除 `ManiSkillVectorEnv` 后，评估循环的 `truncated.all()` 检查出现了以下问题链：

1. **ManiSkill GPU 并行仿真不自动重置已结束的环境**：当 `num_envs>1` 且使用 GPU 后端时，ManiSkill 基础环境在 episode 结束后不会自动重置环境。

2. **`gymnasium.wrappers.TimeLimit` 持续触发 `truncated=True`**：`TimeLimit` 在 `_elapsed_steps >= max_episode_steps` 时设置 `truncated=True`，但不做重置。由于底层环境没有重置，`_elapsed_steps` 持续递增，导致超时后 **每一步** 都返回 `truncated=True`。

3. **`truncated.all()` 在首次超时后每步都为 True**：3 个环境同步超时后，`truncated.all()` 在后续每一步都为 True，因为所有环境都处于已超时但未重置的状态。

4. **`flush_video()` 每步调用，仅产出 2 帧视频**：`RecordEpisode.step()` 在 `_video_steps == 0` 时捕获初始帧，step 后捕获第二帧。evaluate.py 每步调用 `flush_video()`，清空帧缓冲区。下一帧又从 `_video_steps == 0` 开始，再次捕获 2 帧——循环往复。

### 调用链对比

**修复前（有 ManiSkillVectorEnv）：**
```
env.step(action)
  → ManiSkillVectorEnv.step()
    → base_env.step()  → terminated, truncated, info
    → if done.any(): env.reset(options={"env_idx": done_envs})  # auto-reset
    → 注入 final_info, final_observation 到 info
  → 返回 obs_after_reset, rew, terminated, truncated, info_with_final_info
```
评估循环中 `truncated.all()` 仅在所有环境同时完成时为 True（因为已重置的环境 `truncated=False`）。

**修复前（移除 ManiSkillVectorEnv 后，broken）：**
```
env.step(action)
  → TimeLimit.step()
    → base_env.step()  → terminated, truncated, info
    → if elapsed_steps >= max_episode_steps: truncated = True
  → 返回 obs, rew, terminated, truncated=True (每步都是!), info
```
没有 auto-reset，已超时的环境持续返回 `truncated=True`。

**修复后（手动 auto-reset）：**
```
env.step(action)
  → TimeLimit.step()
    → base_env.step()  → terminated, truncated, info
    → truncated 仅在 elapsed_steps == max_episode_steps 时为 True
  → evaluate.py: if done.any(): env.reset(options={"env_idx": done_envs})
  → 重置后的环境 truncated=False
```

### 验证证据

通过 3 个环境的测试脚本（`max_episode_steps=10`）确认：

| 步数 | 无 auto-reset | 有 auto-reset |
|------|--------------|--------------|
| 1-9 | truncated=[F,F,F] | truncated=[F,F,F] |
| 10 | truncated=[T,T,T] | truncated=[T,T,T] |
| 11+ | truncated=[T,T,T] **每步** | truncated=[F,F,F] (已重置) |

## 修改的文件

### evaluation/evaluate.py

#### 修改 1：主循环重写——手动 auto-reset

**移除的逻辑：**
```python
# 旧逻辑：依赖 truncated.all() 和 ManiSkillVectorEnv 的 final_info
if truncated.all():
    if "final_info" in info and "episode" in info["final_info"]:
        for k, v in info["final_info"]["episode"].items():
            eval_metrics[k].append(v.float())
        ...
    if eval_cfg.save_video:
        eval_envs.flush_video()
    policy.reset()
```

**新增的逻辑：**
```python
# 新逻辑：手动检测 per-env done，partial reset，自行追踪指标
done = terminated | truncated
if done.any():
    done_idx = torch.where(done)[0]
    done_np = done_idx.cpu().numpy()

    # 记录每个完成环境的指标
    for i in done_idx:
        ep_metrics = {
            "success_once": success_once[i].item(),
            "return": returns[i].item(),
            "episode_len": episode_lens[i].item(),
            "reward": returns[i].item() / episode_lens[i].item(),
        }
        ...

    # 重置已完成环境的追踪器
    returns[done_idx] = 0
    success_once[done_idx] = False
    episode_lens[done_idx] = 0

    # Auto-reset: partial reset 已完成的环境
    obs, info = eval_envs.reset(options={"env_idx": done_np})
```

#### 修改 2：视频保存改为自然边界刷新

**移除的逻辑：** 在 `truncated.all()` 时调用 `flush_video()`（导致每步 2 帧视频）。

**新增的逻辑：** 依赖 `RecordEpisode` 的 `max_steps_per_video` 自动刷新机制（每 `max_episode_steps` 步自动保存一个视频），评估循环结束后如有残余帧再手动刷新一次。

```python
# 评估循环结束后
if eval_cfg.save_video and len(eval_envs.render_images) > 1:
    eval_envs.flush_video()
```

#### 修改 3：新增 per-env 指标追踪器

```python
returns = torch.zeros(eval_cfg.num_envs, dtype=torch.float32, device=base_env.device)
success_once = torch.zeros(eval_cfg.num_envs, dtype=torch.bool, device=base_env.device)
episode_lens = torch.zeros(eval_cfg.num_envs, dtype=torch.int64, device=base_env.device)
```

这些追踪器在每个 step 更新，在每个环境完成时记录指标并重置。替代了之前依赖 `ManiSkillVectorEnv` 注入 `final_info["episode"]` 的方式。

#### 修改 4：获取 base_env 引用

```python
base_env = eval_envs.unwrapped
```

用于获取 `device` 属性来创建 per-env 追踪张量。

#### 修改 5：`reconfiguration_freq` 从 1 改为 0

```python
# 修改前
reconfiguration_freq=1,

# 修改后
reconfiguration_freq=0,
```

**原因**：`reconfiguration_freq=1` 表示每次 reset 都重新配置场景。但 partial reset（`reset(options={"env_idx": ...})`）只重置部分环境，不能与 reconfigure 同时进行——ManiSkill 在 `BaseEnv.reset()` 中检查并抛出 `RuntimeError: Cannot do a partial reset and reconfigure the environment`。

`reconfiguration_freq=1` 适用于数据收集/重放（全量 reset，每次重新生成场景），评估时不需要重新配置场景，设为 `0`（默认值）即可。

## 视频输出对比

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| 第一个视频 | 201 帧 / 6.7s | 51 帧 / 1.7s (50 步 + 1 初始帧) |
| 后续视频 | 2 帧 / 0.067s | 51 帧 / 1.7s |
| 视频分割方式 | 每步 flush 一次 | 每 max_episode_steps 步自然分割 |

## 参考

- `data_processing/replay.py` 中的 `replay_parallelized_sim()` 函数：使用 `env.flush_trajectory(env_idxs_to_flush=...)` 进行 per-env 轨迹刷新，但不处理视频（视频通过 `max_steps_per_video` 自动分割）。
- `mani_skill/vector/wrappers/gymnasium.py` 中 `ManiSkillVectorEnv.step()`：原始 auto-reset 实现参考。