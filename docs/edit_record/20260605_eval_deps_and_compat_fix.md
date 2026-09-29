# 2026-06-05: 评估脚本依赖修复与兼容性适配

## 概述

修复 `evaluation/evaluate.py` 使其在双 GPU 系统、`gymnasium>=1.3.0`、`sapien>=3.0.3` 环境下成功运行。

## 修改的文件

### evaluation/evaluate.py

#### 修改 1：添加 `CUDA_VISIBLE_DEVICES` 环境变量（第 1-9 行）

```python
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

**原因**：在多 GPU 系统上，SAPIEN PhysX GPU 并行仿真（`physx.enable_gpu()`）会因 CUDA/Vulkan 设备选择冲突而死锁。限制为单 GPU 确保 PhysX 运行时和渲染器使用同一物理设备。

**参考**：SAPIEN #139, #142, #101; ManiSkill #460

#### 修改 2：移除 `ManiSkillVectorEnv` 的导入和使用

移除内容：
```python
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
```

同时移除了之前临时添加的 `VectorEnv.__init__` 猴子补丁。

**原因**：
1. `ManiSkillVectorEnv` 继承自 `gymnasium.vector.VectorEnv`。在 `gymnasium>=1.3.0` 中，`VectorEnv.__init__` 不再接受位置参数（它从 `Generic` 继承，没有自定义 `__init__`），导致 `TypeError: object.__init__() takes exactly one argument`。
2. `RecordEpisode`（`gymnasium.Wrapper` 子类）断言被包裹的环境必须是 `gymnasium.Env` 类型，而 `ManiSkillVectorEnv` 是 `VectorEnv`，不是 `Env`，导致 `AssertionError`。
3. 调换包裹顺序（先 RecordEpisode 后 ManiSkillVectorEnv）后，`ManiSkillVectorEnv` 不代理底层 `RecordEpisode` 的 `flush_video()` 方法，导致 `AttributeError`。

**解决方案**：参照 `data_processing/replay.py` 的做法，完全不使用 `ManiSkillVectorEnv`。当 ManiSkill 以 `num_envs>1` 和 `sim_backend='auto'`（解析为 `physx_cuda`）创建环境时，基础环境已经在 GPU 上完成了向量化，无需额外包裹。

#### 修改 3：重构 `build_eval_env()` 的包裹顺序

修改前：
```python
env = gym.make(...)
env = ManiSkillVectorEnv(env, ...)
env = RecordEpisode(env, ...)  # 失败：VectorEnv 不是 Env
```

修改后：
```python
env = gym.make(...)
if cfg.save_video:
    env = RecordEpisode(env, ...)
# 不需要 ManiSkillVectorEnv 包裹
return env
```

#### 修改 4：`flush_video()` 调用方式不变

```python
if eval_cfg.save_video:
    eval_envs.flush_video()
```

现在直接调用 `RecordEpisode.flush_video()`（不需要 `._env` 代理），因为 `RecordEpisode` 是最外层包裹器。

---

## 系统级变更（不在仓库中）

### 安装缺失的 Python 包到 `emix` conda 环境

1. **`loguru`**：`evaluate.py` 的日志依赖，环境中缺失。
2. **`rich_argparse`**：`evaluate.py` 的命令行格式化依赖，环境中缺失。

```bash
/home/rogerspyke/miniforge3/envs/emix/bin/pip install loguru rich_argparse
```

### 下载 PhysX GPU 共享库

`physx.enable_gpu()` 需要 `libPhysXGpu_64.so`（压缩包约 77MB，解压后约 226MB）存放在 `~/.sapien/physx/<版本号>/`。本系统中 sapien 的自动下载机制因无法访问 GitHub 而静默阻塞，需要手动下载并解压：

```bash
wget -O ~/.sapien/physx/105.1-physx-5.3.1.patch0/linux-so.zip \
  "https://github.com/sapien-sim/physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-so.zip"
cd ~/.sapien/physx/105.1-physx-5.3.1.patch0/ && unzip linux-so.zip
```

**重要性**：缺少 `libPhysXGpu_64.so` 时，调用 `physx.enable_gpu()`（ManiSkill 在 `num_envs>1` 时内部会调用）会因下载阻塞而永久卡住。在 `num_envs=1` 时 ManiSkill 回退到 CPU 仿真，不需要此库。

---

## 根因分析

评估命令的失败由 7 个问题串联导致：

| # | 错误 | 根因 | 修复 |
|---|------|------|------|
| 1 | `ModuleNotFoundError: No module named 'loguru'` | 环境缺少依赖 | `pip install loguru` |
| 2 | `ModuleNotFoundError: No module named 'rich_argparse'` | 环境缺少依赖 | `pip install rich_argparse` |
| 3 | 进程在 `physx.enable_gpu()` 卡死 | `libPhysXGpu_64.so` 不存在；自动下载因网络问题阻塞 | 手动下载 + 解压 |
| 4 | 双 GPU 环境下 `num_envs>1` 创建环境卡死 | PhysX CUDA/Vulkan 设备选择死锁 | `CUDA_VISIBLE_DEVICES=0` |
| 5 | `TypeError: object.__init__() takes exactly one argument` | `gymnasium>=1.3.0` 中 `VectorEnv` 移除了 `__init__`，`ManiSkillVectorEnv` 仍传递位置参数 | 移除 `ManiSkillVectorEnv`，直接使用 `RecordEpisode` |
| 6 | `AssertionError: Expected env to be gymnasium.Env` | `RecordEpisode` 断言要求 `Env` 类型，但 `ManiSkillVectorEnv` 是 `VectorEnv` | 同 #5 |
| 7 | `AttributeError: 'ManiSkillVectorEnv' has no attribute 'flush_video'` | `ManiSkillVectorEnv` 不代理底层 `RecordEpisode` 的方法 | 同 #5 |

### 问题 4 详解：多 GPU 系统下 PhysX GPU 初始化死锁

本系统有 2 张 RTX 4070。SAPIEN 通过 Vulkan 检测到 3 个设备（2 张 GPU + 1 个 llvmpipe 软件渲染器）。PhysX GPU 并行仿真初始化时需要 CUDA 和 Vulkan 在同一物理 GPU 上创建 context，但多 GPU 可见时设备选择产生冲突，导致死锁。这是 SAPIEN 的已知 bug。

**调用链**：
```
gym.make("StackCube-v1", num_envs=5, sim_backend="auto")
  → ManiSkill BaseEnv.__init__()
    → 检测 num_envs>1 → 选择 "physx_cuda"
      → physx.enable_gpu()
        → CUDA 在 device 0 上创建 context
        → Vulkan 在检测到的 3 个设备中选择 → 选择冲突 → 死锁
```

**设置 `CUDA_VISIBLE_DEVICES=0` 后**：系统只暴露 1 张 GPU，SAPIEN 只看到 1 个 Vulkan GPU 设备，CUDA 和 Vulkan 在同一张卡上初始化，正常工作。

### 问题 5-7 详解：`ManiSkillVectorEnv` 与 `gymnasium>=1.3.0` 的不兼容

这三个错误是同一个架构问题的三种表现形式。

**类型关系**：
```
gymnasium.Env                    ← RecordEpisode 要求被包裹的是这个类型
gymnasium.vector.VectorEnv       ← ManiSkillVectorEnv 是这个类型（与 Env 平级，无继承关系）
```

**问题 5（TypeError）**：

```python
# ManiSkillVectorEnv 的 __init__ 源码（简化）
class ManiSkillVectorEnv(gym.vector.VectorEnv):
    def __init__(self, env, ...):
        self._env = env
        super().__init__(
            num_envs,                          # 位置参数 1
            single_observation_space,          # 位置参数 2
            single_action_space,               # 位置参数 3
        )
```

```python
# gymnasium 1.3.0 中 VectorEnv 的继承链
class VectorEnv(Generic[ObsType, ActType, ArrayType]):
    # 没有自定义 __init__
    # Generic.__init__ → 最终是 object.__init__
    # object.__init__() 只接受 self，不接受额外参数
```

在旧版 gymnasium（0.x 或 1.0-1.2）中，`VectorEnv` 有自定义 `__init__` 来接收并存储 `num_envs`、`single_observation_space`、`single_action_space`。gymnasium 1.3.0 移除了这个 `__init__`，改用 `@property` 从子类获取。`super().__init__(3个参数)` 直接穿透到 `object.__init__()`，后者不接受额外参数，导致 `TypeError`。

**问题 6（AssertionError）**：

```python
# RecordEpisode 的 __init__ 断言
class RecordEpisode(gymnasium.Wrapper):
    def __init__(self, env):
        assert isinstance(env, Env), ...  # ← 要求 Env 类型
```

原始代码的包裹顺序：
```python
env = gym.make(...)                    # gymnasium.Env ✓
env = ManiSkillVectorEnv(env, ...)     # 变为 VectorEnv 类型
env = RecordEpisode(env, ...)          # ← 断言失败：VectorEnv 不是 Env
```

**问题 7（AttributeError）**：

调换包裹顺序后：
```python
env = gym.make(...)                    # gymnasium.Env ✓
env = RecordEpisode(env, ...)          # gymnasium.Wrapper → Env ✓
env = ManiSkillVectorEnv(env, ...)     # ← 触发问题 5 的 TypeError
```

即使修了问题 5，`ManiSkillVectorEnv` 也不会自动代理底层 `RecordEpisode` 的 `flush_video()` 方法，需手动 `._env` 穿透。

**统一解决方案**：

ManiSkill 在 `num_envs>1` + GPU 后端时，基础环境本身已经是向量化的（`env.step()` 同时推进 N 个环境，`env.action_space.shape == (N, D)`），`ManiSkillVectorEnv` 对 GPU 后端是多余的包裹层。参照 `data_processing/replay.py` 的做法，直接用 `RecordEpisode` 包裹基础环境即可。

---

## 验证

评估确认成功运行：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py \
  --policy lerobot \
  --pretrained_path outputs/train/2026-06-04/16-13-39_diffusion/checkpoints/030000/pretrained_model/ \
  --repo_id stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --dataset_root data/lerobot/maniskill/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --control_mode pd_ee_delta_pose
```

- 环境在 `num_envs=5` 下成功创建并运行在 GPU 上
- 策略加载并执行推理
- 100 个评估回合的指标正常记录（success_once, return, episode_len, reward, success_at_end）
- 视频正常保存