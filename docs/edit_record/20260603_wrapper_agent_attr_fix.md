# Bug Fix: Wrapper 透传导致 AttributeError

- **日期**: 2026-06-03
- **触发命令**:
  ```bash
  python data_processing/replay.py \
    --traj-path demos/StackCube-v1/motionplanning/20260603_165716.h5 \
    -c pd_ee_delta_pose \
    -o rgb \
    --save-traj
  ```
- **报错信息**:
  ```
  AttributeError: 'TimeLimitWrapper' object has no attribute 'agent'
  ```

---

## 1. 背景

`replay.py` 支持 action 控制模式转换（如 `pd_joint_pos` -> `pd_ee_delta_pose`）。转换逻辑位于 ManiSkill 库的 `conversion.py`，由 replay.py 调用。

涉及的关键环境对象：

| 变量 | 实际类型 | 来源 |
|------|----------|------|
| `ori_env` | `TimeLimitWrapper` | `gym.make()` 自动包装 |
| `env` | `RecordEpisode` -> `FlattenActionSpaceWrapper` -> `TimeLimitWrapper` -> `BaseEnv` | `gym.make()` + `RecordEpisode` 包装 |

环境包装链（最外层 -> 最内层）：

```
TimeLimitWrapper       -- 无 .agent, 只截断超时回合
    -> BaseEnv          -- 有 .agent, 核心环境逻辑

RecordEpisode           -- 无 .agent, 记录轨迹和视频
    -> FlattenActionSpaceWrapper
        -> TimeLimitWrapper
            -> BaseEnv  -- 有 .agent
```

## 2. 问题原因

Gymnasium 的 `Wrapper` 类**没有** `__getattr__` 属性委托机制。旧版 `gym` 库有此机制，但 `gymnasium` 移除了它，改为显式的 `get_wrapper_attr()` 方法。

因此，当 `ori_env`（类型为 `TimeLimitWrapper`）被传入 `conversion.py` 的函数后：

```python
ori_controller = ori_env.agent.controller   # AttributeError!
```

`TimeLimitWrapper` 自己没有 `.agent` 属性，也不会自动委托到内层的 `BaseEnv`。

## 3. 影响范围

`conversion.py` 中有 3 个函数存在同样的问题，共 6 处：

| 函数 | 行号 | 问题代码 |
|------|------|----------|
| `from_pd_joint_pos_to_ee` | 97-98 | `ori_env.agent` / `env.agent` |
| `from_pd_joint_pos` | 233-234 | `ori_env.agent` / `env.agent` |
| `from_pd_joint_delta_pos` | 301-302 | `ori_env.agent` / `env.agent` |

## 4. 约束条件

- **不能修改 `site-packages`**：项目基于 ManiSkill 和 LeRobot 的 pip 库开发，不允许改动库源码。
- `conversion.py` 中同一个环境参数存在**两种深度的访问**：

| 操作 | 示例 | 需要的层级 |
|------|------|-----------|
| 访问 `.agent` | `ori_env.agent.controller` | 需要穿透到 **BaseEnv**（最内层） |
| 调用 `.step()` | `env.step(action)` | 需要保留 **最外层 wrapper**（RecordEpisode 录制轨迹） |
| 调用 `.render_human()` | `env.render_human()` | 穿透到 **BaseEnv**（wrapper 没有） |
| 访问 `.device` | `env.device` | 穿透到 **BaseEnv**（wrapper 没有） |

如果直接传 `env.unwrapped`，`.step()` 会绕过 `RecordEpisode`，导致轨迹和视频录制丢失。

## 5. 修复方案

在 `replay.py` 中定义一个属性代理类 `_UnwrappedAttrProxy`，在调用 `conversion` 函数时用代理包裹环境对象。

代理逻辑：
1. 先在 wrapper 层查找属性（如 `.step()`、`.reset()`）
2. wrapper 层找不到时，穿透到 `.unwrapped`（BaseEnv）查找（如 `.agent`、`.device`、`.render_human()`）

这样：
- `.step()` 走 RecordEpisode wrapper，轨迹/视频录制不受影响
- `.agent` 穿透到 BaseEnv，不再 AttributeError
- 不修改任何 `site-packages` 代码

### 5.1 新增代理类

```python
class _UnwrappedAttrProxy:
    """Attribute proxy: resolve attributes on the wrapper first,
    then fall through to the unwrapped BaseEnv.

    This ensures that .step() and .reset() go through the full
    wrapper chain (e.g. RecordEpisode for trajectory recording),
    while .agent, .device, .render_human() etc. are resolved on
    the underlying BaseEnv where they actually live.
    """

    def __init__(self, wrapped_env):
        self._wrapped = wrapped_env

    def __getattr__(self, name):
        try:
            return getattr(self._wrapped, name)
        except AttributeError:
            return getattr(self._wrapped.unwrapped, name)
```

### 5.2 修改调用点

```python
# 修改前
info = action_conversion.from_pd_joint_pos(
    ...,
    ori_env,
    env,
    ...
)

# 修改后
info = action_conversion.from_pd_joint_pos(
    ...,
    _UnwrappedAttrProxy(ori_env),
    _UnwrappedAttrProxy(env),
    ...
)
```

## 6. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `data_processing/replay.py` | 新增 `_UnwrappedAttrProxy` 代理类；两处 conversion 调用参数用代理包裹 |