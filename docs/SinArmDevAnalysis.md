# RoboTwin 单臂模式开发计划

## 1. 目标与结论

目标是在当前第三方库 RoboTwin 中增加原生单臂运行模式：场景中只加载一只机械臂，只创建一套 planner，动作、观测和数据文件也只包含活动臂。

可行性结论：可行。任务范围限定为单臂可完成的任务后，不需要改造任务语义；主要工作集中在 RoboTwin 核心的双臂生命周期、控制接口和数据 schema。现有 `click_bell_left` 可作为第一阶段验收任务，但它当前只是逻辑单臂，右臂实体仍会被加载。

当前关键限制：

- `script/collect_data.py` 只接受长度为 1 或 3 的 `embodiment` 配置；长度为 1 表示一个双臂 URDF，长度为 3 表示两个独立机械臂。
- `envs/robot/robot.py` 无条件读取左右两套 embodiment 配置，并初始化左右 entity、joint、gripper 和 planner。
- `envs/_base_task.py` 的观测固定写入 `left_arm + left_gripper + right_arm + right_gripper`，动作执行固定按左右两段切分。
- 当前 EMIX 转换和评估代码通过冻结右臂状态实现行为上的单臂，并非物理场景上的单臂。

## 2. 设计边界

本计划只增加显式的单臂模式，不要求原有双臂任务自动兼容单臂，也不修改单臂任务本身的专家动作逻辑。

建议配置增加以下显式字段：

```yaml
arm_mode: single
active_arm: left
```

`arm_mode: dual` 保持现有行为；`arm_mode: single` 时必须明确 `active_arm`，不使用隐式 fallback。第一版只实现 `left`，随后再验证 `right`。

## 3. 分阶段开发计划

### 阶段 A：基线与接口冻结

目的：在任何核心修改前，固定现有双臂行为和单臂任务基线。

工作项：

1. 记录 `click_bell_left` 的随机 seed、成功率、episode 长度、动作维度和 HDF5 字段。
2. 明确单臂输出 schema：活动臂 qpos、活动臂 gripper、活动臂 endpose、head camera，以及单臂 action vector。
3. 保存当前 RoboTwin commit 和本项目已有的第三方修改状态。

验收标准：同一 seed 下，双臂旧模式仍能复现；单臂 schema 和动作维度形成书面约定。

### 阶段 B：Robot 实体与 planner 解耦

主要文件：`third_party/RoboTwin/envs/robot/robot.py`、`third_party/RoboTwin/script/collect_data.py`、评估入口中的 embodiment 构造逻辑。

工作项：

1. 将 embodiment 解析从“配置长度决定模式”改为“显式 `arm_mode` 决定模式”。
2. 单臂模式只加载活动臂的 URDF；不创建右侧 SAPIEN entity。
3. 只初始化活动臂的 active joints、arm joints、gripper、camera link 和 origin pose。
4. 只创建活动臂的 Curobo/MPLib planner，不启动另一侧 planner process。
5. 为 `move_to_homestate`、`set_arm_joints`、`set_gripper`、planner reset 和 close 流程增加单臂分支。

验收标准：场景 entity 数量只有一只机器人；进程列表没有第二个 planner；单臂机器人能够初始化、回 home、打开/关闭 gripper 并正常关闭环境。

### 阶段 C：Base_Task 单臂控制路径

主要文件：`third_party/RoboTwin/envs/_base_task.py`。

工作项：

1. 将 `get_obs` 的 endpose 和 qpos 输出改为按 `arm_mode` 生成。
2. 将 `take_action` 的 qpos/ee 解析改为单臂向量，不再构造右臂 path、TOPP 或 gripper path。
3. 将 `together_open_gripper`、`together_close_gripper` 在单臂模式下映射为活动臂操作。
4. 检查 `move`、`take_dense_action`、`back_to_origin`、`get_arm_pose` 等通用 API，保证单臂任务只触发活动臂路径。
5. 对非活动臂 API 在单臂模式下显式报错，避免静默忽略错误调用。

验收标准：`click_bell_left.play_once()` 只生成左臂控制轨迹；不存在右臂关节访问；qpos 和 action 维度等于左臂关节数加 gripper 维度。

### 阶段 D：Camera 与观测采集

主要文件：`third_party/RoboTwin/envs/camera/camera.py` 及 `Base_Task` 的 observation/save 逻辑。

工作项：

1. head camera 保持不变。
2. wrist camera 只绑定活动臂 camera link；若配置关闭 wrist camera，不引入额外依赖。
3. 移除单臂 HDF5 中的 `right_arm`、`right_gripper`、`right_endpose` 字段。
4. 保持 RGB、depth、segmentation 等可选观测字段语义一致。

验收标准：单臂 episode 可以被 HDF5 读取；每个开启的 camera 都有连续帧；不会访问不存在的 right camera link。

### 阶段 E：EMIX 数据、训练和评估接入

主要文件：`data_processing/convert_robotwin_to_lerobot.py`、`evaluation/robotwin_evaluate.py`、相关 shell 脚本。

工作项：

1. 转换器读取单臂 `joint_action/vector`，不再通过 `left_qpos_dim` 截取并冻结右臂。
2. LeRobot `observation.state` 和 `action` 改为单臂维度。
3. 保持 `observation.eef_pose` 与活动臂 endpose 的映射。
4. 评估器不再检查双臂 action shape，也不再把右半段动作覆盖为 home qpos。
5. 为单臂数据集更新 repo id、统计量和数据完整性检查。

验收标准：单臂数据可以完成转换、训练前读取、policy rollout；模型输出维度与 RoboTwin 单臂 action schema 完全一致。

### 阶段 F：右臂模式与回归

工作项：

1. 将 `active_arm` 切换为 `right`，复用同一套单臂核心代码。
2. 对左右活动臂分别运行初始化、专家采集、HDF5 读取、训练输入和评估回放。
3. 用原有 `arm_mode: dual` 运行一组回归测试，确认未破坏双臂模式。

验收标准：left/right 单臂模式均可运行；双臂旧模式行为无回归；单臂模式场景中不存在非活动臂实体和 planner。

## 4. 稳定性与工作量拆分

| 模块 | 预计工作量 | 稳定性判断 | 主要原因 |
|---|---:|---|---|
| embodiment 解析 | 低 | 高 | 当前分支集中在 collect/eval 入口 |
| Robot 装载与 planner | 中 | 高 | 左右字段较多，但生命周期边界清晰 |
| Base_Task 动作执行 | 中 | 中 | qpos/ee/TOPP 路径都默认双臂 |
| observation 与 HDF5 | 中 | 高 | 字段明确，适合 schema 驱动改造 |
| camera | 低 | 高 | head camera 独立，wrist camera 需单侧选择 |
| EMIX 转换与评估 | 低至中 | 高 | 当前已存在冻结右臂逻辑，容易替换 |
| 单臂任务验证 | 低 | 高 | 任务范围已限定，不处理协作任务 |
| 双臂回归 | 中 | 中 | 需要保留旧路径并测试两种模式 |

总体判断：核心代码改造为中等工作量，风险集中在 `Base_Task.take_action` 和数据 schema；不需要重写 RoboTwin 的任务库，也不需要修改 URDF 资产。

## 5. 推荐的最小验证闭环

第一轮只实现左臂 Piper，配置 `click_bell_left`，关闭 wrist camera，仅使用 head camera 和 qpos action。验证顺序如下：

1. 单臂环境初始化并打印 entity、planner、active joint 数量。
2. 专家轨迹收集 1 个 episode。
3. 读取并检查 HDF5 字段和 action shape。
4. 转换为一个最小 LeRobot dataset。
5. 使用固定 action 或短 rollout 完成 bell success 判定。
6. 再开启 wrist camera 和 endpose，确认可选观测不会改变核心动作维度。

只有该闭环稳定后，才扩展到右臂和更多单臂任务。

## 6. 主要风险与控制措施

- 非活动臂 API 被任务或通用 helper 隐式调用：单臂模式下显式抛出错误，并通过最小任务逐步补齐调用路径。
- planner close/reset 遗留第二侧进程：单臂模式只创建活动侧资源，并在初始化和关闭阶段检查进程数。
- 数据集旧 schema 与新 schema 混用：单臂数据使用独立 dataset root 和 repo id，不覆盖旧数据。
- 双臂模式回归：所有单臂修改都置于显式 `arm_mode` 分支，完成双臂 smoke test 后再合并。
- 第三方目录已有未提交修改：开发前应单独记录当前状态，避免把既有改动与单臂改造混淆。

## 7. 最终判断

在限定单臂任务的前提下，真正移除某一臂具有明确开发路径，且可以按模块逐步验收。建议先做“左臂 Piper + click_bell_left + qpos + head camera”的最小闭环；该闭环通过后，再扩展 camera、endpose、右臂模式和双臂回归。
