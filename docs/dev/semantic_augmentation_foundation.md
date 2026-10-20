# 需求 2/3：仿真自动标注的实施设计

日期：2026-09-29。
最后修改：2026-10-20。状态：通用模块已实现；采集、转换与 patch 训练已接入，真实仿真 ID 对齐验收待完成。
本文按用户最新要求收紧范围，替代此前关于通用语义系统与监督权重的讨论。
实验身份、参数和运行结果仍只维护在 [唯一实验协议](../EXP.md)。

## 1. 已确定的范围

针对少数已知任务，开发者逐个写固定规则即可。仿真器提供对象身份和像素归属，专家脚本提供操作顺序；无需训练阶段分类器，也无需外部分割模型或 Agent。
共同部分只负责保存、对齐和读取标注，不推断通用任务语义。
拼贴训练固定使用两条动作分支，损失为 `L = L_A + L_B`，不按面积或透明度加权，也不再除以 2。

## 2. 只补三种标注

| 标注 | 从哪里取得 | 用途 |
|---|---|---|
| 每个像素属于哪个对象 | 相机原始 Segmentation buffer | 精确取出对象可见像素 |
| 每轮操作抓谁、放到哪里 | 任务脚本显式登记 | 将像素 ID 对应到当前目标 |
| 每帧属于哪轮、哪个阶段 | 每个任务固定的开合事件规则 | topick 选择被抓物，toplace 选择放置目标 |

建议统一输出 `frame_index, operation_id, phase, target_object_id`，以及实例 ID 图和 episode 对象表。
对象表把逻辑名称（如 bread_0、basket）映射到本次仿真中的渲染 ID；资产型号不是渲染 ID。
不同 episode 的数字 ID 不可直接比较，配对按任务规则中的角色对应。

## 3. 仿真端怎么接

### 对象像素

在 `third_party/RoboTwin/envs/camera/camera.py` 增加原始实例 ID 读取接口。
现有 `get_segmentation()` 会转 uint8 并着色，只适合展示；新接口保留整数 ID，使用无损存储。
RGB 和实例图从同一次 `update_picture()` 的缓冲区读取，在 `_base_task.get_obs()` 一起返回。
普通 Actor 的实体可从 `Actor.actor` 取得；多 link 对象登记全部相关实体。
实际 ID 字段与 Segmentation 通道的对应关系要用小场景验证，不能拿资产编号代替。

### 任务规则

每个任务提供很小的规则文件或函数，列出本次 episode 的对象和操作顺序。例如：

```text
place_bread_basket:
  operation 0: manipulated=bread_0, destination=basket
  operation 1: manipulated=bread_1, destination=basket
```

顺序若由随机布局决定，直接登记专家脚本实际选择的顺序。
`place_actor()` 当前只收到目标坐标，无法可靠反推出目标物体；应在任务调用处明确登记 basket，而不是再做图像识别或最近物体猜测。

### 阶段规则

首版采用任务专属有限状态规则。以两次放面包为例：

```text
初始 → topick_0（目标 bread_0）
第一次有效抓取闭合完成 → toplace_0（目标 basket）
第一次有效释放张开完成 → topick_1（目标 bread_1）
第二次有效抓取闭合完成 → toplace_1（目标 basket）
第二次有效释放张开完成 → terminal
```

有效开合由任务规则指定，忽略初始化张开、重复指令等非抓放事件。
这里的完成指对应控制序列执行结束，是确定的标注约定，不额外推断接触成功时刻。
阶段切换记录为下一张将保存的观测帧编号；开合执行期间仍属于切换前阶段。
中途回程自然归入下一次 topick；最终回程归 terminal。特殊任务直接另写规则。

`grasp_actor()`、`place_actor()` 只构造动作列表，调用它们时不能立刻切阶段。
在 `_base_task.move()` 中对应开合的 `take_dense_action()` 执行结束后记录事件，关联 `_take_picture()` 使用的 `FRAME_IDX`。
规划和轨迹回放两条路径都要覆盖；每次 episode 初始化重置计数。
采集结束后，用事件帧号展开逐帧阶段标签，避免稀疏拍照遗漏边界。
旧数据只有夹爪数值时，可为具体任务固定阈值、变化方向和事件顺序规则，产生同一格式；无需通用识别器。

## 4. 数据怎么存、怎么对齐

原始 HDF5 新增：

```text
observation/head_camera/instance_id  [T,H,W] 紧凑整数编号
annotation/operation_id            [T] int32
annotation/phase                   [T] int8
annotation/target_object_id        [T] int32
```

episode 对象表、操作表、边界事件和规则版本存独立 JSON。逐帧字段固定，避免对象数量变化导致不规则数组。
`pkl2hdf5.py` 目前只将名称含 rgb 的字段编码为 JPEG；实例图应另加无损压缩，不能走有损图像编码。

`data_processing/convert_to_lerobot.py` 目前使用 `image[t], state[t] → action=state[t+1]`，并删除最后一张观测。
标注执行相同长度裁剪，实例图同步缩放到 84×84，使用最近邻。
转换后的标注放独立伴随目录，按 episode/frame 查询，并记录对应数据集名称、episode、相机、原始帧号、规则版本及尺寸；只做字段、形状、编号范围等基本校验，不增加哈希或内容指纹校验。保持原数据目录内容不变。
只给启用阶段拼贴的加载器配置标注路径；缺少标注时明确报错。

旧归档 RGB 没有实例图，不能凭任务阶段规则恢复精确物体像素。
新采集直接保存；旧数据只有具备足够状态并验证重放逐帧一致时才能补渲染，不能仅凭相同 seed 宣称对齐。

## 5. 训练端怎么用

增加一个轻量 `AnnotationReader`，按源 episode/frame 返回阶段、目标实例和掩码。
任务规则负责选择目标；拼贴函数只接收两张图和目标掩码，执行像素复制。
对象拼贴的最小实现为以 A 为底图，将 B 的目标可见像素按原坐标覆盖进去：

```text
mask_B = instance_id_B == target_object_id_B
image_out = where(mask_B, image_B, image_A)
```

mask 外严格保持 A 原值；重叠处由 B 覆盖。此规则只描述硬拼贴，不额外引入背景修补、重规划或物理一致性系统。
阶段模式在候选读取时要求规则允许的阶段对应：topick 配 topick，toplace 配 toplace，并使用对应目标。
阶段区间必须按每次抓放事件分开，不能把两轮同名阶段合成一个区间。
本次实施设计不擅自改变现有未来动作窗口；保存边界后，窗口限制可独立接入。

DP 保留现有两路动作与扩散训练流程，各分支按自己的有效位置计算均值，然后直接相加：

```text
L_A = mean(valid diffusion errors for branch A)
L_B = mean(valid diffusion errors for branch B)
L = L_A + L_B
```

不采样拼贴 alpha，不按像素面积配权，不通过除二抵消损失变化。原 alpha-blend 路径保持自己的定义。
标注仅用于训练数据构造，不要求策略评估时读取阶段或实例图。

## 6. 开发落点和验收

按依赖顺序实施：

1. 相机原始 ID 输出与对象登记；检查可见面包和篮子的 ID 掩码对应正确，ID 大于 255 仍无损。
2. 一个任务的固定阶段规则与事件记录；两次抓放得到四段操作阶段，初始化开合不误触发。
3. HDF5 保存和转换；检查 RGB、实例图、阶段和 qpos 时间对齐，末帧裁剪一致。
4. 标注读取、目标选择与硬拼贴；检查目标选择随阶段切换，mask 外像素完全不变。
5. DP 双分支求和；用已知误差检查求和结果，确保没有残留 alpha 或除二。
6. 为剩余选定任务补各自的对象表和规则，复用相同读写、拼贴代码。

先检查几个边界帧及拼贴样本，再做小批量训练通路验证。这些是开发验收，不是新增实验设计。
未来外部数据只需产生相同标注格式；外部分割、阶段识别或 Agent 的实现留到实际需要时处理。


## 7. 已实现的通用模块（2026-09-29）

代码：[instance_annotations.py](../../data_processing/instance_annotations.py)。
测试：[test_instance_annotations.py](../../tests/test_instance_annotations.py)。

### 紧凑存储

仿真原始 ID 可大于 255，但保存前按 episode 对象表转换为从 0 开始的逻辑编号；0 固定为 background。
一个逻辑对象可登记多个渲染 ID。未登记的 ID 报错，不悄悄当背景。
背景在内至多 256 个对象时使用单通道 uint8，更多时自动使用 uint16/uint32，不截断。
4 位最多表示 16 类，5 位最多 32 类；首版采用标准整数字节类型加 gzip 压缩，暂不增加自定义位打包格式。
HDF5 按帧分块，压缩级别 1；对象表、帧号和变换记录一并保存，写完临时文件后原子替换。
归属图不走 JPEG 或有损视频。压缩比例依内容而定，不承诺总文件大小一定小于压缩 RGB 视频。
不计算哈希、不建立内容指纹、不添加逐帧校验和；只检查必要字段、版本、维度和编号范围。压缩容器自身的标准机制不额外扩展。

### 同步接口

`transform_pair()` 一次接收 RGB 和归属图，并要求调用方提供数据集/episode/相机以及 RGB 的原始帧号。
先检查两者身份字段、帧号顺序和尺寸，再共同抽帧、裁剪、缩放、翻转，返回配套结果。
原始帧号在抽帧和重排后保留，空间变换写入 history。缩放中 RGB 使用双线性，归属图使用同坐标中心的最近邻。
接口当前接收 uint8 RGB；应在浮点归一化前使用。仅修改色值的后续处理无需修改归属图。

```python
from data_processing.instance_annotations import InstanceAnnotations, transform_pair

annotation = InstanceAnnotations.from_renderer(
    raw_ids,  # [T,H,W]，相机输出的整数实例图
    renderer_objects={0: "background", 1031: "bread_0", 8029: "basket"},
    dataset_id="bread_capture_v1", episode_id="0", camera="head_camera",
    frame_ids=source_frame_ids,
)
rgb_out, annotation_out = transform_pair(
    rgb, annotation,
    stream_key=("bread_capture_v1", "0", "head_camera"),
    frame_ids=source_frame_ids,
    indices=list(range(len(rgb) - 1)),  # 对应现有转换器删除末帧
    size=(84, 84),
)
annotation_out.save("episode_0.instances.h5")
```

示例中的渲染 ID 和背景 ID 必须替换为实际场景登记值，不能假定所有后端的背景都是 0。
每个相机、episode 分别保存；相机参数或物体位置变化后重新渲染须生成新归属图。
格式转换由调用方保留原始帧号，不依据 FPS 猜测对应关系。
只做基本校验无法识别“身份、尺寸和帧号都一样，但内容被替换”的错误，这是明确边界。
缩小造成小物体丢失及边缘颜色混合也保留为边界，当前不修补、不新增高分辨率拼贴流程。

### 验收与接入范围

11 项 CPU 测试覆盖：大渲染 ID 压缩编号、多部件合并、对象数量溢出升级、无损读写与压缩、
抽帧重排、裁剪翻转、最近邻缩放、RGB 色值变化、删除末帧、变换记录持久化，以及错误元数据拒绝。

运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_instance_annotations.py -v
```

本次只新增独立模块，没有导入到正在运行的采集、训练或评估路径，也没有修改这些路径的代码与数据。
尚未声称已验证当前渲染后端的真实实例输出；下一步在空闲资源下验证相机输出、登记实体 ID，再接通采集和转换调用。
后续阶段标签应复用同一组帧索引；本模块当前只实现像素归属，不包含任务阶段规则。


## 8. sin-v0-patch 链路接入（2026-10-20）

- RoboTwin 在启用 EMIX_INSTANCE_ANNOTATIONS=1 时从当前相机 Segmentation buffer 保存原始 actor ID，并按 episode 写入对象表；不截断到 8 位。
- 十个 central-wide 任务通过 patch_target_actor() 显式登记一个固定目标 actor。该目标角色转换为 episode-local 紧凑标签 patch_target。
- convert_to_lerobot.py --instance-annotations 校验 RGB、ID 图和对象表，按相同帧索引裁掉末帧并同步缩放，写入 annotations/episode_<n>.instances.h5。
- EMIX patch loader 根据源数据集全局 frame index 读取 B 图目标掩码；缺文件、缺角色或越界时报错。线性 EMIX 仍沿用原 alpha 混合。
- patch DP loss 对两路有效动作误差分别取均值后直接相加。CPU 单测覆盖掩码、loader、loss、标注转换基础接口和配置；真实 SAPIEN segmentation channel 与 actor ID 对齐尚未验证。
- 当前 configs/scripts/run.yaml 已登记 300 episodes/task、标注采集及 vanilla/mix/patch 三臂。环境安装完成并通过代表任务的 ID 检查后，才允许开始正式采集。
