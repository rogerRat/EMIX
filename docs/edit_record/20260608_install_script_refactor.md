# 2026-06-08: 安装脚本重构 — 动态依赖提取与 ML 库补装

## 概述

重构 `scripts/_install.sh`，解决三个核心问题：
1. gymnasium 版本冲突的处理策略（保留 lerobot 方案，不降级）
2. `--no-deps` 导致的 mani_skill 依赖硬编码过时问题
3. 缺失 ML 库的补装

同时更新 `docs/edit_record/ENV_DEV.md` 中与依赖相关的描述。

---

## 问题背景

### 问题 1：gymnasium 版本冲突

| 包 | 对 gymnasium 的约束 |
|---|---|
| `lerobot==0.4.4` | `gymnasium>=1.1.1` |
| `mani_skill==3.0.0b22` | `gymnasium<1.0`（需要 0.29.x） |

两个约束**不存在交集**，pip resolver 无法自动解决。

**为何无法通过安装顺序绕过**：

- 先装 lerobot → gymnasium 升到 1.x（必然）
- 再装 mani_skill（不加 `--no-deps`）→ pip 扫描到 lerobot 的 `gymnasium>=1.1.1` 约束，与 mani_skill 的 `<1.0` 冲突，直接报错
- 卸载 gymnasium 再装 mani_skill → 卸载只删除包文件，lerobot 的元数据仍声明 `gymnasium>=1.1.1`，pip resolver 仍会报冲突

**最终策略**：保留 lerobot 拉入的 `gymnasium>=1.1.1`，不做降级。经实际测试，当前代码在 gymnasium 1.x 下运行正常，降级对项目无实际帮助。使用 `--no-deps` 安装 mani_skill 以绕过声明层面的冲突。

### 问题 2：`--no-deps` 的硬编码依赖过时

原脚本在 `--no-deps` 安装 mani_skill 后，手动硬编码了依赖列表：

```bash
python -m pip install sapien dacite fast_kinematics mplib==0.1.1 ...
```

一旦 mani_skill 版本更新、增删依赖，脚本无法感知，导致漏装或版本不对。

**解决方案**：用 `importlib.metadata.requires('mani_skill')` 动态提取当前安装版本声明的依赖，过滤掉 gymnasium 后交给 pip 正常解析安装。

### 问题 3：缺失 ML 库

对比 requirements.txt 与实际 emix 环境，以下 ML 库未被安装：

- `scikit-learn`、`seaborn`、`umap-learn`、`pynndescent`
- `joblib`、`threadpoolctl`
- `numba`、`llvmlite`

---

## 修改的文件

### 1. `scripts/_install.sh` — 安装脚本重构

#### 变更总览

步骤从 6 步调整为 7 步：

| 步骤 | 内容 | 变化 |
|------|------|------|
| 1/7 | 创建 conda 环境 | 不变 |
| 2/7 | 安装 PyTorch + CUDA | 不变 |
| 3/7 | 安装 lerobot | 不变 |
| 4/7 | 安装 mani_skill --no-deps + 动态依赖 | **重构** |
| 5/7 | 安装剩余 PyPI 依赖 | **增强过滤** |
| **6/7** | **补装 ML 库** | **新增** |
| 7/7 | 验证 + 导出 emix_requ.txt | **增强** |

#### Step 4 重构：从硬编码到动态提取

**之前**（硬编码）：

```bash
python -m pip install mani_skill==3.0.0b22 --no-deps

python -m pip install \
    sapien dacite fast_kinematics mplib==0.1.1 \
    nvidia-ml-py pytorch-kinematics tyro h5py ipython tabulate trimesh
```

**之后**（动态提取 + 过滤）：

```bash
# 4a: 无依赖安装
python -m pip install mani_skill==3.0.0b22 --no-deps

# 4b: 从已安装的元数据动态提取依赖，过滤掉 gymnasium
python -c "
import importlib.metadata
requires = importlib.metadata.requires('mani_skill') or []
for dep in requires:
    if 'extra ==' in dep: continue               # 跳过可选依赖
    name = dep.split(';')[0].split('>')[0]...    # 提取包名
    if name.lower() == 'gymnasium': continue     # 跳过冲突包（保留 lerobot 版本）
    print(dep)
" > .mani_skill_deps.txt

# 4c: pip 正常解析并安装
python -m pip install -r .mani_skill_deps.txt
```

**为何能解决过时问题**：

| 机制 | 效果 |
|------|------|
| `importlib.metadata.requires()` | 读取**当前安装版本**的依赖声明，版本变更后自动更新 |
| 过滤 `gymnasium` | 绕过版本冲突，保留 lerobot 拉入的版本 |
| 过滤 `extra ==` | 跳过测试/开发等可选依赖，只装必需的 |
| `pip install -r` | 让 pip 正常解析版本约束，不硬编码 |
| 已安装的依赖（numpy、torch 等） | pip 检测到已满足，直接跳过 |

#### Step 5 增强：过滤逻辑

从 requirements.txt 中排除：

- `torch`、`torchaudio`、`torchvision`、`lerobot`、`mani_skill`（已在前序步骤安装）
- cu13 系列 nvidia 包（16 个，保留 cu12 版本）
- ML 库（在 Step 6 单独安装）
- `annotated-doc`（不在 PyPI）、`cuda-toolkit`（非必要）

过滤的 cu13 包列表：

```
nvidia-cublas, nvidia-cuda-cupti, nvidia-cuda-nvrtc, nvidia-cuda-runtime,
nvidia-cudnn-cu13, nvidia-cufft, nvidia-cufile, nvidia-curand,
nvidia-cusolver, nvidia-cusparse, nvidia-cusparselt-cu13, nvidia-nccl-cu13,
nvidia-nvjitlink, nvidia-nvshmem-cu13, nvidia-nvtx, cuda-toolkit
```

#### Step 6 新增：补装 ML 库

```bash
python -m pip install scikit-learn seaborn umap-learn pynndescent joblib threadpoolctl numba llvmlite
```

不钉版本，由 pip 解析与当前环境兼容的最新版本。

#### Step 7 增强：验证 + 导出

新增验证项：`sklearn`、`seaborn`、`umap`。

新增环境导出：`pip freeze > emix_requ.txt`，锁定最终环境的所有包版本。

### 2. `docs/edit_record/ENV_DEV.md` — 更新依赖描述

更新问题 #3（gymnasium 冲突）和 #4（--no-deps 硬编码）的描述，反映动态提取方案和保留 lerobot 版本的策略。

---

## 安装顺序（最终版）

```
Step 1: conda create -n emix python=3.10
Step 2: pip install torch/torchaudio/torchvision        （CUDA 索引）
Step 3: pip install lerobot==0.4.4                       （拉入 gymnasium>=1.1.1）
Step 4: pip install mani_skill==3.0.0b22 --no-deps       （绕过冲突）
        + 动态提取 mani_skill 依赖，过滤 gymnasium，pip install
Step 5: pip install -r requirements_filtered.txt         （排除已装/cu13/ML库）
Step 6: pip install scikit-learn seaborn umap-learn ...  （补装 ML 库）
Step 7: 验证关键导入 + pip freeze > emix_requ.txt
```

---

## 环境对比差异（requirements.txt vs emix）

关键差异摘要：

| 项目 | requirements.txt | emix 环境 | 说明 |
|------|-----------------|-----------|------|
| `gymnasium` | 0.29.1 | 1.3.0 | 保留 lerobot 方案，不降级 |
| `opencv-python` | 4.13.0.92 | 4.11.0.86 | emix 版本更旧 |
| `pytorch_kinematics` | 0.7.6 | 0.10.0 | 动态安装，使用兼容最新版 |
| `huggingface_hub` | 1.11.0 | — | 包名差异：emix 中为 `huggingface-hub==0.35.3` |
| cu13 nvidia 包 | 16 个 | 无 | **已排除**：仅保留 cu12 版本 |
| ML 库 | 有 | 无 | **已补装**：Step 6 安装 |

次要版本差异（emix 环境更新）：`click`、`pillow`、`numpy`、`datasets` 等约 40 个包有小版本升级，向下兼容，不影响运行。