# ENV_DEV - 环境搭建与排障文档

> 记录 Embodied-Mixup 环境安装过程中的依赖冲突、根因分析及解决方案。
> 脚本详见 `_install.sh`，本文档包含其全部设计说明与注释。

---

## 快速参考

| 项目 | 值 |
|------|-----|
| Conda 环境名 | `emix` |
| Python 版本 | 3.10 |
| PyTorch | 2.10.0+cu128 |
| CUDA 标签 | cu128 (CUDA 12.8) |
| 安装脚本 | `_install.sh`（推荐 `source` 执行以保持环境激活） |

---

## 脚本使用方式

```bash
source _install.sh    # 推荐：source 执行，脚本结束后 conda 环境仍然激活
bash _install.sh      # 也可用，但脚本结束后环境不会保留在当前 shell
```

---

## 问题 - 根因 - 解决方案 一览表

| # | 问题 | 根因 | 解决方案 |
|---|------|------|----------|
| 1 | `pip install -r requirements.txt` 失败：`No matching distribution found for torch==2.10.0+cu128` | PyTorch CUDA 轮子（`+cu128` 后缀）**不在 PyPI 上发布**，而是托管在 PyTorch 自身的包索引 `https://download.pytorch.org/whl/cu128`。PyPI 只有不含 CUDA 后缀的 CPU 版本。 | 先用 `--index-url` 指向 CUDA 轮子索引安装 PyTorch，再从 PyPI 安装其余依赖。见 `_install.sh` Step 2。 |
| 2 | `pip install torch` 即使提供了 `--extra-index-url`，仍拉取 CPU 版本 | `--extra-index-url` 让 pip **同时**考虑 PyPI 和额外索引。pip 可能依据 PyPI 上的 CPU 版 `torch` 解析依赖，静默选择错误的轮子。 | 安装 PyTorch 时使用 `--index-url`（非 `--extra-index-url`）。`--index-url` 限制 pip **仅**访问指定索引，确保选中 CUDA 轮子。 |
| 3 | `gymnasium` 版本冲突：lerobot 要求 `>=1.1.1`，mani_skill 要求 `==0.29.1` | 两项约束**根本不兼容**——不存在同时满足两者的版本。卸载 gymnasium 也无法绕过：lerobot 元数据仍声明 `>=1.1.1`，pip resolver 扫描全量依赖图仍会报错。 | 使用 `--no-deps` 安装 `mani_skill` 以绕过冲突，动态提取其依赖（过滤 gymnasium）后安装。**保留 lerobot 拉入的 `gymnasium>=1.1.1`**，不做降级——经实测当前代码在 1.x 下运行正常，降级无实际帮助。见 `_install.sh` Step 3-4。 |
| 4 | `mani_skill --no-deps` 后依赖硬编码容易过时 | `--no-deps` 跳过全部依赖解析，手动列举依赖在版本更新后会漏装或版本错误。 | 改用 `importlib.metadata.requires('mani_skill')` 动态提取当前版本声明的依赖，过滤掉 `gymnasium` 和可选依赖后交由 pip 正常解析安装。见 `_install.sh` Step 4b-4c。 |
| 5 | `mplib` 版本分歧：mani_skill 要求 `==0.1.1`，但最新版为 `0.2.1` | mani_skill 硬性指定 `mplib==0.1.1`（旧版本），`0.2.1` 有 API 变更。 | 显式钉住 `mplib==0.1.1` 以匹配 mani_skill 的预期版本。见 Step 4 手动依赖安装。 |
| 6 | `pytorch_kinematics` 版本分歧：mani_skill 要求 `==0.7.6`，最新版为 `0.10.0` | mani_skill 指定 `pytorch_kinematics==0.7.6`，较新版本可能有破坏性 API 变更。 | 安装 `pytorch-kinematics` 时不钉版本（解析到最新兼容版本）。安装脚本故意省略 `==0.7.6` 钉住，以使用更新的运行时兼容版本。 |
| 7 | bash 中 `set -u` 在 `conda activate` 时导致 `unbound variable` 崩溃 | conda 内部的 `conda.sh` 引用了 `ZSH_VERSION` 等 bash 中不存在的 shell 特定变量。启用 `set -u` 后，这些引用触发 `unbound variable` 错误。 | 使用 `set -eo pipefail`（省略 `-u`）。脚本中的所有变量均已显式初始化，省略 `-u` 是安全的。 |
| 8 | 裸 `pip` 命令将包装到错误 Python 解释器的 `site-packages` | 若存在多个 Python 环境，裸 `pip` 可能解析到与运行代码不同的 Python，导致"安装成功但 ImportError"。 | 始终使用 `python -m pip` 代替裸 `pip`。这保证包被安装到**当前激活**解释器的 `site-packages`，避免回退到用户级 `~/.local/lib/pythonX.Y/site-packages`。 |
| 9 | `pip --exclude` 标志在旧版 pip 中不可用 | `--exclude` 标志于 pip 24.2 引入。conda 创建的环境可能自带更旧的 pip。 | 使用 `grep -v` 从 `requirements.txt` 过滤掉 PyTorch/lerobot/mani_skill 行后再 `pip install -r`。此方式全版本兼容。 |
| 10 | `opencv-python` 与 `opencv-python-headless` 同时出现在 `requirements.txt` 中可能冲突 | 两个包均提供 `cv2` 模块，同时安装会互相干扰。在服务器/Docker 环境下 `opencv-python-headless` 更优。 | 两者以重叠版本范围列出，pip 通常解析为安装其中一个；无头环境下 `opencv-python-headless` 优先。若需 GUI 支持，仅保留 `opencv-python`。 |
| 11 | `huggingface_hub` 必须钉住在 `<0.36.0` | `lerobot==0.4.4` 要求 `huggingface-hub>=0.34.2,<0.36.0`。更新版本可能破坏 lerobot 的 API 用法。 | 在 requirements 中显式约束 `huggingface_hub>=0.34.2,<0.36.0`。 |
| 12 | `wandb` 须严格钉住在 `<0.25.0` | `lerobot==0.4.4` 或其传递依赖要求特定的 `wandb` 范围。`0.25.0+` 可能引入破坏性 API 变更。 | 在 requirements 中约束 `wandb>=0.24.0,<0.25.0`。 |
| 13 | `numpy` 须保持在 `2.0.0` 以下 | 多个依赖（numba、scipy、opencv）尚不兼容 `numpy>=2.0.0`。使用 numpy 2.x 会导致运行时段错误或 `ImportError`。 | 在 requirements 中约束 `numpy>=1.26.0,<2.0.0`。 |
| 14 | 使用 CUDA 索引时 PyTorch 版本必须钉住 `major.minor.patch` | 不显式钉住版本时，pip 会从 CUDA 索引安装**最新**可用版本（如 2.11.0），其 API 或依赖可能与其它包不兼容。 | 显式钉住 `torch==2.10.0`、`torchaudio==2.10.0`、`torchvision==0.25.0`。见 `_install.sh` Step 2。 |
| 15 | `tyro` 同时出现在动态依赖（Step 4）与过滤后的 requirements（Step 6）中 | `tyro` 由 mani_skill 声明依赖，同时也在 requirements.txt 中。重复安装无害但冗余。 | pip 跳过已满足的包，安全但冗余。 |

---

## 安装顺序详解

安装顺序至关重要，偏离此顺序将破坏依赖解析：

```
Step 1: conda create -n emix python=3.10
Step 2: pip install torch/torchaudio/torchvision  （从 CUDA 索引安装）
Step 3: pip install lerobot==0.4.4                  （拉入 gymnasium>=1.1.1，保留此版本）
Step 4: pip install mani_skill==3.0.0b22 --no-deps  （绕过 gymnasium 冲突）
        + 动态提取 mani_skill 依赖（importlib.metadata），过滤 gymnasium 后 pip install
Step 5: pip install -r requirements_filtered.txt   （排除已装/cu13/ML库）
Step 6: pip install scikit-learn seaborn umap-learn ...  （补装 ML 库）
Step 7: 验证关键导入 + pip freeze > emix_requ.txt
```

### 各步骤设计说明

#### Step 1 — Conda 环境创建

脚本搜索以下位置加载 `conda.sh`：

1. `$HOME/miniconda3/etc/profile.d/conda.sh`
2. `$HOME/anaconda3/etc/profile.d/conda.sh`
3. `/opt/conda/etc/profile.d/conda.sh`
4. `$HOME/miniforge3/etc/profile.d/conda.sh`
5. 回退：`eval "$(conda shell.bash hook 2>/dev/null || true)"`

若环境已存在则先移除（幂等设计）。

#### Step 2 — PyTorch CUDA 安装

`requirements.txt` 中列出了带 CUDA 后缀的版本：

```
torch==2.10.0+cu128
torchaudio==2.10.0+cu128
torchvision==0.25.0+cu128
```

`+cu128` 后缀表示 CUDA 12.8 构建版本，这些轮子**不在 PyPI 上发布**，仅托管在 PyTorch 的包索引 `https://download.pytorch.org/whl/cu128`。

若直接 `pip install -r requirements.txt`，pip 会：
1. 在 PyPI 搜索 `torch==2.10.0+cu128`
2. 找不到匹配（PyPI 仅有不含 CUDA 后缀的 `torch==2.10.0`）
3. 失败并报 `No matching distribution found`

解决方案：先以 `--index-url` 指向 CUDA 轮子索引安装 PyTorch，再从 PyPI 安装其余依赖并过滤掉已安装的三个 PyTorch 包。

使用 `--index-url`（非 `--extra-index-url`）是因为：`--index-url` 让 pip **仅**访问指定索引，防止其从 PyPI 取到 CPU 版 `torch` 并据此解析依赖。

版本号必须显式钉住 `major.minor.patch`（如 `torch==2.10.0`），否则 pip 会安装 CUDA 索引上的最新版本（可能为 2.11.0），导致与其它包不兼容。版本对应关系为 `torch X.Y.Z+cuNN <-> torchaudio X.Y.Z+cuNN <-> torchvision`（版本号方案不同），详见 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

#### Step 3 — 安装 lerobot

lerobot==0.4.4 要求 `gymnasium>=1.1.1`，而 mani_skill 要求 `gymnasium==0.29.1`，二者根本不兼容。先安装 lerobot 确保 `gymnasium>=1.1.1` 进入环境。

#### Step 4 — 安装 mani_skill（--no-deps）+ 动态依赖提取

使用 `--no-deps` 阻止 mani_skill 触发 gymnasium 版本冲突。然后通过 `importlib.metadata.requires('mani_skill')` 动态提取当前版本声明的全部依赖，过滤掉 `gymnasium`（保留 lerobot 拉入的版本）和可选依赖（`extra ==`），交由 pip 正常解析安装。

此方案取代之前的硬编码依赖列表，确保 mani_skill 版本更新后依赖自动跟随，不会因过时而漏装。

过滤逻辑（Python 内联）：

```python
requires = importlib.metadata.requires('mani_skill') or []
for dep in requires:
    if 'extra ==' in dep: continue               # 跳过可选依赖
    name = dep.split(';')[0].split('>')[0]...    # 提取包名
    if name.lower() == 'gymnasium': continue     # 跳过冲突包（保留 lerobot 版本）
    # 输出到临时文件，pip install -r 安装
```

已安装的依赖（如 numpy、torch 等 lerobot 已拉入的）由 pip 检测并跳过，不会重复安装。

**gymnasium 版本策略**：保留 lerobot 拉入的 `gymnasium>=1.1.1`，不做降级。经实测当前代码在 gymnasium 1.x 下运行正常。mani_skill 的依赖声明要求 `<1.0`，但 `--no-deps` 安装后该约束不生效，运行时兼容。

#### Step 5 — 安装其余依赖

从 `requirements.txt` 过滤掉以下内容后安装：

- `torch`、`torchaudio`、`torchvision`（Step 2 已装）
- `lerobot`、`mani_skill`（Step 3-4 已特殊安装）
- cu13 系列 nvidia 包（16 个，保留 cu12 版本）
- ML 库（Step 6 单独安装）
- `annotated-doc`（不在 PyPI）、`cuda-toolkit`（非必要）

使用 `grep -v` 而非 `pip --exclude` 的原因：`--exclude` 标志于 pip 24.2 引入，conda 创建的环境可能自带更旧的 pip，`grep -v` 全版本兼容。

安装完成后删除临时文件。

#### Step 6 — 补装 ML 库

安装 requirements.txt 中缺失但项目可能使用的 ML 库：

```
scikit-learn, seaborn, umap-learn, pynndescent,
joblib, threadpoolctl, numba, llvmlite
```

不钉版本，由 pip 解析与当前环境兼容的最新版本。

#### Step 7 — 验证与导出

脚本验证以下关键导入：`mani_skill`、`lerobot`、`torch`、`torchvision`、`sapien`、`gymnasium`、`h5py`、`cv2`、`numpy`、`diffusers`、`accelerate`、`datasets`、`sklearn`、`seaborn`、`umap`。

验证后执行 `pip freeze > emix_requ.txt`，将最终环境的所有包版本锁定导出。

---

## CUDA 索引 URL 参考

根据驱动版本在 `_install.sh` 中修改 `TORCH_CUDA_TAG`：

| 标签 | CUDA 版本 | 说明 |
|------|-----------|------|
| `cu128` | CUDA 12.8 | 最新 |
| `cu126` | CUDA 12.6 | cu128 未发布时的回退 |
| `cu124` | CUDA 12.4 | 通用性最广 |
| `cu121` | CUDA 12.1 | 旧驱动 |

URL 格式：`https://download.pytorch.org/whl/<TAG>`

---

## `set -eo pipefail` 说明

脚本使用 `set -eo pipefail`（`-e` 遇错退出 + `-o pipefail` 管道错误传播），但**故意省略 `-u`**（将未设置变量视为错误）。

原因：conda 内部的 `conda.sh` 引用了 `ZSH_VERSION` 等 bash 中不存在的 shell 特定变量。启用 `-u` 后这些引用触发 `unbound variable` 错误。脚本中所有变量均已显式初始化，省略 `-u` 是安全的。

---

## `python -m pip` 说明

使用 `python -m pip` 而非裸 `pip`，保证包被安装到**当前激活** Python 解释器的 `site-packages`，而非用户级 `~/.local/lib/pythonX.Y/site-packages`。这避免了常见的"已安装但 ImportError"问题——裸 `pip` 可能解析到与运行代码不同的 Python。

---

## 已知限制

1. **gymnasium 版本声明冲突（运行时兼容）**：lerobot 要求 `gymnasium>=1.1.1`，mani_skill 要求 `<1.0`，二者声明不兼容。实际保留 lerobot 拉入的 `gymnasium>=1.1.1`，不做降级——经实测当前代码在 1.x 下运行正常。mani_skill 使用 `--no-deps` 安装以绕过声明冲突。

2. **opencv 双包问题**：`requirements.txt` 中同时包含 `opencv-python` 和 `opencv-python-headless` 可能导致导入歧义。在无头/服务器环境下，建议移除 `opencv-python` 仅保留 `opencv-python-headless`。

3. **mplib 钉住**：`mplib==0.1.1` 钉住以匹配 mani_skill 预期。若 `mplib` 从 PyPI 下架，可能需要替代源或 fork。

4. **tyro 冗余**：Step 4 动态提取的 mani_skill 依赖中可能包含 `tyro`，同时它也在过滤后的 requirements（Step 6）中。无害但冗余。