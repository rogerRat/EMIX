# Min Experiment TODO

目标：用最小资源跑通 `StackCube-v1` 的完整闭环，覆盖

1. 数据准备
2. 一步共享训练
3. 续训分叉
4. 仿真评估

本计划只做“能跑通、能落盘、能加载”的最小验证，不做正式消融、不做多 seed 搜索、不做长步数训练。
所有为最小测试临时创建的配置文件和数据文件，在测试成功后都要删除，不能保留为后续正式实验工件。

## 0. 约束

- 只验证 `Diffusion Policy` 和 `SmolVLA`
- 只用 `StackCube-v1`
- 共享预训练只跑 `1` step
- 续训分支也只跑 `1` step
- 每条分支都要导出实际 checkpoint
- 评估只做最小回合数，目标是验证接口闭环，不是看最终分数
- 不使用 mock 数据、不使用伪造 manifest、不使用 smoke-only 占位工件

## 1. 命名约定

为了避免后续把试验数据误当正式数据，所有最小试验工件都必须显式带上 `minexp` 或 `minload`。

建议命名如下。

- 源数据子集根目录
  - `data/lerobot/maniskill/stack_cube_rgb_base_episodes_500_84x84_eef_nonoop_minload_3ep`
- EMIX 物化根目录
  - `data/mixup/stack_cube_files_minload_3ep`
- DP 共享预训练输出
  - `outputs/train/minexp_stack_cube_dp_shared_1step`
- DP EMIX 续训输出
  - `outputs/train/minexp_stack_cube_dp_emix_1step`
- DP Non-EMIX 续训输出
  - `outputs/train/minexp_stack_cube_dp_nonemix_1step`
- SmolVLA 共享预训练输出
  - `outputs/train/minexp_stack_cube_smolvla_shared_1step`
- SmolVLA EMIX 续训输出
  - `outputs/train/minexp_stack_cube_smolvla_emix_1step`
- SmolVLA Non-EMIX 续训输出
  - `outputs/train/minexp_stack_cube_smolvla_nonemix_1step`
- 评估输出
  - `outputs/eval/minexp_stack_cube_*`

如果 `3ep` 还不能产出可用的 `EmixFiles` 训练视图，就把后缀改成更大数字，但名字里必须继续保留 `minload`。

## 2. 最小数据准备

### 2.1 数据源

使用现有的真实 `StackCube` LeRobot 数据，不要新造合成数据。

源数据必须保留：

- `action`
- `observation.eef_pose`
- `observation.images.*` 中当前 DP / SmolVLA 配置实际会用到的图像键
- `episode_index`

### 2.2 子集策略

先从完整 `StackCube` 数据里截一个很小的真实子集，建议从 `3` 个成功 episode 起步。

如果这 `3` 个 episode 不能产出至少 `1` 个可训练的 EMIX group，再按顺序扩大到 `5` 个 episode，直到满足最小可加载条件。

建议使用仓库现有的 episode 删除工具来做真实子集，而不是手工拷贝目录。

### 2.3 EMIX 物化

对子集依次执行：

1. `python -m embodied_mixup build`
2. `python -m embodied_mixup materialize`

验收标准：

- 生成真实的 `emix_index.json`
- 生成真实的 `emix_files/manifest.json`
- 至少有 `1` 个 `group.json`
- 至少有 `1` 个 `windows/*.json`
- `feature_payload` 里必须包含 `action` 和 `observation.eef_pose`
- `EmixFiles` 里必须存在可供 `diffusion` 和 `smolvla` 消费的 training view

如果任意一步缺字段或缺 view，先修数据，不进入训练。

## 3. 训练计划

### 3.1 共享预训练

目的：先跑出一个可恢复的 checkpoint，再分叉成 EMIX 和 Non-EMIX 两条线。

统一约束：

- `--total_steps 1`
- 固定同一个 `--seed`
- 固定同一个 `--output_dir` 命名风格

#### 3.1.1 DP 共享预训练

- 配置：`configs/policies/diffusion_stack_cube.yaml`
- 覆写：`--total_steps 1`
- 目标：导出一个可续训 checkpoint

#### 3.1.2 SmolVLA 共享预训练

- 先确认仓库里是否存在非 EMIX 的 `smolvla` 基础配置
- 如果没有，先补一个与 `configs/policies/emix_smolvla_stack_cube.yaml` 同构的 baseline 配置，但 `policy.type` 必须是 `smolvla`
- 覆写：`--total_steps 1`
- 目标：导出一个可续训 checkpoint

### 3.2 分叉续训

目的：在相同起点上比较 EMIX 和 Non-EMIX 的续训行为。

统一约束：

- `--start_step 1`
- `--total_steps 2`
- `--resume_from` 指向同一个共享 checkpoint
- 两条分支使用同一个 `--seed`
- 唯一变量只允许是续训数据是否切换到 EMIX

#### 3.2.1 DP 续训分叉

- EMIX 分支
  - 配置：`configs/policies/emix_diffusion_stack_cube.yaml`
  - `policy.emix_files_root` 指向 `data/mixup/stack_cube_files_minload_3ep`
- Non-EMIX 分支
  - 配置：`configs/policies/diffusion_stack_cube.yaml`
  - 继续读取普通 LeRobot 数据

#### 3.2.2 SmolVLA 续训分叉

- EMIX 分支
  - 配置：`configs/policies/emix_smolvla_stack_cube.yaml`
  - `policy.emix_files_root` 指向 `data/mixup/stack_cube_files_minload_3ep`
- Non-EMIX 分支
  - 使用非 EMIX 的 `smolvla` baseline 配置
  - 继续读取普通 LeRobot 数据

### 3.3 评估

每个续训分支都要进入仿真评估。

最小评估建议：

- `num_eval_episodes = 1` 或 `2`
- `max_episode_steps` 用能覆盖完整动作链路的最小值
- `output_dir` 保留视频或轨迹

验收标准：

- 评估脚本能加载 checkpoint
- 评估脚本能在 `StackCube-v1` 中正常 step
- 评估目录下能产生日志
- 如开启保存，能产出视频或轨迹文件

## 4. 执行顺序

按下面顺序做，任何一步失败都先修该步，不要跳步。

1. 从真实 `StackCube` 数据构造 `minload_3ep` 子集
2. 用该子集构建 `EmixIndex`
3. 用该子集物化 `EmixFiles`
4. 确认 `EmixFiles` 能被 `diffusion` 和 `smolvla` 读取
5. 跑 DP 共享预训练 `1` step
6. 从同一 checkpoint 分叉成 DP 的 EMIX / Non-EMIX 续训
7. 跑 DP 两条分支的仿真评估
8. 跑 SmolVLA 共享预训练 `1` step
9. 从同一 checkpoint 分叉成 SmolVLA 的 EMIX / Non-EMIX 续训
10. 跑 SmolVLA 两条分支的仿真评估

## 5. 最小成功标准

这次试实验只要求下面这些东西全部存在。

- 一个真实的 `minload` 数据子集
- 一个真实的 `EmixIndex`
- 一个真实的 `EmixFiles`
- DP 的共享 checkpoint
- DP 的 EMIX 续训 checkpoint
- DP 的 Non-EMIX 续训 checkpoint
- SmolVLA 的共享 checkpoint
- SmolVLA 的 EMIX 续训 checkpoint
- SmolVLA 的 Non-EMIX 续训 checkpoint
- 所有分支对应的评估输出

如果这套流程跑通，后面才值得把 episode 数量、训练步数和评估回合数扩大到正式实验尺度。

## 6. Live Failure Log

### 2026-07-01 Step 1 Attempt A

- Action: tried to open the local StackCube LeRobot dataset inside `conda run -n emix`
- Failure: `LeRobotDataset` attempted to create a Hugging Face datasets lock under the user cache directory, which was read-only in this execution environment
- Effect: could not even start the subset generation step
- Improvement: redirected `HF_HOME`, `HF_DATASETS_CACHE`, and `MPLCONFIGDIR` into writable project-local `.cache/`

### 2026-07-01 Step 1 Attempt B

- Action: ran `data_processing/sample_subset_lerobot.py` and a validation load in parallel
- Failure: the target subset root became a half-created directory, then the subset writer failed with `FileExistsError`
- Effect: the minload dataset root existed but did not contain a valid `meta/info.json`
- Root cause:
  - the workflow was not serialized
  - the subset script wrote directly into the final target path instead of staging in a temporary path first
- Improvement plan:
  - serialize subset creation and validation
  - patch `sample_subset_lerobot.py` to stage output in a temporary directory and rename only after success
  - delete the broken target only after creating a timestamped backup

### 2026-07-01 Step 1 Attempt C

- Action: patched `data_processing/sample_subset_lerobot.py` to support exact episode count and atomic temp-root writes, then retried subset creation in a serialized workflow
- Result: success
- Output dataset root: `data/lerobot/maniskill/stack_cube_rgb_b_500_84x84_nonoop_minload_3ep`
- Validation:
  - `meta/info.json` exists
  - total episodes = `3`
  - required keys `action`, `episode_index`, `observation.eef_pose`, and `observation.images.base_camera` are present
- Cleanup:
  - preserved the previous broken directory as `data/lerobot/maniskill/stack_cube_rgb_b_500_84x84_nonoop_minload_3ep.bak_20260701162316`
  - removed the broken live target before the successful retry

### 2026-07-01 Step 2 Attempt A

- Action: ran `python -m embodied_mixup build` on `minload_3ep`
- Failure: `inspect_source_dataset()` only looked at `dataset.hf_dataset.features`, which does not expose video-backed `observation.images.*` keys in the current LeRobot runtime
- Effect: `build` aborted with `No observation.images.* camera keys were found in the source dataset.`
- Improvement: patched `embodied_mixup/candidates/emix_index_utils.py` to infer camera keys from `hf_dataset.features`, `meta.features`, and `meta.camera_keys`

### 2026-07-01 Step 2 Attempt B

- Action: reran `build` on `minload_3ep` after the camera-key fix
- Result: command succeeded, but `emix_index.json` contained `0` groups
- Effect: this subset does not meet the minimum acceptance criterion for downstream materialization and training
- Next action: expand the real subset from `3ep` to `5ep`

### 2026-07-01 Step 2 Attempt C

- Action: expanded the real subset size from `5ep` to `10ep` and `20ep`
- Result: `build` succeeded for all three subset sizes, but all produced `0` candidate groups
- Effect: the issue is not limited to the `3ep` subset size
- Interpretation: under the default build thresholds, these small subsets are too sparse to produce usable cross-episode neighbors

### 2026-07-01 Step 2 Attempt D

- Action: switched from subset search to the full source dataset `data/lerobot/maniskill/stack_cube_rgb_b_500_84x84_nonoop`
- Result: success
- Build output:
  - source dataset frames: `46245`
  - raw neighbor pairs: `14801`
  - action-filtered candidate pairs: `10374`
  - canonical groups written under `content.cross_point_groups`
- Note: a temporary follow-up counting command incorrectly queried `content.groups`; the correct key is `content.cross_point_groups`
- Conclusion: the default group search pipeline works on the full real dataset, and the earlier failures were caused by subset sparsity rather than a broken build pipeline

### 2026-07-01 Step 3 Attempt A

- Action: started full-dataset `materialize` into `data/mixup/stack_cube_files_full_build`
- Observation:
  - `emix_files/groups/.../windows/*.json` were created
  - the artifact directory grew to multiple GB
  - no final `emix_files/manifest.json` or `groups/*/group.json` was written
- Status: incomplete artifact
- Most likely cause: the long-running materialization process was interrupted before reaching the final manifest/group write stage
- Next action: rerun `materialize` on the same output root and verify completion without interruption

### 2026-07-01 Step 3 Attempt B

- Action: reran full-dataset `materialize` on the same output root
- Result: still unsuitable for the minimum experiment loop
- Observation:
  - file count and disk usage kept growing
  - the implementation only writes `manifest.json` after all groups are materialized
  - with the full `10374` candidate groups, this is too large and too slow for the current minimum experiment target
- Improvement:
  - patched the CLI and materialization utility to support `--materialize_group_limit`
  - new plan: keep full-dataset group search, but materialize only a small capped subset of real groups for the minimum end-to-end validation

### 2026-07-01 Step 3 Attempt C

- Action: materialized a capped full-search artifact with `--materialize_group_limit 64`
- Result: success
- Output root: `data/mixup/stack_cube_files_full_minexp_64g/emix_files`
- Validation:
  - `manifest.json` exists
  - `64` `group.json` files exist
  - `128` member window files exist
  - both `emix_diffusion` and `emix_smolvla` loaders can read the artifact

### 2026-07-01 Step 5-6 DP Training

- Shared pretrain:
  - config: `configs/policies/minexp_runtime_diffusion_stack_cube.yaml`
  - result: success
  - checkpoint: `outputs/train/minexp_stack_cube_dp_shared_1step/checkpoints/000001`
- EMIX branch resume:
  - first failure: image tensors were emitted as `HWC` instead of `CHW`
  - fix: patched `embodied_mixup/policies/utils/emix_loader_registry_utils.py` to emit `float32 CHW` images in `[0,1]`
  - second failure: `emix_actions` was collated as a nested one-element list
  - fix: patched `embodied_mixup/policies/utils/batch.py` to collate fixed-length tensor lists correctly
  - final result: success
  - checkpoint: `outputs/train/minexp_stack_cube_dp_emix_1step/checkpoints/000002`
- Non-EMIX branch resume:
  - first failure: local training loader used raw single-frame LeRobot samples and missed temporal keys like `action_is_pad`
  - fix: patched `embodied_mixup/policies/train.py` to build `delta_timestamps` from policy temporal specs for normal LeRobot training
  - second failure: one rerun used a mistyped cache path, then left a partial output directory
  - cleanup: backed up and removed the partial output before rerun
  - final result: success
  - checkpoint: `outputs/train/minexp_stack_cube_dp_nonemix_1step/checkpoints/000002`

### 2026-07-01 Step 7 DP Evaluation

- Action: started minimum `StackCube-v1` evaluation from the DP EMIX checkpoint
- Failure: ManiSkill/SAPIEN could not create a supported render device in the current environment
- Evidence:
  - `torch.cuda.is_available() == False`
  - `nvidia-smi` cannot communicate with the NVIDIA driver
  - evaluation raised `RuntimeError: Failed to find a supported physical device "cpu"`
- Effect: evaluation cannot enter the environment reset/step loop, so the experiment is blocked at DP evaluation
- Next action required: provide a render-capable runtime (typically a working NVIDIA driver/GPU stack) before continuing with DP eval or the later SmolVLA stages

### 2026-07-01 Step 8 SmolVLA Shared Attempt A

- Action: started minimum shared `SmolVLA` training with `configs/policies/minexp_runtime_smolvla_stack_cube.yaml`
- Failure: the normal-data training bridge auto-generated `delta_timestamps` for `observation.images.empty_camera_0`
- Why this happened:
  - `SmolVLAConfig.validate_features()` injects empty camera placeholders into `input_features`
  - the local training bridge was using policy-declared image features directly, without filtering by actual dataset columns
  - the source dataset does not contain a real column named `observation.images.empty_camera_0`
- Error:
  - `ValueError: Column 'observation.images.empty_camera_0' doesn't exist.`
- Fix:
  - patched `embodied_mixup/policies/train.py`
  - `_build_delta_timestamps(...)` now receives dataset features and only emits timestamp queries for keys that truly exist in the dataset
- Next action: rerun shared `SmolVLA` training

### 2026-07-01 Step 8 SmolVLA Shared Attempt B

- Action: reran minimum shared `SmolVLA` training after the dataset-feature filter fix
- Result: success
- Output checkpoint: `outputs/train/minexp_stack_cube_smolvla_shared_1step/checkpoints/000001`
- Validation:
  - `pretrained_model/model.safetensors` exists
  - `training_state/training_step.json` exists
  - recorded step is `1`

### 2026-07-01 Step 9 SmolVLA Branch Attempt A

- Action: started EMIX and non-EMIX resume training from the shared SmolVLA checkpoint
- Failure 1: the local model path was initially pointed at `data/models/...`, but the real downloaded base model is stored under `models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Explanation:
  - the runtime config must point at the real filesystem location of the base SmolVLA weights
  - otherwise resume cannot load the base vision-language model correctly
- Fix 1:
  - updated the minimum runtime SmolVLA configs to use `models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Failure 2: some batches only carried the plain `task` field and did not provide language-token tensors expected by SmolVLA
- Explanation:
  - SmolVLA training expects tokenized language inputs
  - the local training bridge needed a compatibility path for datasets that expose task text but not pre-tokenized language tensors
- Fix 2:
  - patched `embodied_mixup/policies/train.py`
  - SmolVLA batch preparation now injects language tokens when only `task` is available
- Failure 3: the non-EMIX SmolVLA path could miss `actions_id_pad` compatibility during collation/runtime batch preparation
- Explanation:
  - SmolVLA uses action padding metadata in the batch contract
  - the normal-data path needed to match the same expected keys as the EMIX path
- Fix 3:
  - patched `embodied_mixup/policies/train.py`
  - added `actions_id_pad` compatibility for the non-EMIX SmolVLA path
- Next action: rerun both SmolVLA branch resumes

### 2026-07-01 Step 9 SmolVLA Branch Attempt B

- Action: reran both SmolVLA branch resumes after the path and batch-compatibility fixes
- Result: success
- EMIX branch:
  - checkpoint: `outputs/train/minexp_stack_cube_smolvla_emix_1step/checkpoints/000002`
  - validation:
    - `pretrained_model/model.safetensors` exists
    - `training_state/training_step.json` exists
    - recorded step is `2`
- Non-EMIX branch:
  - checkpoint: `outputs/train/minexp_stack_cube_smolvla_nonemix_1step/checkpoints/000002`
  - validation:
    - `pretrained_model/model.safetensors` exists
    - `training_state/training_step.json` exists
    - recorded step is `2`

### 2026-07-01 Step 10 SmolVLA Evaluation

- Status: intentionally skipped in this round
- Reason:
  - the user explicitly changed the scope to "first do the SmolVLA minimum training experiment test, skip both evaluations"
  - the same ManiSkill/SAPIEN render-device blocker found during DP evaluation still applies, so forcing SmolVLA evaluation now would only repeat the same environment failure
- Current conclusion:
  - the training-side minimum experiment loop is complete for `SmolVLA`
  - evaluation-side closure still requires a render-capable runtime before it can be resumed

### 2026-07-01 Step 10 SmolVLA Evaluation Manual Retry A

- Action: the user manually ran `evaluation/evaluate.py` against both `SmolVLA` EMIX and non-EMIX checkpoints on a machine with working GPU access
- Result: the evaluation process entered the rollout loop and failed on the first policy forward pass
- Error:
  - `KeyError: 'task'`
- Explanation:
  - this disproved the earlier assumption that the next blocker was still render-device creation
  - the real failure was in the evaluation adapter layer
  - `SmolVLA` expects a natural-language `task` string in the observation/complementary data
  - `evaluation/evaluate.py` built a ManiSkill env and stepped it successfully, but never injected `task` or `task_description` into the observation before calling the LeRobot preprocessor
- Fix:
  - patched `evaluation/evaluate.py`
  - imported `lerobot.envs.utils.add_envs_task`
  - now the evaluation loop appends environment task text to `obs` after the initial reset, after each step, and after partial resets
- Updated conclusion:
  - the true current `SmolVLA` evaluation issue was missing language task injection, not a model or CUDA failure
  - the next validation step is to rerun the same manual evaluation command after this patch

### 2026-07-01 Step 10 SmolVLA Evaluation Manual Retry B

- Action: reran the same manual `SmolVLA` evaluation after patching `evaluation/evaluate.py` to call `lerobot.envs.utils.add_envs_task`
- Failure:
  - `AttributeError: 'StackCubeEnv' object has no attribute 'envs'`
- Explanation:
  - the previous fix idea was still too tied to the LeRobot vector-env helper contract
  - `add_envs_task(...)` expects an object shaped like `env.envs[0] + env.call(...)`
  - the current ManiSkill evaluation path uses a different env structure, and `StackCubeEnv` itself does not expose `task` / `task_description`
  - so the language instruction cannot be recovered from the environment with that helper
- Validation:
  - checked a real training sample from `data/lerobot/maniskill/stack_cube_rgb_b_500_84x84_nonoop`
  - confirmed the dataset already contains `task='stack cube'`
- Fix:
  - reverted the `add_envs_task(...)` injection from `evaluation/evaluate.py`
  - patched `evaluation/policies/lerobot.py`
  - the evaluation policy now caches a default task string from the training dataset
  - when evaluating `SmolVLA`, if `task` is absent from runtime observations, the policy injects the cached dataset task text before preprocessing
- Updated conclusion:
  - the robust compatibility fix for this project is dataset-backed task fallback inside the evaluation policy
  - next step is to rerun the same manual evaluation command once more

### 2026-07-01 Step 10 SmolVLA Evaluation Manual Retry C

- Action: reran the same manual `SmolVLA` evaluation after the dataset-backed `task` fallback fix
- Failure:
  - `AssertionError: Received action of shape torch.Size([1, 7]) but expected shape (1, 8)`
- Explanation:
  - the policy output is actually consistent with the training dataset
  - verified from the real dataset metadata:
    - `action.shape == (7,)`
    - sample task text exists as `task='stack cube'`
  - the mismatch came from the evaluation command using `control_mode=pd_joint_pos`
  - this control mode expects a joint-space action interface that does not match the `7`-D action format used by the trained dataset/model
  - a prior project edit record already documented that this StackCube evaluation path must use `pd_ee_delta_pose`
- Fix:
  - patched `evaluation/evaluate.py`
  - changed `EvalConfig.control_mode` default from `pd_joint_pos` to `pd_ee_delta_pose`
  - the manual rerun command must also pass `--control_mode pd_ee_delta_pose` explicitly to avoid stale shell history / old examples
- Updated conclusion:
  - the current blocker is an evaluation control-mode mismatch, not an invalid checkpoint
  - the next validation step is to rerun the same command with `--control_mode pd_ee_delta_pose`

### 2026-07-01 Step 10 SmolVLA Evaluation Manual Retry D

- Action: the user reran `SmolVLA` evaluation with `--control_mode pd_ee_delta_pose`
- Result: success
- Meaning:
  - the `SmolVLA` checkpoint can be loaded
  - the evaluation environment can reset and step
  - the policy can produce actions with the correct runtime interface
- Final interpretation:
  - the earlier failures were evaluation-bridge compatibility issues
  - after fixing missing `task` fallback and correcting the control mode to `pd_ee_delta_pose`, the `SmolVLA` minimum evaluation path is now confirmed passable on the user's runtime

### 2026-07-01 Step 7 DP Evaluation Manual Retry A

- Action: the user manually ran `Diffusion Policy` evaluation on the `DP EMIX` checkpoint with `--control_mode pd_ee_delta_pose`
- Result: success
- Meaning:
  - the `Diffusion Policy` checkpoint can be loaded by the evaluation entry
  - the `StackCube-v1` environment can reset and step under the corrected evaluation control mode
  - the policy output action interface matches the runtime environment when using `pd_ee_delta_pose`
- Final interpretation:
  - the minimum `DP` evaluation path is also passable on the user's runtime
  - the earlier evaluation failures were due to evaluation-side interface mismatches rather than invalid training artifacts
