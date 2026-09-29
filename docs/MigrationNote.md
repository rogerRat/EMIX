# RoboTwin migration execution log

Date: 2026-07-15 (UTC+8)

Format: action -> issue -> modification -> result.

## 16:23 - Start collection

Action: Started the configured 200-episode `click_bell_left` collection.

Issue: `scripts/robotwin_collect.sh` was not executable and returned exit code
126 (`Permission denied`).

Modification: Kept file permissions unchanged and invoked the same script with
`bash scripts/robotwin_collect.sh`.

Result: RoboTwin collection initialization started.

## 16:23 - Restore object assets

Action: Ran collection after the permission workaround.

Issue: RoboTwin could not find `assets/objects/objaverse/list.json`; only
`third_party/RoboTwin/assets/objects.zip` was present.

Modification: Verified the archive content and extracted `objects.zip` into
`third_party/RoboTwin/assets/`. The target `objects/` directory was absent, so
no existing object files were overwritten.

Result: The required Objaverse list became available. Available disk space was
checked before extraction (about 538 GB free).

## 16:24 - Generate Curobo runtime YAML files

Action: Restarted collection after restoring object assets.

Issue: Piper required `assets/embodiments/piper/curobo.yml`, but only
`curobo_tmp.yml` existed. The native collector then retried with an invalid
robot instance and repeatedly reported that `left_planner` was absent.

Modification: Stopped the invalid retry loop and ran RoboTwin's bundled
`script/update_embodiment_config_path.py`. It generated six `curobo*.yml`
files from templates and replaced `ASSETS_PATH` with the absolute RoboTwin path.

Result: Piper Curobo configuration was valid and planning could start.

## 16:27-16:32 - Collect native demonstrations

Action: Ran the full collection protocol with repaired assets and Curobo YAML.

Issue: The optional language-instruction generator failed after collection
because `description/task_instruction/click_bell_left.json` does not exist.

Modification: No source change. The active configuration has `language_num: 0`
and the policy protocol has no language input.

Result: Collection completed successfully: 200 HDF5 episodes were created in
`third_party/RoboTwin/data/click_bell_left/emix_click_bell_left/data/`.
Planning accepted 200 seeds and rejected 3. HDF5 validation confirmed JPEG
head-camera frames, 14D joint vectors, and 7D left EEF poses.

## 16:33 - Fix converter import

Action: Ran `scripts/robotwin_convert.sh`.

Issue: Executing the converter as a file path made
`from data_processing.fix_stats` fail with `ModuleNotFoundError`.

Modification: Backed up the converter as
`data_processing/convert_robotwin_to_lerobot.py.bak_20260715163327`. Changed
the import to `from fix_stats import fix_lerobot_image_stats`.

Result: The converter import smoke test passed.

## 16:34 - Convert to LeRobot

Action: Re-ran `scripts/robotwin_convert.sh`.

Issue: Dependency DEBUG output was excessively verbose during video encoding,
but there was no functional conversion error.

Modification: No functional code change for the DEBUG output.

Result: Conversion completed: 200 episodes and 10,861 frames at 17 FPS were
written to `data/lerobot/robotwin/robotwin_click_bell_left_piper_200_head_84/`.
Image statistics were recomputed. The stats and episode metadata writers made
timestamped backups before updating metadata.

## 16:35 - Fix EMIX build argument

Action: Ran the build command from `scripts/robotwin_train.sh`.

Issue: The script used `--repo-id`; the current CLI accepts `--repo_id`.

Modification: Backed up the script as
`scripts/robotwin_train.sh.bak_20260715163512` and changed the option to
`--repo_id`.

Result: EMIX index build succeeded: 1,369 candidate groups, 47 clusters, and
162 noise pairs. The index was written to
`data/emix_index/robotwin_click_bell_left_piper_200_head_84/emix_index.json`.

## 16:36 - Fix 17 FPS training timestamps

Action: Started configured baseline training.

Issue: LeRobot rejected temporal offsets because the dataset is 17 FPS while
`_build_delta_timestamps` was hardcoded to 30 FPS.

Modification: Backed up `embodied_mixup/policies/train.py` as
`embodied_mixup/policies/train.py.bak_20260715163656`. Added a positive
`dataset_fps` parameter and passed `float(ds_meta.info["fps"])` from dataset
metadata.

Result: The 17 FPS temporal offsets passed LeRobot validation and the file
passed an AST parse check.

## 16:37-16:43 - Validate training and create a checkpoint

Action: Restarted baseline training.

Issue: A failed run had left a non-empty output directory; the trainer refuses
to reuse non-empty output directories. The full protocol also saves only every
5,000 steps, which is unsuitable for smoke testing.

Modification: Preserved the stale directory by renaming it to
`outputs/train/robotwin-click-bell-left-dp-baseline-30k.bak_20260715163741`.
The long run reached about 1,373 stable steps and was intentionally stopped.
Then ran a separate CLI smoke train with `--total_steps 10` and
`--output_dir outputs/train/robotwin-click-bell-left-dp-baseline-smoke-10`.

Result: Training data loading, CUDA, forward/backward execution, and loss
reduction were verified. The 10-step run saved a checkpoint at
`outputs/train/robotwin-click-bell-left-dp-baseline-smoke-10/checkpoints/000010/pretrained_model/`.

## 16:44-16:46 - Repair the isolated evaluation runtime

Action: Started evaluation in the dedicated `emix-twin` environment.

Issue: `emix-twin` lacked the editable project package, LeRobot, and LeRobot
runtime modules including `datasets`, `draccus`, and `deepdiff`.

Modification: Installed the editable project package and `lerobot==0.4.4`
without dependencies. Installed only actual missing evaluation modules with
`--no-deps`: `datasets`, `dill`, `multiprocess`, `diffusers`, `accelerate`,
`einops`, `huggingface-hub`, `draccus`, and `deepdiff`, plus their direct
pure-Python dependencies.

Issue: An attempted automatic LeRobot dependency resolver incorrectly parsed
the Torch constraint and began replacing the RoboTwin-pinned CUDA stack.

Modification: Interrupted the resolver immediately. Restored the pinned runtime
with `torch==2.4.1`, `torchvision==0.19.1`, `numpy==1.26.4`, and
`pillow==11.3.0` in `emix-twin`.

Result: The evaluation environment passed joint imports with Torch 2.4.1+cu121,
torchvision 0.19.1+cu121, NumPy 1.26.4, SAPIEN 3.0.0b1, and MPLib 0.2.1. The
`emix` training environment was not changed.

## 16:46 - Fix evaluator relative input paths

Action: Ran the one-episode evaluation with the 10-step checkpoint.

Issue: The evaluator changed directory to RoboTwin before resolving relative
checkpoint and dataset paths, so it searched under `third_party/RoboTwin/`.

Modification: Backed up the evaluator as
`evaluation/robotwin_evaluate.py.bak_20260715164629`. Resolved checkpoint and
dataset paths to absolute paths before `os.chdir(robotwin_root)`.

Result: The evaluator loaded the checkpoint and LeRobot dataset.

## 16:47 - Fix evaluator step-limit initialization

Action: Continued the one-episode evaluation.

Issue: The evaluator did not pass `eval_mode=True`, leaving
`task.step_lim=None` and causing an integer-versus-None comparison failure.

Modification: Backed up the evaluator as
`evaluation/robotwin_evaluate.py.bak_20260715164750` and added
`config["eval_mode"] = True` in `_build_task`.

Result: RoboTwin loaded the task limit (400 steps) and the policy action loop
ran.

## 16:48 - Complete initial train-to-evaluation smoke test

Action: Ran one valid RoboTwin seed with the 10-step checkpoint and Piper
separation 0.60 m.

Issue: Informational warnings about optional `pytorch3d`, display input, and
Curobo USD support were emitted but did not block execution.

Modification: No source change was required.

Result: The evaluator exited with code 0. It loaded the checkpoint, constructed
the RoboTwin scene, mapped observations, selected actions, executed the action
loop, and wrote metrics. The smoke result was 1/1 success for seed 100000. This
is not an experimental metric because the checkpoint had only 10 training steps.

## 16:49 - Identify evaluation output-directory issue

Action: Located the successful smoke-test metrics.

Issue: `args.output_dir` is still resolved after `os.chdir(robotwin_root)`, so
the metrics were written under `third_party/RoboTwin/outputs/eval/` instead of
project-level `outputs/eval/`.

Modification: At this time no fix had been applied yet; the following
time-ordered entry records the fix and its validation.

Result: The successful current metrics file is
`third_party/RoboTwin/outputs/eval/click_bell_left_pretrained_model_20260715164815/metrics.json`.

## 16:53 - Fix evaluation output directory resolution (in progress)

Action: Started a second one-episode smoke evaluation to validate the result
location.

Issue: The evaluator resolved `args.output_dir` after changing its working
directory to RoboTwin, placing output under the submodule.

Modification: Backed up the evaluator as
`evaluation/robotwin_evaluate.py.bak_20260715165330`. Resolved `args.output_dir`
to an absolute path before `os.chdir(robotwin_root)` and used that resolved path
to create the evaluation output directory.

Result: The validation exited with code 0 and wrote
`outputs/eval/click_bell_left_pretrained_model_20260715165410/metrics.json`.
The output contains one evaluated seed (100000) and a smoke-only 1/1 result.
The location is now the project-level `outputs/eval/` directory, so the output
path issue is resolved.

## 17:04 - Train EMIX smoke checkpoint

Action: Started the EMIX Diffusion Policy using the existing RoboTwin EMIX
index and the same smoke-test budget used for the baseline.

Issue: No new training error occurred. The full 30,000-step protocol is not
required for this validation task, and its 5,000-step checkpoint interval is
too large for a smoke test.

Modification: Kept the committed EMIX YAML unchanged and invoked the trainer
with `--total_steps 10` and the isolated output directory
`outputs/train/robotwin-click-bell-left-dp-emix30-smoke-10`.

Result: The `emix_diffusion` mixed-training loop completed 10 CUDA steps with
exit code 0 and saved a valid pretrained checkpoint at
`outputs/train/robotwin-click-bell-left-dp-emix30-smoke-10/checkpoints/000010/pretrained_model/`.
The checkpoint was then submitted to one-episode RoboTwin evaluation.

## 17:04 - Evaluate EMIX smoke checkpoint

Action: Evaluated the EMIX 10-step checkpoint with one valid RoboTwin seed,
Piper separation 0.60 m, and the same dataset/observation contract as the
baseline smoke test.

Issue: No new blocking issue occurred. Optional `pytorch3d`, display-input, and
Curobo USD warnings remained non-blocking.

Modification: No source change was required for the EMIX evaluation.

Result: The evaluator exited with code 0 and wrote
`outputs/eval/click_bell_left_pretrained_model_20260715170459/metrics.json`.
The smoke run recorded 1/1 success for seed 100000. This is only a functional
pipeline result, not a comparison metric, because both policies used 10-step
checkpoints and one evaluation seed.
