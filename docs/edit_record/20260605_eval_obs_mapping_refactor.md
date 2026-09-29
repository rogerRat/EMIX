# 2026-06-05: Evaluate Obs Mapping Refactor & Eval Command Research

## Modified Files

### evaluation/policies/constants.py (NEW)
- Extracted `_DEFAULT_OBS_MAPPING` from `lerobot.py` into this dedicated module
- Defined three observation mapping presets:
  - `OBS_MAPPING_BASE_AND_HAND`: base camera + hand camera + proprio (for "bh" datasets)
  - `OBS_MAPPING_BASE_ONLY`: base camera + proprio (for "nohandcamera" datasets)
  - `OBS_MAPPING_HAND_ONLY`: hand camera + proprio (for "nobasecamera" datasets)
- `_DEFAULT_OBS_MAPPING` is assigned to one of the above presets (currently `OBS_MAPPING_BASE_ONLY`)
- To switch presets, only need to change the assignment line in `constants.py`

### evaluation/policies/lerobot.py
- Removed inline `_DEFAULT_OBS_MAPPING` definition
- Added `from .constants import _DEFAULT_OBS_MAPPING`
- `LeRobotPolicy.__init__` default parameter remains `obs_rules: list = _DEFAULT_OBS_MAPPING`

## Research Findings

### Why evaluate needs dataset directory

Evaluate runs in ManiSkill simulation environment, NOT on training data. However, loading the dataset is required for:

1. **Normalization statistics** (`meta/stats.json`): `state_min`, `state_max`, `action_min`, `action_max` — used by `preprocessor` (input normalization) and `postprocessor` (output denormalization)
2. **Dataset metadata** (`ds_meta`): required by `make_policy()` to construct the model with correct input/output feature shapes
3. **`pretrained_model/` directory** alone does NOT contain these normalization stats — they come from the dataset that trained the model

### Why both `repo_id` and `dataset_root` are needed

- `repo_id`: LeRobot dataset identifier. If `root` is `None`, LeRobot treats it as a HuggingFace Hub dataset name and downloads to `~/.cache/huggingface/lerobot/{repo_id}`. With `root` specified, `repo_id` serves as the local dataset directory name for locating metadata under `root/meta/`
- `root`: Local path to the dataset. Without it, LeRobot attempts remote download which fails for local-only datasets
- Both must match the dataset used during training to ensure consistent normalization

### Current trained model details

| Item | Value |
|---|---|
| Checkpoint | `outputs/train/2026-06-04/16-13-39_diffusion/checkpoints/030000/pretrained_model/` |
| Dataset repo_id | `stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop` |
| Dataset root | `data/lerobot/maniskill/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop` |
| Control mode | `pd_ee_delta_pose` (action dim=7) |
| Observation | `observation.images.base_camera` (84x84) + `observation.state` (dim=9), NO hand camera |
| Obs mapping preset | `OBS_MAPPING_BASE_ONLY` |

### Eval command

```bash
python evaluation/evaluate.py \
  --policy lerobot \
  --pretrained_path outputs/train/2026-06-04/16-13-39_diffusion/checkpoints/030000/pretrained_model/ \
  --repo_id stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --dataset_root data/lerobot/maniskill/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop \
  --control_mode pd_ee_delta_pose
```

Key notes:
- `--control_mode pd_ee_delta_pose` is MANDATORY — `EvalConfig` defaults to `pd_joint_pos` which has action dim=9, mismatching the trained model's action dim=7
- `_DEFAULT_OBS_MAPPING` must match the model's training input structure (currently `OBS_MAPPING_BASE_ONLY`, no hand camera)