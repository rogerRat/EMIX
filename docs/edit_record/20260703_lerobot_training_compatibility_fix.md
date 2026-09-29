# LeRobot Training Compatibility Fix Record

## Scope

This record summarizes the code changes made on 2026-07-03 to make
`python -m embodied_mixup train --config configs/policies/ours/DP_stack-cube.yaml`
compatible with the current `lerobot` version installed in the `emix`
environment.

The fixes were limited to the current training failures. No new dependency,
framework, or extra abstraction layer was introduced.

## User-Facing Failures

### Failure 1

Training crashed before the first batch with:

`ValueError: Source LeRobotDataset does not expose episode_data_index`

### Failure 2

After the first fix, mixed training crashed when the first EMIX batch was used:

`KeyError: 'observation.images'`

### Failure 3

After the second fix, EMIX forward crashed with a device mismatch:

`RuntimeError: Expected all tensors to be on the same device`

## Root Causes

### Root Cause 1: Old LeRobot Episode Boundary Assumption

The EMIX reader assumed that `LeRobotDataset` exposes
`dataset.episode_data_index["from"]` and `dataset.episode_data_index["to"]`.

The current `lerobot` version in the `emix` environment does not expose that
field. Instead, episode boundaries are available through
`dataset.meta.episodes`, with `dataset_from_index` and `dataset_to_index`.

### Root Cause 2: EMIX Window Loading Bypassed `LeRobotDataset.__getitem__`

The EMIX reader used `hf_dataset.select(...)` to load a frame window directly.

In the current `lerobot` version, image observations such as
`observation.images.base_camera` are injected by `LeRobotDataset.__getitem__`.
They are not present as direct columns in `hf_dataset`.

As a result, non-EMIX batches contained image tensors, while EMIX batches did
not, and diffusion training failed when it tried to encode images.

### Root Cause 3: Device Transfer Only Covered Top-Level Tensors

The training loop only moved top-level tensor fields to the target device.

EMIX batches include nested tensor containers such as:

- `emix_actions: list[Tensor]`

Those nested tensors remained on CPU while the rest of the batch moved to CUDA.

## Code Changes

### Change 1: Add Multi-Path Episode Boundary Resolution

File:

- `embodied_mixup/policies/utils/emix_loader_registry_utils.py`

Changes:

- Replaced the direct dependency on `dataset.episode_data_index`.
- Added `_resolve_episode_bounds()`.
- Added `_resolve_episode_bounds_from_meta()`.
- Added `_resolve_episode_bounds_from_hf_dataset()`.

Resolution order:

1. `dataset.episode_data_index`
2. `dataset.meta.episodes[{dataset_from_index,dataset_to_index}]`
3. `hf_dataset["episode_index"]` scan fallback

This keeps compatibility with both the old and current `lerobot` layouts.

### Change 2: Load EMIX Frame Windows Through `dataset[i]`

File:

- `embodied_mixup/policies/utils/emix_loader_registry_utils.py`

Changes:

- Removed the direct `hf_dataset.select(...)` path inside `_load_frame_range`.
- Unified frame loading through:

  `frames = [self.dataset[i] for i in range(start, end)]`

- Added an explicit empty-range guard.

This ensures EMIX batches include all fields that the current
`LeRobotDataset.__getitem__` contract provides, including image observations and
task strings.

### Change 3: Add Recursive Batch-to-Device Transfer

File:

- `embodied_mixup/policies/train.py`

Changes:

- Added `_move_batch_to_device(value, device)`.
- Replaced the three repeated top-level-only `.to(device)` loops in:
  - mixed training
  - pure EMIX training
  - default training

The new helper recursively moves tensors inside:

- `dict`
- `list`
- `tuple`

This specifically fixes `emix_actions` remaining on CPU.

## Validation Performed

### Validation 1: Syntax Check

Commands run:

- `python -m compileall embodied_mixup/policies/utils/emix_loader_registry_utils.py`
- `python -m compileall embodied_mixup/policies/train.py`

### Validation 2: EMIX Dataset Sample Check

Checked that an EMIX sample now contains:

- `observation.state`
- `observation.images.base_camera`
- `observation.images`
- `action`
- `emix_actions`
- `action_is_pad`

### Validation 3: EMIX Policy Forward Check

Built one EMIX dataloader batch and executed:

- `policy.forward(batch)`

This succeeded after:

- image keys were restored by window loading through `dataset[i]`
- nested EMIX tensors were moved to the target device correctly

## Modified Files

- `embodied_mixup/policies/utils/emix_loader_registry_utils.py`
- `embodied_mixup/policies/train.py`

## Notes

- The EMIX window loading fix prioritizes correctness with the current
  `lerobot` contract over speed.
- Reading frame windows through `dataset[i]` is slower than direct
  `hf_dataset.select(...)`, but it is the minimal correct fix for the current
  environment.
- No unrelated files were changed as part of this record.
