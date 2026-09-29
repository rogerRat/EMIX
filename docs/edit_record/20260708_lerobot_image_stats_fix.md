# LeRobot Image Stats Fix

## Problem

LeRobot dataset creation can write incorrect image standard deviation values
for `observation.images.*` features.

The observed PickCube dataset had:

```text
meta/stats.json image std:
[0.001217, 0.001238, 0.001826]
```

Manual decoding of dataset frames showed the actual values were around:

```text
[0.1789, 0.0979, 0.1320]
```

Episode metadata confirmed the root symptom:

```text
stats/observation.images.base_camera/std = [[[0.0]], [[0.0]], [[0.0]]]
```

The likely cause is uint8 overflow in the current LeRobot stats path. The
implementation computes a mean of squares from image arrays. If the image array
is `uint8`, `batch ** 2` overflows before the mean is computed.

## Impact

Diffusion Policy uses:

```text
VISUAL: MEAN_STD
STATE: MIN_MAX
ACTION: MIN_MAX
```

Therefore an image std that is two orders of magnitude too small causes:

```text
(image - mean) / wrong_std
```

to amplify visual inputs by roughly 70x to 150x. This can destabilize visual
policy training and evaluation.

## Decision

Do not patch `site-packages`.

Add a project-local repair script:

```text
data_processing/fix_stats.py
```

The script:

- loads a standard LeRobot dataset
- detects visual keys from dataset metadata
- reads frames through `LeRobotDataset`, so both image-backed and video-backed
  storage are supported
- converts decoded frames to float before computing stats
- rewrites image stats in `meta/stats.json`
- rewrites image stats in `meta/episodes/chunk-*/file-*.parquet`
- creates timestamped backups before overwriting metadata files

`data_processing/convert_to_lerobot_v2.py` now calls the repair step after all
episodes have been saved.

## Files Changed

- `data_processing/fix_stats.py`
  - New standalone stats repair CLI and importable function.

- `data_processing/convert_to_lerobot_v2.py`
  - Calls `fix_lerobot_image_stats(...)` after conversion.
  - Adds a comment explaining the uint8 overflow issue.

- `docs/sop/SOP-01-data-generation-pipeline.md`
  - Documents the automatic repair step and manual command.

- `docs/sop/data-structure-and-transforms.md`
  - Documents the image stats contract and repair behavior.

## Validation

Ran:

```bash
python -m py_compile data_processing/fix_stats.py data_processing/convert_to_lerobot_v2.py
python data_processing/fix_stats.py \
  --root data/lerobot/maniskill/pick_cube_rgb_b_84x84_full \
  --repo_id pick_cube_rgb_b_84x84_full
```

The script created backups:

```text
meta/stats.json.bak_20260708124628
meta/episodes/chunk-000/file-000.parquet.bak_20260708124628
```

Corrected global image stats:

```text
mean=[0.6396639796907341, 0.4462936139071866, 0.34889153966473363]
std=[0.1786343624838237, 0.09771325777522127, 0.13192767889304347]
```

Episode 0 image std after repair:

```text
[0.17893621242222227, 0.09814048617708374, 0.13194974647704003]
```
