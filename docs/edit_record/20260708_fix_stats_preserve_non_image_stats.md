# Fix Stats Preserve Non-Image Stats

## Problem

`data_processing/fix_stats.py` was intended to recompute only image statistics,
but it wrote the aggregated result back as the whole `meta/stats.json`.

For datasets whose episode metadata does not contain full per-feature stats,
the aggregation input only contains image stats. This can replace a complete
`stats.json` with one that only has keys such as:

```text
observation.images.base_camera
```

This breaks LeRobot policy evaluation because Diffusion Policy expects
normalization stats for all checkpoint features:

```text
observation.images.base_camera
observation.state
observation.eef_pose
action
```

The observed failure was:

```text
KeyError: 'observation.state'
```

## Decision

Keep the repair script focused on the current problem: image stats repair.

The script now:

- loads the existing full `meta/stats.json`
- recomputes image stats from decoded frames
- replaces only the image feature entries
- preserves all non-image entries such as state, eef pose, action, indices, and
  task stats
- continues creating timestamped backups before overwriting metadata

No new abstraction or dependency was added.

## Files Changed

- `data_processing/fix_stats.py`
  - Loads existing stats with `load_stats(dataset_root)`.
  - Builds `corrected_stats` from the existing full stats dict.
  - Replaces only entries listed in `dataset.meta.camera_keys`.
  - Writes the merged stats dict back to `meta/stats.json`.

- `evaluation/policies/lerobot.py`
  - Removes unused direct reads of `observation.state` and `action` min/max.
  - Adds explicit dataset stats validation against checkpoint input/output
    features so future metadata errors report the missing keys directly.

## Dataset Repaired

Repaired dataset:

```text
data/lerobot/maniskill/pick_cube-b-pd_ee_delta_pose-84x84
```

Command:

```bash
/home/rogerspyke/miniforge3/envs/emix/bin/python data_processing/fix_stats.py \
  --root data/lerobot/maniskill/pick_cube-b-pd_ee_delta_pose-84x84 \
  --repo_id pick_cube-b-pd_ee_delta_pose-84x84
```

Backups created:

```text
meta/stats.json.bak_20260708155743
meta/episodes/chunk-000/file-000.parquet.bak_20260708155743
```

The previously broken image-only stats file had also been backed up as:

```text
meta/stats.json.bak_20260708155432
```

## Validation

Syntax check:

```bash
/home/rogerspyke/miniforge3/envs/emix/bin/python -m py_compile \
  data_processing/fix_stats.py \
  evaluation/policies/lerobot.py
```

Post-repair `meta/stats.json` contains:

```text
action
episode_index
frame_index
index
observation.eef_pose
observation.images.base_camera
observation.state
task_index
timestamp
```

Corrected image stats:

```text
observation.images.base_camera mean=[0.6396663547659349, 0.44629638635993885, 0.34889758799237985]
observation.images.base_camera std=[0.17863052791619397, 0.09771202145200995, 0.13193041045491194]
```

