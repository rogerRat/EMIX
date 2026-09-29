# Materialize Export Problem Summary For Agents

## Purpose

This document explains the current `materialize` failure mode in simple engineering terms.
It is intended for an agent that will later redesign or replace the export path.

The goal is to prevent future work from preserving the wrong artifact format.

## One-Line Summary

The current `python -m embodied_mixup materialize` implementation does not export a
small LeRobot-style dataset. Instead, it builds a custom JSON sidecar tree and writes
decoded image pixel data directly into window JSON files. This is the main design error.

## Expected Output

The intended output is:

- multiple small LeRobotDataset-style datasets
- scalar and structured features stored in parquet
- image data stored in mp4
- directory structure compatible with the existing `data/lerobot/...` format

In other words, the exporter should produce `parquet + mp4`, not `json + embedded pixels`.

## Actual Output

The current code path is:

- `embodied_mixup/cli.py`
  - `cmd_materialize()`
  - `_export_emix_artifacts()`
- `embodied_mixup/candidates/emix_files_utils.py`
  - `build_emix_files()`
  - `_materialize_group()`
  - `_extract_member_window()`

The current artifact shape is:

```text
{output_dir}/
  emix_index.json
  emix_files/
    manifest.json
    groups/
      {group_id}/
        group.json
        windows/
          {member_id}.json
```

This is a custom `EmixFiles` sidecar format. It is not a LeRobotDataset export.

## Critical Design Error

The current implementation loads image windows and serializes them into JSON.

Relevant code behavior in `embodied_mixup/candidates/emix_files_utils.py`:

1. `_extract_member_window()` builds `feature_payload`.
2. For each image key `observation.images.{camera_key}`, it calls
   `load_optional_feature_window(...)`.
3. That reaches `embodied_mixup/candidates/loaders.py::_load_video_backed_feature_window()`.
4. `_load_video_backed_feature_window()` iterates frame by frame and calls `dataset[frame_idx]`.
5. Frames are converted into numpy arrays.
6. Back in `_extract_member_window()`, arrays are converted with `.tolist()`.
7. `_write_json()` stores that payload as pretty-printed JSON.

This means the exporter is doing all of the following:

- decoding source video frames
- expanding them into raw pixel arrays
- converting raw arrays into Python lists
- writing image pixels into large text JSON files

This is not a valid media export strategy for this project.

## Why It Looks Like A Hang

The CLI log can appear stuck at:

```text
[candidates.emix_files] Group materialization progress: 1/10374 groups
```

This does not necessarily mean the process is deadlocked.

The first visible progress line is emitted before expensive per-group work starts. That work includes:

- reading multiple frame windows from the source dataset
- decoding video-backed features
- converting arrays to Python lists
- writing large JSON files

Because each group has two members and each member window may include image tensors,
the first group can take a long time before the next progress log appears.

## Observed Evidence

Observed during investigation on the reported run:

- output files were already being created under
  `data/emix_files/stack_cube_rgb_b_500_84x84_nonoop/emix_files/groups/...`
- approximately `725` JSON files had already been written
- total output size had already reached about `5.7G`
- individual window JSON files were about `8.4MB`

Conclusion:

- the process was not stuck on the first group
- the export path is simply extremely heavy
- the heavy cost comes from the storage format and serialization strategy

## Root Cause

The root cause is not merely "export is slow".

The root cause is that the current materialization target is wrong:

- it exports `EmixFiles` JSON sidecars instead of LeRobotDataset artifacts
- it stores image content inside JSON instead of mp4
- it duplicates decoded media in the most inefficient representation available

There is also a semantic mismatch in the current metadata:

- `materialization_policy["duplicate_media"]` is set to `False`
- actual behavior duplicates media content into JSON sidecars

So both the artifact definition and the implementation are inconsistent with the intended design.

## What The Exporter Should Do Instead

The exporter should be redesigned around dataset export, not JSON window export.

The target behavior should be:

1. produce one or more small LeRobot-style datasets
2. store scalar features in parquet
3. store visual streams in mp4
4. preserve enough metadata for later training and traceability
5. keep output layout compatible with the existing LeRobot loader expectations

## Acceptable Implementation Directions

### Preferred Direction

Export a real LeRobotDataset-compatible artifact:

- parquet for tabular or scalar features
- mp4 for image sequences
- proper `meta/` files and dataset structure

### Temporary Direction

If a full LeRobotDataset export cannot be completed immediately, a temporary intermediate
format may be used, but it must still satisfy these constraints:

- image data must not be stored in JSON
- binary media should be written as mp4 or another compact binary format
- JSON may only contain metadata, indices, and lightweight scalar fields

## Non-Goals For The Fix

The fix should not try to optimize the current JSON-with-pixels design.

Specifically, do not treat the following as the final solution:

- reducing JSON indentation only
- compressing giant JSON files only
- keeping image arrays in JSON but chunking them differently
- adding more progress logs without changing the data format

Those changes may reduce symptoms slightly, but they preserve the wrong export target.

## Files Most Relevant To The Redesign

- `embodied_mixup/cli.py`
- `embodied_mixup/candidates/emix_files_utils.py`
- `embodied_mixup/candidates/loaders.py`
- `embodied_mixup/policies/utils/emix_loader_registry_utils.py`

These files currently assume the custom `EmixFiles` materialization path and will likely
need coordinated redesign if the exporter is changed to a true LeRobotDataset output.

## Decision For Future Agents

Do not continue evolving the current materialize path as if the JSON sidecar format were correct.

Treat the current behavior as a design bug:

- wrong artifact format
- wrong media representation
- wrong scaling properties

The correct target is a LeRobot-style dataset export with `parquet + mp4`.
