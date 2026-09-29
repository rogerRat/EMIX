# EMIX Materialized LeRobotDataset Export Implementation Plan

Date: 2026-07-02

Status: User decisions resolved. Ready for implementation planning.

This document is intended to be executable by a lower-level implementation
agent. Code, comments, log messages, schema keys, and implementation-facing
technical strings must remain English and ASCII-only.

## 1. Objective

Replace the current `materialize` output with a standard LeRobotDataset root and
one EMIX marker file:

```text
{dataset_root}/emix/emix_index.json
```

The exported dataset root must be directly loadable by LeRobot as a normal
LeRobotDataset. EMIX-specific training must load the same root through
`policy.emix_files_root`, even though that field name remains legacy.

## 2. Final User Decisions

The following implementation decisions are fixed:

1. Source datasets without video/image data are invalid for this materialized
   export. If the original source dataset has no video folder or no video/image
   features, fail fast.
2. Remove `--materialize_group_limit`. No group limit is supported.
3. `--materialized_repo_id` is required.
4. Visualization export must be based on the new materialized LeRobotDataset
   files.
5. Schema-v2 implementation may use dataclasses or plain dict validation.
6. `source_index_id` and `source_index_schema_version` are required metadata.
7. `public.source_dataset` must include media metadata.
8. Reuse source group ids whenever possible.
9. Store explicit member ids.
10. A trainable group with member count other than two must fail immediately.
11. Clip boundary windows and record actual valid length. Do not pad.
12. Padding is not supported.
13. Always export `observation.eef_pose` if it exists.
14. Copy all source `observation.*` features.
15. Copy arbitrary non-observation source features when they can be represented
    by the target LeRobot feature schema.
16. Use original dataset text/task information by default. If the exact field is
    unclear, locate it in source metadata/frame data and document the lookup.
    The user may provide it manually later.
17. Target the LeRobot writer API pattern used by
    `data_processing/convert_to_lerobot_v2.py`.
18. Copy `fps` from the original dataset and keep it identical.
19. Keep image values compatible with LeRobot and rely on LeRobot writer code as
    much as possible.
20. Use LeRobot finalization/consolidation code if available and appropriate.
21. Include output root path in the materialized index id.
22. Remove old `EmixFiles` compatibility immediately. Do not provide backward
    compatibility.
23. Keep the existing sample id format if no implementation reason requires a
    change.
24. For `emix_smolvla`, `batch["task"]` must be the task string, not the group
    id.
25. Validate LeRobot episode count from `meta/info.json`.
26. If no trainable groups exist, fail.
27. Implement file logging in this task.
28. Tests may be mock-only. No LeRobot integration test is required.

## 3. Non-Goals

- Do not preserve the old `EmixFiles` artifact format for `materialize`.
- Do not generate `manifest.json`.
- Do not generate `groups/*/windows/*.json`.
- Do not generate any EMIX JSON sidecar except `emix/emix_index.json`.
- Do not store frame pixel payloads, full action arrays, or per-frame feature
  payloads inside the EMIX index.
- Do not rename `policy.emix_files_root` in this change.
- Do not provide any backward compatibility path for old `EmixFiles` roots.

## 4. Current Code Facts

The current repository state is:

- `embodied_mixup/cli.py` exposes `materialize`, but it currently calls
  `_export_emix_artifacts(...)`.
- `_export_emix_artifacts(...)` writes a schema-v1 `emix_index.json` plus an old
  `emix_files/` tree through `build_emix_files(...)`.
- `embodied_mixup/candidates/emix_index.py` defines schema-v1 source index
  dataclasses.
- `embodied_mixup/candidates/emix_files.py` and
  `embodied_mixup/candidates/emix_files_utils.py` define the old materialized
  JSON artifact.
- `embodied_mixup/policies/utils/emix_loader_registry_utils.py` reads the old
  `EmixFiles` structure via `EmixFilesReader`.
- `embodied_mixup/policies/train.py` still passes
  `cfg.policy_cfg.emix_files_root` to `make_emix_loader(...)`.
- Existing tests are in `tests/test_emix_artifacts.py`.
- `data_processing/convert_to_lerobot_v2.py` is the writer API reference. It
  imports `LeRobotDataset` from `lerobot.datasets.lerobot_dataset`, calls
  `LeRobotDataset.create(repo_id=..., root=..., fps=..., features=...,
  image_writer_threads=...)`, adds frames with `dataset.add_frame(frame)`, and
  calls `dataset.save_episode()` once per episode.

## 5. Output Contract

`--output_dir` is the final LeRobotDataset root. It is not a parent directory.

Required root layout:

```text
{output_dir}/
  meta/
  data/
  videos/
  emix/
    emix_index.json
```

Allowed top-level entries:

- `meta`
- `data`
- `videos`
- `emix`

Rules:

- If the source dataset has no video folder, fail before writing output.
- If the source dataset has no image/video features, fail before writing output.
- If `output_dir` exists, fail before writing anything.
- Create `output_dir` only after all source validation passes.
- Write all LeRobot files through LeRobotDataset APIs.
- Write `emix/emix_index.json` after LeRobot episode export succeeds.
- If `emix/emix_index.json` writing fails, leave the dataset root on disk and
  raise an exception. Do not silently return a partial success.
- `emix/` must contain exactly one file: `emix_index.json`.
- No other file matching `*.json` is allowed under `emix/`.
- No old `emix_files/` directory is allowed anywhere under `output_dir`.
- No old `manifest.json` is allowed anywhere under `output_dir`.
- `videos/**/*.mp4` must exist in the materialized root.
- `data/**/*.parquet` must exist after export.

## 6. CLI Contract

Keep the existing subcommand:

```bash
python -m embodied_mixup materialize \
  --repo_id SOURCE_REPO_ID \
  --root /abs/source_lerobot_root \
  --output_dir /abs/materialized_lerobot_root \
  --materialized_repo_id MATERIALIZED_REPO_ID
```

Required changes in `embodied_mixup/cli.py`:

- Change `cmd_materialize(args)` to call the new materialized exporter.
- Keep `_resolve_config(args)` unchanged unless required by exporter inputs.
- Remove `--materialize_group_limit`.
- Add required `--materialized_repo_id`.
- If `--materialized_repo_id` is missing, argparse must fail.
- Update help text from `EmixFiles` to materialized LeRobotDataset.
- Update `visualize --export_dir` so exported visualization artifacts are based
  on the new materialized LeRobotDataset format.
- If `visualize --export_dir` needs a materialized repo id, add a required or
  explicitly validated `--materialized_repo_id` argument for that path.

## 7. New Module Layout

Create new exporter modules. Do not extend old `emix_files_utils.py` for the new
format.

Recommended files:

```text
embodied_mixup/candidates/materialized_lerobot.py
embodied_mixup/candidates/materialized_lerobot_utils.py
```

Responsibilities:

- `materialized_lerobot.py`
  - Dataclasses or lightweight schema helpers for schema-v2 materialized EMIX
    index. Either approach is acceptable.
  - Constants such as `EMIX_MATERIALIZED_INDEX_SCHEMA_VERSION = 2`.
  - Payload validation for the schema-v2 index.

- `materialized_lerobot_utils.py`
  - Export orchestration.
  - Source feature inspection.
  - Source window extraction.
  - LeRobotDataset writer construction.
  - Post-export validation.
  - JSON save/load helpers for `emix/emix_index.json`.

Required public functions:

```python
def export_materialized_lerobot_dataset(
    index,
    source_dataset,
    output_dir,
    materialized_repo_id,
) -> tuple[object, Path]:
    """Export trainable EMIX member windows as a LeRobotDataset.

    @input: index [EmixIndex], source_dataset [LeRobotDataset-like],
        output_dir [str|Path, non-existing final dataset root],
        materialized_repo_id [str, non-empty].
    @output: (materialized_index, output_root_path). Success raises no error.
    @scenario: Materialize schema-v2 EMIX training data as a normal
        LeRobotDataset plus one EMIX marker JSON.
    """
```

```python
def validate_materialized_lerobot_root(output_dir) -> None:
    """Validate root layout and EMIX schema-v2 sidecar.

    @input: output_dir [str|Path, existing materialized LeRobotDataset root].
    @output: None on success; raises FileNotFoundError or ValueError on failure.
    @scenario: Enforce the materialized root contract after export and before
        training loader use.
    """
```

```python
def load_materialized_emix_index(output_dir):
    """Load schema-v2 materialized EMIX index from a dataset root.

    @input: output_dir [str|Path, existing materialized LeRobotDataset root].
    @output: materialized index object or validated dict.
    @scenario: Resolve EMIX group metadata for policy loaders.
    """
```

## 8. Schema-V2 Materialized Index Contract

Use schema version `2`.

Required top-level structure:

```json
{
  "public": {
    "id": "emix_materialized_index_xxx",
    "schema_version": 2,
    "created_at": "YYYY-MM-DDTHH:MM:SS+0800",
    "source_dataset": {
      "repo_id": "source_repo_id",
      "root": "/abs/source/root",
      "hash": "source_info_hash_or_null",
      "media": {
        "camera_keys": ["base_camera"],
        "frame_alignment": "per_frame",
        "video_path_pattern": "videos/observation.images.{camera_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "storage_format": "mp4"
      }
    },
    "materialized_dataset": {
      "repo_id": "materialized_repo_id",
      "root": "/abs/outputs/root"
    },
    "metadata": {
      "window_len": 16,
      "group_count": 123,
      "members_per_group": 2,
      "episode_unit": "member_window",
      "range_semantics": "start_inclusive_end_exclusive",
      "global_index_basis": "parquet_index_column",
      "source_index_id": "emix_index_xxx",
      "source_index_schema_version": 1,
      "source_group_count": 200,
      "source_training_group_count": 123,
      "exported_episode_count": 246
    }
  },
  "content": {
    "cross_point_groups": []
  }
}
```

Required `metadata` fields:

- `window_len`: integer greater than zero.
- `group_count`: number of exported trainable groups.
- `members_per_group`: always `2`.
- `episode_unit`: always `member_window`.
- `range_semantics`: always `start_inclusive_end_exclusive`.
- `global_index_basis`: always `parquet_index_column`.
- `source_index_id`: schema-v1 source `EmixIndex.public.id`.
- `source_index_schema_version`: schema-v1 source
  `EmixIndex.public.schema_version`.
- `source_group_count`: total groups in source index.
- `source_training_group_count`: training groups before export.
- `exported_episode_count`: `group_count * 2`.
- `fps`: copied from source dataset metadata.
- `task_source`: string describing where task text was resolved from.

Required `source_dataset.media` fields:

- `camera_keys`
- `frame_alignment`
- `video_path_pattern`
- `storage_format`

## 9. Group Contract

Each exported group must contain exactly two members:

- `anchor`
- `neighbor`

Required group fields:

- `group_id`
- `source_group_id`
- `export_group_index`
- `state`
- `members`

Group id rule:

- Reuse source `EmixCrossPointGroup.group_id` whenever possible.
- Store `source_group_id` explicitly even when it equals `group_id`.

Required member fields:

- `member_id`
- `member_role`: `anchor` or `neighbor`
- `source_member_index`: `0` for anchor, `1` for neighbor.
- `source_dataset_repo_id`
- `source_episode_index`
- `source_window_start_index`
- `source_window_end_index`
- `source_cross_frame_index`
- `source_valid_length`
- `export_episode_index`
- `export_window_start_index`
- `export_window_end_index`
- `export_cross_frame_index`
- `export_valid_length`

Member id rule:

- Use explicit deterministic member ids.
- Recommended values are `{group_id}_anchor` and `{group_id}_neighbor`.

Exported episode order must be contiguous and deterministic:

```text
group_0 anchor   -> export episode 0
group_0 neighbor -> export episode 1
group_1 anchor   -> export episode 2
group_1 neighbor -> export episode 3
```

For every member:

- `source_window_start_index <= source_cross_frame_index`
- `source_cross_frame_index < source_window_end_index`
- `export_window_start_index <= export_cross_frame_index`
- `export_cross_frame_index < export_window_end_index`
- `source_valid_length == source_window_end_index - source_window_start_index`
- `export_valid_length == export_window_end_index - export_window_start_index`
- `source_valid_length == export_valid_length`
- `export_valid_length` must equal the number of frames written to the export
  episode.

## 10. Source Group Filtering

Only groups with `group.state.training == True` are exported.

Implementation rules:

- Skip aborted groups.
- If a training group does not contain exactly two members, raise `ValueError`.
- Preserve source group order from `index.content.cross_point_groups`.
- No group limit is supported.
- If no training groups remain after filtering, raise `ValueError`.

## 11. Window Semantics

Use existing helper:

```python
compute_window_bounds(
    frame_idx=member.source_frame_index,
    window_len=window_len,
    episode_start=source_episode_start,
    episode_end=source_episode_end,
)
```

Source window range is half-open:

```text
[source_window_start_index, source_window_end_index)
```

Export window range is also half-open:

```text
[export_window_start_index, export_window_end_index)
```

Export index semantics:

- `export_window_start_index` is the first global frame index of the exported
  member episode in the materialized LeRobotDataset parquet index.
- `export_window_end_index` is one past the final global frame index.
- `export_cross_frame_index` is
  `export_window_start_index + source_cross_frame_index - source_window_start_index`.
- Boundary windows are clipped to the source episode boundary.
- Do not pad boundary windows.
- Record actual valid length in `source_valid_length` and
  `export_valid_length`.

## 12. Feature Export Contract

The materialized LeRobotDataset should preserve source data as much as possible.

Required frame keys to export:

- `action`
- Every source key that starts with `observation.`.
- Every source key that starts with `observation.images.`.
- `observation.eef_pose` if present, even when `observation.state` exists.
- Arbitrary non-observation source features when they can be represented in the
  target LeRobot feature schema and passed through `add_frame(...)` safely.
- `task`, resolved from original dataset text/task information.

Required task resolution policy:

- First try source frame task text if available.
- Then try source episode task text from source metadata if available.
- Then try dataset-level task metadata if available.
- If no task text can be found, raise `ValueError` unless the user provides a
  manual task value in a later implementation update.
- Record the selected source in schema-v2 metadata field `task_source`.
- For `emix_smolvla`, runtime `batch["task"]` must use this task string.

Recommended frame keys to exclude:

- Source bookkeeping fields generated by LeRobot, such as `index`,
  `episode_index`, `frame_index`, and `timestamp`, unless
  `LeRobotDataset.add_frame(...)` for the target API requires user-supplied
  values.

Feature schema rules:

- Infer target `features` from the first exportable source frame, following the
  pattern in `data_processing/convert_to_lerobot_v2.py`.
- Image/video features must use LeRobot-compatible `dtype: video` feature
  declarations.
- Preserve image values in a LeRobot-compatible form and rely on LeRobot writer
  code for video encoding.
- If a source feature cannot be represented by the target LeRobot feature
  schema, raise a clear `ValueError` with the feature key and reason.

## 13. LeRobotDataset Writer Contract

Use the LeRobot API pattern from `data_processing/convert_to_lerobot_v2.py`:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id=materialized_repo_id,
    root=str(output_root),
    fps=source_fps,
    features=features,
    image_writer_threads=image_writer_threads,
)

for frame in member_window_frames:
    frame["task"] = resolved_task_string
    dataset.add_frame(frame)
dataset.save_episode()
```

Required behavior:

- The writer must create a local dataset at `output_dir`.
- The writer repo id must be `materialized_repo_id`.
- One EMIX member window equals one LeRobot episode.
- Call `save_episode()` after all frames for one member are added.
- Copy `fps` from source dataset metadata and keep it identical.
- Use LeRobot code for image/video handling as much as possible.
- Use LeRobot finalization/consolidation methods if the target API provides a
  documented method required to complete local dataset metadata or video writes.
- Do not manually write parquet or mp4 files unless LeRobot API version requires
  a workaround. If a workaround is needed, stop and document it before coding.

Implementation must inspect the active API before coding details:

- Confirm `LeRobotDataset.create(...)` signature.
- Confirm how source `fps` is exposed.
- Confirm whether a writer finalization method exists.
- Confirm how source tasks are represented in this repository's LeRobot
  datasets.

## 14. Materialized Index ID Contract

Use a deterministic, path-specific id that includes output root:

```text
emix_materialized_index_{md5(source_index_id, materialized_repo_id, output_root, group_count)[:12]}
```

Rules:

- `output_root` must be the resolved absolute output root string.
- Including `output_root` is intentional; materialized indexes are path-specific.

## 15. Loader Migration Contract

Keep runtime batch keys unchanged:

- `observation.state`
- `observation.images.*`
- `observation.images`
- `action`
- `emix_actions`
- `emix_alpha`
- `emix_mask`
- `action_is_pad`
- `sample_id`
- `group_id`
- `member_ids`

Required loader behavior:

- `policy.emix_files_root` now points to the materialized LeRobotDataset root.
- The loader reads `{root}/emix/emix_index.json`.
- The loader also loads the same root as a LeRobotDataset.
- The registry entry point remains `make_emix_loader(...)`.
- The returned dataset object must still be a PyTorch `Dataset`.
- The loader must reconstruct two-member samples from the schema-v2 index.
- Member frames must be fetched from materialized LeRobot episodes, not from
  JSON sidecars.
- Remove old `EmixFilesReader` compatibility. Do not accept old
  `manifest.json` roots.
- If a user passes an old `EmixFiles` root, fail immediately. A concise error
  mentioning unsupported legacy `EmixFiles` is acceptable, but no fallback path
  may be implemented.

Recommended implementation:

- Replace `EmixFilesReader` with `MaterializedEmixDatasetReader`.
- Keep `EmixPolicyDataset` class name if this minimizes training changes.
- Keep `make_emix_loader(...)` signature unchanged.
- In `EmixPolicyDataset.__getitem__`, load frames from materialized LeRobot
  global indices or episode-local indices according to schema-v2 member fields.
- Keep existing `sample_id` format if possible:
  `{group_id}:{member_0_id}:{member_1_id}`.
- For `emix_smolvla`, set `batch["task"]` to the resolved task string.

## 16. Validation Contract

After export, run `validate_materialized_lerobot_root(output_dir)`.

Checks:

- `output_dir` exists and is a directory.
- Top-level entries are exactly `meta`, `data`, `videos`, and `emix`.
- `meta/info.json` exists.
- At least one `data/**/*.parquet` file exists.
- At least one `videos/**/*.mp4` file exists.
- `emix/emix_index.json` exists.
- `emix/emix_index.json` has `public.schema_version == 2`.
- `emix/` contains only `emix_index.json`.
- No `manifest.json` exists anywhere below `output_dir`.
- No `emix_files/` directory exists anywhere below `output_dir`.
- Every group has exactly two members.
- Export episodes for each group are contiguous.
- Export window lengths match exported episode lengths.
- Cross-frame indices fall inside declared window bounds.
- `public.materialized_dataset.root` equals `output_dir.resolve()`.
- `public.materialized_dataset.repo_id` equals CLI `--materialized_repo_id`.
- `public.metadata.group_count` equals the number of exported groups.
- `public.metadata.exported_episode_count == group_count * 2`.
- LeRobot episode count in `meta/info.json` equals
  `public.metadata.exported_episode_count`.
- Source dataset validation must fail if source video folder or source image
  features are absent.

## 17. Error Handling

Fail fast with clear exceptions.

Required errors:

- Existing output root: `FileExistsError`.
- Missing source root: `FileNotFoundError`.
- Source has no video folder: `FileNotFoundError` or `ValueError`.
- Source has no image/video features: `ValueError`.
- Missing required source feature: `KeyError`.
- No trainable groups after filtering: `ValueError`.
- Training group member count is not exactly two: `ValueError`.
- Missing task text and no manual task fallback: `ValueError`.
- Unsupported schema-v1 source index: `ValueError`.
- Unsupported schema-v2 materialized index during load: `ValueError`.
- Legacy `EmixFiles` root passed to loader: `ValueError`.
- LeRobot writer API import/signature mismatch: `ImportError` or `RuntimeError`
  with exact remediation text.

## 18. Logging

Use Python `logging`.

Required console and file logging:

- Logs must be stored in `./logs/`.
- Log file name format is `<script_name>_<YYYYMMDDHHMMSS>.log`.
- Timestamp timezone is UTC+8.
- Log format is `[Stage/Module] [Level] Message`.
- Default verbosity is `DEBUG`.
- Capture subprocess/console output when subprocesses are introduced.

Required log stages:

- `[materialize.validate]`
- `[materialize.index]`
- `[materialize.export]`
- `[materialize.validate_output]`
- `[loader.materialized]`

Minimum messages:

- Source repo id and root.
- Materialized repo id and output root.
- Source fps.
- Source task resolution path.
- Source video folder validation result.
- Source total group count.
- Training group count.
- Exported group count.
- Exported episode count.
- Export progress every 5 percent or every N groups.
- Validation success/failure.

Color coding is required for console output if practical without destabilizing
file logs:

- Green: success.
- Red: warning/error.
- Blue: URL/args.

File logs must remain plain ASCII without terminal control codes.

## 19. Documentation Updates

Update these docs after code changes:

- `docs/DEV.md`
- `docs/sop/SOP-02-intersection-clustering-visualization.md`
- `docs/sop/SOP-03-model-training.md`
- `docs/sop/data-structure-and-transforms.md`

Required doc changes:

- Replace `EmixFiles` materialize output descriptions.
- State that `policy.emix_files_root` is a legacy field name pointing to the
  materialized LeRobotDataset root.
- Document the exact `materialize` command.
- Document the schema-v2 `emix/emix_index.json` contract.
- Document that old `EmixFiles` artifacts are unsupported.
- Document that source datasets without video/image data fail materialization.
- Document that `emix_smolvla` task values are task strings from the dataset.

## 20. Test Plan

Add or update tests under `tests/`.

Recommended test files:

```text
tests/test_materialized_lerobot_export.py
tests/test_materialized_emix_loader.py
tests/test_emix_artifacts.py
```

Tests may be mock-only. Do not require a real LeRobot installation for the base
test suite.

Required unit tests:

- Schema-v2 index round trip or dict validation.
- Reject schema version `1` when loading materialized index.
- Reject missing `public`.
- Reject missing `content.cross_point_groups`.
- Reject group without exactly two members.
- Reject cross-frame outside window bounds.
- Reject non-contiguous export episode assignment.
- Reject missing source media metadata.
- Reject missing required source index metadata.

Required export tests:

- Export fails if output root already exists.
- Export fails if source video folder is missing.
- Export fails if source image/video features are missing.
- Export creates only allowed top-level layout.
- Export writes `meta/info.json`.
- Export writes at least one `data/**/*.parquet`.
- Export writes at least one `videos/**/*.mp4`.
- Export writes only `emix/emix_index.json` under `emix/`.
- Export does not generate `manifest.json`.
- Export does not generate `emix_files/`.
- Export emits exactly two episodes per group.
- Export order is deterministic.
- Boundary windows are clipped and actual valid length is recorded.
- No padding is emitted.
- `meta/info.json` episode count matches schema-v2 exported episode count.
- `fps` is copied from source metadata.
- Task string is resolved from source data.

Required loader tests:

- `make_emix_loader("emix_diffusion", root, ...)` reads schema-v2 materialized
  root.
- `make_emix_loader("emix_smolvla", root, ...)` reads schema-v2 materialized
  root.
- Runtime batch keys match the existing contract.
- `emix_actions` contains two tensors.
- `emix_alpha` is scalar float tensor.
- `emix_mask` and `action_is_pad` have action horizon length.
- `member_ids` are deterministic explicit ids.
- `sample_id` keeps the old format if possible.
- `batch["task"]` is the task string for `emix_smolvla`.
- Old `EmixFiles` roots are rejected without compatibility fallback.

## 21. Implementation Sequence

Do implementation in this order:

1. Inspect active LeRobot API in the target environment and compare it with
   `data_processing/convert_to_lerobot_v2.py`.
2. Locate source dataset fps metadata.
3. Locate source dataset task text metadata/frame field.
4. Add schema-v2 materialized index helpers.
5. Add materialized LeRobot exporter.
6. Add source validation for video folder and image/video features.
7. Wire `cmd_materialize(...)` to the new exporter.
8. Remove `--materialize_group_limit`.
9. Add `--materialized_repo_id` as required.
10. Update `visualize --export_dir` to use the new materialized file format.
11. Add output validation.
12. Migrate EMIX loader to read materialized root only.
13. Remove old `EmixFiles` loader compatibility path.
14. Add file logging.
15. Update tests for schema/export/loader.
16. Update docs and SOPs.
17. Run targeted tests.
18. Run the full test suite if runtime dependencies are available.
