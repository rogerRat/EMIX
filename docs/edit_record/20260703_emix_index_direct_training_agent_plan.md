# EMIX Index Direct Training Migration Agent Plan

Date: 2026-07-03

Status: Ready for implementation by another agent.

Audience: implementation agent only. Optimize for maximum agent readability, not
human prose. All implementation-facing names, code comments, log messages,
schema keys, and technical strings must remain English and ASCII-only.

## 0. Task Objective

Replace the current EMIX export/materialization training path with direct
training from:

```text
source LeRobotDataset root + EmixIndex JSON
```

The final active path must be:

```text
build EmixIndex -> visualize/tune EmixIndex -> train directly from source data using EmixIndex
```

The final active path must not be:

```text
build EmixIndex -> export/materialize EmixFiles or materialized LeRobotDataset -> train
```

## 1. Hard Requirements

1. Remove the EmixFiles/materialized export design from the active code path.
2. Train directly from source `LeRobotDataset` frames referenced by `EmixIndex`.
3. Strengthen `EmixIndex` so it records a strict source dataset fingerprint.
4. At training load time, recompute and validate the fingerprint before reading
   any EMIX training samples.
5. Remove parameters that only exist for export/window materialization.
6. Remove `window_len` from candidate generation config, CLI, index metadata,
   docs, and tests unless a remaining active code path proves it is required.
7. Do not add new dependencies.
8. Do not add compatibility layers for old EmixFiles or materialized roots.
9. Do not preserve old config keys silently.
10. Fail fast with explicit errors when old fields or wrong artifacts are used.

## 2. Non-Goals

Do not implement any of the following:

- No `EmixFiles` output.
- No custom `emix_files/` tree.
- No materialized EMIX LeRobotDataset output.
- No `materialize` CLI command.
- No `visualize --export_dir`.
- No `policy.emix_files_root`.
- No `emix.files_root`.
- No fallback from direct index training to materialized dataset training.
- No automatic source dataset root guessing beyond the existing exact root
  validation behavior.
- No window export, no fixed global window length, no stored training windows.
- No media duplication.
- No migration command for old artifacts.

## 3. Current Repository Facts

These facts were observed before writing this plan:

- `docs/sop/SOP-02-intersection-clustering-visualization.md` still documents
  `materialize`, `window_len`, and export-oriented workflow.
- `embodied_mixup/cli.py` exposes `build`, `materialize`, `visualize`, and
  `tune`.
- `embodied_mixup/cli.py::_add_index_args` includes `--window_len` and
  `--raw_candidate_feature_keys`.
- `embodied_mixup/cli.py::cmd_materialize` exports a materialized LeRobotDataset.
- `embodied_mixup/cli.py::cmd_visualize` can export a materialized dataset via
  `--export_dir`.
- `embodied_mixup/configs/config.py::EmbodiedMixupConfig` contains
  `window_len` and `raw_candidate_feature_keys`.
- `embodied_mixup/candidates/emix_index.py` schema version is currently `1`.
- `EmixSourceDatasetDeclaration.hash` exists but is currently too weak because
  it is only based on `meta/info.json`.
- `embodied_mixup/candidates/emix_index_utils.py::_hash_dataset_info` hashes
  only `meta/info.json` with md5.
- `build_emix_index` includes `window_len` and `raw_candidate_feature_keys` in
  index id input and metadata.
- `embodied_mixup/candidates/materialized_lerobot.py` and
  `embodied_mixup/candidates/materialized_lerobot_utils.py` implement the
  current materialized LeRobotDataset path.
- `embodied_mixup/candidates/emix_files.py` and
  `embodied_mixup/candidates/emix_files_utils.py` implement the older custom
  EmixFiles path.
- `embodied_mixup/policies/utils/emix_loader_registry_utils.py` contains
  `EmixFilesReader`, but the reader currently reads materialized
  LeRobotDataset roots and `emix/emix_index.json`.
- `embodied_mixup/policies/utils/emix_config.py` requires `emix.files_root`.
- EMIX policy config dataclasses contain `emix_files_root`.
- `embodied_mixup/policies/train.py` passes
  `cfg.policy_cfg.emix_files_root` to `make_emix_loader`.
- Existing tests include materialized artifact tests in
  `tests/test_emix_artifacts.py` and loader patches in
  `tests/test_mix_training.py`.

## 4. Final Active Interface

### 4.1 Build CLI

Keep:

```bash
python -m embodied_mixup build \
  --repo_id SOURCE_REPO_ID \
  --root SOURCE_LEROBOT_ROOT \
  --output_dir EMIX_INDEX_OUTPUT_DIR
```

Output:

```text
{EMIX_INDEX_OUTPUT_DIR}/emix_index.json
```

### 4.2 Visualize CLI

Keep:

```bash
python -m embodied_mixup visualize \
  --repo_id SOURCE_REPO_ID \
  --root SOURCE_LEROBOT_ROOT \
  --output vis_out/result.html
```

Output:

```text
vis_out/result.html
vis_out/plotly.min.js
```

### 4.3 Tune CLI

Keep:

```bash
python -m embodied_mixup tune \
  --repo_id SOURCE_REPO_ID \
  --root SOURCE_LEROBOT_ROOT
```

### 4.4 Removed CLI

Remove:

```bash
python -m embodied_mixup materialize
```

Remove these options from all active commands:

```text
--window_len
--raw_candidate_feature_keys
--export_dir
--materialized_repo_id
```

### 4.5 Training YAML

Use this final schema:

```yaml
dataset:
  repo_id: stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop
  root: data/lerobot/maniskill/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop

emix:
  enabled: true
  index_path: data/emix_data/stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop/emix_index.json
  mixup:
    alpha_mode: beta
    beta: 0.4
    alpha_value: null
    loss_weight: 1.0
    visual_keys: null
    state_mode: anchor
  training:
    training_mode: mixed
    seed: 0
    scheduler:
      type: constant
      batch_prob: 0.10
```

Reject this legacy schema:

```yaml
emix:
  files_root: data/emix_files/anything
```

Reject this legacy policy field:

```yaml
policy:
  emix_files_root: anything
```

## 5. Implementation Order

Follow this order to reduce breakage:

1. Implement strong source dataset fingerprint helpers.
2. Upgrade `EmixIndex` schema and index build metadata.
3. Remove export-only config and CLI parameters.
4. Implement direct `EmixIndexReader`.
5. Change training config schema from `files_root` to `index_path`.
6. Wire direct reader into training dataset construction.
7. Remove active exports from package `__init__` files.
8. Disable or delete materialized/EmixFiles active code paths.
9. Update tests.
10. Update docs.

## 6. Source Dataset Fingerprint

### 6.1 Required Behavior

The fingerprint must prove that the `EmixIndex` was built from exactly the
source dataset currently used for training.

Use sha256 only. Do not use md5 for the new strict fingerprint.

Fingerprint must include files under:

```text
meta/
data/
videos/
```

If `videos/` is absent, still compute the fingerprint for existing roots. Do not
require videos for index fingerprinting unless another existing validation
already requires them. The fingerprint function must not invent dataset layout
rules beyond exact file matching.

Ignore:

```text
logs/
outputs/
vis_out/
emix/
emix_files/
```

because these are not source dataset content.

### 6.2 Fingerprint Record Shape

Add a fingerprint payload shaped like this:

```json
{
  "algorithm": "sha256",
  "included_roots": ["meta", "data", "videos"],
  "file_count": 123,
  "total_bytes": 456789,
  "dataset_sha256": "hex-string",
  "files": [
    {
      "relative_path": "meta/info.json",
      "size_bytes": 1234,
      "sha256": "hex-string"
    }
  ]
}
```

All files must be sorted by POSIX-style relative path before hashing the dataset
summary.

Dataset summary hash algorithm:

1. For every included file, compute file sha256.
2. Build a list of records with `relative_path`, `size_bytes`, `sha256`.
3. JSON serialize the list with `sort_keys=True` and compact separators.
4. Compute sha256 of that JSON byte string.
5. Store as `dataset_sha256`.

Use POSIX relative paths, even on macOS:

```python
relative_path = path.relative_to(dataset_root).as_posix()
```

### 6.3 Function Placement

Implement in `embodied_mixup/candidates/emix_index_utils.py` unless doing so
would create clear duplication. Do not create a new module unless the file
becomes unreasonably large.

Required public or private helpers:

```python
def compute_source_dataset_fingerprint(dataset_root: str | Path) -> dict[str, Any]:
    """Compute a strict source dataset fingerprint.

    @input: dataset_root [str|Path], exact LeRobotDataset root.
    @output: dict with algorithm, included_roots, file_count, total_bytes,
        dataset_sha256, files.
    @scenario: Bind an EmixIndex to the exact source dataset content used for
        index construction and training.
    """
```

```python
def validate_source_dataset_fingerprint(
    expected: dict[str, Any],
    dataset_root: str | Path,
) -> None:
    """Validate a source dataset root against an expected fingerprint.

    @input: expected [dict], dataset_root [str|Path].
    @output: None on exact match; raises ValueError on mismatch.
    @scenario: Stop EMIX training when the source data no longer matches the
        EmixIndex.
    """
```

Error message on mismatch must include:

```text
Source dataset fingerprint mismatch
expected dataset_sha256=...
actual dataset_sha256=...
dataset_root=...
```

Do not silently continue.

## 7. EmixIndex Schema Changes

### 7.1 Schema Version

Change:

```python
EMIX_INDEX_SCHEMA_VERSION = 1
```

to:

```python
EMIX_INDEX_SCHEMA_VERSION = 2
```

No backward compatibility is required for schema-v1 indexes.

### 7.2 Source Dataset Declaration

Keep `EmixSourceDatasetDeclaration.hash` for minimal field churn, but change its
meaning:

```text
hash == source_dataset_fingerprint["dataset_sha256"]
```

Also store the full fingerprint in:

```text
index.public.metadata["source_dataset_fingerprint"]
```

Do not add another source declaration field unless necessary.

### 7.3 Index ID

Update `make_emix_index_id` inputs so the index id includes:

```text
source_repo_id
dataset_length
source_dataset_sha256
candidate_generation_config_snapshot
```

Remove from id inputs:

```text
window_len
raw_candidate_feature_keys
```

### 7.4 Index Metadata

`public.metadata` must include:

```json
{
  "dataset_repo_id": "...",
  "dataset_root": "...",
  "source_dataset_fingerprint": {...},
  "total_frames": 0,
  "total_raw_neighbor_pairs": 0,
  "total_pairs": 0,
  "num_clusters": 0,
  "num_noise": 0,
  "total_groups": 0,
  "config": {
    "xyz_epsilon": 0.02,
    "rpy_epsilon": 0.01,
    "future_horizon": 8,
    "future_angle_thresh": 0.5235987755982988,
    "min_future_disp": 0.002,
    "gripper_action_filter_enabled": true,
    "hdbscan_min_cluster_size_ratio": 0.01,
    "hdbscan_min_samples": 3,
    "hdbscan_cluster_selection_epsilon": 0.05,
    "hdbscan_alpha": 1.0,
    "hdbscan_cluster_selection_method": "eom",
    "training_candidates_total_budget": null,
    "training_candidates_cap_per_cluster": null,
    "training_candidates_include_noise": false,
    "training_candidates_seed": 0
  }
}
```

Do not include:

```text
window_len
raw_candidate_feature_keys
```

## 8. Config Changes

File:

```text
embodied_mixup/configs/config.py
```

Remove fields:

```python
window_len: int = 16
raw_candidate_feature_keys: list[str] = field(default_factory=list)
```

Remove validation:

```python
if self.window_len <= 0:
    ...
```

Remove unused import:

```python
field
```

if no longer needed.

Keep candidate-generation fields that are still used by neighbor search,
action filtering, clustering, and training candidate metadata.

## 9. CLI Changes

File:

```text
embodied_mixup/cli.py
```

### 9.1 Remove Export Helpers

Remove:

```python
_export_materialized_lerobot_artifacts
cmd_materialize
```

Remove materialized imports caused by those functions.

### 9.2 Remove Args

From `_resolve_config(args)`, remove:

```python
window_len=args.window_len
raw_candidate_feature_keys=...
```

From `_add_index_args(parser)`, remove:

```python
--window_len
--raw_candidate_feature_keys
```

From `build_parser()`, remove the whole `materialize` subparser.

From visualize parser, remove:

```python
--export_dir
--materialized_repo_id
```

From `cmd_visualize`, remove the export block:

```python
if args.export_dir is not None:
    ...
```

### 9.3 CLI Help Must Not Mention

After the change, the following strings must not appear in CLI help output:

```text
materialize
window_len
raw_candidate_feature_keys
export_dir
materialized_repo_id
EmixFiles
```

## 10. Direct EmixIndex Training Loader

File:

```text
embodied_mixup/policies/utils/emix_loader_registry_utils.py
```

### 10.1 Rename or Replace Reader

Replace `EmixFilesReader` with:

```python
class EmixIndexReader:
    """Read EMIX samples directly from a source LeRobotDataset and EmixIndex."""
```

Do not keep `EmixFilesReader` as a compatibility alias.

### 10.2 Reader Constructor

Required constructor:

```python
def __init__(
    self,
    source_root: str | Path,
    source_repo_id: str,
    emix_index_path: str | Path,
    policy_temporal_spec: PolicyTemporalSpec,
) -> None:
```

Constructor responsibilities:

1. Resolve and validate `source_root` using existing exact LeRobot root helper.
2. Load `EmixIndex` using `load_emix_index`.
3. Validate `index.public.schema_version == EMIX_INDEX_SCHEMA_VERSION`.
4. Validate `index.public.source_datasets` has exactly one source dataset.
5. Validate source declaration repo id equals `source_repo_id`.
6. Validate source declaration root equals resolved `source_root` as a string, or
   fail. Do not silently accept a different root.
7. Validate source dataset fingerprint using
   `validate_source_dataset_fingerprint`.
8. Load source `LeRobotDataset(repo_id=source_repo_id, root=str(source_root))`.
9. Build `group_id -> group` map from `index.content.cross_point_groups`.
10. Build episode boundary maps for source dataset.
11. Fail if no groups exist.

### 10.3 Training Samples

Implement:

```python
def training_samples(self, policy_type: str) -> list[EmixTrainingSample]:
```

Behavior:

1. Iterate `index.content.cross_point_groups`.
2. Skip groups where `group.state.training` is false.
3. Require exactly two members per trainable group.
4. Create member ids deterministically without relying on exported records:

```text
{group_id}:member_0
{group_id}:member_1
```

5. `sample_id`:

```text
{group_id}:{member_0_id}:{member_1_id}
```

6. Fail if final sample list is empty.

### 10.4 Member Lookup

Because `EmixGroupMemberRef` has no `member_id`, resolve member id by suffix:

```text
member_0 -> group.members[0]
member_1 -> group.members[1]
```

Reject unknown member ids with `KeyError`.

### 10.5 Source Span Loading

Implement:

```python
def load_member_window(self, group_id: str, member_id: str) -> dict[str, Any]:
```

Even if the method name remains `load_member_window` internally, it must load
directly from source dataset frames, not from exported windows.

Span length:

```python
span_len = max(
    self.policy_temporal_spec.n_obs_steps,
    self.policy_temporal_spec.action_steps,
)
```

Start:

```python
start = member.source_frame_index
```

End:

```python
end = min(episode_end, start + span_len)
```

Rules:

1. Do not read before `source_frame_index`.
2. Do not cross episode boundary.
3. Do not use `window_len`.
4. Do not pad inside the reader.
5. Return the same payload shape currently consumed by
   `_frame_dict_from_window`:

```python
{"feature_payload": payload}
```

6. `payload[key]` is a stacked sequence for each source dataset column.
7. If source dataset frame data lacks `task` but `dataset[start]` has task-like
   access as current code assumes, preserve existing behavior of adding `task`
   when possible.
8. If the span is empty, raise `ValueError`.

### 10.6 EmixPolicyDataset Constructor

Change:

```python
def __init__(self, emix_files_root: str, policy_temporal_spec, loader_cfg)
```

to:

```python
def __init__(
    self,
    source_root: str,
    source_repo_id: str,
    emix_index_path: str,
    policy_temporal_spec: PolicyTemporalSpec,
    loader_cfg: EmixLoaderSettings,
) -> None:
```

Use:

```python
self.reader = EmixIndexReader(
    source_root=source_root,
    source_repo_id=source_repo_id,
    emix_index_path=emix_index_path,
    policy_temporal_spec=policy_temporal_spec,
)
```

### 10.7 Existing Batch Logic

Keep existing batch behavior unless a test proves it is wrong:

- image mixup behavior
- state selection behavior
- `observation.images` stacking behavior
- `action`
- `emix_actions`
- `emix_alpha`
- `emix_mask`
- policy-specific action pad key
- `action_is_pad`
- `sample_id`
- `group_id`
- `member_ids`
- `task` for SmolVLA

The existing `_resolve_obs_indices` and `_resolve_action_indices` already handle
short spans by repeating the last valid frame and marking padded action steps.
Reuse that behavior.

## 11. Loader Registry Changes

File:

```text
embodied_mixup/policies/utils/emix_loader_registry.py
```

Change:

```python
def make_emix_loader(policy_type, emix_files_root, policy_temporal_spec, loader_cfg)
```

to:

```python
def make_emix_loader(
    policy_type: str,
    source_root: str,
    source_repo_id: str,
    emix_index_path: str,
    policy_temporal_spec: PolicyTemporalSpec,
    loader_cfg,
) -> EmixPolicyDataset:
```

Return:

```python
return EmixPolicyDataset(
    source_root,
    source_repo_id,
    emix_index_path,
    policy_temporal_spec,
    settings,
)
```

Update module docstring from `EmixFiles` roots to `EmixIndex` direct loading.

## 12. EMIX YAML Schema Changes

File:

```text
embodied_mixup/policies/utils/emix_config.py
```

### 12.1 Required New Key

Require:

```text
emix.index_path
```

### 12.2 Rejected Old Key

If `emix.files_root` exists, raise:

```text
emix.files_root is no longer supported. Use emix.index_path and dataset.root.
```

### 12.3 Policy Config Attribute

Set:

```python
policy_cfg.emix_index_path = emix_payload["index_path"]
```

Do not set:

```python
policy_cfg.emix_files_root
```

Update docstrings accordingly.

## 13. Policy Config Dataclass Changes

Files:

```text
embodied_mixup/policies/DP/cfg_emix_diffusion.py
embodied_mixup/policies/ACT/cfg_emix_act.py
embodied_mixup/policies/SmolVLA/cfg_emix_smolvla.py
```

Change field:

```python
emix_files_root: str = ""
```

to:

```python
emix_index_path: str = ""
```

Do not keep both fields.

## 14. Training Wiring Changes

File:

```text
embodied_mixup/policies/train.py
```

### 14.1 Legacy Policy Key Rejection

Update `_reject_legacy_policy_emix_keys` to include:

```text
emix_files_root
```

Keep rejecting legacy policy-level mixup keys.

Do not add `emix_index_path` under `policy`; it must live under top-level
`emix.index_path`.

### 14.2 EMIX Dataset Construction

In `_create_dataset_and_policy`, for `training_mode == "emix"`, change
`make_emix_loader` call to pass:

```python
source_root=dataset_root
source_repo_id=cfg.dataset_repo_id
emix_index_path=cfg.policy_cfg.emix_index_path
```

In `_build_emix_dataset`, pass the same values:

```python
source_root=str(require_lerobot_dataset_root(cfg.dataset_root))
source_repo_id=cfg.dataset_repo_id
emix_index_path=cfg.policy_cfg.emix_index_path
```

Remove all references to:

```text
cfg.policy_cfg.emix_files_root
```

## 15. Remove Active Materialized/EmixFiles Exports

Files:

```text
embodied_mixup/candidates/__init__.py
embodied_mixup/__init__.py
```

Remove imports and `__all__` entries for:

```text
EmixFiles
EmixFilesPublic
EmixFilesContent
EmixFilesGroup
EmixWindowRecord
EmixMaterializedIndex
EmixMaterializedIndexPublic
EmixMaterializedIndexContent
EmixMaterializedCrossPointGroup
EmixMaterializedMemberRecord
export_materialized_lerobot_dataset
load_materialized_emix_index
validate_materialized_lerobot_root
```

If the implementation agent deletes files, create timestamped backups first:

```text
embodied_mixup/candidates/emix_files.py.bak_<YYYYMMDDHHMMSS>
embodied_mixup/candidates/emix_files_utils.py.bak_<YYYYMMDDHHMMSS>
embodied_mixup/candidates/materialized_lerobot.py.bak_<YYYYMMDDHHMMSS>
embodied_mixup/candidates/materialized_lerobot_utils.py.bak_<YYYYMMDDHHMMSS>
```

Deletion is optional if removing imports and active references is enough. The
final active code path must not import or call them.

## 16. Remove Remaining `window_len` Active Uses

Search command:

```bash
rg -n "window_len|--window_len|raw_candidate_feature_keys|--raw_candidate_feature_keys" embodied_mixup configs tests docs
```

Required final state:

- No active CLI/config/training docs mention `window_len`.
- No active config dataclass field named `window_len`.
- No index metadata includes `window_len`.
- No direct training loader uses `window_len`.

Allowed final occurrences only if clearly quarantined in backup files or removed
legacy files outside active imports. Prefer no occurrences in tracked active
files.

## 17. Config YAML Updates

Files:

```text
configs/policies/emix_mixed_diffusion_stack_cube_eef_base_nonoop.yaml
configs/policies/template_emix_mixed_dp1_stack_cube.yaml
configs/policies/template_emix_mixed_smolvla_stack_cube.yaml
```

Change comments and fields:

From:

```yaml
emix:
  enabled: true
  files_root: data/emix_files/<materialized_lerobot_root>
```

To:

```yaml
emix:
  enabled: true
  index_path: data/emix_data/<repo_id>/emix_index.json
```

Do not mention materialized roots in these YAML files.

## 18. Documentation Updates

Update at least:

```text
docs/sop/SOP-02-intersection-clustering-visualization.md
docs/sop/SOP-03-model-training.md
docs/sop/data-structure-and-transforms.md
docs/DEV.md
```

### 18.1 SOP-02 Final Message

SOP-02 must say:

- Build `EmixIndex`.
- Visualize/tune `EmixIndex`.
- Use `EmixIndex` directly for training.
- No `materialize`.
- No `EmixFiles`.
- No fixed `window_len`.
- `EmixIndex` contains a strict source dataset fingerprint.

### 18.2 SOP-03 Final Message

SOP-03 must say:

- EMIX training requires original `dataset.root`.
- EMIX training requires `emix.index_path`.
- Training validates source dataset fingerprint before constructing samples.
- `emix.files_root` is invalid.
- Materialized EMIX dataset roots are invalid.

### 18.3 DEV Final Message

DEV must describe active architecture:

```text
source LeRobotDataset -> EmixIndex -> direct EMIX training loader
```

Do not describe a file materialization layer as active.

## 19. Tests

### 19.1 Update tests/test_emix_artifacts.py

Remove tests that assert materialized export behavior:

- materialized index round trip
- export materialized dataset writes sidecar
- policy loader uses materialized root

Add tests:

1. `test_source_dataset_fingerprint_is_stable`
   - Create temp root with `meta/info.json`, `data/chunk-000/file-000.parquet`
     or simple dummy file if no parquet dependency is used, and optional
     `videos/...mp4`.
   - Compute fingerprint twice.
   - Assert same `dataset_sha256`.

2. `test_source_dataset_fingerprint_changes_on_content_change`
   - Compute fingerprint.
   - Modify one included file.
   - Compute again.
   - Assert `dataset_sha256` changed.

3. `test_validate_source_dataset_fingerprint_rejects_mismatch`
   - Use expected fingerprint.
   - Modify file.
   - Assert `ValueError` includes
     `Source dataset fingerprint mismatch`.

4. `test_emix_index_round_trip_schema_v2`
   - Build minimal `EmixIndex` with schema version 2.
   - Include full fingerprint metadata.
   - Save/load.
   - Assert `hash == dataset_sha256`.

5. `test_emix_index_reader_rejects_fingerprint_mismatch`
   - Mock LeRobotDataset class if needed.
   - Build index with old fingerprint.
   - Modify source file.
   - Construct reader.
   - Assert `ValueError`.

6. `test_emix_index_reader_loads_source_span_without_crossing_episode`
   - Mock dataset with two episodes.
   - Member near episode end.
   - Assert loaded span stays inside episode.

### 19.2 Update tests/test_mix_training.py

Replace patches:

```text
EmixFilesReader
```

with:

```text
EmixIndexReader
```

Update config payloads from:

```yaml
emix:
  files_root: ...
```

to:

```yaml
emix:
  index_path: ...
```

Add rejection test:

```yaml
emix:
  files_root: legacy
```

Expected error:

```text
emix.files_root is no longer supported
```

### 19.3 CLI Tests or Smoke Commands

At minimum run:

```bash
python -m embodied_mixup --help
python -m embodied_mixup build --help
python -m embodied_mixup visualize --help
python -m embodied_mixup tune --help
```

Assert output does not include removed terms:

```text
materialize
window_len
raw_candidate_feature_keys
export_dir
materialized_repo_id
EmixFiles
```

## 20. Validation Commands

Use the active virtual environment for all Python commands. Do not install
packages globally.

Recommended commands:

```bash
python -m pytest tests/test_emix_artifacts.py tests/test_mix_training.py
```

If dependencies are missing and installation is required, ask for the target
venv/conda path first. Do not install into global or user Python.

Static search after implementation:

```bash
rg -n "EmixFiles|emix_files_root|files_root|materialize|materialized|window_len|raw_candidate_feature_keys" embodied_mixup configs docs tests
```

Expected:

- No active code references to `emix_files_root`.
- No active YAML uses `files_root`.
- No CLI docs mention `materialize`.
- No active code path imports materialized exporter.
- Some historical docs may still mention old terms only if explicitly marked
  as removed/deprecated. Prefer removing active SOP references entirely.

## 21. Error Handling Requirements

Use explicit errors. Do not infer or fallback.

Required failure cases:

1. `emix.index_path` missing for EMIX policy.
2. `emix.files_root` present.
3. `policy.emix_files_root` present.
4. `EmixIndex` file missing.
5. `EmixIndex` schema version unsupported.
6. `EmixIndex` has zero groups.
7. `EmixIndex` has zero trainable groups.
8. Trainable group has member count other than two.
9. Source dataset root does not match source declaration root.
10. Source repo id does not match source declaration repo id.
11. Source dataset fingerprint mismatch.
12. Member frame index outside source dataset length.
13. Member frame span is empty.

## 22. Logging Requirements

Keep existing logging style. Do not introduce a new logging framework.

Useful log points:

```text
[candidates.index] Computing source dataset fingerprint
[candidates.index] Source dataset fingerprint complete: files=... bytes=... sha256=...
[emix.loader] Loading EmixIndex: ...
[emix.loader] Validating source dataset fingerprint
[emix.loader] Source dataset fingerprint validated
[emix.loader] Loaded direct EMIX samples: groups=... samples=...
```

Use standard `logging`.

## 23. Backup Requirements For Destructive Changes

Before deleting any tracked file or overwriting a file in a destructive way,
create a timestamped backup:

```text
<file>.bak_<YYYYMMDDHHMMSS>
```

This applies especially to:

```text
embodied_mixup/candidates/emix_files.py
embodied_mixup/candidates/emix_files_utils.py
embodied_mixup/candidates/materialized_lerobot.py
embodied_mixup/candidates/materialized_lerobot_utils.py
```

If the implementation chooses to keep these files but remove active imports and
references, backups are not required for non-destructive edits.

## 24. Acceptance Criteria

Implementation is complete only when all are true:

1. `python -m embodied_mixup --help` has no `materialize` command.
2. `build`, `visualize`, and `tune` help have no `window_len` parameter.
3. EMIX YAML uses `emix.index_path`, not `emix.files_root`.
4. Training code passes source dataset root and index path to direct EMIX loader.
5. Direct EMIX loader reads from original `LeRobotDataset`.
6. Direct EMIX loader validates strict source dataset fingerprint.
7. `EmixIndex` schema is v2 and records full fingerprint metadata.
8. `EmixIndex` id changes when source dataset content changes.
9. No active code path writes EMIX materialized datasets.
10. No active code path writes or reads `EmixFiles`.
11. Tests cover fingerprint stability, mismatch rejection, index round-trip, and
    direct source span loading.
12. SOP-02 no longer instructs users to run `materialize`.
13. SOP-03 no longer instructs users to set `policy.emix_files_root` or
    `emix.files_root`.

## 25. Suggested File Edit Checklist

Use this as an execution checklist:

```text
[ ] embodied_mixup/candidates/emix_index.py
[ ] embodied_mixup/candidates/emix_index_utils.py
[ ] embodied_mixup/candidates/ids.py
[ ] embodied_mixup/configs/config.py
[ ] embodied_mixup/cli.py
[ ] embodied_mixup/policies/utils/emix_loader_registry_utils.py
[ ] embodied_mixup/policies/utils/emix_loader_registry.py
[ ] embodied_mixup/policies/utils/emix_config.py
[ ] embodied_mixup/policies/DP/cfg_emix_diffusion.py
[ ] embodied_mixup/policies/ACT/cfg_emix_act.py
[ ] embodied_mixup/policies/SmolVLA/cfg_emix_smolvla.py
[ ] embodied_mixup/policies/train.py
[ ] embodied_mixup/candidates/__init__.py
[ ] embodied_mixup/__init__.py
[ ] configs/policies/emix_mixed_diffusion_stack_cube_eef_base_nonoop.yaml
[ ] configs/policies/template_emix_mixed_dp1_stack_cube.yaml
[ ] configs/policies/template_emix_mixed_smolvla_stack_cube.yaml
[ ] tests/test_emix_artifacts.py
[ ] tests/test_mix_training.py
[ ] docs/sop/SOP-02-intersection-clustering-visualization.md
[ ] docs/sop/SOP-03-model-training.md
[ ] docs/sop/data-structure-and-transforms.md
[ ] docs/DEV.md
```

## 26. Critical Do-Not-Do List

Do not:

- Do not keep `EmixFilesReader` as an alias.
- Do not accept `emix.files_root`.
- Do not accept materialized LeRobotDataset roots for EMIX training.
- Do not compute fingerprint only from `meta/info.json`.
- Do not use md5 for strict source fingerprint.
- Do not store image arrays in JSON.
- Do not add a new wrapper abstraction around loaders.
- Do not add a registry/factory for source fingerprinting.
- Do not keep `window_len` as a hidden default.
- Do not silently rebuild the index if fingerprint validation fails.
- Do not silently skip invalid groups during loader initialization if they are
  marked training and malformed.
- Do not mutate source datasets.
- Do not install dependencies globally.
