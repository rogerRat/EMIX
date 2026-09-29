# Minimal Dataset Root and Repo ID Consistency Plan

## Purpose

This document defines the minimal implementation plan for handling the two dataset identifiers used in this repository:

- `root`: the exact local LeRobot dataset directory.
- `repo_id`: the logical dataset identifier passed to LeRobot APIs and stored in downstream artifacts.

The goal is to avoid guessing `repo_id` from paths during later processing, while keeping the code change small and compatible with the current pipeline.

## Design Decision

Use an explicit-call contract:

- Every consumer command must receive both `root` and `repo_id`.
- `root` must point to the exact dataset directory containing `meta/info.json`.
- `repo_id` must be the logical repo id used to load the dataset.
- Code must not infer `repo_id` from `root` during consumption.

Use a simple production default:

- For commands that create a new dataset, `target_repo_id` may be omitted.
- If `target_repo_id` is omitted, derive it from the target dataset folder name.
- The derived `target_repo_id` must be logged and used consistently when creating the output LeRobot dataset.

Do not add a new identity schema, sidecar file, or complex resolver in this change.

## Non-Goals

Do not implement the following in this minimal change:

- No new `DatasetIdentity` dataclass.
- No new metadata sidecar such as `meta/emix_identity.json`.
- No schema migration for existing `emix_index.json`.
- No automatic repo id discovery from parent directories.
- No rename or migration of existing dataset folders.
- No automatic modification of existing datasets.
- No broad refactor of training, EMIX files, or visualization code.

## Required Behavior

### Consumer Commands

Consumer commands are commands that read an existing dataset. Examples:

- `python -m embodied_mixup build`
- `python -m embodied_mixup materialize`
- `python -m embodied_mixup visualize`
- `python -m embodied_mixup tune`
- training scripts that load a LeRobot dataset
- evaluation scripts that load a LeRobot dataset
- visualization scripts that load a LeRobot dataset
- filtering/subsetting scripts when reading the source dataset

For all consumer commands:

1. Require both `--root` and `--repo_id`.
2. Treat `--root` as the exact local dataset directory.
3. Check that `<root>/meta/info.json` exists before calling `LeRobotDataset`.
4. Do not guess `repo_id` from `root`.
5. Do not silently replace the user-provided `repo_id`.
6. Use the exact user-provided `repo_id` when calling `LeRobotDataset(repo_id=..., root=...)`.
7. Log both values before loading:

```text
[module] [INFO] Loading LeRobot dataset: repo_id=<repo_id> root=<root>
```

### Producer Commands

Producer commands are commands that create a new dataset. Examples:

- `data_processing/convert_to_lerobot_v2.py`
- `data_processing/filter_nonoop_lerobot.py`
- `data_processing/sample_subset_lerobot.py`

For all producer commands:

1. Source dataset inputs must still require both source `--root` and source `--repo_id`.
2. Target dataset output must have a target root or output path that determines the final dataset folder.
3. `--target_repo_id` should be optional where practical.
4. If `--target_repo_id` is omitted, set it to the target dataset folder name.
5. If `--target_repo_id` is provided, use it exactly.
6. Log the final target values before creating the dataset:

```text
[module] [INFO] Creating target LeRobot dataset: target_repo_id=<target_repo_id> target_root=<target_root>
```

## Target Repo ID Derivation Rule

Use this helper rule wherever a producer command needs a default `target_repo_id`:

```python
def default_repo_id_from_dataset_root(dataset_root: str | Path) -> str:
    root = Path(dataset_root).expanduser()
    repo_id = root.name
    if not repo_id:
        raise ValueError("Cannot derive repo_id from an empty dataset root name.")
    return repo_id
```

Important:

- Derive from the final dataset directory name, not the parent directory.
- Do not strip semantic suffixes.
- Do not normalize beyond basic path expansion.
- Do not inspect metadata to derive this value.

Examples:

```text
target_root=data/lerobot/maniskill/stack_cube_v1
default target_repo_id=stack_cube_v1

target_root=/mnt/datasets/renamed_local_folder
default target_repo_id=renamed_local_folder
```

## Minimal Helper Module

Create a small helper module only if it avoids duplicate logic:

```text
embodied_mixup/datasets/path_utils.py
```

Suggested contents:

```python
from pathlib import Path


def require_lerobot_dataset_root(root: str | Path) -> Path:
    dataset_root = Path(root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            "The root argument must point to the exact LeRobot dataset root. "
            f"Missing file: {info_path}"
        )
    return dataset_root


def default_repo_id_from_dataset_root(root: str | Path) -> str:
    dataset_root = Path(root).expanduser()
    repo_id = dataset_root.name
    if not repo_id:
        raise ValueError("Cannot derive repo_id from an empty dataset root name.")
    return repo_id
```

This module must not read or modify dataset metadata.

## Implementation Steps

### Step 1: Add Small Path Helper

Add:

```text
embodied_mixup/datasets/__init__.py
embodied_mixup/datasets/path_utils.py
```

Keep the module focused on path checks and default repo id derivation only.

### Step 2: Update EMIX CLI Loading

File:

```text
embodied_mixup/cli.py
```

Current function:

```python
def _load_dataset(repo_id: str, root: str):
```

Keep the same signature.

Required changes:

1. Use `require_lerobot_dataset_root(root)` to validate `root`.
2. Keep `repo_id = repo_id.rstrip("/")`.
3. Do not infer or rewrite `repo_id` from `root`.
4. Log both `repo_id` and resolved `dataset_root`.
5. Call:

```python
dataset = LeRobotDataset(repo_id, root=str(dataset_root))
```

Expected result:

- Existing EMIX commands keep the same CLI interface.
- Bad `root` fails early.
- Wrong `repo_id` is not guessed or corrected.

### Step 3: Ensure EMIX Output Uses Explicit Repo ID

File:

```text
embodied_mixup/cli.py
```

Keep:

```python
_default_emix_output_dir(args.repo_id)
```

Do not change this to use folder names.

Rationale:

- Consumer calls must be explicit.
- Output artifacts are keyed by the repo id the user explicitly requested.

### Step 4: Update Convert Script Target Default

File:

```text
data_processing/convert_to_lerobot_v2.py
```

Current behavior:

- `--repo_id` is required.
- It is used to create the output dataset.

Minimal change:

- Keep `--repo_id` required if this script does not already expose a distinct target root.
- If the script has or adds an output dataset root argument, allow `--repo_id` to be omitted and default it from that output dataset root folder name.

Implementation rule:

```python
repo_id = args.repo_id
if repo_id is None:
    repo_id = default_repo_id_from_dataset_root(args.root)
```

Log:

```text
[convert_to_lerobot_v2] [INFO] Creating target LeRobot dataset: target_repo_id=<repo_id> target_root=<root>
```

Do not add metadata sidecar files.

### Step 5: Update Filter Script Source and Target Handling

File:

```text
data_processing/filter_nonoop_lerobot.py
```

Source side:

- Keep `--repo_id` required.
- Keep `--root` or equivalent source root required.
- Validate source root with `require_lerobot_dataset_root`.
- Load source using exact `args.repo_id`.

Target side:

- Make `--target_repo_id` optional if a target dataset root is available.
- If `--target_repo_id` is omitted, derive it from `target_root`.
- If `--target_root` is omitted but the script only has `target_repo_id`, keep `--target_repo_id` required.

Required logic:

```python
target_repo_id = args.target_repo_id
if target_repo_id is None:
    if args.target_root is None:
        raise ValueError("--target_repo_id is required when --target_root is not provided.")
    target_repo_id = default_repo_id_from_dataset_root(args.target_root)
```

Log source and target identities before loading/creating.

### Step 6: Update Sample Subset Script Source and Target Handling

File:

```text
data_processing/sample_subset_lerobot.py
```

Apply the same rule as Step 5:

- Source dataset requires explicit `--root` and `--repo_id`.
- Target repo id defaults to target folder name only when target root is provided.
- No metadata sidecar.
- No source repo id guessing.

### Step 7: Update Other Direct Dataset Loaders Only If Needed

Search for direct uses of:

```text
LeRobotDataset(
LeRobotDataset.create(
```

For each direct consumer load:

- Ensure there is an explicit `root`.
- Ensure there is an explicit `repo_id`.
- Ensure `root/meta/info.json` is checked before loading.

Known files to inspect:

```text
visualization/draw_trajectories.py
evaluation/policies/lerobot.py
embodied_mixup/policies/train.py
data_processing/remove_feature_lerobot.py
```

Keep changes minimal:

- Do not redesign config formats unless necessary.
- If a YAML config already has `dataset.root` and `dataset.repo_id`, preserve it.
- Add early validation only.

### Step 8: Update Documentation Examples

Update only the relevant command examples in:

```text
README.md
docs/sop/SOP-01-data-generation-pipeline.md
docs/sop/SOP-02-intersection-clustering-visualization.md
docs/sop/SOP-03-model-training.md
docs/sop/data-structure-and-transforms.md
```

Minimal wording:

```text
All consumer commands require both --root and --repo_id. The root argument is the exact LeRobot dataset directory. The repo_id argument is never inferred from the folder name.
```

For producer examples:

```text
When --target_repo_id is omitted and --target_root is provided, the target repo id defaults to the target folder name.
```

Do not introduce a new identity model in the docs.

## Validation Checklist

After implementation, run these checks.

### Static Search

Command:

```bash
rg -n "LeRobotDataset\\(|LeRobotDataset.create\\(" .
```

Confirm every consumer load uses explicit `root` and `repo_id`.

### EMIX CLI Root Validation

Expected failure:

```bash
python -m embodied_mixup build --repo_id stack_cube --root /tmp/not_a_dataset
```

Expected behavior:

- Fails before calling `LeRobotDataset`.
- Error mentions missing `/tmp/not_a_dataset/meta/info.json`.

### No Repo ID Guessing In Consumers

Use:

```bash
python -m embodied_mixup build --repo_id explicit_id --root data/lerobot/maniskill/different_folder_name
```

Expected behavior:

- Logs `repo_id=explicit_id`.
- Does not replace it with `different_folder_name`.
- Output defaults to `data/emix_data/explicit_id` unless `--output_dir` is provided.

### Producer Default Target Repo ID

Use a producer command with:

```text
--target_root data/lerobot/maniskill/new_dataset_folder
```

and without:

```text
--target_repo_id
```

Expected behavior:

- Derived `target_repo_id` is `new_dataset_folder`.
- Log states the derived value.
- `LeRobotDataset.create` receives `repo_id="new_dataset_folder"`.

## Acceptance Criteria

The change is complete when:

1. Every dataset consumer command requires explicit `root` and `repo_id`.
2. No consumer code guesses `repo_id` from a dataset folder name.
3. Producer code can default `target_repo_id` from the target folder name when `target_root` is available.
4. Existing commands with explicit `--repo_id` keep working.
5. Invalid roots fail before LeRobot loading.
6. Documentation states the explicit `root + repo_id` contract.
7. No new identity schema, metadata sidecar, or migration system is introduced.

## Risk Notes

Potential issue:

- Some producer scripts may not have a clear `target_root`.

Required handling:

- Do not invent a path layout.
- Keep `--target_repo_id` required for those scripts.
- Add the default only where the target dataset folder is explicit.

Potential issue:

- Some LeRobot versions interpret `root` as the dataset root, while others may expect a cache root.

Required handling:

- Preserve the current call shape where it already works.
- Only add validation around the existing call.
- Do not change the broader LeRobot storage layout in this task.

