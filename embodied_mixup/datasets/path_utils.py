"""Small helpers for LeRobot dataset path validation.

@input: root [str|Path] dataset directory candidates.
@output: validated Path or derived repo id string.
@scenario: Keep consumer commands strict about exact dataset roots while
           providing a stable default repo_id for producer commands.
"""

from __future__ import annotations

from pathlib import Path


def require_lerobot_dataset_root(root: str | Path) -> Path:
    """Validate that `root` points to an exact LeRobot dataset directory."""

    dataset_root = Path(root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            "The root argument must point to the exact LeRobot dataset root. "
            f"Missing file: {info_path}"
        )
    return dataset_root


def default_repo_id_from_dataset_root(root: str | Path) -> str:
    """Derive a repo_id from the final dataset directory name."""

    dataset_root = Path(root).expanduser()
    repo_id = dataset_root.name
    if not repo_id:
        raise ValueError("Cannot derive repo_id from an empty dataset root name.")
    return repo_id
