"""Load compatible local LeRobot datasets as one SmolVLA training source."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from embodied_mixup.datasets import require_lerobot_dataset_root


def validate_sources(root: str | Path, repo_ids: list[str]) -> list[Path]:
    """Require distinct local dataset roots under one parent directory."""
    parent = Path(root).expanduser().resolve()
    if not parent.is_dir() or not repo_ids or len(set(repo_ids)) != len(repo_ids):
        raise ValueError("dataset.root must exist and dataset.repo_ids must be distinct and nonempty")
    if any(not isinstance(repo_id, str) or not repo_id or "/" in repo_id or repo_id in {".", ".."}
           for repo_id in repo_ids):
        raise ValueError("dataset.repo_ids must contain local child directory names")
    return [require_lerobot_dataset_root(parent / repo_id) for repo_id in repo_ids]


def load_multi_dataset(root: str, repo_ids: list[str], policy_cfg: Any, video_backend: str | None):
    """Return aggregate metadata and a LeRobot multi-dataset without copying videos."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
    from embodied_mixup.policies.train import _build_delta_timestamps

    roots = validate_sources(root, repo_ids)
    metas = [LeRobotDatasetMetadata(repo_id=repo_id, root=source)
             for repo_id, source in zip(repo_ids, roots, strict=True)]
    canonical_features = metas[0].features
    fps = float(metas[0].info["fps"])
    for repo_id, meta in zip(repo_ids[1:], metas[1:], strict=True):
        if meta.features != canonical_features or float(meta.info["fps"]) != fps:
            raise ValueError(f"Incompatible features or fps in multi-task dataset {repo_id}")
    delta_timestamps = {key: values for key, values in _build_delta_timestamps(policy_cfg, canonical_features, fps).items() if values}
    dataset = MultiLeRobotDataset(
        repo_ids=repo_ids,
        root=str(Path(root).expanduser().resolve()),
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )
    metadata = SimpleNamespace(features=canonical_features, stats=dataset.stats, info=metas[0].info)
    return metadata, dataset


def write_dataset_stats(path: Path, stats: dict[str, Any]) -> None:
    """Persist pooled normalization statistics next to every checkpoint."""
    payload = {key: {name: np.asarray(value).tolist() for name, value in fields.items()}
               for key, fields in stats.items()}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def read_dataset_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: {name: np.asarray(value) for name, value in fields.items()}
            for key, fields in payload.items()}
