"""A mixed within-task and cross-task EMIX training view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset

from .emix_loader_registry import make_emix_loader
from .emix_loader_registry_utils import EmixLoaderSettings


class MultiTaskEmixDataset(ConcatDataset):
    """Combine validated source indexes with explicit cross-task pairs."""

    def __init__(self, manifest_path: str | Path, repo_ids: list[str], policy_type: str,
                 temporal_spec: Any, settings: EmixLoaderSettings, dataset_root: str | None = None):
        path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("pair_scope") != "within_and_cross_task":
            raise ValueError("Unsupported multi-task index manifest")
        sources = payload.get("sources")
        if not isinstance(sources, list) or [source.get("repo_id") for source in sources] != repo_ids:
            raise ValueError("Multi-task index source order must match dataset.repo_ids")
        datasets = []
        for source in sources:
            root = str(Path(source["root"]).expanduser().resolve())
            if dataset_root is not None and Path(root) != (Path(dataset_root).expanduser().resolve() / source["repo_id"]):
                raise ValueError(f"Index source root mismatch for {source['repo_id']}")
            index_path = str(Path(source["index_path"]).expanduser().resolve())
            annotation_root = str(Path(source["annotation_root"]).expanduser().resolve())
            source_settings = EmixLoaderSettings(
                **{**vars(settings), "annotation_root": annotation_root}
            )
            datasets.append(make_emix_loader(
                policy_type=policy_type,
                source_root=root,
                source_repo_id=source["repo_id"],
                emix_index_path=index_path,
                policy_temporal_spec=temporal_spec,
                loader_cfg=source_settings,
            ))
        from .multi_cross_loader import CrossTaskEmixDataset
        self.cross_dataset = CrossTaskEmixDataset(
            payload["cross_pairs_path"], sources, temporal_spec, settings)
        self.repo_ids = list(repo_ids)
        super().__init__(datasets)
        self.within_size = self.cumulative_sizes[-1]

    def __len__(self):
        return 4 * ((self.within_size + 2) // 3)

    def __getitem__(self, idx):
        from bisect import bisect_right
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if idx % 4 == 0:
            return self.cross_dataset[(idx // 4) % len(self.cross_dataset)]
        within_index = (idx - idx // 4 - 1) % self.within_size
        dataset_index = bisect_right(self.cumulative_sizes, within_index)
        item = dict(super().__getitem__(within_index))
        for auxiliary_key in ("observation.eef_pose", "observation.left_gripper", "emix_eef_poses"):
            item.pop(auxiliary_key, None)
        repo_id = self.repo_ids[dataset_index]
        for key in ("sample_id", "group_id"):
            if key in item:
                item[key] = f"{repo_id}:{item[key]}"
        return item


def write_manifest(path: Path, repo_ids: list[str], dataset_root: Path, index_root: Path) -> None:
    """Reference ten validated indexes and build a cross-task pair index."""
    from .multi_dataset import validate_sources

    roots = validate_sources(dataset_root, repo_ids)
    sources = []
    for repo_id, root in zip(repo_ids, roots, strict=True):
        index_path = index_root / repo_id / "emix_index.json"
        annotations = root / "annotations"
        if not index_path.is_file() or not annotations.is_dir():
            raise FileNotFoundError(f"Missing index or annotations for {repo_id}")
        sources.append({"repo_id": repo_id, "root": str(root),
                        "index_path": str(index_path.resolve()),
                        "annotation_root": str(annotations.resolve())})
    cross_path = path.parent / "cross_task_pairs.json"
    if not cross_path.is_file():
        from .multi_cross_index import build_cross_pairs
        build_cross_pairs(cross_path, repo_ids, dataset_root)
    cross = json.loads(cross_path.read_text(encoding="utf-8"))
    if (cross.get("source_repo_ids") != repo_ids or cross.get("pair_scope") != "cross_task"
            or not cross.get("pairs")):
        raise ValueError("Cross-task index does not match source order or has no pairs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "pair_scope": "within_and_cross_task",
                                "cross_pairs_path": str(cross_path.resolve()),
                                "cross_fraction": 0.25, "cross_pair_count": len(cross["pairs"]),
                                "sources": sources}, indent=2), encoding="utf-8")
