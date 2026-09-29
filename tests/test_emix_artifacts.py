"""Unit tests for canonical EMIX artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from embodied_mixup.candidates.emix_index import (
    EMIX_DATASET_VERSION,
    EMIX_INDEX_SCHEMA_VERSION,
    EmixCrossPointGroup,
    EmixGroupMemberRef,
    EmixGroupState,
    EmixIndex,
    EmixIndexContent,
    EmixIndexPublic,
    EmixSourceDatasetDeclaration,
    EmixSourceMediaContract,
)
from embodied_mixup.candidates.materialized_lerobot import (
    EMIX_MATERIALIZED_INDEX_SCHEMA_VERSION,
    EmixMaterializedCrossPointGroup,
    EmixMaterializedDatasetDeclaration,
    EmixMaterializedGroupState,
    EmixMaterializedIndex,
    EmixMaterializedIndexContent,
    EmixMaterializedIndexPublic,
    EmixMaterializedMemberRecord,
)
from embodied_mixup.candidates.materialized_lerobot_utils import (
    export_materialized_lerobot_dataset,
    load_materialized_emix_index,
    validate_materialized_lerobot_root,
)
from embodied_mixup.policies.utils.batch import PolicyTemporalSpec
from embodied_mixup.policies.utils.emix_loader_registry import make_emix_loader
from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixLoaderSettings


class FakeExportLeRobotDataset:
    """Small stand-in for LeRobotDataset.create used by unit tests."""

    def __init__(self, repo_id: str, root: str | Path, fps: int, features: dict, **kwargs) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        self.fps = fps
        self.features = features
        self._episode_index = 0
        self._buffer: list[dict[str, object]] = []
        self.meta = type("Meta", (), {})()
        self.meta.info = {
            "fps": fps,
            "total_episodes": 0,
            "total_frames": 0,
            "features": features,
        }
        self.meta.episodes = {}
        self.episode_data_index = {"from": [], "to": []}
        self.hf_dataset = type("HF", (), {"column_names": list(features.keys())})()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)
        self._write_info()

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        payload = dict(frame)
        payload["task"] = task
        self._buffer.append(payload)

    def save_episode(self, episode_data: dict | None = None) -> None:
        episode_index = self._episode_index
        episode_length = len(self._buffer)
        episode_dir = self.root / "data" / "chunk-000"
        episode_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = episode_dir / f"episode_{episode_index:06d}.parquet"
        parquet_path.write_text(
            json.dumps(self._serialize(self._buffer), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        video_dir = self.root / "videos" / "chunk-000" / "observation.images.base_camera"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(b"mp4")
        self.meta.episodes[episode_index] = {"episode_index": episode_index, "length": episode_length}
        self.episode_data_index["from"].append(0 if episode_index == 0 else self.episode_data_index["to"][-1])
        self.episode_data_index["to"].append(self.episode_data_index["from"][-1] + episode_length)
        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length
        self._write_info()
        self._write_episodes_jsonl()
        self._episode_index += 1
        self._buffer = []

    def stop_image_writer(self) -> None:
        return None

    @property
    def num_episodes(self) -> int:
        return self.meta.info["total_episodes"]

    def _write_info(self) -> None:
        (self.root / "meta" / "info.json").write_text(
            json.dumps(self.meta.info, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_episodes_jsonl(self) -> None:
        lines = [
            json.dumps({"episode_index": idx, "length": ep["length"], "tasks": ["pick cube"]})
            for idx, ep in sorted(self.meta.episodes.items())
        ]
        (self.root / "meta" / "episodes.jsonl").write_text("\n".join(lines), encoding="utf-8")

    def _serialize(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._serialize(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._serialize(item) for item in value]
        return value


class FakeMaterializedReader:
    """Mock reader for loader tests."""

    def __init__(self, emix_files_root: str | Path) -> None:
        self.root = Path(emix_files_root)
        self.group = type(
            "Group",
            (),
            {
                "group_id": "emix_group_test",
                "state": type("State", (), {"training": True})(),
                "members": [
                    type("Member", (), {"member_id": "emix_group_test_anchor", "export_episode_index": 0})(),
                    type("Member", (), {"member_id": "emix_group_test_neighbor", "export_episode_index": 1})(),
                ],
            },
        )()
        self.dataset = type("Dataset", (), {"__getitem__": lambda self, idx: {"task": "pick cube"}})()

    def training_samples(self, policy_type: str):
        return [
            type(
                "Record",
                (),
                {
                    "group": self.group,
                    "member_ids": ["emix_group_test_anchor", "emix_group_test_neighbor"],
                    "sample_id": "emix_group_test:emix_group_test_anchor:emix_group_test_neighbor",
                },
            )()
        ]

    def load_member_window(self, group_id: str, member_id: str) -> dict[str, object]:
        return {
            "feature_payload": {
                "action": np.asarray(
                    [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]], dtype=np.float32
                ),
                "observation.state": np.asarray(
                    [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]] * 3, dtype=np.float32
                ),
                "observation.images.base_camera": np.asarray(
                    [np.zeros((4, 4, 3), dtype=np.uint8)] * 3, dtype=np.uint8
                ),
                "task": ["pick cube", "pick cube", "pick cube"],
            }
        }

    def load_group(self, group_id: str):
        return self.group


class TestEmixArtifacts(unittest.TestCase):
    def test_materialized_index_round_trip(self) -> None:
        index = EmixMaterializedIndex(
            public=EmixMaterializedIndexPublic(
                id="emix_materialized_index_test",
                schema_version=EMIX_MATERIALIZED_INDEX_SCHEMA_VERSION,
                created_at="2026-07-01T00:00:00+0800",
                source_dataset=EmixSourceDatasetDeclaration(
                    repo_id="demo_repo",
                    root="/tmp/demo",
                    hash="abc123",
                    media=EmixSourceMediaContract(
                        camera_keys=["base_camera"],
                        frame_alignment="per_frame",
                        video_path_pattern="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                        storage_format="mp4",
                    ),
                ),
                materialized_dataset=EmixMaterializedDatasetDeclaration(
                    repo_id="materialized_repo",
                    root="/tmp/materialized",
                ),
                metadata={
                    "window_len": 16,
                    "group_count": 1,
                    "members_per_group": 2,
                    "episode_unit": "member_window",
                    "range_semantics": "start_inclusive_end_exclusive",
                    "global_index_basis": "parquet_index_column",
                    "source_index_id": "emix_index_test",
                    "source_index_schema_version": EMIX_INDEX_SCHEMA_VERSION,
                    "source_group_count": 1,
                    "source_training_group_count": 1,
                    "exported_episode_count": 2,
                    "fps": 30,
                    "task_source": "frame.task -> episode.tasks -> dataset.meta.tasks",
                },
            ),
            content=EmixMaterializedIndexContent(
                cross_point_groups=[
                    EmixMaterializedCrossPointGroup(
                        group_id="emix_group_test",
                        source_group_id="emix_group_test",
                        export_group_index=0,
                        state=EmixMaterializedGroupState(state="training", abort_reasons=[]),
                        members=[
                            EmixMaterializedMemberRecord(
                                member_id="emix_group_test_anchor",
                                member_role="anchor",
                                source_member_index=0,
                                source_dataset_repo_id="demo_repo",
                                source_episode_index=1,
                                source_window_start_index=0,
                                source_window_end_index=3,
                                source_cross_frame_index=1,
                                source_valid_length=3,
                                export_episode_index=0,
                                export_window_start_index=0,
                                export_window_end_index=3,
                                export_cross_frame_index=1,
                                export_valid_length=3,
                            ),
                            EmixMaterializedMemberRecord(
                                member_id="emix_group_test_neighbor",
                                member_role="neighbor",
                                source_member_index=1,
                                source_dataset_repo_id="demo_repo",
                                source_episode_index=2,
                                source_window_start_index=3,
                                source_window_end_index=6,
                                source_cross_frame_index=4,
                                source_valid_length=3,
                                export_episode_index=1,
                                export_window_start_index=3,
                                export_window_end_index=6,
                                export_cross_frame_index=4,
                                export_valid_length=3,
                            ),
                        ],
                    )
                ]
            ),
        )
        payload = index.to_dict()
        restored = EmixMaterializedIndex.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(restored.public.id, "emix_materialized_index_test")
        self.assertEqual(restored.public.metadata["exported_episode_count"], 2)

    def test_export_materialized_dataset_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_root = root / "source"
            output_root = root / "outputs"
            self._write_source_dataset_fixture(source_root)
            source_dataset = self._make_source_dataset_stub(source_root)
            index = self._make_source_index(source_root)

            with mock.patch(
                "embodied_mixup.candidates.materialized_lerobot_utils._get_lerobot_dataset_class",
                return_value=FakeExportLeRobotDataset,
            ):
                materialized_index, materialized_root = export_materialized_lerobot_dataset(
                    index=index,
                    source_dataset=source_dataset,
                    output_dir=output_root,
                    materialized_repo_id="materialized_repo",
                )

            self.assertEqual(materialized_root, output_root.resolve())
            self.assertEqual(materialized_index.public.schema_version, 2)
            validate_materialized_lerobot_root(materialized_root)
            loaded = load_materialized_emix_index(materialized_root)
            self.assertEqual(loaded.public.metadata["group_count"], 1)
            self.assertEqual(loaded.content.cross_point_groups[0].members[0].member_role, "anchor")

    def test_policy_loader_uses_materialized_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with mock.patch(
                "embodied_mixup.policies.utils.emix_loader_registry_utils.EmixFilesReader",
                FakeMaterializedReader,
            ):
                dataset = make_emix_loader(
                    "emix_smolvla",
                    str(root),
                    PolicyTemporalSpec(n_obs_steps=2, action_steps=3),
                    EmixLoaderSettings(policy_type="emix_smolvla", include_task=True),
                )
                self.assertEqual(len(dataset), 1)
                batch = dataset[0]
                self.assertIn("task", batch)
                self.assertEqual(batch["task"], "pick cube")

    def _make_source_index(self, source_root: Path) -> EmixIndex:
        return EmixIndex(
            public=EmixIndexPublic(
                id="emix_index_test",
                schema_version=EMIX_INDEX_SCHEMA_VERSION,
                created_at="2026-07-01T00:00:00+0800",
                source_datasets=[
                    EmixSourceDatasetDeclaration(
                        repo_id="demo_repo",
                        root=str(source_root),
                        hash="abc123",
                        media=EmixSourceMediaContract(
                            camera_keys=["base_camera"],
                            frame_alignment="per_frame",
                            video_path_pattern="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                            storage_format="mp4",
                        ),
                    )
                ],
                source_dataset_version=EMIX_DATASET_VERSION,
                metadata={"window_len": 3},
            ),
            content=EmixIndexContent(
                cross_point_groups=[
                    EmixCrossPointGroup(
                        group_id="emix_group_test",
                        member_count=2,
                        members=[
                            EmixGroupMemberRef("demo_repo", 0, 1),
                            EmixGroupMemberRef("demo_repo", 1, 4),
                        ],
                        state=EmixGroupState(state="training", abort_reasons=[]),
                    )
                ]
            ),
        )

    def _write_source_dataset_fixture(self, root: Path) -> None:
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (root / "videos" / "chunk-000" / "observation.images.base_camera").mkdir(
            parents=True, exist_ok=True
        )
        info = {
            "codebase_version": "v2.1",
            "robot_type": None,
            "total_episodes": 2,
            "total_frames": 6,
            "total_tasks": 1,
            "total_videos": 1,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 30,
            "splits": {"train": "0:2"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "action": {"dtype": "float32", "shape": [3], "names": ["a0", "a1", "a2"]},
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": [f"s{i}" for i in range(6)],
                },
                "observation.images.base_camera": {
                    "dtype": "video",
                    "shape": [4, 4, 3],
                    "names": ["height", "width", "channels"],
                },
                "task": {"dtype": "string", "shape": [1], "names": None},
            },
        }
        (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        (root / "meta" / "episodes.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"episode_index": 0, "tasks": ["pick cube"], "length": 3}),
                    json.dumps({"episode_index": 1, "tasks": ["pick cube"], "length": 3}),
                ]
            ),
            encoding="utf-8",
        )
        (root / "meta" / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "pick cube"}) + "\n",
            encoding="utf-8",
        )
        for episode_index in range(2):
            parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            parquet_path.write_bytes(b"parquet")
            video_path = (
                root
                / "videos"
                / "chunk-000"
                / "observation.images.base_camera"
                / f"episode_{episode_index:06d}.mp4"
            )
            video_path.write_bytes(b"mp4")


    def _make_source_dataset_stub(self, root: Path):
        class DatasetStub:
            def __init__(self, root: Path) -> None:
                self.root = root
                self.repo_id = "demo_repo"
                self.meta = type("Meta", (), {})()
                self.meta.info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
                self.meta.episodes = {
                    0: {"episode_index": 0, "tasks": ["pick cube"], "length": 3},
                    1: {"episode_index": 1, "tasks": ["pick cube"], "length": 3},
                }
                self.meta.total_episodes = 2
                self.episode_data_index = {"from": [0, 3], "to": [3, 6]}
                self._frames = [
                    {
                        "action": [float(idx), float(idx + 1), float(idx + 2)],
                        "observation.state": [float(idx + j) for j in range(6)],
                        "observation.images.base_camera": np.zeros((4, 4, 3), dtype=np.uint8),
                        "task": "pick cube",
                    }
                    for idx in range(6)
                ]

            def __getitem__(self, idx: int):
                return self._frames[idx]

        return DatasetStub(root)


if __name__ == "__main__":
    unittest.main()
