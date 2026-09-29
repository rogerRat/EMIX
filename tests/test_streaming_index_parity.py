"""Parity tests: vectorized streaming index build vs the scalar object path.

The streaming path (run_neighbor_index_pairs_arrays, run_action_filter_arrays,
build_emix_index_streaming) must produce the same pair set, the same filter
metrics, the same cluster partition, and a structurally identical
emix_index.json as the original per-pair Python path on a synthetic dataset.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest import mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embodied_mixup.configs.config import DEFAULT_EMBODIED_MIXUP_CONFIG
from embodied_mixup.candidates.action_filter import (
    precompute_future_indices,
    run_action_filter,
    run_action_filter_arrays,
)
from embodied_mixup.candidates.emix_index import EmixIndex
from embodied_mixup.candidates.emix_index_utils import (
    _frame_membership_validator,
    _pair_group_payload,
    build_emix_index,
    build_emix_index_streaming,
)
from embodied_mixup.candidates.episode_utils import (
    build_episode_bound_maps,
    build_episode_start_end,
    safe_future_index,
)
from embodied_mixup.candidates.ids import make_emix_group_id
from embodied_mixup.candidates.loaders import (
    load_actions,
    load_episode_indices,
    load_eef_poses,
    load_gripper_signal,
)
from embodied_mixup.candidates.pairs import (
    run_neighbor_index_pairs,
    run_neighbor_index_pairs_arrays,
)
from embodied_mixup.candidates.emix_index import (
    EmixGroupMemberRef,
)


N_EPISODES = 6
HOVER_FRAMES = 16  # frames at each waypoint; must exceed future_horizon (8)
# so in-block future motion (per-episode drift direction) exists for the
# angle gate instead of always pointing at the next waypoint.
N_WAYPOINTS = 8
EP_LEN = HOVER_FRAMES * N_WAYPOINTS
N_FRAMES = N_EPISODES * EP_LEN
SEED = 7


def _synthetic_arrays():
    """Build a synthetic dataset with unambiguous waypoint clusters.

    Every episode hovers at the same eight discrete waypoints (tight jitter,
    well separated), so cross-episode pairs form isolated blobs around each
    waypoint.  Per-episode drift directions differ, so pairs of different
    episodes carry sufficiently divergent future motion to pass the angle
    gate, and the gripper alternates per hover so the consistency gate keeps
    half of each blob.  The clustering partition is robust to input order.
    """
    rng = np.random.default_rng(SEED)
    grid = np.array(
        [
            [0.40 + 0.09 * (w % 4), -0.10 + 0.09 * (w // 4), 0.25]
            for w in range(N_WAYPOINTS)
        ]
    )
    drift = rng.normal(scale=1.0, size=(N_EPISODES, 3))
    drift /= np.linalg.norm(drift, axis=1, keepdims=True)
    drift *= 0.0015  # per-frame hover drift; future direction differs per episode

    poses = np.zeros((N_FRAMES, 6), dtype=np.float64)
    for ep in range(N_EPISODES):
        for w in range(N_WAYPOINTS):
            start = ep * EP_LEN + w * HOVER_FRAMES
            hover = grid[w] + drift[ep] * np.arange(HOVER_FRAMES)[:, None]
            hover += rng.normal(scale=0.0008, size=hover.shape)
            poses[start : start + HOVER_FRAMES, :3] = hover
            poses[start : start + HOVER_FRAMES, 3:6] = rng.normal(
                scale=0.004, size=(HOVER_FRAMES, 3)
            )
    episode_indices = np.repeat(np.arange(N_EPISODES), EP_LEN).astype(np.int64)
    actions = rng.normal(scale=0.05, size=(N_FRAMES, 4)).astype(np.float32)
    # Alternate gripper state per hover block so matching-state pairs exist.
    block_state = np.tile(
        np.repeat([0.9, 0.1], HOVER_FRAMES), N_WAYPOINTS
    ).astype(np.float32)
    gripper = np.tile(block_state, N_EPISODES)
    gripper += rng.normal(scale=0.0005, size=gripper.shape).astype(np.float32)
    gripper_states = np.sign(gripper)
    return poses, episode_indices, actions, gripper, gripper_states


class _FakeHF:
    """Minimal hf_dataset stand-in: column access, row access, features, len."""

    def __init__(self, columns, features):
        self._columns = columns
        self.features = features

    def __len__(self):
        return len(next(iter(self._columns.values())))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._columns[key]
        row = {name: np.asarray(values[key]) for name, values in self._columns.items()}
        return row


class _FakeDataset:
    """Minimal LeRobotDataset stand-in for the builders."""

    def __init__(self, root):
        poses, episode_indices, actions, gripper, _ = _synthetic_arrays()
        self.repo_id = "synthetic_parity_ds"
        self.root = root
        self._poses = poses
        self.hf_dataset = _FakeHF(
            columns={
                "observation.eef_pose": poses,
                "episode_index": episode_indices,
                "action": actions,
                "observation.left_gripper": gripper,
            },
            features={
                "observation.eef_pose": object(),
                "episode_index": object(),
                "action": object(),
                "observation.left_gripper": object(),
                "observation.images.head_camera": object(),
                "observation.images.hand_camera": object(),
            },
        )

    def __getitem__(self, idx):
        return {
            "observation.images.head_camera": np.zeros((2, 2, 3)),
            "observation.images.hand_camera": np.zeros((2, 2, 3)),
        }


def _canonical_partition(labels):
    """Relabel a clustering by first-occurrence order so partitions compare equal."""
    mapping: dict[int, int] = {}
    canonical = np.full(len(labels), -1, dtype=np.int64)
    for idx, label in enumerate(labels):
        if label < 0:
            continue
        if label not in mapping:
            mapping[label] = len(mapping)
        canonical[idx] = mapping[label]
    return canonical


def _partition_mismatch_fraction(labels_a, labels_b):
    """Fraction of pairs whose cluster assignment disagrees after best matching.

    HDBSCAN boundary points are sensitive to input order and float precision,
    so backend/order parity is asserted as a small mismatch fraction after
    greedily mapping each label of B onto the majority label of A (noise -1
    participates as its own label).
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape[0] != labels_b.shape[0]:
        return 1.0
    if labels_a.shape[0] == 0:
        return 0.0
    canonical_a = _canonical_partition(labels_a)
    mapping: dict[int, int] = {}
    for label_b in np.unique(labels_b):
        mask = labels_b == label_b
        values, counts = np.unique(canonical_a[mask], return_counts=True)
        mapping[int(label_b)] = int(values[np.argmax(counts)])
    mapped_b = np.array([mapping[int(x)] for x in labels_b])
    return float(np.mean(mapped_b != canonical_a))


def _assert_partitions_compatible(testcase, labels_a, labels_b, max_mismatch):
    mismatch = _partition_mismatch_fraction(labels_a, labels_b)
    testcase.assertLessEqual(
        mismatch,
        max_mismatch,
        f"partition mismatch {mismatch:.4f} exceeds {max_mismatch}",
    )


class StreamingParityTest(unittest.TestCase):
    def setUp(self):
        (
            self.poses,
            self.episode_indices,
            self.actions,
            self.gripper,
            self.gripper_states,
        ) = _synthetic_arrays()
        # Parity tests compare full-pipeline outputs; disable the default
        # per-episode pair cap here (the default would subsample this dense
        # synthetic collection differently on each path's pair ordering —
        # that behaviour is covered by the dedicated cap test).
        self.config = replace(
            DEFAULT_EMBODIED_MIXUP_CONFIG,
            max_candidate_pairs=None,
            max_candidate_pairs_per_episode=None,
        )

    def test_future_indices_match_scalar(self):
        _, episode_end_map = build_episode_bound_maps(
            episode_indices=self.episode_indices,
            episode_starts=build_episode_start_end(self.episode_indices)[0],
            episode_ends=build_episode_start_end(self.episode_indices)[1],
        )
        vectorized = precompute_future_indices(
            episode_indices=self.episode_indices,
            episode_end_map=episode_end_map,
            future_horizon=self.config.future_horizon,
        )
        for frame_idx in range(N_FRAMES):
            scalar = safe_future_index(
                frame_idx=frame_idx,
                episode_indices=self.episode_indices,
                episode_end_map=episode_end_map,
                future_horizon=self.config.future_horizon,
            )
            expected = -1 if scalar is None else scalar
            self.assertEqual(int(vectorized[frame_idx]), expected, f"frame {frame_idx}")

    def test_neighbor_pair_arrays_match_scalar(self):
        scalar_pairs = run_neighbor_index_pairs(
            poses=self.poses,
            episode_indices=self.episode_indices,
            gripper_states=self.gripper_states,
            config=self.config,
        )
        scalar_set = {(int(a), int(b)) for a, b in scalar_pairs}
        anchors, neighbors = run_neighbor_index_pairs_arrays(
            poses=self.poses,
            episode_indices=self.episode_indices,
            gripper_states=self.gripper_states,
            config=self.config,
        )
        array_set = {(int(a), int(b)) for a, b in zip(anchors, neighbors)}
        self.assertEqual(scalar_set, array_set)
        # Sorted (anchor, neighbor) ordering for deterministic group ids.
        order = np.lexsort((neighbors, anchors))
        np.testing.assert_array_equal(anchors[order], anchors)
        np.testing.assert_array_equal(neighbors[order], neighbors)

    def test_action_filter_arrays_match_scalar(self):
        raw_pairs = run_neighbor_index_pairs(
            poses=self.poses,
            episode_indices=self.episode_indices,
            gripper_states=self.gripper_states,
            config=self.config,
        )
        starts, ends = build_episode_start_end(self.episode_indices)
        _, episode_end_map = build_episode_bound_maps(
            episode_indices=self.episode_indices,
            episode_starts=starts,
            episode_ends=ends,
        )
        scalar_records = run_action_filter(
            neighbor_pairs=raw_pairs,
            poses=self.poses,
            actions=self.actions,
            gripper_signal=self.gripper,
            episode_indices=self.episode_indices,
            episode_end_map=episode_end_map,
            config=self.config,
        )
        future_idx = precompute_future_indices(
            episode_indices=self.episode_indices,
            episode_end_map=episode_end_map,
            future_horizon=self.config.future_horizon,
        )
        anchors, neighbors = run_neighbor_index_pairs_arrays(
            poses=self.poses,
            episode_indices=self.episode_indices,
            gripper_states=self.gripper_states,
            config=self.config,
        )
        vectorized = run_action_filter_arrays(
            anchors=anchors,
            neighbors=neighbors,
            poses=self.poses,
            actions=self.actions,
            gripper_signal=self.gripper,
            episode_indices=self.episode_indices,
            future_idx=future_idx,
            config=self.config,
        )

        vector_set = {
            (int(a), int(b))
            for a, b in zip(vectorized["anchor_idx"], vectorized["neighbor_idx"])
        }
        scalar_set = {(int(r.anchor_idx), int(r.neighbor_idx)) for r in scalar_records}
        self.assertEqual(scalar_set, vector_set)

        vector_by_pair = {
            (int(a), int(b)): i
            for i, (a, b) in enumerate(
                zip(vectorized["anchor_idx"], vectorized["neighbor_idx"])
            )
        }
        for record in scalar_records:
            key = (int(record.anchor_idx), int(record.neighbor_idx))
            idx = vector_by_pair[key]
            for array_name, attr in (
                ("action_divergence", "action_divergence"),
                ("pose_distance_xyz", "pose_distance_xyz"),
                ("pose_distance_rot", "pose_distance_rpy"),
                ("future_angle_xyz", "future_angle_xyz"),
                ("future_disp_a", "future_disp_a"),
                ("future_disp_b", "future_disp_b"),
            ):
                # pose_distance_rpy is a CandidateRecord property delegating
                # to relation.pose_distance_rot; the rest are direct fields.
                value = getattr(record, attr, None)
                if value is None:
                    value = getattr(record.relation, attr)
                np.testing.assert_allclose(
                    vectorized[array_name][idx],
                    value,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"{array_name} mismatch at {key}",
                )
            self.assertEqual(
                bool(vectorized["gripper_action_filter_flag"][idx]),
                record.relation.gripper_action_filter_flag,
                f"gripper flag mismatch at {key}",
            )

    def test_group_payload_matches_scalar_group_id(self):
        members = [
            EmixGroupMemberRef(
                source_dataset_repo_id="ds",
                source_episode_index=3,
                source_frame_index=11,
            ),
            EmixGroupMemberRef(
                source_dataset_repo_id="ds",
                source_episode_index=4,
                source_frame_index=222,
            ),
        ]
        expected = make_emix_group_id("ds", members, salt=0)
        filtered = {
            "anchor_idx": np.array([11], dtype=np.int32),
            "neighbor_idx": np.array([222], dtype=np.int32),
            "anchor_episode": np.array([3], dtype=np.int32),
            "neighbor_episode": np.array([4], dtype=np.int32),
            "pose_distance_xyz": np.array([0.01]),
            "pose_distance_rot": np.array([0.02]),
            "action_divergence": np.array([0.3]),
        }
        validator = _frame_membership_validator(None, [])
        self.assertFalse(validator(np.array([0]))[0])  # no hf_dataset -> invalid
        payload = _pair_group_payload(
            filtered=filtered,
            poses=np.zeros((300, 6)),
            episode_indices=np.zeros(300, dtype=np.int64),
            cluster_labels=np.array([2], dtype=np.int64),
            frame_degree=np.zeros(300, dtype=np.int32),
            source_repo_id="ds",
            pair_index=0,
            member_frames_ok=lambda indices: np.ones(indices.shape, dtype=bool),
        )
        self.assertEqual(payload["group_id"], expected)
        self.assertEqual(payload["state"]["state"], "training")
        self.assertTrue(payload["state"]["training"])

    def test_streaming_build_matches_object_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("meta", "data", "videos"):
                (root / name).mkdir()
            dataset = _FakeDataset(root=str(root))

            scalar_index = build_emix_index(dataset=dataset, config=self.config)
            scalar_path = root / "scalar_index.json"
            from embodied_mixup.candidates.emix_index_utils import save_emix_index

            save_emix_index(scalar_index, scalar_path)

            streaming_path = build_emix_index_streaming(
                dataset=dataset,
                config=self.config,
                output_path=root / "streaming_index.json",
            )

            with open(scalar_path, encoding="utf-8") as handle:
                scalar_payload = json.load(handle)
            with open(streaming_path, encoding="utf-8") as handle:
                streaming_payload = json.load(handle)

            # Both artifacts must load through the strict reader.
            EmixIndex.from_dict(scalar_payload)
            EmixIndex.from_dict(streaming_payload)

            # Same aggregates (created_at and ids may differ in principle, but
            # the deterministic id inputs here are identical).
            self.assertEqual(
                scalar_payload["public"]["metadata"]["total_pairs"],
                streaming_payload["public"]["metadata"]["total_pairs"],
            )
            self.assertEqual(
                scalar_payload["public"]["metadata"]["total_raw_neighbor_pairs"],
                streaming_payload["public"]["metadata"]["total_raw_neighbor_pairs"],
            )
            self.assertEqual(
                scalar_payload["public"]["metadata"]["num_clusters"],
                streaming_payload["public"]["metadata"]["num_clusters"],
            )
            self.assertEqual(
                scalar_payload["public"]["id"],
                streaming_payload["public"]["id"],
            )

            # Same pair membership sets and a compatible clustering partition.
            def pair_set(payload):
                return {
                    (
                        g["members"][0]["source_frame_index"],
                        g["members"][1]["source_frame_index"],
                    )
                    for g in payload["content"]["cross_point_groups"]
                }

            self.assertEqual(pair_set(scalar_payload), pair_set(streaming_payload))

            def ordered_labels(payload):
                groups = payload["content"]["cross_point_groups"]
                members = [
                    (
                        g["members"][0]["source_frame_index"],
                        g["members"][1]["source_frame_index"],
                    )
                    for g in groups
                ]
                labels = np.array(
                    [g["group_metrics"]["cluster_id"] for g in groups]
                )
                order = sorted(range(len(members)), key=lambda i: members[i])
                return labels[order]

            _assert_partitions_compatible(
                self,
                ordered_labels(scalar_payload),
                ordered_labels(streaming_payload),
                max_mismatch=0.10,
            )

            # Same per-pair metrics on the shared pair set.
            def metric_map(payload):
                return {
                    (
                        g["members"][0]["source_frame_index"],
                        g["members"][1]["source_frame_index"],
                    ): (
                        g["group_metrics"]["pose_distance_xyz"],
                        g["group_metrics"]["pose_distance_rot"],
                        g["group_metrics"]["action_divergence"],
                        g["group_metrics"]["local_neighbors"],
                    )
                    for g in payload["content"]["cross_point_groups"]
                }

            scalar_metrics = metric_map(scalar_payload)
            streaming_metrics = metric_map(streaming_payload)
            self.assertEqual(set(scalar_metrics), set(streaming_metrics))
            for key, expected in scalar_metrics.items():
                actual = streaming_metrics[key]
                for exp, act in zip(expected, actual):
                    np.testing.assert_allclose(exp, act, rtol=1e-5, atol=1e-8)


    def test_sklearn_backend_matches_hdbscan_partition(self):
        """Force the fast sklearn backend and check the partition matches."""
        import embodied_mixup.candidates.emix_index_utils as index_utils

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("meta", "data", "videos"):
                (root / name).mkdir()
            dataset = _FakeDataset(root=str(root))

            scalar_index = build_emix_index(dataset=dataset, config=self.config)
            scalar_labels = [
                group.group_metrics["cluster_id"]
                for group in scalar_index.content.cross_point_groups
            ]
            scalar_members = [
                (
                    group.members[0].source_frame_index,
                    group.members[1].source_frame_index,
                )
                for group in scalar_index.content.cross_point_groups
            ]

            with mock.patch.object(
                index_utils, "_SKLEARN_HDBSCAN_MIN_PAIRS", 0
            ):
                streaming_path = index_utils.build_emix_index_streaming(
                    dataset=dataset,
                    config=self.config,
                    output_path=root / "sklearn_backend_index.json",
                )
            with open(streaming_path, encoding="utf-8") as handle:
                streaming_payload = json.load(handle)

            order = {
                pair: idx for idx, pair in enumerate(scalar_members)
            }
            streaming_labels = np.zeros(len(scalar_members), dtype=np.int64)
            for group in streaming_payload["content"]["cross_point_groups"]:
                pair = (
                    group["members"][0]["source_frame_index"],
                    group["members"][1]["source_frame_index"],
                )
                streaming_labels[order[pair]] = group["group_metrics"]["cluster_id"]

            _assert_partitions_compatible(
                self,
                np.array(scalar_labels),
                streaming_labels,
                max_mismatch=0.10,
            )


    def test_stage_and_external_labels_roundtrip(self):
        """Staging exits cleanly; an external sidecar's labels complete the build."""
        import os
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        import embodied_mixup.candidates.emix_index_utils as index_utils

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("meta", "data", "videos"):
                (root / name).mkdir()
            dataset = _FakeDataset(root=str(root))
            stage_dir = root / "stage"

            # Scalar reference partition.
            scalar_index = build_emix_index(dataset=dataset, config=self.config)
            scalar_labels = [
                group.group_metrics["cluster_id"]
                for group in scalar_index.content.cross_point_groups
            ]

            # Phase 1: stage midpoints; the build must exit with instructions.
            env = {
                "EMIX_CLUSTER_STAGE_DIR": str(stage_dir),
                "EMIX_CLUSTER_LABELS_PATH": "",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with self.assertRaises(SystemExit):
                    build_emix_index_streaming(
                        dataset=dataset,
                        config=self.config,
                        output_path=root / "staged_index.json",
                    )
            self.assertTrue((stage_dir / "midpoints_scaled.npy").exists())
            self.assertTrue((stage_dir / "params.json").exists())

            # Phase 2: sidecar clustering (sklearn stands in for cuML).
            points = np.load(stage_dir / "midpoints_scaled.npy")
            params = json.loads((stage_dir / "params.json").read_text())
            labels = SklearnHDBSCAN(
                min_cluster_size=params["min_cluster_size"],
                min_samples=params["min_samples"],
                cluster_selection_epsilon=params["cluster_selection_epsilon"],
                cluster_selection_method=params["cluster_selection_method"],
                metric="euclidean",
            ).fit_predict(points)
            np.save(stage_dir / "labels.npy", np.asarray(labels, dtype=np.int64))

            # Phase 3: resume the build with the external labels.
            os.environ["EMIX_CLUSTER_LABELS_PATH"] = str(stage_dir / "labels.npy")
            try:
                streaming_path = build_emix_index_streaming(
                    dataset=dataset,
                    config=self.config,
                    output_path=root / "resumed_index.json",
                )
            finally:
                os.environ.pop("EMIX_CLUSTER_LABELS_PATH", None)

            with open(streaming_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            streaming_labels = [
                group["group_metrics"]["cluster_id"]
                for group in payload["content"]["cross_point_groups"]
            ]
            _assert_partitions_compatible(
                self,
                np.array(scalar_labels),
                np.array(streaming_labels),
                max_mismatch=0.10,
            )


    def test_candidate_pair_cap_subsamples_deterministically(self):
        """The per-episode cap keeps a seeded subsample inside the full set."""
        from dataclasses import replace

        import embodied_mixup.candidates.emix_index_utils as index_utils_module

        # 6 synthetic episodes; ratio 40 -> cap 240 (above the natural count
        # would be a no-op, below it subsamples).
        capped_config = replace(self.config, max_candidate_pairs_per_episode=40.0)
        self.assertEqual(
            index_utils_module._effective_candidate_pair_cap(capped_config, 6), 240
        )
        self.assertIsNone(
            index_utils_module._effective_candidate_pair_cap(
                replace(self.config,
                        max_candidate_pairs=None,
                        max_candidate_pairs_per_episode=None),
                6,
            )
        )
        # Absolute override wins when it is the smaller one.
        self.assertEqual(
            index_utils_module._effective_candidate_pair_cap(
                replace(capped_config, max_candidate_pairs=100), 6
            ),
            100,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("meta", "data", "videos"):
                (root / name).mkdir()
            dataset = _FakeDataset(root=str(root))

            scalar_index = build_emix_index(dataset=dataset, config=capped_config)
            streaming_path = build_emix_index_streaming(
                dataset=dataset,
                config=capped_config,
                output_path=root / "capped_index.json",
            )
            with open(streaming_path, encoding="utf-8") as handle:
                streaming_payload = json.load(handle)

            scalar_groups = scalar_index.content.cross_point_groups
            streaming_groups = streaming_payload["content"]["cross_point_groups"]
            self.assertLessEqual(len(scalar_groups), 240)
            self.assertEqual(len(scalar_groups), len(streaming_groups))

            meta = streaming_payload["public"]["metadata"]
            self.assertEqual(meta["candidate_pair_cap"], 240)
            self.assertEqual(meta["total_episodes"], 6)
            self.assertGreater(meta["total_pairs_uncapped"], 240)

            def member_pair(g):
                members = g["members"] if isinstance(g, dict) else g.members
                getter = (lambda m, k: m[k]) if isinstance(g, dict) else (lambda m, k: getattr(m, k))
                return (getter(members[0], "source_frame_index"),
                        getter(members[1], "source_frame_index"))

            def pair_set(groups):
                return {member_pair(g) for g in groups}

            # Each capped subset stays inside the shared uncapped pair set
            # (the two paths order pairs differently by design, so the same
            # seeded indices select different-but-equivalent subsets).  The
            # reference build disables capping explicitly: the default
            # per-episode ratio would itself cap this collection.
            uncapped_config = replace(
                self.config,
                max_candidate_pairs=None,
                max_candidate_pairs_per_episode=None,
            )
            uncapped_path = build_emix_index_streaming(
                dataset=dataset,
                config=uncapped_config,
                output_path=root / "uncapped_index.json",
            )
            with open(uncapped_path, encoding="utf-8") as handle:
                uncapped_payload = json.load(handle)
            full_pairs = pair_set(uncapped_payload["content"]["cross_point_groups"])
            self.assertLessEqual(pair_set(streaming_groups), full_pairs)
            self.assertLessEqual(pair_set(scalar_groups), full_pairs)

            # Same seed -> stable subset across streaming rebuilds.
            again = build_emix_index_streaming(
                dataset=dataset,
                config=capped_config,
                output_path=root / "capped_index_again.json",
            )
            with open(again, encoding="utf-8") as handle:
                again_payload = json.load(handle)
            self.assertEqual(
                pair_set(streaming_groups), pair_set(again_payload["content"]["cross_point_groups"])
            )

            # The uncapped build keeps every pair.
            self.assertIsNone(uncapped_payload["public"]["metadata"]["candidate_pair_cap"])
            self.assertEqual(
                uncapped_payload["public"]["metadata"]["total_pairs"],
                uncapped_payload["public"]["metadata"]["total_pairs_uncapped"],
            )
            self.assertGreater(
                uncapped_payload["public"]["metadata"]["total_pairs"], 240
            )


if __name__ == "__main__":
    unittest.main()
