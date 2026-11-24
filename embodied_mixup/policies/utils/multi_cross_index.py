"""Build deterministic, pose-compatible cross-task pairs for multi-v0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _source_frames(root: Path, stride: int):
    import pyarrow.parquet as pq

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet frames in {root}")
    columns = ("index", "episode_index", "observation.eef_pose", "observation.state", "action")
    rows = []
    for file in files:
        table = pq.read_table(file, columns=list(columns))
        rows.extend(table.to_pylist())
    if not rows:
        raise ValueError(f"Empty source dataset {root}")
    indexes = np.asarray([row["index"] for row in rows], dtype=np.int64)
    if not np.array_equal(indexes, np.arange(len(rows))):
        raise ValueError(f"Non-contiguous frame index in {root}")
    episodes = np.asarray([row["episode_index"] for row in rows], dtype=np.int32)
    poses = np.asarray([row["observation.eef_pose"] for row in rows], dtype=np.float32)
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    valid = np.flatnonzero(np.arange(len(rows)) % stride == 0)
    valid = valid[np.isfinite(poses[valid]).all(axis=1)]
    return episodes, poses, states, actions, valid


def build_cross_pairs(path: Path, repo_ids: list[str], dataset_root: Path, *,
                      stride: int = 30, max_pairs: int = 20000, seed: int = 0) -> int:
    """Create cross-task pair records with independent source provenance.

    Close EEF states anchor the pairs. Action windows must diverge, so the
    second target carries a real alternative trajectory from another task.
    """
    from scipy.spatial import cKDTree
    from .multi_dataset import validate_sources

    roots = validate_sources(dataset_root, repo_ids)
    arrays = [_source_frames(root, stride) for root in roots]
    rng = np.random.default_rng(seed)
    pairs = []
    for a in range(len(repo_ids)):
        ep_a, pose_a, state_a, action_a, valid_a = arrays[a]
        anchors = valid_a[rng.permutation(len(valid_a))]
        for b in range(a + 1, len(repo_ids)):
            ep_b, pose_b, state_b, action_b, valid_b = arrays[b]
            # A shared metric gives 2 cm position and 0.26 rad orientation gates.
            scaled_b = np.concatenate((pose_b[valid_b, :3] / 0.02,
                                       pose_b[valid_b, 3:] / 0.26), axis=1)
            tree = cKDTree(scaled_b)
            query = np.concatenate((pose_a[anchors, :3] / 0.02,
                                    pose_a[anchors, 3:] / 0.26), axis=1)
            distances, neighbors = tree.query(query, k=4, distance_upper_bound=1.5)
            if neighbors.ndim == 1:
                neighbors = neighbors[:, None]
            count = 0
            for row, fa in enumerate(anchors):
                for neighbor in neighbors[row]:
                    if neighbor >= len(valid_b):
                        continue
                    fb = int(valid_b[neighbor])
                    xyz = float(np.linalg.norm(pose_a[fa, :3] - pose_b[fb, :3]))
                    rot = float(np.linalg.norm(pose_a[fa, 3:] - pose_b[fb, 3:]))
                    if xyz > 0.02 or rot > 0.26:
                        continue
                    if abs(float(state_a[fa, -1] - state_b[fb, -1])) > 0.25:
                        continue
                    # Keep full 50-step targets inside their respective episodes.
                    horizon = 50
                    if fa + horizon > len(ep_a) or fb + horizon > len(ep_b):
                        continue
                    if ep_a[fa + horizon - 1] != ep_a[fa] or ep_b[fb + horizon - 1] != ep_b[fb]:
                        continue
                    divergence = float(np.mean(np.linalg.norm(
                        action_a[fa:fa+horizon] - action_b[fb:fb+horizon], axis=1)))
                    if not np.isfinite(divergence) or divergence < 0.05:
                        continue
                    pairs.append({"a": {"repo_id": repo_ids[a], "frame_index": int(fa),
                                         "episode_index": int(ep_a[fa])},
                                  "b": {"repo_id": repo_ids[b], "frame_index": fb,
                                         "episode_index": int(ep_b[fb])},
                                  "pose_distance_xyz": xyz, "pose_distance_rot": rot,
                                  "action_divergence": divergence})
                    count += 1
                    break
                if count >= max(100, max_pairs // (len(repo_ids) * (len(repo_ids) - 1) // 2)):
                    break
    if not pairs:
        raise ValueError("No compatible cross-task pairs passed pose/action filters")
    # Both tasks must occur as language/state anchor, not only as donor.
    reverse_pairs = [{**pair, "a": pair["b"], "b": pair["a"]} for pair in pairs]
    pairs.extend(reverse_pairs)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    payload = {"schema_version": 1, "pair_scope": "cross_task",
               "source_repo_ids": repo_ids, "stride": stride, "seed": seed,
               "criteria": {"xyz_max_m": 0.02, "rot_max_rad": 0.26,
                            "gripper_max": 0.25, "action_divergence_min": 0.05,
                            "action_horizon": 50},
               "pairs": pairs}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return len(pairs)
