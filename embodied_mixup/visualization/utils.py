"""Visualization data preparation utilities.

Transforms EmixIndex and source dataset arrays into Plotly-friendly data
structures for 3D/2D rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..candidates.emix_index import EmixCrossPointGroup, EmixIndex
from ..candidates.loaders import load_actions, load_eef_poses, load_episode_indices, load_gripper_signal

logger = logging.getLogger(__name__)

_MAX_RAW_REJECTED_FOR_VISUALIZATION = 10_000

_FONT_FAMILY = "Times New Roman, Times, serif"
_PLOTLY_FONT = dict(family=_FONT_FAMILY)


def _get_cluster_colors(n_clusters: int) -> list[str]:
    """Return a deterministic color palette for cluster plots."""

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if n_clusters <= 0:
        return []
    return [palette[idx % len(palette)] for idx in range(int(n_clusters))]


def _format_member_id(member) -> str:
    """Build a stable display identifier for one index member."""

    return f"{member.source_dataset_repo_id}:{int(member.source_episode_index)}:{int(member.source_frame_index)}"


@dataclass
class VizMemberPair:
    """Lightweight two-member adapter for plotting."""

    group_id: str
    member_ids: list[str]
    member_episodes: list[int]
    member_indices: list[int]
    member_poses: list[list[float]]
    action_divergence: float
    cluster_id: int


def compute_raw_pair_data(
    index: EmixIndex,
    dataset,
    config,
    *,
    groups: list[EmixCrossPointGroup] | None = None,
    poses: np.ndarray | None = None,
    episode_indices: np.ndarray | None = None,
    actions: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Recompute raw neighbor pairs and match them against the action-filtered
    pairs stored in the EmixIndex.

    Raw pairs are regenerated via KD-tree neighbor search (the same method
    used during index building) and then compared with the filtered pairs in
    the index to determine which were accepted/rejected by the action filter.

    @input:  index: EmixIndex containing action-filtered cross-point groups.
             dataset: LeRobotDataset-like object providing frame access.
             config: EmbodiedMixupConfig with threshold parameters.
    @output: Dict with raw_midpoints, raw_accepted_mask, raw_divergences,
             raw_cluster_ids, raw_frame_pairs, and per-member episode/frame
             arrays for building Plotly customdata.
    @scenario: Enable raw-vs-filtered toggle in 3D/2D visualizations.
    """
    from ..candidates.pairs import run_neighbor_index_pairs

    # ``prepare_visualization_data`` already loaded these arrays.  Reuse them
    # when available so raw-vs-filtered preparation does not briefly retain a
    # second complete copy of the dataset in memory.
    if poses is None:
        poses = load_eef_poses(dataset)
    if episode_indices is None:
        episode_indices = load_episode_indices(dataset)
    if actions is None:
        actions = load_actions(dataset)
    gripper_states = np.sign(load_gripper_signal(actions))

    logger.info("Re-running neighbor search for raw pair visualization...")
    raw_pairs = run_neighbor_index_pairs(
        poses=poses,
        episode_indices=episode_indices,
        gripper_states=gripper_states,
        config=config,
    )
    logger.info("Raw neighbor search produced %d pairs", len(raw_pairs))

    # Build set of accepted (frame_a, frame_b) tuples from EmixIndex groups.
    # ``groups`` may be a capped subset (--max-points) so the raw accepted
    # pairs stay consistent with the reduced filtered plot; default to the
    # full index.
    cross_point_groups = (
        groups if groups is not None else index.content.cross_point_groups
    )
    filtered_pair_set: set[tuple[int, int]] = set()
    frame_to_cluster: dict[tuple[int, int], int] = {}
    for group in cross_point_groups:
        if len(group.members) != 2:
            continue
        f0 = int(group.members[0].source_frame_index)
        f1 = int(group.members[1].source_frame_index)
        pair_key = (min(f0, f1), max(f0, f1))
        filtered_pair_set.add(pair_key)
        cluster_id = int(group.group_metrics.get("cluster_id", -1))
        frame_to_cluster[pair_key] = cluster_id

    n_raw_total = len(raw_pairs)
    render_pairs: list[tuple[int, int]] = []
    render_accepted: list[bool] = []
    n_accepted_total = 0
    n_rejected_total = 0
    for anchor_idx, neighbor_idx in raw_pairs:
        pair_key = (min(int(anchor_idx), int(neighbor_idx)), max(int(anchor_idx), int(neighbor_idx)))
        is_accepted = pair_key in filtered_pair_set
        if is_accepted:
            n_accepted_total += 1
            render_pairs.append((anchor_idx, neighbor_idx))
            render_accepted.append(True)
        else:
            n_rejected_total += 1
            if n_rejected_total <= _MAX_RAW_REJECTED_FOR_VISUALIZATION:
                render_pairs.append((anchor_idx, neighbor_idx))
                render_accepted.append(False)

    if n_rejected_total > _MAX_RAW_REJECTED_FOR_VISUALIZATION:
        logger.info(
            "Raw rejected visualization capped at %d points: rendering %d / %d "
            "rejected pairs; all %d accepted pairs retained.",
            _MAX_RAW_REJECTED_FOR_VISUALIZATION,
            _MAX_RAW_REJECTED_FOR_VISUALIZATION,
            n_rejected_total,
            n_accepted_total,
        )

    n_rendered = len(render_pairs)
    raw_midpoints = np.empty((n_rendered, 6), dtype=np.float64) if n_rendered > 0 else np.empty((0, 6))
    raw_accepted_mask = np.asarray(render_accepted, dtype=bool) if n_rendered > 0 else np.empty(0, dtype=bool)
    raw_divergences = np.empty(n_rendered, dtype=np.float64) if n_rendered > 0 else np.empty(0)
    raw_cluster_ids = np.full(n_rendered, -1, dtype=np.int64) if n_rendered > 0 else np.empty(0, dtype=np.int64)
    raw_ep_a = np.empty(n_rendered, dtype=np.int64) if n_rendered > 0 else np.empty(0, dtype=np.int64)
    raw_frame_a = np.empty(n_rendered, dtype=np.int64) if n_rendered > 0 else np.empty(0, dtype=np.int64)
    raw_ep_b = np.empty(n_rendered, dtype=np.int64) if n_rendered > 0 else np.empty(0, dtype=np.int64)
    raw_frame_b = np.empty(n_rendered, dtype=np.int64) if n_rendered > 0 else np.empty(0, dtype=np.int64)

    if n_rendered > 0:
        anchor_indices = np.fromiter(
            (pair[0] for pair in render_pairs), dtype=np.intp, count=n_rendered,
        )
        neighbor_indices = np.fromiter(
            (pair[1] for pair in render_pairs), dtype=np.intp, count=n_rendered,
        )
        # Use bounded chunks: fancy indexing otherwise creates two full pose
        # and action-difference temporaries in addition to the Plotly arrays.
        # The chunk size is an implementation detail and does not affect any
        # floating-point result or output ordering.
        chunk_size = 65536
        for start in range(0, n_rendered, chunk_size):
            end = min(start + chunk_size, n_rendered)
            anchor_chunk = anchor_indices[start:end]
            neighbor_chunk = neighbor_indices[start:end]
            np.add(
                poses[anchor_chunk], poses[neighbor_chunk],
                out=raw_midpoints[start:end],
            )
            raw_midpoints[start:end] *= 0.5
            raw_ep_a[start:end] = episode_indices[anchor_chunk]
            raw_frame_a[start:end] = anchor_chunk
            raw_ep_b[start:end] = episode_indices[neighbor_chunk]
            raw_frame_b[start:end] = neighbor_chunk
            raw_divergences[start:end] = np.linalg.norm(
                actions[anchor_chunk] - actions[neighbor_chunk], axis=1,
            )
        for idx, (anchor_idx, neighbor_idx) in enumerate(render_pairs):
            pair_key = (min(int(anchor_idx), int(neighbor_idx)), max(int(anchor_idx), int(neighbor_idx)))
            raw_cluster_ids[idx] = frame_to_cluster.get(pair_key, -1)

    logger.info(
        "Raw pair matching: %d total, %d accepted (filtered), %d rejected",
        n_raw_total, n_accepted_total, n_rejected_total,
    )

    return {
        "raw_midpoints": raw_midpoints,
        "raw_accepted_mask": raw_accepted_mask,
        "raw_divergences": raw_divergences,
        "raw_cluster_ids": raw_cluster_ids,
        "raw_ep_a": raw_ep_a,
        "raw_frame_a": raw_frame_a,
        "raw_ep_b": raw_ep_b,
        "raw_frame_b": raw_frame_b,
        "n_raw": n_raw_total,
        "n_accepted": n_accepted_total,
        "n_rejected": n_rejected_total,
        "n_raw_rendered": n_rendered,
        "n_rejected_rendered": int((~raw_accepted_mask).sum()) if n_rendered else 0,
    }


def prepare_visualization_data(
    index: EmixIndex,
    dataset,
    config=None,
    *,
    max_points: int | None = None,
) -> Dict[str, Any]:
    """Extract Plotly-friendly visualization data from EmixIndex and dataset.

    @input:  index: EmixIndex with cross-point groups.
             dataset: LeRobotDataset-like object providing frame access.
             config: EmbodiedMixupConfig (enables raw-vs-filtered toggle).
             max_points: optional cap on the number of cross-point groups
                 rendered.  A deterministic spread sample is taken across the
                 whole group list so a large index (hundreds of thousands of
                 intersection points) still produces a small, smooth HTML.
    @output: Dict with trajectories, midpoints, divergences, cluster_labels,
             member_pairs, cluster_info, and (when config is set) raw-pair data.
    @scenario: The EmixIndex for a 500-episode task holds ~500k cross-point
               groups; embedding every one as a hoverable WebGL marker bloats
               the HTML to hundreds of MB.  Capping keeps delivery snappy.
    """

    episode_indices = load_episode_indices(dataset)
    poses = load_eef_poses(dataset)
    actions = load_actions(dataset)
    gripper_states = np.sign(load_gripper_signal(actions))
    # Raw [0, 1] signal (1 = fully open) for graded-threshold consumers
    # such as pick/place transition detection.
    gripper_signal = load_gripper_signal(actions, dataset)

    unique_eps = np.unique(episode_indices)
    trajectories: Dict[int, dict] = {}
    for ep_idx in unique_eps:
        mask = episode_indices == ep_idx
        frame_ids = np.flatnonzero(mask)
        # LeRobot episodes are stored contiguously.  Slices preserve the
        # historical per-episode values while avoiding four full boolean
        # indexing copies for every episode.  Retain the mask fallback for
        # unusual datasets whose episode frames are interleaved.
        if frame_ids.size and frame_ids[-1] - frame_ids[0] + 1 == frame_ids.size:
            episode_data = slice(int(frame_ids[0]), int(frame_ids[-1]) + 1)
        else:
            episode_data = mask
        trajectories[int(ep_idx)] = {
            "xyz": poses[episode_data, :3],
            "rpy": poses[episode_data, 3:6],
            "actions": actions[episode_data],
            "gripper": gripper_states[episode_data],
            "gripper_signal": gripper_signal[episode_data],
        }

    # Capped (--max-points): take a deterministic spread sample across the
    # whole group list so a large index still renders a representative,
    # smooth scatter.  The same capped set feeds the raw-pair accepted set.
    all_groups = index.content.cross_point_groups
    n_total_groups = len(all_groups)
    if max_points is not None and 0 < max_points < n_total_groups:
        sample_indices = np.unique(
            np.linspace(0, n_total_groups - 1, int(max_points))
            .round().astype(np.int64)
        )
        groups = [all_groups[i] for i in sample_indices]
        logger.info(
            "Capping visualization to %d / %d cross-point groups (--max-points=%d)",
            len(groups), n_total_groups, max_points,
        )
    else:
        groups = all_groups

    member_pairs: List[VizMemberPair] = []
    cluster_labels: list[int] = []
    for group in groups:
        if len(group.members) != 2:
            raise ValueError(f"Group '{group.group_id}' must contain exactly two members.")
        member_0, member_1 = group.members
        member_pair = VizMemberPair(
            group_id=group.group_id,
            member_ids=[_format_member_id(member_0), _format_member_id(member_1)],
            member_episodes=[
                int(member_0.source_episode_index),
                int(member_1.source_episode_index),
            ],
            member_indices=[
                int(member_0.source_frame_index),
                int(member_1.source_frame_index),
            ],
            member_poses=[
                list(np.asarray(poses[int(member_0.source_frame_index)], dtype=np.float32)),
                list(np.asarray(poses[int(member_1.source_frame_index)], dtype=np.float32)),
            ],
            action_divergence=float(group.group_metrics.get("action_divergence", 0.0)),
            cluster_id=int(group.group_metrics.get("cluster_id", -1)),
        )
        member_pairs.append(member_pair)
        cluster_labels.append(member_pair.cluster_id)

    n_pairs = len(member_pairs)
    midpoints = np.empty((n_pairs, 6), dtype=np.float64) if n_pairs > 0 else np.empty((0, 6))
    divergences = np.empty(n_pairs, dtype=np.float64) if n_pairs > 0 else np.empty(0)
    for idx, member_pair in enumerate(member_pairs):
        point_0 = np.asarray(member_pair.member_poses[0])
        point_1 = np.asarray(member_pair.member_poses[1])
        midpoints[idx] = (point_0 + point_1) / 2.0
        divergences[idx] = float(member_pair.action_divergence)

    cluster_labels_arr = np.asarray(cluster_labels, dtype=np.int64) if cluster_labels else np.empty(0, dtype=np.int64)
    local_density = _compute_local_point_density(midpoints[:, :3]) if len(midpoints) else np.empty(0, dtype=np.int32)
    frame_neighbors: Dict[int, List[int]] = {}
    for member_pair in member_pairs:
        frame_neighbors.setdefault(member_pair.member_indices[0], []).append(member_pair.member_indices[1])
        frame_neighbors.setdefault(member_pair.member_indices[1], []).append(member_pair.member_indices[0])

    cluster_info = [group.group_metrics for group in groups]

    logger.info(
        "Prepared visualization data: %d episodes, %d groups, %d labels",
        len(trajectories),
        n_pairs,
        len(np.unique(cluster_labels_arr)) if len(cluster_labels_arr) else 0,
    )

    result = {
        "trajectories": trajectories,
        "midpoints": midpoints,
        "divergences": divergences,
        "cluster_labels": cluster_labels_arr,
        "local_density": local_density,
        "frame_neighbors": frame_neighbors,
        "member_pairs": member_pairs,
        "cluster_info": cluster_info,
    }

    if config is not None:
        raw_data = compute_raw_pair_data(
            index,
            dataset,
            config,
            groups=groups,
            poses=poses,
            episode_indices=episode_indices,
            actions=actions,
        )
        result.update(raw_data)
        result["has_raw_data"] = bool(raw_data.get("n_raw", 0) > 0)
    else:
        result["has_raw_data"] = False

    return result


def _compute_local_point_density(points_xyz: np.ndarray, radius: float | None = None, k_neighbors: int = 16) -> np.ndarray:
    if points_xyz.size == 0:
        return np.empty(0, dtype=np.int32)
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3).")
    if len(points_xyz) == 1:
        return np.zeros(1, dtype=np.int32)
    resolved_radius = float(radius) if radius is not None else _estimate_density_radius(points_xyz, k_neighbors=k_neighbors)
    if resolved_radius <= 0.0:
        return np.zeros(len(points_xyz), dtype=np.int32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points_xyz)
        try:
            # SciPy can return counts directly, avoiding an intermediate
            # Python list of neighbor-index lists (which is substantial for
            # large raw-pair visualizations).
            neighbor_counts = tree.query_ball_point(
                points_xyz, r=resolved_radius, return_length=True,
            )
            return np.maximum(np.asarray(neighbor_counts, dtype=np.int32) - 1, 0)
        except TypeError:
            # Keep compatibility with older SciPy releases.
            neighbor_ids = tree.query_ball_point(points_xyz, r=resolved_radius)
            return np.asarray([max(len(ids) - 1, 0) for ids in neighbor_ids], dtype=np.int32)
    except Exception as exc:
        logger.warning("Falling back to brute-force density computation: %s", exc)
        return _compute_local_point_density_bruteforce(points_xyz, radius=resolved_radius)


def _estimate_density_radius(points_xyz: np.ndarray, k_neighbors: int = 16) -> float:
    if len(points_xyz) < 2:
        return 0.0
    effective_k = max(2, min(int(k_neighbors), len(points_xyz)))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points_xyz)
        distances, _ = tree.query(points_xyz, k=effective_k)
        if distances.ndim == 1:
            distances = distances[:, None]
        kth_distances = distances[:, -1]
        positive_distances = kth_distances[kth_distances > 0.0]
        if positive_distances.size > 0:
            return float(np.median(positive_distances))
    except Exception:
        pass
    centroid = np.mean(points_xyz, axis=0)
    workspace_scale = np.linalg.norm(points_xyz - centroid, axis=1)
    positive_scale = workspace_scale[workspace_scale > 0.0]
    if positive_scale.size == 0:
        return 0.0
    return float(np.median(positive_scale) * 0.1)


def _compute_local_point_density_bruteforce(points_xyz: np.ndarray, radius: float, chunk_size: int = 1024) -> np.ndarray:
    del chunk_size
    if points_xyz.size == 0:
        return np.empty(0, dtype=np.int32)
    result = np.zeros(len(points_xyz), dtype=np.int32)
    radius_sq = float(radius) ** 2
    for idx, point in enumerate(points_xyz):
        distances_sq = np.sum((points_xyz - point) ** 2, axis=1)
        result[idx] = int(np.count_nonzero(distances_sq <= radius_sq) - 1)
    return result
