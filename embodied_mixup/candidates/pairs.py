"""Neighbor search and pair construction."""

import logging

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from ..configs import EmbodiedMixupConfig
from .schemas import CandidateParticipant, CandidateRecord, CandidateRelation, CandidatePublic

logger = logging.getLogger(__name__)


def tail_eligible_mask(episode_indices: np.ndarray, exclude_frames: int) -> np.ndarray:
    """Mark the frames allowed into pairing: everything but each episode's tail.

    The final home-return frames of an episode sit at the same pose as its
    start, so without the trim the search pairs "start task" with "go home"
    and teaches a spurious do-nothing mode.  Frames are stored
    episode-contiguous, so an episode ends where its index stops repeating.

    @input: episode_indices np.ndarray (N,), exclude_frames int.
    @output: (N,) bool mask, True where the frame may join a pair.
    """
    n = len(episode_indices)
    if exclude_frames <= 0 or n == 0:
        return np.ones(n, dtype=bool)
    is_episode_end = np.empty(n, dtype=bool)
    is_episode_end[-1] = True
    np.not_equal(episode_indices[1:], episode_indices[:-1], out=is_episode_end[:-1])
    episode_last = np.flatnonzero(is_episode_end)
    frame_episode_last = episode_last[
        np.searchsorted(episode_last, np.arange(n), side="left")
    ]
    return (frame_episode_last - np.arange(n)) >= exclude_frames


def run_neighbor_search(
    poses: np.ndarray,
    episode_indices: np.ndarray,
    gripper_states: np.ndarray,
    config: EmbodiedMixupConfig,
) -> list[CandidateRecord]:
    """Generate raw cross-episode neighbor pairs from EEF pose space."""
    xyz_tree = KDTree(poses[:, :3])
    return _query_neighbor_pairs(
        xyz_tree=xyz_tree,
        poses=poses,
        episode_indices=episode_indices,
        gripper_states=gripper_states,
        xyz_eps=config.xyz_epsilon,
        rpy_eps=config.rpy_epsilon,
        cross_episode_only=config.cross_episode_only,
        gripper_consistency=config.gripper_consistency,
        eligible=tail_eligible_mask(episode_indices, config.exclude_tail_frames),
    )


def run_neighbor_index_pairs(
    poses: np.ndarray,
    episode_indices: np.ndarray,
    gripper_states: np.ndarray,
    config: EmbodiedMixupConfig,
) -> list[tuple[int, int]]:
    """Return raw neighbor frame pairs without materializing candidate records.

    This is intentionally a separate API from :func:`run_neighbor_search`.
    Candidate construction is required by index building, but visualization
    only needs frame indices.  Keeping the neighbor list for one anchor at a
    time and returning compact integer pairs avoids retaining the nested
    ``CandidateRecord`` object graph used by the normal build path.
    """
    xyz_tree = KDTree(poses[:, :3])
    return _query_neighbor_index_pairs(
        xyz_tree=xyz_tree,
        poses=poses,
        episode_indices=episode_indices,
        gripper_states=gripper_states,
        xyz_eps=config.xyz_epsilon,
        rpy_eps=config.rpy_epsilon,
        cross_episode_only=config.cross_episode_only,
        gripper_consistency=config.gripper_consistency,
        eligible=tail_eligible_mask(episode_indices, config.exclude_tail_frames),
    )


def _query_neighbor_index_pairs(
    xyz_tree: KDTree,
    poses: np.ndarray,
    episode_indices: np.ndarray,
    gripper_states: np.ndarray,
    xyz_eps: float,
    rpy_eps: float,
    cross_episode_only: bool,
    gripper_consistency: bool,
    eligible: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Query valid pairs while keeping only compact frame-index tuples."""
    n_frames = poses.shape[0]
    pairs: list[tuple[int, int]] = []
    logger.info("[candidates.pairs] Querying position KD-Tree (r=%.4f m)...", xyz_eps)

    # Preserve the same neighbor traversal order as the normal record path so
    # raw visualization trace ordering remains unchanged.  The memory saving
    # comes from retaining compact integer pairs instead of CandidateRecord
    # objects with nested dataclasses, strings, and pose lists.
    for anchor_idx in tqdm(range(n_frames), desc="Filtering neighbors", leave=False):
        if eligible is not None and not eligible[anchor_idx]:
            continue
        candidate_indices = xyz_tree.query_ball_point(poses[anchor_idx, :3], r=xyz_eps)
        for neighbor_idx in candidate_indices:
            if neighbor_idx <= anchor_idx:
                continue
            if eligible is not None and not eligible[neighbor_idx]:
                continue
            if np.linalg.norm(poses[anchor_idx, 3:6] - poses[neighbor_idx, 3:6]) >= rpy_eps:
                continue
            if cross_episode_only and episode_indices[anchor_idx] == episode_indices[neighbor_idx]:
                continue
            if gripper_consistency and gripper_states[anchor_idx] != gripper_states[neighbor_idx]:
                continue
            pairs.append((anchor_idx, neighbor_idx))
    return pairs


def run_neighbor_index_pairs_arrays(
    poses: np.ndarray,
    episode_indices: np.ndarray,
    gripper_states: np.ndarray,
    config: EmbodiedMixupConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw neighbor frame pairs as compact int32 index arrays.

    Vectorized twin of :func:`run_neighbor_index_pairs` with identical gates
    (tail eligibility, rpy threshold, cross-episode, gripper consistency) and
    the same pair set, but no per-pair Python loop.  At 500-episode scale the
    tuple-list path retains >100M Python objects (~15 GB) and spends hours in
    the interpreter; the array path keeps two int32 arrays (~1 GB at 100M
    pairs) and runs the gates in NumPy.  Pairs are ordered by (anchor,
    neighbor) so downstream group ids are deterministic.

    @input: poses (N, 6) float64; episode_indices (N,); gripper_states (N,);
        config with xyz_epsilon / rpy_epsilon / cross_episode_only /
        gripper_consistency / exclude_tail_frames.
    @output: (anchor_idx, neighbor_idx) int32 arrays, anchor < neighbor.
    @scenario: Feed the action filter at 100M-pair scale within host memory.
    """
    xyz_tree = KDTree(poses[:, :3])
    eligible = tail_eligible_mask(episode_indices, config.exclude_tail_frames)
    logger.info(
        "[candidates.pairs] Querying position KD-Tree pairs (r=%.4f m)...", config.xyz_epsilon
    )
    # output_type='ndarray' keeps the raw pair set in one C-backed (P, 2) array.
    raw_pairs = xyz_tree.query_pairs(r=config.xyz_epsilon, output_type="ndarray")
    logger.info(
        "[candidates.pairs] Position radius query produced %d unordered pairs",
        raw_pairs.shape[0],
    )

    anchor_arr = np.asarray(raw_pairs[:, 0], dtype=np.int64)
    neighbor_arr = np.asarray(raw_pairs[:, 1], dtype=np.int64)
    del raw_pairs

    # Apply the remaining gates in chunks so each boolean temporary stays small.
    chunk_size = 20_000_000
    keep_parts: list[np.ndarray] = []
    rpy_eps = config.rpy_epsilon
    for start in range(0, anchor_arr.shape[0], chunk_size):
        stop = min(start + chunk_size, anchor_arr.shape[0])
        a = anchor_arr[start:stop]
        b = neighbor_arr[start:stop]
        keep = eligible[a] & eligible[b]
        if rpy_eps is not None:
            rpy_sq = np.einsum(
                "ij,ij->i",
                poses[a, 3:6] - poses[b, 3:6],
                poses[a, 3:6] - poses[b, 3:6],
            )
            keep &= rpy_sq < (rpy_eps * rpy_eps)
        if config.cross_episode_only:
            keep &= episode_indices[a] != episode_indices[b]
        if config.gripper_consistency:
            keep &= gripper_states[a] == gripper_states[b]
        keep_parts.append(keep.nonzero()[0] + start)
        del a, b, keep
    kept = (
        np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    )
    del keep_parts

    anchors = anchor_arr[kept].astype(np.int32, copy=False)
    neighbors = neighbor_arr[kept].astype(np.int32, copy=False)
    del anchor_arr, neighbor_arr, kept
    # Deterministic (anchor, neighbor) ordering for stable group ids.
    order = np.lexsort((neighbors, anchors))
    anchors = anchors[order]
    neighbors = neighbors[order]
    del order
    logger.info(
        "[candidates.pairs] Raw gate filtering retained %d candidate pairs",
        anchors.shape[0],
    )
    return anchors, neighbors


def build_frame_to_neighbors(pairs: list[CandidateRecord]) -> dict[int, list[int]]:
    """Build frame-to-neighbor lookup from pair records."""
    frame_to_neighbors: dict[int, list[int]] = {}
    for pair in pairs:
        anchor_idx = int(pair.anchor_idx)
        neighbor_idx = int(pair.neighbor_idx)
        frame_to_neighbors.setdefault(anchor_idx, []).append(neighbor_idx)
        frame_to_neighbors.setdefault(neighbor_idx, []).append(anchor_idx)
    return frame_to_neighbors


def _query_neighbor_pairs(
    xyz_tree: KDTree,
    poses: np.ndarray,
    episode_indices: np.ndarray,
    gripper_states: np.ndarray,
    xyz_eps: float,
    rpy_eps: float,
    cross_episode_only: bool,
    gripper_consistency: bool,
    eligible: np.ndarray | None = None,
) -> list[CandidateRecord]:
    """Run KD-tree search and construct pair records that satisfy pose thresholds."""
    n_frames = poses.shape[0]
    pairs: list[CandidateRecord] = []
    logger.info("[candidates.pairs] Querying position KD-Tree (r=%.4f m)...", xyz_eps)

    for anchor_idx in tqdm(range(n_frames), desc="Filtering neighbors", leave=False):
        # Query one anchor at a time. Materializing all N neighbor lists is
        # quadratic in dense repeated trajectories and exhausts host memory.
        if eligible is not None and not eligible[anchor_idx]:
            continue
        candidate_indices = xyz_tree.query_ball_point(poses[anchor_idx, :3], r=xyz_eps)

        for neighbor_idx in candidate_indices:
            # Every unordered pair is visited once, which removes the need for
            # a second unbounded Python set of seen pair ids.
            if neighbor_idx <= anchor_idx:
                continue
            if eligible is not None and not eligible[neighbor_idx]:
                continue

            rpy_dist = np.linalg.norm(poses[anchor_idx, 3:6] - poses[neighbor_idx, 3:6])
            if rpy_dist >= rpy_eps:
                continue

            if cross_episode_only and episode_indices[anchor_idx] == episode_indices[neighbor_idx]:
                continue

            if gripper_consistency and gripper_states[anchor_idx] != gripper_states[neighbor_idx]:
                continue

            xyz_dist = np.linalg.norm(poses[anchor_idx, :3] - poses[neighbor_idx, :3])
            anchor_pose = poses[anchor_idx].astype(np.float32).tolist()
            neighbor_pose = poses[neighbor_idx].astype(np.float32).tolist()
            anchor_participant = CandidateParticipant(
                role="participant_a",
                episode_index=int(episode_indices[anchor_idx]),
                frame_index=int(anchor_idx),
                branch_frame_in_window=0,
                pose=anchor_pose,
                local_window={},
            )
            neighbor_participant = CandidateParticipant(
                role="participant_b",
                episode_index=int(episode_indices[neighbor_idx]),
                frame_index=int(neighbor_idx),
                branch_frame_in_window=0,
                pose=neighbor_pose,
                local_window={},
            )
            candidate_id = f"neighbor_{anchor_idx}_{neighbor_idx}"
            pairs.append(
                CandidateRecord(
                    public=CandidatePublic(
                        candidate_id=candidate_id,
                        sample_id=candidate_id,
                        dataset_repo_id="",
                        source_dataset_id="",
                        schema_version=2,
                        build_stage="neighbor_search",
                        build_config={},
                    ),
                    relation=CandidateRelation(
                        source_pair_index=len(pairs),
                        anchor_participant=anchor_participant,
                        neighbor_participant=neighbor_participant,
                        pose_distance_xyz=float(xyz_dist),
                        pose_distance_rot=float(rpy_dist),
                        action_divergence=0.0,
                    ),
                )
            )

    return pairs
