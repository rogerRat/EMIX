"""Action-based filtering for raw neighbor pairs."""

import logging
import math
from typing import Any

import numpy as np

from ..configs import EmbodiedMixupConfig
from .episode_utils import safe_future_index
from .schemas import (
    CandidateMedia,
    CandidateParticipant,
    CandidatePublic,
    CandidateRecord,
    CandidateRelation,
)

logger = logging.getLogger(__name__)


def precompute_future_indices(
    episode_indices: np.ndarray,
    episode_end_map: dict[int, int],
    future_horizon: int,
) -> np.ndarray:
    """Vectorized :func:`safe_future_index` over every frame.

    @input: episode_indices (N,); episode_end_map episode id -> exclusive end;
        future_horizon positive frame offset.
    @output: (N,) int32 array of future frame indices, -1 where the horizon
        would cross the episode end (the safe_future_index None case).
    @scenario: Replace per-pair future-index lookups in the vectorized filter.
    """
    n = episode_indices.shape[0]
    future_idx = np.arange(n, dtype=np.int64) + int(future_horizon)
    # Frames are episode-contiguous, so the episode id changes exactly at the
    # starts; map each frame onto its episode slot and read the shared end.
    boundaries = np.flatnonzero(np.not_equal(episode_indices[1:], episode_indices[:-1]))
    episode_starts = np.concatenate(([0], boundaries + 1))
    episode_ends = np.concatenate((boundaries + 1, [n]))
    slot = np.searchsorted(episode_starts, np.arange(n), side="right") - 1
    frame_episode_end = episode_ends[slot]
    ok = future_idx < frame_episode_end
    out = np.where(ok, future_idx, -1)
    # Cross-check against the scalar reference for the map-based end values.
    if episode_end_map:
        map_ends = np.array(
            [episode_end_map.get(int(ep), -1) for ep in np.unique(episode_indices)],
            dtype=np.int64,
        )
        map_slots = np.searchsorted(np.unique(episode_indices), episode_indices)
        mismatch = (map_ends[map_slots] >= 0) & (map_ends[map_slots] != frame_episode_end)
        if mismatch.any():
            raise ValueError(
                "Episode end map disagrees with episode_indices contiguity at "
                f"{int(mismatch.sum())} frames."
            )
    return out.astype(np.int32)


def run_action_filter_arrays(
    anchors: np.ndarray,
    neighbors: np.ndarray,
    poses: np.ndarray,
    actions: np.ndarray,
    gripper_signal: np.ndarray,
    episode_indices: np.ndarray,
    future_idx: np.ndarray,
    config,
) -> dict[str, np.ndarray]:
    """Vectorized twin of :func:`run_action_filter` for compact pair arrays.

    Computes exactly the same metrics and validity flags per pair, in chunks,
    and returns only the pairs whose ``action_filter_valid`` is True.

    @input: anchors/neighbors (P,) int frame indices; poses (N, 6); actions
        (N, A) float32; gripper_signal (N,); episode_indices (N,); future_idx
        (N,) from :func:`precompute_future_indices`; config with the filter
        thresholds.
    @output: dict of compact arrays for surviving pairs: anchor_idx,
        neighbor_idx, anchor_episode, neighbor_episode, pose_distance_xyz,
        pose_distance_rot, action_divergence, future_angle_xyz, future_disp_a,
        future_disp_b, gripper_action_filter_flag, valid_by_future_angle,
        valid_by_current_action.
    @scenario: Keep the 100M-pair action filter inside host memory and NumPy
        speed instead of one dict-building Python call per pair.
    """
    pair_count = int(anchors.shape[0])
    pose_xyz = poses[:, :3]
    pose_rpy = poses[:, 3:6]
    chunk_size = 10_000_000
    out: dict[str, list[np.ndarray]] = {key: [] for key in (
        "anchor_idx", "neighbor_idx", "anchor_episode", "neighbor_episode",
        "pose_distance_xyz", "pose_distance_rot", "action_divergence",
        "future_angle_xyz", "future_disp_a", "future_disp_b",
        "gripper_action_filter_flag", "valid_by_future_angle",
        "valid_by_current_action",
    )}

    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        a = anchors[start:stop].astype(np.int64, copy=False)
        b = neighbors[start:stop].astype(np.int64, copy=False)

        # Current action divergence gate
        action_diff = actions[a] - actions[b]
        cad = np.sqrt(np.einsum("ij,ij->i", action_diff, action_diff, optimize=True))
        del action_diff
        current_valid = cad >= config.action_delta_threshold

        fa = future_idx[a]
        fb = future_idx[b]
        has_future = (fa >= 0) & (fb >= 0)

        # Future motion gates (safe on masked-out lanes; values discarded)
        fa_safe = np.where(has_future, fa, 0)
        fb_safe = np.where(has_future, fb, 0)
        v_a = pose_xyz[fa_safe] - pose_xyz[a]
        v_b = pose_xyz[fb_safe] - pose_xyz[b]
        disp_a = np.sqrt(np.einsum("ij,ij->i", v_a, v_a))
        disp_b = np.sqrt(np.einsum("ij,ij->i", v_b, v_b))
        enough_disp = (disp_a >= config.min_future_disp) & (disp_b >= config.min_future_disp)
        denom = np.maximum(disp_a * disp_b, 1e-12)
        cosine = np.clip(np.einsum("ij,ij->i", v_a, v_b) / denom, -1.0, 1.0)
        del v_a, v_b, denom
        angle = np.arccos(cosine)
        del cosine
        angle_valid = has_future & enough_disp & (angle >= config.future_angle_thresh)

        valid = current_valid & angle_valid

        # Gripper transition code: 1 opening, -1 closing, 0 steady.
        if config.gripper_action_filter_enabled:
            thr = 1e-4
            delta_a = (
                gripper_signal[np.where(has_future, fa_safe, a)] - gripper_signal[a]
            )
            delta_b = (
                gripper_signal[np.where(has_future, fb_safe, b)] - gripper_signal[b]
            )
            # Mirror compute_gripper_transition's dead-band exactly: only
            # deltas beyond ±1e-4 count as opening/closing.
            code_a = np.where(delta_a > thr, 1.0, np.where(delta_a < -thr, -1.0, 0.0))
            code_b = np.where(delta_b > thr, 1.0, np.where(delta_b < -thr, -1.0, 0.0))
            gripper_flag = code_a != code_b
            del delta_a, delta_b, code_a, code_b
        else:
            gripper_flag = np.zeros(stop - start, dtype=bool)

        keep = valid.nonzero()[0]
        out["anchor_idx"].append(a[keep].astype(np.int32))
        out["neighbor_idx"].append(b[keep].astype(np.int32))
        out["anchor_episode"].append(episode_indices[a[keep]].astype(np.int32))
        out["neighbor_episode"].append(episode_indices[b[keep]].astype(np.int32))
        out["pose_distance_xyz"].append(
            np.sqrt(np.einsum("ij,ij->i", pose_xyz[a[keep]] - pose_xyz[b[keep]],
                              pose_xyz[a[keep]] - pose_xyz[b[keep]]))
        )
        out["pose_distance_rot"].append(
            np.sqrt(np.einsum("ij,ij->i", pose_rpy[a[keep]] - pose_rpy[b[keep]],
                              pose_rpy[a[keep]] - pose_rpy[b[keep]]))
        )
        out["action_divergence"].append(cad[keep])
        out["future_angle_xyz"].append(angle[keep])
        out["future_disp_a"].append(disp_a[keep])
        out["future_disp_b"].append(disp_b[keep])
        out["gripper_action_filter_flag"].append(gripper_flag[keep])
        out["valid_by_future_angle"].append(angle_valid[keep])
        out["valid_by_current_action"].append(current_valid[keep])
        del a, b, fa, fb, fa_safe, fb_safe, disp_a, disp_b, angle, angle_valid
        del current_valid, valid, gripper_flag, keep

    result = {key: (np.concatenate(parts) if parts else np.empty(0)) for key, parts in out.items()}
    logger.info(
        "[candidates.action_filter] Retained %d action-filtered candidate pairs from %d raw neighbor pairs",
        result["anchor_idx"].shape[0],
        pair_count,
    )
    return result


def run_action_filter(
    neighbor_pairs: list[CandidateRecord | tuple[int, int]],
    poses: np.ndarray,
    actions: np.ndarray,
    gripper_signal: np.ndarray,
    episode_indices: np.ndarray,
    episode_end_map: dict[int, int],
    config: EmbodiedMixupConfig,
) -> list[CandidateRecord]:
    """Filter raw neighbor pairs into action-filtered candidate pairs.

    @input: neighbor_pairs is a list of raw frame-index pairs or CandidateRecord objects; poses is
        an (N, D) array with xyz in columns 0:3; actions is an (N, A) array;
        gripper_signal is an (N,) array; episode_indices maps each frame to an
        episode id; episode_end_map maps episode id to final frame index;
        config contains current-action and future-motion thresholds.
    @output: List of CandidateRecord objects that pass the action filter.
    @scenario: Keep nearby pose pairs only when they have enough current action
        divergence and enough short-horizon motion direction divergence.
    """
    candidate_records: list[CandidateRecord] = []
    for source_pair_index, pair in enumerate(neighbor_pairs):
        if isinstance(pair, tuple):
            anchor_idx, neighbor_idx = int(pair[0]), int(pair[1])
            anchor_episode = int(episode_indices[anchor_idx])
            neighbor_episode = int(episode_indices[neighbor_idx])
            pose_distance_xyz = float(
                np.linalg.norm(poses[anchor_idx, :3] - poses[neighbor_idx, :3])
            )
            pose_distance_rot = float(
                np.linalg.norm(poses[anchor_idx, 3:6] - poses[neighbor_idx, 3:6])
            )
            candidate_id = f"neighbor_{anchor_idx}_{neighbor_idx}"
            public = CandidatePublic(
                candidate_id=candidate_id,
                sample_id=candidate_id,
                dataset_repo_id="",
                source_dataset_id="",
                schema_version=2,
                build_stage="neighbor_search",
                build_config={},
            )
            anchor_participant = CandidateParticipant(
                role="participant_a",
                episode_index=anchor_episode,
                frame_index=anchor_idx,
                branch_frame_in_window=0,
                pose=poses[anchor_idx].astype(np.float32).tolist(),
                local_window={},
            )
            neighbor_participant = CandidateParticipant(
                role="participant_b",
                episode_index=neighbor_episode,
                frame_index=neighbor_idx,
                branch_frame_in_window=0,
                pose=poses[neighbor_idx].astype(np.float32).tolist(),
                local_window={},
            )
            media = CandidateMedia()
            source_pair_id = source_pair_index
        else:
            anchor_idx = int(pair.anchor_idx)
            neighbor_idx = int(pair.neighbor_idx)
            public = pair.public
            anchor_participant = pair.relation.anchor_participant
            neighbor_participant = pair.relation.neighbor_participant
            pose_distance_xyz = pair.pose_distance_xyz
            pose_distance_rot = pair.pose_distance_rpy
            media = pair.media
            source_pair_id = pair.relation.source_pair_index
        action_filter_metrics = compute_action_filter_metrics(
            poses=poses,
            actions=actions,
            gripper_signal=gripper_signal,
            episode_indices=episode_indices,
            episode_end_map=episode_end_map,
            anchor_idx=anchor_idx,
            neighbor_idx=neighbor_idx,
            future_horizon=config.future_horizon,
            min_future_disp=config.min_future_disp,
            future_angle_thresh=config.future_angle_thresh,
            action_delta_threshold=config.action_delta_threshold,
            gripper_action_filter_enabled=config.gripper_action_filter_enabled,
        )
        if not action_filter_metrics["action_filter_valid"]:
            continue

        action_filtered_pair = CandidateRecord(
            public=public,
            relation=CandidateRelation(
                source_pair_index=source_pair_id,
                anchor_participant=anchor_participant,
                neighbor_participant=neighbor_participant,
                pose_distance_xyz=pose_distance_xyz,
                pose_distance_rot=float(pose_distance_rot),
                action_divergence=float(action_filter_metrics["current_action_divergence"]),
                future_angle_xyz=float(action_filter_metrics["future_angle_xyz"]),
                future_disp_a=float(action_filter_metrics["future_disp_a"]),
                future_disp_b=float(action_filter_metrics["future_disp_b"]),
                gripper_action_filter_flag=bool(
                    action_filter_metrics["gripper_action_filter_flag"]
                ),
                action_filter_valid_by_future_angle=bool(
                    action_filter_metrics["future_angle_valid"]
                ),
                action_filter_valid_by_current_action=bool(
                    action_filter_metrics["current_action_valid"]
                ),
                action_filter_valid=bool(action_filter_metrics["action_filter_valid"]),
                cluster_id=-1,
                midpoint_pose=None,
            ),
            media=media,
        )
        candidate_records.append(action_filtered_pair)

    logger.info(
        "[candidates.action_filter] Retained %d action-filtered candidate pairs from %d raw neighbor pairs",
            len(candidate_records),
            len(neighbor_pairs),
        )
    return candidate_records


def compute_action_filter_metrics(
    poses: np.ndarray,
    actions: np.ndarray,
    gripper_signal: np.ndarray,
    episode_indices: np.ndarray,
    episode_end_map: dict[int, int],
    anchor_idx: int,
    neighbor_idx: int,
    future_horizon: int,
    min_future_disp: float,
    future_angle_thresh: float,
    action_delta_threshold: float,
    gripper_action_filter_enabled: bool,
) -> dict[str, Any]:
    """Compute action-filter metrics for one raw neighbor pair.

    @input: poses is an (N, D) array with xyz in columns 0:3; actions is an
        (N, A) array; gripper_signal is an (N,) array; episode_indices maps
        each frame to an episode id; episode_end_map maps episode id to final
        frame index; anchor_idx and neighbor_idx are valid frame indices;
        future_horizon is a positive frame offset; min_future_disp,
        future_angle_thresh, and action_delta_threshold are non-negative
        thresholds; gripper_action_filter_enabled controls diagnostic gripper
        transition computation.
    @output: Dict with scalar metrics and boolean validity flags.
    @scenario: Evaluate whether a pose-neighbor pair has both current action
        divergence and future xyz direction divergence.
    """
    current_action_divergence = float(
        np.linalg.norm(actions[anchor_idx] - actions[neighbor_idx])
    )
    current_action_valid = current_action_divergence >= action_delta_threshold
    anchor_future_idx = safe_future_index(
        frame_idx=anchor_idx,
        episode_indices=episode_indices,
        episode_end_map=episode_end_map,
        future_horizon=future_horizon,
    )
    neighbor_future_idx = safe_future_index(
        frame_idx=neighbor_idx,
        episode_indices=episode_indices,
        episode_end_map=episode_end_map,
        future_horizon=future_horizon,
    )

    if anchor_future_idx is None or neighbor_future_idx is None:
        return {
            "current_action_divergence": current_action_divergence,
            "current_action_valid": current_action_valid,
            "future_angle_xyz": 0.0,
            "future_disp_a": 0.0,
            "future_disp_b": 0.0,
            "future_angle_valid": False,
            "gripper_action_filter_flag": False,
            "action_filter_valid": False,
        }

    v_a = poses[anchor_future_idx, :3] - poses[anchor_idx, :3]
    v_b = poses[neighbor_future_idx, :3] - poses[neighbor_idx, :3]
    disp_a = float(np.linalg.norm(v_a))
    disp_b = float(np.linalg.norm(v_b))

    future_angle_valid = False
    future_angle = 0.0
    if disp_a >= min_future_disp and disp_b >= min_future_disp:
        denom = max(disp_a * disp_b, 1e-12)
        cosine = float(np.clip(np.dot(v_a, v_b) / denom, -1.0, 1.0))
        future_angle = float(math.acos(cosine))
        future_angle_valid = future_angle >= future_angle_thresh

    gripper_action_filter_flag = False
    if gripper_action_filter_enabled:
        anchor_transition = compute_gripper_transition(
            gripper_signal=gripper_signal,
            start_idx=anchor_idx,
            future_idx=anchor_future_idx,
        )
        neighbor_transition = compute_gripper_transition(
            gripper_signal=gripper_signal,
            start_idx=neighbor_idx,
            future_idx=neighbor_future_idx,
        )
        gripper_action_filter_flag = anchor_transition != neighbor_transition

    return {
        "future_angle_xyz": future_angle,
        "future_disp_a": disp_a,
        "future_disp_b": disp_b,
        "future_angle_valid": future_angle_valid,
        "current_action_divergence": current_action_divergence,
        "current_action_valid": current_action_valid,
        "gripper_action_filter_flag": gripper_action_filter_flag,
        "action_filter_valid": current_action_valid and future_angle_valid,
    }


def compute_gripper_transition(
    gripper_signal: np.ndarray,
    start_idx: int,
    future_idx: int,
) -> str:
    """Quantize short-horizon gripper behavior into a simple discrete label.

    @input: gripper_signal is an (N,) array; start_idx and future_idx are valid
        frame indices in the same episode.
    @output: One of "opening", "closing", "steady", or "unknown".
    @scenario: Record gripper transition diagnostics for one candidate pair.
    """
    if gripper_signal.size == 0:
        return "unknown"
    start_value = float(gripper_signal[start_idx])
    future_value = float(gripper_signal[future_idx])
    delta = future_value - start_value
    threshold = 1e-4
    if delta > threshold:
        return "opening"
    if delta < -threshold:
        return "closing"
    return "steady"
