"""Utilities for EmixIndex construction and persistence.

@input: source dataset objects and file paths.
@output: validated EmixIndex payloads and JSON on disk.
@scenario: Build the canonical index artifact for direct EmixIndex training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .emix_index import (
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
from .action_filter import run_action_filter
from .episode_utils import build_episode_bound_maps, build_episode_start_end
from .ids import make_emix_group_id, make_emix_index_id
from .loaders import load_actions, load_eef_poses, load_episode_indices, load_gripper_signal
from .pairs import build_frame_to_neighbors, run_neighbor_index_pairs

logger = logging.getLogger(__name__)
UTC_PLUS_8 = timezone(timedelta(hours=8))

_FINGERPRINT_INCLUDED_ROOTS = ["meta", "data", "videos"]
_FINGERPRINT_EXCLUDED_PREFIXES = {"logs", "outputs", "vis_out", "emix", "emix_files"}

# Above this pair count the fast sklearn HDBSCAN backend replaces the
# hdbscan-library backend (see _cluster_pair_arrays).
_SKLEARN_HDBSCAN_MIN_PAIRS = 2_000_000


def compute_source_dataset_fingerprint(dataset_root: str | Path) -> dict[str, Any]:
    """Compute a strict source dataset fingerprint.

    @input: dataset_root [str|Path], exact LeRobotDataset root.
    @output: dict with algorithm, included_roots, file_count, total_bytes,
        dataset_sha256, files.
    @scenario: Bind an EmixIndex to the exact source dataset content used for
        index construction and training.
    """

    dataset_root = Path(dataset_root).expanduser().resolve()
    logger.info("[candidates.index] Computing source dataset fingerprint for %s", dataset_root)

    file_records: list[dict[str, Any]] = []
    for root_name in _FINGERPRINT_INCLUDED_ROOTS:
        subtree = dataset_root / root_name
        if not subtree.exists():
            continue
        for fpath in sorted(subtree.rglob("*")):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(dataset_root).as_posix()
            # Skip excluded prefixes
            top = rel.split("/")[0]
            if top in _FINGERPRINT_EXCLUDED_PREFIXES:
                continue
            size = fpath.stat().st_size
            digest = hashlib.sha256()
            with fpath.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            file_records.append({"relative_path": rel, "sha256": digest.hexdigest(), "size_bytes": size})

    # Sort by POSIX relative path for determinism
    file_records.sort(key=lambda r: r["relative_path"])

    summary_json = json.dumps(file_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    dataset_sha256 = hashlib.sha256(summary_json).hexdigest()
    total_bytes = sum(r["size_bytes"] for r in file_records)

    logger.info(
        "[candidates.index] Source dataset fingerprint complete: files=%d bytes=%d sha256=%s",
        len(file_records),
        total_bytes,
        dataset_sha256,
    )

    return {
        "algorithm": "sha256",
        "included_roots": list(_FINGERPRINT_INCLUDED_ROOTS),
        "file_count": len(file_records),
        "total_bytes": total_bytes,
        "dataset_sha256": dataset_sha256,
        "files": file_records,
    }


def validate_source_dataset_fingerprint(
    expected: dict[str, Any],
    dataset_root: str | Path,
) -> None:
    """Validate a source dataset root against an expected fingerprint.

    @input: expected [dict], dataset_root [str|Path].
    @output: None on exact match; raises ValueError on mismatch.
    @scenario: Stop EMIX training when the source data no longer matches the EmixIndex.
    """

    dataset_root = Path(dataset_root).expanduser().resolve()
    logger.info("[emix.loader] Validating source dataset fingerprint for %s", dataset_root)
    actual = compute_source_dataset_fingerprint(dataset_root)
    expected_sha = expected.get("dataset_sha256", "")
    actual_sha = actual["dataset_sha256"]
    if expected_sha != actual_sha:
        raise ValueError(
            "Source dataset fingerprint mismatch\n"
            f"expected dataset_sha256={expected_sha}\n"
            f"actual dataset_sha256={actual_sha}\n"
            f"dataset_root={dataset_root}"
        )
    logger.info("[emix.loader] Source dataset fingerprint validated")


def inspect_source_dataset(
    dataset,
    root: str | Path | None = None,
) -> tuple[EmixSourceDatasetDeclaration, dict[str, Any]]:
    """Inspect one source dataset and build its declaration with strict fingerprint.

    @input: dataset [LeRobotDataset-like object], root [path-like|None].
    @output: (EmixSourceDatasetDeclaration, fingerprint_dict).
    @scenario: Capture source dataset repo, path, strict sha256 fingerprint, and media contract.
    """

    repo_id = str(getattr(dataset, "repo_id", "") or "")
    root_value = root if root is not None else getattr(dataset, "root", None)
    if root_value is None:
        raise ValueError("A source dataset root is required.")
    dataset_root = Path(root_value).expanduser().resolve()
    if not repo_id:
        raise ValueError("dataset.repo_id is required for EmixIndex construction.")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {dataset_root}")

    features = getattr(getattr(dataset, "hf_dataset", None), "features", {}) or {}
    meta_features = getattr(getattr(dataset, "meta", None), "features", {}) or {}
    meta_camera_keys = getattr(getattr(dataset, "meta", None), "camera_keys", []) or []

    raw_camera_keys = set()
    for key in list(features.keys()) + list(meta_features.keys()) + list(meta_camera_keys):
        if key.startswith("observation.images."):
            raw_camera_keys.add(key.split("observation.images.", 1)[1])

    camera_keys = sorted(raw_camera_keys)
    if not camera_keys:
        raise ValueError("No observation.images.* camera keys were found in the source dataset.")

    media = EmixSourceMediaContract(
        camera_keys=camera_keys,
        frame_alignment="per_frame",
        video_path_pattern="videos/observation.images.{camera_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        storage_format="mp4",
    )

    fingerprint = compute_source_dataset_fingerprint(dataset_root)
    source_sha256 = fingerprint["dataset_sha256"]

    declaration = EmixSourceDatasetDeclaration(
        repo_id=repo_id,
        root=str(dataset_root),
        hash=source_sha256,
        media=media,
    )
    return declaration, fingerprint


def _effective_candidate_pair_cap(config: Any, episode_count: int) -> int | None:
    """Resolve the candidate-pair cap for a collection of ``episode_count`` episodes.

    The absolute ``max_candidate_pairs`` overrides; the per-episode ratio
    ``max_candidate_pairs_per_episode`` scales the cap with collection size.
    With both set the smaller wins; with neither set capping is disabled.

    @input: config [EmbodiedMixupConfig], episode_count [int] >= 0.
    @output: effective cap in pairs, or None for uncapped builds.
    @scenario: Keep clustering/serialization bounded on dense collections.
    """
    absolute = getattr(config, "max_candidate_pairs", None)
    per_episode = getattr(config, "max_candidate_pairs_per_episode", None)
    candidates = []
    if absolute is not None:
        candidates.append(int(absolute))
    if per_episode is not None:
        candidates.append(int(np.ceil(per_episode * max(0, episode_count))))
    return min(candidates) if candidates else None


def common_start_point(poses: np.ndarray, episode_indices: np.ndarray) -> np.ndarray:
    """Estimate the arm's shared start (home) xyz from episode first frames.

    Every episode launches from the same scripted home pose, so the median of
    the per-episode first-frame EEF positions is a robust estimate; a stray
    episode that starts elsewhere cannot drag the point.

    @input: poses (N, >=3) EEF xyz+rpy, episode_indices (N,) sorted frame->episode.
    @output: (3,) float xyz of the common start point.
    @scenario: exclude_start_radius filtering needs the reference point.
    """
    _, first_frames = np.unique(episode_indices, return_index=True)
    return np.median(np.asarray(poses)[first_frames, :3], axis=0)


def apply_start_point_filter(
    filtered: Any,
    poses: np.ndarray,
    episode_indices: np.ndarray,
    config: Any,
) -> dict[str, Any] | None:
    """Drop pairs whose midpoint xyz is within ``exclude_start_radius`` of home.

    Mutates the mapping in place (array- or list-valued) and returns filter
    info for the index metadata, or None when the filter is disabled.  Runs
    before the candidate-pair cap so the cap subsamples surviving pairs.

    @input: filtered pair mapping from run_action_filter[_arrays], poses (N, >=3),
    episode_indices (N,), config [EmbodiedMixupConfig].
    @output: {"start_point_xyz": (3,), "removed": int} or None.
    @scenario: pairs clustered at the shared start supervise "wait at home"
    instead of task behaviour, so they are removed from the index entirely.
    """
    radius = getattr(config, "exclude_start_radius", None)
    if radius is None:
        return None
    start = common_start_point(poses, episode_indices)
    if isinstance(filtered, dict):
        anchors = np.asarray(filtered["anchor_idx"])
        neighbors = np.asarray(filtered["neighbor_idx"])
    else:  # object path: list of CandidateRecord
        anchors = np.asarray([pair.anchor_idx for pair in filtered])
        neighbors = np.asarray([pair.neighbor_idx for pair in filtered])
    midpoints = (np.asarray(poses)[anchors, :3] + np.asarray(poses)[neighbors, :3]) / 2.0
    keep = np.linalg.norm(midpoints - start, axis=1) >= radius
    removed = int((~keep).sum())
    if removed and isinstance(filtered, dict):
        for key in filtered:
            filtered[key] = filtered[key][keep]
    elif removed:
        filtered[:] = [pair for pair, keep_i in zip(filtered, keep) if keep_i]
    logger.info(
        "[candidates.index] Start-point filter (r=%.3f m, home=%s): removed %d of %d pairs",
        radius,
        np.round(start, 4).tolist(),
        removed,
        int(anchors.shape[0]),
    )
    return {"start_point_xyz": start.tolist(), "removed": removed}


def apply_pair_time_span_filter(
    filtered: Any,
    episode_indices: np.ndarray,
    config: Any,
) -> dict[str, Any] | None:
    """Drop pairs whose frames are too far apart in within-episode time.

    The span of a pair is |t_anchor - t_neighbor| where t is the frame's time
    position inside its own episode; pairs are kept only when the span is at
    most ``max_pair_time_ratio`` x the mean episode length of the collection.
    Mutates the mapping in place and returns filter info, or None when the
    filter is disabled.  Runs before the start-point filter and the
    candidate-pair cap.

    @input: filtered pair mapping from run_action_filter[_arrays] (or list of
    CandidateRecord), episode_indices (N,) sorted frame->episode, config.
    @output: {"mean_episode_frames": float, "removed": int} or None.
    @scenario: long-span pairs mix different task phases (start x finish) and
    produce phase-mismatched pseudo-multimodal supervision.
    """
    ratio = getattr(config, "max_pair_time_ratio", None)
    if ratio is None:
        return None
    episode_indices = np.asarray(episode_indices)
    n = episode_indices.shape[0]
    boundaries = np.flatnonzero(np.not_equal(episode_indices[1:], episode_indices[:-1]))
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries + 1, [n]))
    mean_len = float((ends - starts).mean())
    slot = np.searchsorted(starts, np.arange(n), side="right") - 1
    time_pos = np.arange(n, dtype=np.float64) - starts[slot]
    if isinstance(filtered, dict):
        anchors = np.asarray(filtered["anchor_idx"])
        neighbors = np.asarray(filtered["neighbor_idx"])
    else:  # object path: list of CandidateRecord
        anchors = np.asarray([pair.anchor_idx for pair in filtered])
        neighbors = np.asarray([pair.neighbor_idx for pair in filtered])
    span = np.abs(time_pos[anchors] - time_pos[neighbors])
    keep = span <= ratio * mean_len
    removed = int((~keep).sum())
    if removed and isinstance(filtered, dict):
        for key in filtered:
            filtered[key] = filtered[key][keep]
    elif removed:
        filtered[:] = [pair for pair, keep_i in zip(filtered, keep) if keep_i]
    logger.info(
        "[candidates.index] Pair time-span filter (ratio=%.2f, mean episode=%.1f frames): "
        "removed %d of %d pairs",
        ratio,
        mean_len,
        removed,
        int(anchors.shape[0]),
    )
    return {"mean_episode_frames": mean_len, "removed": removed}


def build_emix_index(
    dataset,
    config: Any,
) -> EmixIndex:
    """Build a canonical EmixIndex from source data.

    @input: dataset [LeRobotDataset-like], config [EmbodiedMixupConfig].
    @output: EmixIndex object with schema v2 and strict source fingerprint.
    @scenario: Convert pair-based discovery into group-based canonical index records.
    """
    source_dataset, fingerprint = inspect_source_dataset(
        dataset, root=getattr(dataset, "root", None)
    )
    logger.info("[candidates.index] Starting EmixIndex build for dataset '%s'", source_dataset.repo_id)
    logger.info("[candidates.index] Loading dataset arrays and episode boundaries...")
    poses = load_eef_poses(dataset)
    episode_indices = load_episode_indices(dataset)
    actions = load_actions(dataset)
    gripper_signal = load_gripper_signal(actions, dataset=dataset)
    episode_starts, episode_ends = build_episode_start_end(episode_indices)
    episode_count = int(episode_starts.shape[0])
    episode_start_map, episode_end_map = build_episode_bound_maps(
        episode_indices=episode_indices,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
    )
    gripper_states = np.sign(gripper_signal)

    raw_neighbor_pairs = run_neighbor_index_pairs(
        poses=poses,
        episode_indices=episode_indices,
        gripper_states=gripper_states,
        config=config,
    )
    logger.info(
        "[candidates.index] Neighbor search produced %d raw candidate pairs",
        len(raw_neighbor_pairs),
    )
    raw_neighbor_pair_count = len(raw_neighbor_pairs)
    candidate_records = run_action_filter(
        neighbor_pairs=raw_neighbor_pairs,
        poses=poses,
        actions=actions,
        gripper_signal=gripper_signal,
        episode_indices=episode_indices,
        episode_end_map=episode_end_map,
        config=config,
    )
    del raw_neighbor_pairs
    time_span_info = apply_pair_time_span_filter(candidate_records, episode_indices, config)
    start_filter_info = apply_start_point_filter(candidate_records, poses, episode_indices, config)
    uncapped_pair_count = len(candidate_records)
    cap = _effective_candidate_pair_cap(config, episode_count)
    if cap is not None and uncapped_pair_count > cap:
        rng = np.random.default_rng(config.training_candidates_seed)
        selected = np.sort(rng.choice(uncapped_pair_count, size=cap, replace=False))
        candidate_records = [candidate_records[i] for i in selected]
        logger.info(
            "[candidates.index] Candidate pair cap %d: subsampled %d of %d pairs (seed=%d)",
            cap,
            cap,
            uncapped_pair_count,
            config.training_candidates_seed,
        )
    logger.info(
        "[candidates.index] Action filter retained %d candidate pairs; starting clustering...",
        len(candidate_records),
    )
    cluster_labels, cluster_info = _cluster_candidate_records(candidate_records, config)
    frame_to_neighbors = build_frame_to_neighbors(candidate_records)
    logger.info(
        "[candidates.index] Clustering finished with %d clusters; building %d cross-point groups...",
        len([c for c in np.unique(cluster_labels) if c >= 0]),
        len(candidate_records),
    )

    created_at = datetime.now(UTC_PLUS_8).strftime("%Y-%m-%dT%H:%M:%S%z")
    config_snapshot = {
        "xyz_epsilon": config.xyz_epsilon,
        "rpy_epsilon": config.rpy_epsilon,
        "exclude_tail_seconds": config.exclude_tail_seconds,
        "dataset_fps": config.dataset_fps,
        "exclude_tail_frames": config.exclude_tail_frames,
        "future_horizon": config.future_horizon,
        "future_angle_thresh": config.future_angle_thresh,
        "min_future_disp": config.min_future_disp,
        "action_delta_threshold": config.action_delta_threshold,
        "gripper_action_filter_enabled": config.gripper_action_filter_enabled,
        "hdbscan_min_cluster_size_ratio": config.hdbscan_min_cluster_size_ratio,
        "hdbscan_min_samples": config.hdbscan_min_samples,
        "hdbscan_cluster_selection_epsilon": config.hdbscan_cluster_selection_epsilon,
        "hdbscan_alpha": config.hdbscan_alpha,
        "hdbscan_cluster_selection_method": config.hdbscan_cluster_selection_method,
        "max_candidate_pairs": config.max_candidate_pairs,
        "max_candidate_pairs_per_episode": config.max_candidate_pairs_per_episode,
        "exclude_start_radius": config.exclude_start_radius,
        "max_pair_time_ratio": config.max_pair_time_ratio,
        "training_candidates_total_budget": config.training_candidates_total_budget,
        "training_candidates_cap_per_cluster": config.training_candidates_cap_per_cluster,
        "training_candidates_include_noise": config.training_candidates_include_noise,
        "training_candidates_seed": config.training_candidates_seed,
    }
    index_id = make_emix_index_id(
        source_repo_id=source_dataset.repo_id,
        dataset_length=int(len(dataset.hf_dataset)),
        source_dataset_sha256=fingerprint["dataset_sha256"],
        config_snapshot=config_snapshot,
    )

    groups: list[EmixCrossPointGroup] = []
    group_progress_interval = _compute_progress_interval(len(candidate_records))
    for idx, pair in enumerate(candidate_records):
        if _should_log_progress(idx, len(candidate_records), group_progress_interval):
            logger.info(
                "[candidates.index] Group build progress: %d/%d candidate pairs processed",
                idx + 1,
                len(candidate_records),
            )
        members = [
            EmixGroupMemberRef(
                source_dataset_repo_id=source_dataset.repo_id,
                source_episode_index=int(pair.anchor_episode),
                source_frame_index=int(pair.anchor_idx),
            ),
            EmixGroupMemberRef(
                source_dataset_repo_id=source_dataset.repo_id,
                source_episode_index=int(pair.neighbor_episode),
                source_frame_index=int(pair.neighbor_idx),
            ),
        ]

        abort_reasons: list[str] = []
        state = "training"
        cluster_id = int(cluster_labels[idx]) if idx < len(cluster_labels) else -1
        if cluster_id < 0:
            abort_reasons.append("cluster_id=-1")
            state = "aborted"
        if not _validate_group_members(dataset, members, source_dataset.media.camera_keys):
            abort_reasons.append("frame_alignment_failed")
            state = "aborted"

        groups.append(
            EmixCrossPointGroup(
                group_id=make_emix_group_id(source_dataset.repo_id, members, idx),
                member_count=len(members),
                members=members,
                state=EmixGroupState(state=state, abort_reasons=abort_reasons),
                group_metrics={
                    "cluster_id": cluster_id,
                    "pose_distance_xyz": float(pair.pose_distance_xyz),
                    "pose_distance_rot": float(pair.pose_distance_rpy),
                    "action_divergence": float(pair.action_divergence),
                    "local_neighbors": int(len(frame_to_neighbors.get(pair.anchor_idx, []))),
                },
                build_trace={
                    "source_pair_index": int(idx),
                    "candidate_id": str(pair.candidate_id),
                },
            )
        )

    public = EmixIndexPublic(
        id=index_id,
        schema_version=EMIX_INDEX_SCHEMA_VERSION,
        created_at=created_at,
        source_datasets=[source_dataset],
        source_dataset_version=EMIX_DATASET_VERSION,
        metadata={
            "dataset_repo_id": source_dataset.repo_id,
            "dataset_root": source_dataset.root,
            "source_dataset_fingerprint": fingerprint,
            "total_frames": int(len(dataset.hf_dataset)),
            "total_episodes": episode_count,
            "total_raw_neighbor_pairs": raw_neighbor_pair_count,
            "total_pairs": len(candidate_records),
            "total_pairs_uncapped": uncapped_pair_count,
            "candidate_pair_cap": cap,
            "pair_time_span_removed": (time_span_info or {}).get("removed"),
            "mean_episode_frames": (time_span_info or {}).get("mean_episode_frames"),
            "start_point_xyz": (start_filter_info or {}).get("start_point_xyz"),
            "start_filter_removed": (start_filter_info or {}).get("removed"),
            "num_clusters": len([c for c in np.unique(cluster_labels) if c >= 0]),
            "num_noise": int(np.sum(np.asarray(cluster_labels) == -1)),
            "total_groups": len(groups),
            "config": config_snapshot,
        },
    )
    logger.info(
        "[candidates.index] EmixIndex build complete: %d groups, %d clusters, %d noise pairs",
        len(groups),
        len([c for c in np.unique(cluster_labels) if c >= 0]),
        int(np.sum(np.asarray(cluster_labels) == -1)),
    )
    return EmixIndex(public=public, content=EmixIndexContent(cross_point_groups=groups))


def build_emix_index_streaming(dataset, config: Any, output_path: str | Path) -> Path:
    """Build the canonical EmixIndex and stream it to JSON without materializing per-pair objects.

    Functionally identical to ``build_emix_index`` + ``save_emix_index`` (same
    pair gates, same clustering parameters, same JSON structure, same
    deterministic group ids), but every stage is vectorized and the group
    records are written to disk one at a time.  At 500-episode scale the
    object path holds tens of millions of dataclasses plus a full-tree
    ``asdict`` during ``json.dump`` and exhausts host memory; the streaming
    path peaks at a few compact NumPy arrays.

    Two documented equivalence substitutions:
    - ``_validate_group_members``'s per-pair ``dataset[frame]`` decode is
      replaced by a one-time feature-schema check plus a vectorized bounds
      check.  Camera-key presence in a frame dict is invariant across frames
      (keys derive from ``hf_dataset.features``), so the outcome per pair is
      unchanged; only the O(pairs) video decodes are gone.
    - Per-pair ``local_neighbors`` comes from a ``np.bincount`` degree array
      over the undirected pair graph, which is exactly
      ``build_frame_to_neighbors``'s list lengths.

    @input: dataset [LeRobotDataset-like], config [EmbodiedMixupConfig],
        output_path [str|Path] for emix_index.json.
    @output: Path to the written index (atomic: tmp file + rename).
    @scenario: Flow-scale index building (100M+ raw pairs) on one host.
    """
    import gc

    source_dataset, fingerprint = inspect_source_dataset(
        dataset, root=getattr(dataset, "root", None)
    )
    logger.info(
        "[candidates.index] Starting streaming EmixIndex build for dataset '%s'",
        source_dataset.repo_id,
    )
    logger.info("[candidates.index] Loading dataset arrays and episode boundaries...")
    poses = load_eef_poses(dataset)
    episode_indices = load_episode_indices(dataset)
    actions = load_actions(dataset)
    gripper_signal = load_gripper_signal(actions, dataset=dataset)
    episode_starts, episode_ends = build_episode_start_end(episode_indices)
    episode_count = int(episode_starts.shape[0])
    _episode_start_map, episode_end_map = build_episode_bound_maps(
        episode_indices=episode_indices,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
    )
    del episode_starts, episode_ends
    gripper_states = np.sign(gripper_signal)

    from .pairs import run_neighbor_index_pairs_arrays

    anchors, neighbors = run_neighbor_index_pairs_arrays(
        poses=poses,
        episode_indices=episode_indices,
        gripper_states=gripper_states,
        config=config,
    )
    raw_neighbor_pair_count = int(anchors.shape[0])
    logger.info(
        "[candidates.index] Neighbor search produced %d raw candidate pairs",
        raw_neighbor_pair_count,
    )

    from .action_filter import precompute_future_indices, run_action_filter_arrays

    future_idx = precompute_future_indices(
        episode_indices=episode_indices,
        episode_end_map=episode_end_map,
        future_horizon=config.future_horizon,
    )
    filtered = run_action_filter_arrays(
        anchors=anchors,
        neighbors=neighbors,
        poses=poses,
        actions=actions,
        gripper_signal=gripper_signal,
        episode_indices=episode_indices,
        future_idx=future_idx,
        config=config,
    )
    del anchors, neighbors, future_idx
    gc.collect()
    time_span_info = apply_pair_time_span_filter(filtered, episode_indices, config)
    start_filter_info = apply_start_point_filter(filtered, poses, episode_indices, config)
    uncapped_pair_count = int(filtered["anchor_idx"].shape[0])

    # Candidate-pair cap: beyond a few hundred thousand pairs the marginal
    # EMIX value is nil while clustering/serialization cost explodes.  Keep a
    # seeded uniform subsample (sorted, so pair order and group ids stay
    # deterministic) and record the pre-cap count in the metadata.
    cap = _effective_candidate_pair_cap(config, episode_count)
    if cap is not None and uncapped_pair_count > cap:
        rng = np.random.default_rng(config.training_candidates_seed)
        selected = np.sort(rng.choice(uncapped_pair_count, size=cap, replace=False))
        filtered = {key: value[selected] for key, value in filtered.items()}
        del selected
        logger.info(
            "[candidates.index] Candidate pair cap %d: subsampled %d of %d pairs (seed=%d)",
            cap,
            cap,
            uncapped_pair_count,
            config.training_candidates_seed,
        )
    total_pairs = int(filtered["anchor_idx"].shape[0])
    logger.info(
        "[candidates.index] Action filter retained %d candidate pairs; starting clustering...",
        total_pairs,
    )

    cluster_labels = _cluster_pair_arrays(filtered, poses, config)
    gc.collect()
    num_clusters = int(np.unique(cluster_labels[cluster_labels >= 0]).size)
    num_noise = int(np.count_nonzero(cluster_labels == -1))

    # Degree of every frame in the undirected pair graph (== the length of
    # build_frame_to_neighbors' per-frame lists).
    frame_degree = np.bincount(
        np.concatenate((filtered["anchor_idx"], filtered["neighbor_idx"])).astype(np.int64),
        minlength=poses.shape[0],
    ).astype(np.int32)

    config_snapshot = {
        "xyz_epsilon": config.xyz_epsilon,
        "rpy_epsilon": config.rpy_epsilon,
        "exclude_tail_seconds": config.exclude_tail_seconds,
        "dataset_fps": config.dataset_fps,
        "exclude_tail_frames": config.exclude_tail_frames,
        "future_horizon": config.future_horizon,
        "future_angle_thresh": config.future_angle_thresh,
        "min_future_disp": config.min_future_disp,
        "action_delta_threshold": config.action_delta_threshold,
        "gripper_action_filter_enabled": config.gripper_action_filter_enabled,
        "hdbscan_min_cluster_size_ratio": config.hdbscan_min_cluster_size_ratio,
        "hdbscan_min_samples": config.hdbscan_min_samples,
        "hdbscan_cluster_selection_epsilon": config.hdbscan_cluster_selection_epsilon,
        "hdbscan_alpha": config.hdbscan_alpha,
        "hdbscan_cluster_selection_method": config.hdbscan_cluster_selection_method,
        "max_candidate_pairs": config.max_candidate_pairs,
        "max_candidate_pairs_per_episode": config.max_candidate_pairs_per_episode,
        "exclude_start_radius": config.exclude_start_radius,
        "max_pair_time_ratio": config.max_pair_time_ratio,
        "training_candidates_total_budget": config.training_candidates_total_budget,
        "training_candidates_cap_per_cluster": config.training_candidates_cap_per_cluster,
        "training_candidates_include_noise": config.training_candidates_include_noise,
        "training_candidates_seed": config.training_candidates_seed,
    }
    index_id = make_emix_index_id(
        source_repo_id=source_dataset.repo_id,
        dataset_length=int(len(dataset.hf_dataset)),
        source_dataset_sha256=fingerprint["dataset_sha256"],
        config_snapshot=config_snapshot,
    )
    created_at = datetime.now(UTC_PLUS_8).strftime("%Y-%m-%dT%H:%M:%S%z")
    public_payload = EmixIndexPublic(
        id=index_id,
        schema_version=EMIX_INDEX_SCHEMA_VERSION,
        created_at=created_at,
        source_datasets=[source_dataset],
        source_dataset_version=EMIX_DATASET_VERSION,
        metadata={
            "dataset_repo_id": source_dataset.repo_id,
            "dataset_root": source_dataset.root,
            "source_dataset_fingerprint": fingerprint,
            "total_frames": int(len(dataset.hf_dataset)),
            "total_episodes": episode_count,
            "total_raw_neighbor_pairs": raw_neighbor_pair_count,
            "total_pairs": total_pairs,
            "total_pairs_uncapped": uncapped_pair_count,
            "candidate_pair_cap": cap,
            "pair_time_span_removed": (time_span_info or {}).get("removed"),
            "mean_episode_frames": (time_span_info or {}).get("mean_episode_frames"),
            "start_point_xyz": (start_filter_info or {}).get("start_point_xyz"),
            "start_filter_removed": (start_filter_info or {}).get("removed"),
            "num_clusters": num_clusters,
            "num_noise": num_noise,
            "total_groups": total_pairs,
            "config": config_snapshot,
        },
    ).to_dict()

    camera_keys = source_dataset.media.camera_keys
    member_frames_ok = _frame_membership_validator(dataset, camera_keys)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    progress_interval = _compute_progress_interval(total_pairs)
    group_encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"))
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write('{"content": {"cross_point_groups": [')
        for idx in range(total_pairs):
            group_payload = _pair_group_payload(
                filtered=filtered,
                poses=poses,
                episode_indices=episode_indices,
                cluster_labels=cluster_labels,
                frame_degree=frame_degree,
                source_repo_id=source_dataset.repo_id,
                pair_index=idx,
                member_frames_ok=member_frames_ok,
            )
            if idx:
                handle.write(", ")
            for chunk in group_encoder.iterencode(group_payload):
                handle.write(chunk)
            if _should_log_progress(idx, total_pairs, progress_interval):
                logger.info(
                    "[candidates.index] Group stream progress: %d/%d candidate pairs written",
                    idx + 1,
                    total_pairs,
                )
        handle.write(']}, "public": ')
        json.dump(public_payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("}")
    os.replace(tmp_path, output_path)
    logger.info(
        "[candidates.index] EmixIndex build complete: %d groups, %d clusters, %d noise pairs -> %s",
        total_pairs,
        num_clusters,
        num_noise,
        output_path,
    )
    return output_path


def _cluster_candidate_records(candidate_records, config: Any) -> tuple[np.ndarray, list[dict]]:
    """Cluster candidate records using HDBSCAN."""

    if len(candidate_records) == 0:
        logger.info("[candidates.index] No candidate pairs available for clustering")
        return np.asarray([], dtype=np.int64), []

    record_count = len(candidate_records)
    required_neighbors = max(2, int(config.hdbscan_min_samples))
    if record_count < required_neighbors:
        labels = np.full(record_count, -1, dtype=np.int64)
        logger.info(
            "[candidates.index] Skipping HDBSCAN for %d candidate records; "
            "%d are required for its neighbor query.",
            record_count,
            required_neighbors,
        )
        cluster_info = _build_cluster_info(labels, candidate_records)
        for idx, label in enumerate(labels):
            candidate_records[idx].public.cluster_id = int(label)
            candidate_records[idx].relation.cluster_id = int(label)
        return labels, cluster_info

    try:
        import hdbscan
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "EmixIndex clustering requires optional dependencies 'hdbscan' and "
            "'scikit-learn' to be installed in the active environment."
        ) from exc

    midpoints = np.array(
        [
            (np.asarray(pair.anchor_pose) + np.asarray(pair.neighbor_pose)) / 2.0
            for pair in candidate_records
        ],
        dtype=np.float64,
    )
    logger.info(
        "[candidates.index] Prepared %d midpoint vectors for HDBSCAN clustering",
        len(midpoints),
    )
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(midpoints)
    min_cluster_size = max(2, int(len(candidate_records) * config.hdbscan_min_cluster_size_ratio))
    logger.info(
        "[candidates.index] Running HDBSCAN with min_cluster_size=%d, min_samples=%d, epsilon=%.6f, alpha=%.3f",
        min_cluster_size,
        int(config.hdbscan_min_samples),
        float(config.hdbscan_cluster_selection_epsilon),
        float(config.hdbscan_alpha),
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=int(config.hdbscan_min_samples),
        cluster_selection_epsilon=float(config.hdbscan_cluster_selection_epsilon),
        alpha=float(config.hdbscan_alpha),
        metric="euclidean",
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(x_scaled)
    logger.info("[candidates.index] HDBSCAN fit complete; aggregating cluster metadata...")
    cluster_info = _build_cluster_info(labels, candidate_records)
    for idx, label in enumerate(labels):
        candidate_records[idx].public.cluster_id = int(label)
        candidate_records[idx].relation.cluster_id = int(label)
    return labels, cluster_info


def _cluster_pair_arrays(filtered: dict[str, np.ndarray], poses: np.ndarray, config: Any) -> np.ndarray:
    """Cluster candidate pairs by pose midpoint, mirroring _cluster_candidate_records.

    Same HDBSCAN parameters, standardization, and float64 midpoint arithmetic
    as the scalar path; only the per-pair Python loop is gone.

    @input: filtered arrays from run_action_filter_arrays; poses (N, 6);
        config with the hdbscan_* fields.
    @output: (P,) int64 cluster labels, -1 for noise.
    @scenario: Streaming-build clustering stage.
    """
    pair_count = int(filtered["anchor_idx"].shape[0])
    if pair_count == 0:
        logger.info("[candidates.index] No candidate pairs available for clustering")
        return np.asarray([], dtype=np.int64)
    required_neighbors = max(2, int(config.hdbscan_min_samples))
    if pair_count < required_neighbors:
        logger.info(
            "[candidates.index] Skipping HDBSCAN for %d candidate records; "
            "%d are required for its neighbor query.",
            pair_count,
            required_neighbors,
        )
        return np.full(pair_count, -1, dtype=np.int64)

    try:
        import hdbscan
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "EmixIndex clustering requires optional dependencies 'hdbscan' and "
            "'scikit-learn' to be installed in the active environment."
        ) from exc

    pose_xyz = poses[:, :3]
    chunk_size = 5_000_000

    # External-labels fast path: a sidecar process (e.g. cuML HDBSCAN on GPU)
    # can cluster the staged midpoints; see EMIX_CLUSTER_STAGE_DIR below.
    labels_path = os.environ.get("EMIX_CLUSTER_LABELS_PATH")
    if labels_path:
        labels = np.load(labels_path).astype(np.int64, copy=False)
        if labels.shape[0] != pair_count:
            raise ValueError(
                f"External cluster labels count {labels.shape[0]} does not match "
                f"candidate pair count {pair_count}: {labels_path}"
            )
        logger.info(
            "[candidates.index] Loaded %d external cluster labels from %s (%d clusters)",
            labels.shape[0],
            labels_path,
            int(np.unique(labels[labels >= 0]).size),
        )
        return labels

    midpoints = np.empty((pair_count, 3), dtype=np.float64)
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        a = filtered["anchor_idx"][start:stop].astype(np.int64)
        b = filtered["neighbor_idx"][start:stop].astype(np.int64)
        midpoints[start:stop] = (pose_xyz[a] + pose_xyz[b]) / 2.0

    # Midpoint staging: hand the clustering problem to an external accelerator
    # (GPU cuML sidecar) instead of fitting a CPU backend that cannot finish
    # at tens of millions of points.  Midpoints are standardized before
    # staging so the sidecar needs no sklearn dependency.
    stage_dir = os.environ.get("EMIX_CLUSTER_STAGE_DIR")
    if stage_dir:
        os.makedirs(stage_dir, exist_ok=True)
        mean = midpoints.mean(axis=0)
        std = midpoints.std(axis=0)
        std[std == 0] = 1.0
        scaled = (midpoints - mean) / std
        np.save(os.path.join(stage_dir, "midpoints_scaled.npy"), scaled.astype(np.float32))
        np.save(os.path.join(stage_dir, "mean.npy"), mean)
        np.save(os.path.join(stage_dir, "std.npy"), std)
        with open(os.path.join(stage_dir, "params.json"), "w") as handle:
            json.dump(
                {
                    "pair_count": pair_count,
                    "min_cluster_size": max(
                        2, int(pair_count * config.hdbscan_min_cluster_size_ratio)
                    ),
                    "min_samples": int(config.hdbscan_min_samples),
                    "cluster_selection_epsilon": float(
                        config.hdbscan_cluster_selection_epsilon
                    ),
                    "cluster_selection_method": config.hdbscan_cluster_selection_method,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        del midpoints, scaled
        logger.info(
            "[candidates.index] Staged %d scaled midpoints and params in %s for external clustering",
            pair_count,
            stage_dir,
        )
        raise SystemExit(
            "EmixIndex clustering staged for external accelerator. Re-run the build "
            "with EMIX_CLUSTER_LABELS_PATH pointing at the sidecar's labels.npy."
        )
    logger.info(
        "[candidates.index] Prepared %d midpoint vectors for HDBSCAN clustering",
        pair_count,
    )
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(midpoints)
    del midpoints
    min_cluster_size = max(2, int(pair_count * config.hdbscan_min_cluster_size_ratio))
    logger.info(
        "[candidates.index] Running HDBSCAN with min_cluster_size=%d, min_samples=%d, epsilon=%.6f, alpha=%.3f",
        min_cluster_size,
        int(config.hdbscan_min_samples),
        float(config.hdbscan_cluster_selection_epsilon),
        float(config.hdbscan_alpha),
    )
    # Backend selection: the hdbscan library's Boruvka implementation degrades
    # to effectively single-threaded hours-long runs past a few million points
    # (observed: 24.2M midpoints still unfinished after 38 min at <8% CPU).
    # sklearn's HDBSCAN is the same algorithm (Campello et al.) with a
    # multithreaded implementation and matches the hdbscan library on
    # identical inputs; alpha is only supported by the hdbscan library, and
    # alpha=1.0 (the default) is a no-op scaling factor, so the sklearn path
    # is equivalent for it.
    if pair_count >= _SKLEARN_HDBSCAN_MIN_PAIRS:
        if abs(float(config.hdbscan_alpha) - 1.0) > 1e-12:
            raise ValueError(
                "hdbscan_alpha != 1.0 requires the hdbscan-library backend "
                f"(pair count {pair_count} below {_SKLEARN_HDBSCAN_MIN_PAIRS}); "
                "the fast sklearn backend only supports the alpha=1.0 default."
            )
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=int(config.hdbscan_min_samples),
            cluster_selection_epsilon=float(config.hdbscan_cluster_selection_epsilon),
            metric="euclidean",
            cluster_selection_method=config.hdbscan_cluster_selection_method,
        )
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=int(config.hdbscan_min_samples),
            cluster_selection_epsilon=float(config.hdbscan_cluster_selection_epsilon),
            alpha=float(config.hdbscan_alpha),
            metric="euclidean",
            cluster_selection_method=config.hdbscan_cluster_selection_method,
            # Keep all parallelism inside this single process (shared-memory
            # joblib backend); spawned worker processes would duplicate the
            # point arrays and re-create the OOM this path replaces.
            core_dist_n_jobs=os.cpu_count() or 1,
            approx_min_span_tree=True,
        )
    labels = np.asarray(clusterer.fit_predict(x_scaled), dtype=np.int64)
    del x_scaled, clusterer
    logger.info(
        "[candidates.index] HDBSCAN fit complete; %d clusters found",
        int(np.unique(labels[labels >= 0]).size),
    )
    return labels


def _frame_membership_validator(dataset, camera_keys: list[str]):
    """Return a vectorized membership check for member frame indices.

    The scalar path called ``dataset[frame]`` per pair member and tested
    camera-key presence in the returned frame dict.  Frame dict keys are
    derived from ``hf_dataset.features`` and are therefore identical for every
    frame, so a one-time schema check plus a bounds test produces the same
    per-pair outcome without decoding any video.
    """

    hf = getattr(dataset, "hf_dataset", None)
    if hf is None:
        return lambda indices: np.zeros(np.asarray(indices).shape, dtype=bool)
    total_frames = len(hf)
    # Newer LeRobot keeps video-backed features in meta.features rather than
    # hf_dataset.features; the scalar path's dataset[frame] dict merges both.
    meta_features = getattr(getattr(dataset, "meta", None), "features", {}) or {}
    schema_ok = all(
        f"observation.images.{camera_key}" in hf.features
        or f"observation.images.{camera_key}" in meta_features
        for camera_key in camera_keys
    )

    def validator(frame_indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(frame_indices, dtype=np.int64)
        in_bounds = (indices >= 0) & (indices < total_frames)
        if not schema_ok:
            return np.zeros(indices.shape, dtype=bool)
        return in_bounds

    return validator


def _pair_group_payload(
    filtered: dict[str, np.ndarray],
    poses: np.ndarray,
    episode_indices: np.ndarray,
    cluster_labels: np.ndarray,
    frame_degree: np.ndarray,
    source_repo_id: str,
    pair_index: int,
    member_frames_ok,
) -> dict:
    """Build one group's JSON dict, equivalent to the object-path serialization.

    ``group_id`` replicates :func:`ids.make_emix_group_id` on plain dicts
    (same json.dumps payload, same md5 truncation) so ids match the dataclass
    build for identical members and salt.  ``local_neighbors`` reads the
    bincount degree of the anchor frame, which equals the length of
    ``build_frame_to_neighbors``' list for that frame.
    """
    del poses, episode_indices  # pair metrics are precomputed in filtered arrays
    anchor_idx = int(filtered["anchor_idx"][pair_index])
    neighbor_idx = int(filtered["neighbor_idx"][pair_index])
    anchor_episode = int(filtered["anchor_episode"][pair_index])
    neighbor_episode = int(filtered["neighbor_episode"][pair_index])

    members = [
        {
            "source_dataset_repo_id": source_repo_id,
            "source_episode_index": anchor_episode,
            "source_frame_index": anchor_idx,
        },
        {
            "source_dataset_repo_id": source_repo_id,
            "source_episode_index": neighbor_episode,
            "source_frame_index": neighbor_idx,
        },
    ]
    raw = json.dumps(
        {"source_repo_id": source_repo_id, "members": members, "salt": pair_index},
        sort_keys=True,
    )
    group_id = f"emix_group_{hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]}"

    cluster_id = int(cluster_labels[pair_index]) if pair_index < len(cluster_labels) else -1
    abort_reasons: list[str] = []
    state = "training"
    if cluster_id < 0:
        abort_reasons.append("cluster_id=-1")
        state = "aborted"
    frames_ok = bool(
        member_frames_ok(np.array([anchor_idx, neighbor_idx], dtype=np.int64)).all()
    )
    if not frames_ok:
        abort_reasons.append("frame_alignment_failed")
        state = "aborted"

    return {
        "group_id": group_id,
        "member_count": 2,
        "members": members,
        "state": {
            "state": state,
            "abort_reasons": abort_reasons,
            "training": state == "training" and not abort_reasons,
            "aborted": state == "aborted",
        },
        "group_metrics": {
            "cluster_id": cluster_id,
            "pose_distance_xyz": float(filtered["pose_distance_xyz"][pair_index]),
            "pose_distance_rot": float(filtered["pose_distance_rot"][pair_index]),
            "action_divergence": float(filtered["action_divergence"][pair_index]),
            "local_neighbors": int(frame_degree[anchor_idx]),
        },
        "build_trace": {
            "source_pair_index": int(pair_index),
            "candidate_id": f"neighbor_{anchor_idx}_{neighbor_idx}",
        },
    }


def _compute_progress_interval(total_items: int) -> int:
    """Return a coarse logging interval for long-running loops."""
    if total_items <= 0:
        return 1
    return max(1, total_items // 20)


def _should_log_progress(
    item_idx: int,
    total_items: int,
    progress_interval: int,
) -> bool:
    """Emit progress on first item, fixed intervals, and last item."""
    if total_items <= 0:
        return False
    if item_idx == 0 or item_idx == total_items - 1:
        return True
    return ((item_idx + 1) % progress_interval) == 0


def _build_cluster_info(labels: np.ndarray, pairs) -> list[dict]:
    if len(labels) == 0:
        return []
    cluster_info: list[dict] = []
    for cluster_id in sorted(set(int(x) for x in labels)):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        cluster_pairs = [pairs[i] for i in np.where(mask)[0]]
        pose_dists = np.array([p.pose_distance_xyz for p in cluster_pairs], dtype=np.float64)
        action_divs = np.array([p.action_divergence for p in cluster_pairs], dtype=np.float64)
        rep_idx = int(np.argmin(pose_dists))
        cluster_info.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(mask.sum()),
                "mean_action_divergence": float(np.mean(action_divs)),
                "mean_pose_distance_xyz": float(np.mean(pose_dists)),
                "representative": cluster_pairs[rep_idx].to_dict(),
            }
        )
    return cluster_info


def save_emix_index(index: EmixIndex, path: str | Path) -> Path:
    """Save one EmixIndex to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(index.to_dict(), handle, indent=2, sort_keys=True)
    return path


def load_emix_index(path: str | Path) -> EmixIndex:
    """Load one EmixIndex from JSON."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EmixIndex file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return EmixIndex.from_dict(payload)


def _validate_group_members(dataset, members: list[EmixGroupMemberRef], camera_keys: list[str]) -> bool:
    """Validate that the referenced frames and cameras can be resolved."""

    hf = getattr(dataset, "hf_dataset", None)
    if hf is None:
        return False

    total_frames = len(hf)
    for member in members:
        if member.source_frame_index < 0 or member.source_frame_index >= total_frames:
            return False
        frame = dataset[member.source_frame_index]
        for camera_key in camera_keys:
            if f"observation.images.{camera_key}" not in frame:
                return False
    return True
