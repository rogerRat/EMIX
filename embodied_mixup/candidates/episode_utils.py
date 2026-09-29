"""Episode boundary and frame-window utilities."""

from typing import Optional

import numpy as np


def build_episode_start_end(episode_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build per-episode inclusive-exclusive frame bounds."""
    if episode_indices.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    unique_episodes, first_indices = np.unique(episode_indices, return_index=True)
    order = np.argsort(first_indices)
    first_indices = first_indices[order]
    unique_episodes = unique_episodes[order]

    episode_starts = np.asarray(first_indices, dtype=np.int64)
    episode_ends = np.empty_like(episode_starts)
    episode_ends[:-1] = episode_starts[1:]
    episode_ends[-1] = episode_indices.shape[0]

    if unique_episodes.shape[0] != episode_starts.shape[0]:
        raise ValueError("Failed to derive episode boundaries from episode indices.")

    return episode_starts, episode_ends


def build_episode_bound_maps(
    episode_indices: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
) -> tuple[dict[int, int], dict[int, int]]:
    """Build episode-id to start/end frame maps."""
    unique_episodes, first_indices = np.unique(episode_indices, return_index=True)
    order = np.argsort(first_indices)
    unique_episodes = unique_episodes[order]

    start_map = {
        int(ep_id): int(episode_starts[idx]) for idx, ep_id in enumerate(unique_episodes)
    }
    end_map = {
        int(ep_id): int(episode_ends[idx]) for idx, ep_id in enumerate(unique_episodes)
    }
    return start_map, end_map


def safe_future_index(
    frame_idx: int,
    episode_indices: np.ndarray,
    episode_end_map: dict[int, int],
    future_horizon: int,
) -> Optional[int]:
    """Return a future frame index without crossing episode boundaries."""
    if frame_idx < 0 or frame_idx >= episode_indices.shape[0]:
        return None

    episode_id = int(episode_indices[frame_idx])
    episode_end = episode_end_map.get(episode_id)
    if episode_end is None:
        episode_positions = np.where(episode_indices == episode_id)[0]
        if episode_positions.size == 0:
            return None
        episode_end = int(episode_positions[-1] + 1)

    future_idx = frame_idx + future_horizon
    if future_idx >= episode_end:
        return None
    return future_idx


def compute_window_bounds(
    frame_idx: int,
    window_len: int,
    episode_start: int,
    episode_end: int,
) -> tuple[int, int, int]:
    """Return local window bounds and branch offset inside the window."""
    half_window = window_len // 2
    window_start = max(episode_start, frame_idx - half_window)
    window_end = min(episode_end, window_start + window_len)
    window_start = max(episode_start, window_end - window_len)
    branch_frame_in_window = int(frame_idx - window_start)
    return int(window_start), int(window_end), branch_frame_in_window
