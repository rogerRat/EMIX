"""Dataset loading helpers for candidate generation."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def load_eef_poses(dataset) -> np.ndarray:
    """Load absolute EEF poses from a preprocessed LeRobotDataset."""
    hf = dataset.hf_dataset

    if "observation.eef_pose" not in hf.features:
        available = [k for k in hf.features.keys() if "eef" in k.lower() or "pose" in k.lower()]
        raise KeyError(
            "Feature 'observation.eef_pose' not found in dataset. "
            f"Available related features: {available}. "
            f"All features: {list(hf.features.keys())}"
        )

    eef_poses = np.array(hf["observation.eef_pose"][:], dtype=np.float64)
    if eef_poses.ndim != 2 or eef_poses.shape[1] != 6:
        raise ValueError(
            f"Expected observation.eef_pose shape (N, 6), got {eef_poses.shape}."
        )

    logger.info("[candidates.loaders] Loaded %d EEF poses", eef_poses.shape[0])
    return eef_poses


def load_episode_indices(dataset) -> np.ndarray:
    """Extract per-frame episode indices from the dataset."""
    hf = dataset.hf_dataset
    try:
        # Bulk column read: one pass over parquet instead of one hf[i] row
        # access per frame (minutes at 200k+ frames).
        return np.asarray(hf["episode_index"], dtype=np.int64).reshape(-1)
    except (KeyError, ValueError, TypeError):
        return np.array(
            [hf[i]["episode_index"].item() for i in range(len(hf))],
            dtype=np.int64,
        )


def load_actions(dataset) -> np.ndarray:
    """Load all action vectors from the dataset."""
    return np.array(dataset.hf_dataset["action"][:], dtype=np.float32)


def load_gripper_signal(actions: np.ndarray, dataset=None) -> np.ndarray:
    """Load the explicit left gripper signal or fall back to the last action dimension.

    @input: actions np.ndarray (N, A); optional LeRobot dataset.
    @output: np.ndarray (N,) float32 gripper signal.
    @scenario: Use the active left Piper signal for RoboTwin while preserving old datasets.
    """
    if dataset is not None and "observation.left_gripper" in dataset.hf_dataset.features:
        signal = np.asarray(dataset.hf_dataset["observation.left_gripper"][:], dtype=np.float32)
        return signal.reshape(-1)
    if actions.ndim != 2 or actions.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if actions.shape[1] == 0:
        return np.zeros((actions.shape[0],), dtype=np.float32)
    return np.asarray(actions[:, -1], dtype=np.float32)


def load_optional_feature_window(
    dataset,
    feature_key: str,
    start_idx: int,
    end_idx: int,
):
    """Load one optional feature window from the dataset.

    Scalar parquet-backed features are loaded in bulk from the Hugging Face
    arrow dataset. Video or image features are resolved frame-by-frame through
    the LeRobot dataset access path, because they are stored outside parquet in
    the dataset's videos/ tree.
    """

    feature_meta = _get_feature_meta(dataset, feature_key)
    feature_dtype = str(feature_meta.get("dtype", ""))
    if feature_dtype in {"image", "video"}:
        return _load_video_backed_feature_window(dataset, feature_key, start_idx, end_idx)

    hf = dataset.hf_dataset
    if feature_key not in hf.features:
        raise KeyError(f"Feature '{feature_key}' not found in dataset.")
    return np.asarray(hf[feature_key][start_idx:end_idx])


def _get_feature_meta(dataset, feature_key: str) -> dict[str, Any]:
    """Resolve feature metadata from the dataset wrapper."""

    features = getattr(dataset, "features", None)
    if isinstance(features, dict) and feature_key in features:
        meta = features[feature_key]
        if isinstance(meta, dict):
            return meta

    hf = getattr(dataset, "hf_dataset", None)
    if hf is not None and hasattr(hf, "features") and feature_key in hf.features:
        meta = hf.features[feature_key]
        if isinstance(meta, dict):
            return meta

    raise KeyError(f"Feature '{feature_key}' not found in dataset metadata.")


def _load_video_backed_feature_window(dataset, feature_key: str, start_idx: int, end_idx: int) -> np.ndarray:
    """Load a video-backed feature as a contiguous frame window."""

    frames: list[np.ndarray] = []
    for frame_idx in range(start_idx, end_idx):
        item = dataset[frame_idx]
        if feature_key not in item:
            raise KeyError(
                f"Feature '{feature_key}' missing from dataset frame {frame_idx}."
            )
        value = item[feature_key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        if value.ndim == 3 and value.shape[0] in (1, 3) and value.shape[-1] not in (1, 3):
            value = np.transpose(value, (1, 2, 0))
        if value.dtype != np.uint8:
            value = np.clip(value * 255.0 if np.issubdtype(value.dtype, np.floating) else value, 0, 255).astype(np.uint8)
        frames.append(value)

    return np.asarray(frames)
