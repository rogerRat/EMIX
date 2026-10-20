import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from embodied_mixup.datasets import default_repo_id_from_dataset_root, require_lerobot_dataset_root
from embodied_mixup.policies.utils.bootstrap import bootstrap_lerobot_runtime

LOGGER = logging.getLogger(__name__)
UTC_PLUS_8 = timezone(timedelta(hours=8))
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def _set_default_hf_cache() -> None:
    """Set writable default HF cache locations.

    @input: None.
    @output: None; environment variables are populated when absent.
    @scenario: HuggingFace datasets may create lock files while loading local
        parquet metadata. In restricted workspaces, the default home cache may
        be read-only, so this script uses /tmp unless the caller set a cache.
    """
    os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")


def _backup_file(path: Path, stamp: str) -> Path:
    """Create a timestamped backup beside one file.

    @input: path Path for an existing file; stamp str timestamp suffix.
    @output: Path to the backup file.
    @scenario: Preserve original LeRobot metadata before overwriting stats.
    """
    backup_path = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def _stats_from_tensor(frames: torch.Tensor) -> dict[str, np.ndarray]:
    """Compute image stats from decoded frames.

    @input: frames torch.Tensor shaped (T, C, H, W) or (C, H, W), normally in
        float [0, 1] after LeRobot decoding.
    @output: dict mapping stat names to numpy arrays shaped like LeRobot image
        stats: (3, 1, 1), except count shaped (1,).
    @scenario: Recompute correct visual feature stats without uint8 overflow.
    """
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape {tuple(frames.shape)}")

    data = frames.detach().cpu().to(torch.float32).numpy()
    if data.max(initial=0.0) > 1.5:
        data = data / 255.0

    flat = np.transpose(data, (0, 2, 3, 1)).reshape(-1, data.shape[1]).astype(np.float64)
    stat: dict[str, np.ndarray] = {
        "min": flat.min(axis=0).reshape(3, 1, 1),
        "max": flat.max(axis=0).reshape(3, 1, 1),
        "mean": flat.mean(axis=0).reshape(3, 1, 1),
        "std": flat.std(axis=0).reshape(3, 1, 1),
        "count": np.array([data.shape[0]], dtype=np.int64),
    }
    for quantile in QUANTILES:
        stat[f"q{int(quantile * 100):02d}"] = np.quantile(flat, quantile, axis=0).reshape(3, 1, 1)
    return stat


def _episode_image_stats(dataset, episode_index: int, image_key: str) -> dict[str, np.ndarray]:
    """Decode one episode image feature and compute correct stats.

    @input: dataset LeRobotDataset; episode_index int; image_key str visual
        feature key from dataset.meta.camera_keys.
    @output: dict of LeRobot-compatible image stats for this episode/key.
    @scenario: Support both image-backed and video-backed LeRobot datasets by
        reading through LeRobotDataset.__getitem__.
    """
    episode = dataset.meta.episodes[episode_index]
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    dataset._ensure_hf_dataset_loaded()
    timestamps = [float(value) for value in dataset.hf_dataset[start:end]["timestamp"]]
    frames = dataset._query_videos({image_key: timestamps}, episode_index)[image_key]
    return _stats_from_tensor(frames)


def _load_existing_episode_stats(row: dict, image_keys: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Extract non-image stats from one episode metadata row.

    @input: row dict from an episodes parquet row; image_keys list[str] to
        exclude because they will be recomputed.
    @output: nested stats dict compatible with aggregate_stats.
    @scenario: Preserve existing non-visual stats while replacing visual stats.
    """
    result: dict[str, dict[str, np.ndarray]] = {}
    for column, value in row.items():
        if not column.startswith("stats/"):
            continue
        parts = column.split("/")
        if len(parts) < 3:
            continue
        feature_key = "/".join(parts[1:-1])
        stat_key = parts[-1]
        if feature_key in image_keys:
            continue
        result.setdefault(feature_key, {})[stat_key] = np.asarray(value)
    return result


def _update_episode_tables(root: Path, episode_stats: dict[int, dict[str, dict[str, np.ndarray]]], stamp: str) -> None:
    """Overwrite visual stats in LeRobot episode metadata parquet files.

    @input: root Path dataset root; episode_stats dict keyed by episode index;
        stamp str backup timestamp.
    @output: None; parquet files are backed up and overwritten.
    @scenario: Keep per-episode metadata consistent with corrected stats.json.
    """
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for path in episode_files:
        table = pq.read_table(path)
        df = table.to_pandas()
        changed = False
        for row_index, row in df.iterrows():
            episode_index = int(row["episode_index"])
            if episode_index not in episode_stats:
                continue
            for feature_key, stats in episode_stats[episode_index].items():
                for stat_key, value in stats.items():
                    column = f"stats/{feature_key}/{stat_key}"
                    if column in df.columns:
                        df.at[row_index, column] = value.tolist()
                        changed = True
        if changed:
            backup_path = _backup_file(path, stamp)
            LOGGER.info("Backed up episode metadata: %s", backup_path)
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
            LOGGER.info("Updated episode metadata: %s", path)


def fix_lerobot_image_stats(root: str | Path, repo_id: str | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Recompute image statistics and preserve all non-image statistics.

    @input: root str or Path exact LeRobot dataset root; repo_id optional
        dataset repo id. If omitted, use the dataset root name.
    @output: corrected full dataset stats dict.
    @scenario: Repair LeRobot datasets whose image std was corrupted by uint8
        overflow during stats generation without dropping state/action stats.
    """
    _set_default_hf_cache()
    bootstrap_lerobot_runtime()
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import load_stats, write_stats

    dataset_root = require_lerobot_dataset_root(root)
    resolved_repo_id = (repo_id or default_repo_id_from_dataset_root(dataset_root)).rstrip("/")
    LOGGER.info("Loading dataset: repo_id=%s root=%s", resolved_repo_id, dataset_root)
    dataset = LeRobotDataset(repo_id=resolved_repo_id, root=str(dataset_root))
    image_keys = list(dataset.meta.camera_keys)
    existing_stats = load_stats(dataset_root) or {}
    if not image_keys:
        LOGGER.info("No image/video features found; stats unchanged.")
        return existing_stats

    stamp = datetime.now(UTC_PLUS_8).strftime("%Y%m%d%H%M%S")
    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.exists():
        backup_path = _backup_file(stats_path, stamp)
        LOGGER.info("Backed up stats: %s", backup_path)

    per_episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    total_episodes = int(dataset.meta.total_episodes)
    LOGGER.info("Recomputing image stats for %d episodes and keys=%s", total_episodes, image_keys)
    for episode_index in range(total_episodes):
        row = dict(dataset.meta.episodes[episode_index])
        merged_stats = _load_existing_episode_stats(row, image_keys)
        for image_key in image_keys:
            merged_stats[image_key] = _episode_image_stats(dataset, episode_index, image_key)
        per_episode_stats[episode_index] = merged_stats
        if (episode_index + 1) % 25 == 0 or episode_index + 1 == total_episodes:
            LOGGER.info("Recomputed image stats: %d/%d episodes", episode_index + 1, total_episodes)

    corrected_image_stats = aggregate_stats(list(per_episode_stats.values()))
    corrected_stats = dict(existing_stats)
    for image_key in image_keys:
        corrected_stats[image_key] = corrected_image_stats[image_key]

    write_stats(corrected_stats, dataset_root)
    LOGGER.info("Updated stats.json: %s", stats_path)
    _update_episode_tables(dataset_root, per_episode_stats, stamp)

    for image_key in image_keys:
        LOGGER.info(
            "Corrected %s mean=%s std=%s",
            image_key,
            corrected_stats[image_key]["mean"].reshape(-1).tolist(),
            corrected_stats[image_key]["std"].reshape(-1).tolist(),
        )
    return corrected_stats


def main() -> None:
    """CLI entrypoint.

    @input: CLI args --root and optional --repo_id.
    @output: None; metadata files are backed up and overwritten in place.
    @scenario: Manually repair a LeRobot dataset before training.
    """
    parser = argparse.ArgumentParser(description="Fix LeRobot image stats by decoding images/videos.")
    parser.add_argument("--root", required=True, help="Exact LeRobot dataset root.")
    parser.add_argument("--repo_id", default=None, help="Dataset repo id. Defaults to root directory name.")
    args = parser.parse_args()
    fix_lerobot_image_stats(args.root, args.repo_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] [%(levelname)s] %(message)s")
    main()
