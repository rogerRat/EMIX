import argparse
import logging
import random
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from embodied_mixup.datasets import (
    default_repo_id_from_dataset_root,
    require_lerobot_dataset_root,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a smaller LeRobot subset by episode deletion.")
    parser.add_argument("--repo_id", type=str, default=None, help="Source LeRobot dataset repo ID")
    parser.add_argument("--root", type=str, required=True, help="Source LeRobot dataset root")
    parser.add_argument("--target_repo_id", type=str, default=None, help="Destination LeRobot dataset repo ID")
    parser.add_argument("--target_root", type=str, required=True, help="Destination LeRobot dataset root")
    parser.add_argument("--sample_ratio", type=float, default=0.2, help="Fraction of episodes to keep")
    parser.add_argument("--num_episodes", type=int, default=None, help="Exact number of episodes to keep")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode sampling")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes is not None and args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive when provided")
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("--sample_ratio must be in the interval (0, 1]")


def compute_keep_count(total_episodes: int, args: argparse.Namespace) -> int:
    if args.num_episodes is not None:
        if args.num_episodes > total_episodes:
            raise ValueError("--num_episodes cannot exceed total episode count")
        return args.num_episodes
    keep_count = max(1, int(total_episodes * args.sample_ratio))
    return min(keep_count, total_episodes)


def create_subset_atomically(
    dataset: LeRobotDataset,
    target_root: Path,
    target_repo_id: str,
    keep_count: int,
    seed: int,
) -> None:
    if target_root.exists():
        raise FileExistsError(f"Target root already exists: {target_root}")

    temp_root = target_root.parent / f"{target_root.name}.tmp_seed{seed}"
    if temp_root.exists():
        raise FileExistsError(f"Temporary root already exists: {temp_root}")

    all_episode_indices = [int(ep["episode_index"]) for ep in dataset.meta.episodes]
    kept_episodes = sorted(random.Random(seed).sample(all_episode_indices, keep_count))
    kept_episode_set = set(kept_episodes)
    deleted_episodes = [ep_idx for ep_idx in all_episode_indices if ep_idx not in kept_episode_set]

    try:
        delete_episodes(
            dataset,
            deleted_episodes,
            output_dir=temp_root,
            repo_id=target_repo_id,
        )
        temp_root.rename(target_root)
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise


def main() -> None:
    args = parse_args()
    validate_args(args)

    source_root = require_lerobot_dataset_root(args.root)
    source_repo_id = (args.repo_id or default_repo_id_from_dataset_root(source_root)).rstrip("/")
    LOGGER.info(
        "Loading LeRobot dataset: repo_id=%s root=%s",
        source_repo_id,
        source_root,
    )

    dataset = LeRobotDataset(
        repo_id=source_repo_id,
        root=str(source_root),
    )
    total_episodes = len(dataset.meta.episodes)
    keep_count = compute_keep_count(total_episodes, args)
    print(f"Total episodes before deletion: {total_episodes}")
    print(f"Keeping {keep_count} episodes")

    target_root = Path(args.target_root).expanduser().resolve()
    target_repo_id = (args.target_repo_id or default_repo_id_from_dataset_root(target_root)).rstrip("/")
    LOGGER.info(
        "Creating target LeRobot dataset: target_repo_id=%s target_root=%s",
        target_repo_id,
        target_root,
    )

    create_subset_atomically(
        dataset=dataset,
        target_root=target_root,
        target_repo_id=target_repo_id,
        keep_count=keep_count,
        seed=args.seed,
    )
    print(f"Subset created at: {target_root}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] [%(levelname)s] %(message)s")
    main()
