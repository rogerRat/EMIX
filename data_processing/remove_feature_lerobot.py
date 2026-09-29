import argparse
import logging

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import modify_features

from embodied_mixup.datasets import default_repo_id_from_dataset_root, require_lerobot_dataset_root

LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Remove specified features from a LeRobot dataset and write to output dir"
    )
    parser.add_argument("--repo_id", default=None, help="dataset repo id")
    parser.add_argument(
        "--root",
        required=True,
        help="dataset root directory",
    )
    parser.add_argument(
        "--remove_features",
        nargs="+",
        default=["observation.images.hand_camera"],
        help="one or more feature paths to remove (space-separated)",
    )
    parser.add_argument(
        "--output_dir",
        help="output directory for modified dataset",
    )

    args = parser.parse_args()

    dataset_root = require_lerobot_dataset_root(args.root)
    repo_id = (args.repo_id or default_repo_id_from_dataset_root(dataset_root)).rstrip("/")
    LOGGER.info("Loading LeRobot dataset: repo_id=%s root=%s", repo_id, dataset_root)
    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))
    modify_features(
        dataset, remove_features=args.remove_features, output_dir=args.output_dir
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] [%(levelname)s] %(message)s")
    main()
