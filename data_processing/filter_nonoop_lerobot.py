import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from embodied_mixup.datasets import (
    default_repo_id_from_dataset_root,
    require_lerobot_dataset_root,
)

LOGGER = logging.getLogger(__name__)


class EpisodeLeRobotDataset(Dataset):
    def __init__(self, lerobot_dataset):
        self.dataset = lerobot_dataset
        self.num_episodes = lerobot_dataset.num_episodes
        self.meta = lerobot_dataset.meta
        self.features = lerobot_dataset.features

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, episode_index):
        from_idx = self.dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = self.dataset.meta.episodes["dataset_to_index"][episode_index]
        return [self.dataset[i] for i in range(from_idx, to_idx)]


def compute_action_difference(action_prev, action_curr, norm_type=2):
    if not isinstance(action_prev, torch.Tensor):
        action_prev = torch.tensor(action_prev)
    if not isinstance(action_curr, torch.Tensor):
        action_curr = torch.tensor(action_curr)
    diff = action_curr - action_prev
    return torch.norm(diff, p=norm_type).item()


def filter_redundant_frames(episode_frames, threshold=0.01, action_key="action", norm_type=2):
    if len(episode_frames) <= 1:
        return episode_frames

    filtered_frames = []
    
    filtered_frames.append(episode_frames[0])
    last_kept_action = episode_frames[0][action_key]

    for frame in tqdm(episode_frames[1:], desc="Filtering frames", leave=False):
        current_action = frame[action_key]
        diff = compute_action_difference(last_kept_action, current_action, norm_type)
        if diff >= threshold:
            filtered_frames.append(frame)
            last_kept_action = current_action

    return filtered_frames


def main(args):
    source_root = require_lerobot_dataset_root(args.root)
    source_repo_id = (args.repo_id or default_repo_id_from_dataset_root(source_root)).rstrip("/")
    LOGGER.info(
        "Loading LeRobot dataset: repo_id=%s root=%s",
        source_repo_id,
        source_root,
    )
    src_dataset = LeRobotDataset(
        repo_id=source_repo_id,
        root=str(source_root),
    )
    episode_dataset = EpisodeLeRobotDataset(src_dataset)

    if not args.target_root:
        raise ValueError("--target_root must be provided.")
    target_root = Path(args.target_root).expanduser().resolve()
    target_repo_id = (args.target_repo_id or default_repo_id_from_dataset_root(target_root)).rstrip("/")
    LOGGER.info(
        "Creating target LeRobot dataset: target_repo_id=%s target_root=%s",
        target_repo_id,
        target_root,
    )
    new_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=src_dataset.fps,
        root=str(target_root) if target_root is not None else None,
        features={
            k: v for k, v in src_dataset.features.items() 
            if k.startswith('observation') or k == 'action'
        },
        image_writer_threads=16,
    )

    total_original_frames = 0
    total_filtered_frames = 0

    for episode_idx in tqdm(range(len(episode_dataset)), desc="Processing episodes"):
        episode_frames = episode_dataset[episode_idx]
        total_original_frames += len(episode_frames)

        filtered_episode = filter_redundant_frames(
            episode_frames,
            threshold=args.threshold,
            action_key=args.action_key,
            norm_type=args.norm_type
        )
        print(f"Episode {episode_idx}: Original frames = {len(episode_frames)}, Filtered frames = {len(filtered_episode)}")
        total_filtered_frames += len(filtered_episode)

        for frame in filtered_episode:
            frame_np = {}
            for k, v in frame.items():
                if not (k.startswith('observation') or k == 'action' or k == 'task'):
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[0] in (1, 3):
                    v = np.transpose(v, (1, 2, 0))
                frame_np[k] = v
            
            new_dataset.add_frame(frame_np)
        
        new_dataset.save_episode()

    reduction_rate = (1 - total_filtered_frames / total_original_frames) * 100
    print("\n" + "="*60)
    print("Filtering Complete!")
    print("="*60)
    print(f"Original total frames: {total_original_frames}")
    print(f"Filtered total frames: {total_filtered_frames}")
    print(f"Reduction: {reduction_rate:.2f}%")
    print(f"Threshold used: {args.threshold}")
    print("="*60)
    print(f"Dataset saved to: {target_root if target_root is not None else target_repo_id}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Filter redundant actions from LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default=None, help="Source LeRobot dataset repo ID")
    parser.add_argument("--root", type=str, required=True, help="Root directory for source dataset")
    parser.add_argument("--target_repo_id", type=str, default=None, help="Destination repo ID for the new dataset")
    parser.add_argument("--target_root", type=str, default=None, help="Root directory for saving the new dataset")
    parser.add_argument("--threshold", type=float, default=0.01, help="Threshold for filtering (keep if diff >= threshold)")
    parser.add_argument("--action_key", type=str, default="action", help="Key name for action in dataset")
    parser.add_argument("--norm_type", type=int, default=2, help="Norm type (1 or 2)")
    args = parser.parse_args()
    main(args)
