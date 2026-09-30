"""Save deterministic samples of the actual training image path, before normalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from embodied_mixup.policies.train import _build_train_config, _build_emix_dataset
from embodied_mixup.policies.utils.image_noise import add_image_noise


def sample_indices(length: int, seed: int, count: int = 4) -> list[int]:
    if length <= 0:
        raise ValueError('dataset has no frames')
    return sorted(map(int, np.random.default_rng(seed).choice(length, size=min(count, length), replace=False)))


def image_from_sample(sample: dict, *, noise, seed: int, step: int) -> Image.Image:
    key = 'observation.images.head_camera'
    frame = sample[key]
    if not isinstance(frame, torch.Tensor):
        frame = torch.as_tensor(frame)
    if frame.ndim == 4:
        frame = frame[-1]
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f'expected [3,H,W] or [T,3,H,W] image, got {tuple(frame.shape)}')
    if frame.max() > 1:
        frame = frame.float() / 255
    noisy = add_image_noise({key: frame.float()}, noise, seed=seed, step=step)[key]
    rgb = (noisy.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.train_config.read_text())
    cfg = _build_train_config(payload, args.train_config)
    if cfg.training_mode in ('emix', 'mixed'):
        dataset = _build_emix_dataset(cfg)
        source = 'emix_composed'
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, root=cfg.dataset_root)
        source = 'decoded_dataset'
    seed = int((payload.get('emix') or {}).get('training', {}).get('seed', 0))
    indices = sample_indices(len(dataset), seed)
    images = [image_from_sample(dataset[i], noise=cfg.image_noise, seed=seed, step=j)
              for j, i in enumerate(indices)]
    width, height = images[0].size
    sheet = Image.new('RGB', (width * len(images), height + 18), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, (index, image) in enumerate(zip(indices, images)):
        sheet.paste(image, (j * width, 18))
        draw.text((j * width + 2, 2), f'frame {index}', fill='black')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps({
        'source': source, 'indices': indices, 'image_noise': cfg.image_noise.to_dict(),
        'stage': 'decoded/composed plus augmentation, before policy normalization',
    }, indent=2))


if __name__ == '__main__':
    main()
