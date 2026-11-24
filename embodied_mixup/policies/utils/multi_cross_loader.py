"""Training samples from cross-task pair records with source-specific masks."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .emix_loader_registry_utils import hard_patch_image
from .patch_annotations import PatchAnnotationReader


class CrossTaskEmixDataset(Dataset):
    def __init__(self, pair_path, sources, temporal_spec, settings):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        payload = json.loads(Path(pair_path).read_text(encoding="utf-8"))
        ids = [source["repo_id"] for source in sources]
        if (payload.get("schema_version") != 1 or payload.get("pair_scope") != "cross_task"
                or payload.get("source_repo_ids") != ids or not payload.get("pairs")):
            raise ValueError("Invalid cross-task pair index")
        if temporal_spec.n_obs_steps != 1 or temporal_spec.action_steps != 50:
            raise ValueError("Cross-task pair index requires one observation and 50 action steps")
        if settings.detect_overlap_enable or settings.segment_wise_enable:
            raise ValueError("Cross-task masked overlap and segment filtering are not configured")
        self.pairs = payload["pairs"]
        self.settings = settings
        self.temporal_spec = temporal_spec
        self.datasets = {}
        self.annotations = {}
        for source in sources:
            repo_id = source["repo_id"]
            dataset = LeRobotDataset(
                repo_id=repo_id, root=source["root"],
                delta_timestamps={
                    "observation.state": [0.0],
                    "action": [i / 17 for i in range(50)],
                    "observation.images.head_camera": [0.0],
                },
                video_backend=settings.video_backend,
            )
            self.datasets[repo_id] = dataset
            if settings.mixup_image_mode == "patch":
                episodes = dataset.meta.episodes
                starts = [int(episodes[i]["dataset_from_index"]) for i in range(len(episodes))]
                ends = [int(episodes[i]["dataset_to_index"]) for i in range(len(episodes))]
                self.annotations[repo_id] = PatchAnnotationReader(
                    Path(source["annotation_root"]), dataset, (starts, ends))
        if settings.mixup_alpha_mode == "beta":
            self.beta = torch.distributions.Beta(settings.mixup_beta, settings.mixup_beta)
        elif settings.mixup_alpha_mode == "fixed" and settings.mixup_alpha_value is not None:
            self.beta = None
        else:
            raise ValueError("Invalid cross-task alpha settings")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        a, b = pair["a"], pair["b"]
        if a["repo_id"] == b["repo_id"]:
            raise ValueError("Cross-task pair must name different sources")
        sample_a = self.datasets[a["repo_id"]][a["frame_index"]]
        sample_b = self.datasets[b["repo_id"]][b["frame_index"]]
        image_key = "observation.images.head_camera"
        image_a = sample_a[image_key].to(torch.float32)
        image_b = sample_b[image_key].to(torch.float32)
        if self.settings.mixup_image_mode == "patch":
            mask = self.annotations[b["repo_id"]].target_mask(b["frame_index"])
            image = hard_patch_image(image_a, image_b, mask)
            alpha = None
        else:
            alpha = (float(self.beta.sample().item()) if self.beta is not None
                     else float(self.settings.mixup_alpha_value))
            image = alpha * image_a + (1.0 - alpha) * image_b
        action_a = sample_a["action"].to(torch.float32)
        action_b = sample_b["action"].to(torch.float32)
        if bool(sample_a["action_is_pad"].any()) or bool(sample_b["action_is_pad"].any()):
            raise ValueError("Cross-task index includes an incomplete action window")
        valid = torch.ones(self.temporal_spec.action_steps, dtype=torch.bool)
        result = {
            image_key: image.unsqueeze(0),
            "observation.images": image.unsqueeze(0).unsqueeze(0),
            "observation.state": sample_a["observation.state"].to(torch.float32),
            "action": action_a.clone(),
            "emix_actions": [action_a, action_b],
            "emix_mask": valid,
            "action_is_pad": ~valid,
            self.temporal_spec.action_pad_key: ~valid,
            "task": sample_a["task"],
            "sample_id": f"cross:{a['repo_id']}:{a['frame_index']}:{b['repo_id']}:{b['frame_index']}",
            "group_id": f"cross:{index}",
            "member_ids": [
                f"{a['repo_id']}:{a['frame_index']}",
                f"{b['repo_id']}:{b['frame_index']}",
            ],
            "emix_source_images": [
                {image_key: image_a.unsqueeze(0)},
                {image_key: image_b.unsqueeze(0)},
            ],
            "emix_states": [sample_a["observation.state"], sample_b["observation.state"]],
        }
        if alpha is not None:
            result["emix_alpha"] = torch.tensor(alpha, dtype=torch.float32)
        return result
