"""Low-cost contract checks for multi-task SmolVLA preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from embodied_mixup.policies.utils.bootstrap import bootstrap_lerobot_runtime

bootstrap_lerobot_runtime()

from embodied_mixup.policies.SmolVLA.emix_smolvla import EmixSmolVLAPolicy
from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixLoaderSettings
from embodied_mixup.policies.utils.multi_dataset import read_dataset_stats, write_dataset_stats
from embodied_mixup.policies.utils.multi_index import MultiTaskEmixDataset
from evaluation.evaluate import _make_policy_observation
from scripts.run_multi_v0 import ARMS, load_spec, make_eval_config, make_training_config


class _FakeFlow(torch.nn.Module):
    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, count, device):
        return torch.zeros(count, device=device)

    def forward(self, *args):
        action = args[5]
        return action.unsqueeze(-1)


class MultiV0MockTests(unittest.TestCase):
    def test_three_train_and_thirty_eval_configs(self):
        spec = load_spec()
        self.assertEqual(len(ARMS), 3)
        self.assertEqual(len(spec["sources"]["tasks"]), 10)
        configs = [make_training_config(spec, arm) for arm in ARMS]
        self.assertEqual([c["policy"]["type"] for c in configs],
                         ["smolvla", "emix_smolvla", "emix_smolvla"])
        self.assertEqual({c["policy"]["pretrained_path"] for c in configs},
                         {"lerobot/smolvla_base"})
        self.assertNotIn("emix", configs[0])
        self.assertEqual(configs[1]["emix"]["mixup"]["image_mode"], "linear")
        self.assertEqual(configs[2]["emix"]["mixup"]["image_mode"], "patch")
        evaluations = [make_eval_config(spec, arm, task, prompt)
                       for arm in ARMS for task, prompt in spec["sources"]["tasks"].items()]
        self.assertEqual(len(evaluations), 30)
        self.assertTrue(all(c["evaluation"]["task_prompt"] for c in evaluations))

    def test_multi_index_routes_within_and_cross_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            sources = [{"repo_id": name, "root": f"/tmp/{name}",
                        "index_path": f"/tmp/{name}.json", "annotation_root": f"/tmp/{name}/annotations"}
                       for name in ("task_a", "task_b")]
            path.write_text(json.dumps({"schema_version": 1,
                                        "pair_scope": "within_and_cross_task",
                                        "cross_pairs_path": str(Path(temp) / "cross.json"),
                                        "sources": sources}))
            class FakeDataset:
                def __init__(self, name):
                    self.name = name
                def __len__(self):
                    return 2
                def __getitem__(self, index):
                    return {"sample_id": "sample", "group_id": "group", "task": self.name}
            class FakeCrossDataset:
                def __init__(self, *args):
                    pass
                def __len__(self):
                    return 1
                def __getitem__(self, index):
                    return {"sample_id": "cross:task_a:0:task_b:1",
                            "member_ids": ["task_a:0", "task_b:1"], "task": "task_a"}
            with patch("embodied_mixup.policies.utils.multi_index.make_emix_loader",
                       side_effect=lambda **kwargs: FakeDataset(kwargs["source_repo_id"])) as maker, \
                 patch("embodied_mixup.policies.utils.multi_cross_loader.CrossTaskEmixDataset", FakeCrossDataset):
                dataset = MultiTaskEmixDataset(path, ["task_a", "task_b"], "emix_smolvla",
                                                object(), EmixLoaderSettings(policy_type="emix_smolvla"))
            self.assertEqual(maker.call_count, 2)
            self.assertEqual(dataset[0]["member_ids"], ["task_a:0", "task_b:1"])
            self.assertEqual(dataset[1]["sample_id"], "task_a:sample")
            self.assertEqual(dataset[3]["group_id"], "task_b:group")
            self.assertEqual(dataset[3]["task"], "task_b")

    def test_pooled_stats_roundtrip(self):
        stats = {"action": {"mean": np.array([1.0, 2.0]), "std": np.array([0.2, 0.5])}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stats.json"
            write_dataset_stats(path, stats)
            result = read_dataset_stats(path)
        np.testing.assert_allclose(result["action"]["mean"], [1.0, 2.0])

    def test_mix_and_patch_flow_loss(self):
        policy = EmixSmolVLAPolicy.__new__(EmixSmolVLAPolicy)
        torch.nn.Module.__init__(policy)
        policy.model = _FakeFlow()
        policy.prepare_images = lambda batch: ([batch["observation.images.head_camera"]],
                                               [torch.ones(1, dtype=torch.bool)])
        policy.prepare_state = lambda batch: batch["observation.state"]
        policy.prepare_action = lambda batch: batch["action"]
        batch = {
            "observation.state": torch.zeros(1, 7),
            "observation.images.head_camera": torch.zeros(1, 3, 2, 2),
            "observation.language.tokens": torch.zeros(1, 1, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(1, 1, dtype=torch.bool),
            "emix_actions": [torch.ones(1, 2, 1), torch.full((1, 2, 1), 2.0)],
            "emix_alpha": torch.tensor([0.25]),
        }
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        batch[OBS_LANGUAGE_TOKENS] = batch.pop("observation.language.tokens")
        batch[OBS_LANGUAGE_ATTENTION_MASK] = batch.pop("observation.language.attention_mask")
        for mode, expected in (("linear", 1.75), ("patch", 3.0)):
            policy.config = SimpleNamespace(adapt_to_pi_aloha=False,
                mixup_image_mode=mode, max_action_dim=1, mixup_loss_weight=1.0)
            loss, _ = policy.forward(batch)
            self.assertAlmostEqual(loss.item(), expected)

    def test_eval_prompt_in_observation(self):
        observation = {"observation": {"head_camera": {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}},
                       "joint_action": {"vector": np.zeros(14)}}
        batch = _make_policy_observation(observation, 4, 4, task_prompt="Stack two bowls.")
        self.assertEqual(batch["task"], ["Stack two bowls."])
        self.assertEqual(tuple(batch["observation.state"].shape), (1, 7))


if __name__ == "__main__":
    unittest.main()
