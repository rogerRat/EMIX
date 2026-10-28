"""CPU-only tests for hard-patch image mixing and diffusion loss modes."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

import torch

from embodied_mixup.policies.utils.emix_loader_registry_utils import hard_patch_image, mask_overlap_ratio


class HardPatchImageTests(unittest.TestCase):
    def test_mask_interior_copies_b_and_exterior_stays_equal_to_a(self):
        image_a = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
        image_b = torch.full_like(image_a, -7.0)
        target_mask = torch.zeros((4, 5), dtype=torch.bool)
        target_mask[1:3, 2:5] = True

        result = hard_patch_image(image_a, image_b, target_mask)

        torch.testing.assert_close(result[:, ~target_mask], image_a[:, ~target_mask])
        torch.testing.assert_close(result[:, target_mask], image_b[:, target_mask])
        self.assertTrue(torch.equal(result, torch.where(target_mask.unsqueeze(0), image_b, image_a)))

    def test_rejects_invalid_image_or_mask_shape(self):
        image = torch.zeros((3, 4, 5))
        with self.assertRaisesRegex(ValueError, "matching CxHxW"):
            hard_patch_image(image, torch.zeros((3, 4, 4)), torch.zeros((4, 5)))
        with self.assertRaisesRegex(ValueError, "match image height and width"):
            hard_patch_image(image, image, torch.zeros((5, 4)))



class LinearImageMixTests(unittest.TestCase):
    def test_linear_mode_still_blends_images_with_sampled_alpha(self):
        from embodied_mixup.policies.utils.batch import PolicyTemporalSpec
        from embodied_mixup.policies.utils.emix_loader_registry_utils import (
            EmixLoaderSettings,
            EmixPolicyDataset,
            EmixTrainingSample,
        )

        class ReaderStub:
            def load_member_window(self, group_id, member_id):
                return windows[member_id]

        member_a = SimpleNamespace(source_frame_index=0)
        member_b = SimpleNamespace(source_frame_index=1)
        group = SimpleNamespace(group_id="group-0", members=[member_a, member_b])
        record = EmixTrainingSample(group=group, member_ids=["a", "b"], sample_id="sample-0")

        image_a = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)
        image_b = torch.full((1, 2, 2, 3), 255, dtype=torch.uint8)
        windows = {
            "a": {
                "feature_payload": {
                    "observation.images.head": image_a.numpy(),
                    "observation.state": torch.zeros((1, 2)).numpy(),
                    "action": torch.zeros((1, 1)).numpy(),
                }
            },
            "b": {
                "feature_payload": {
                    "observation.images.head": image_b.numpy(),
                    "observation.state": torch.ones((1, 2)).numpy(),
                    "action": torch.ones((1, 1)).numpy(),
                }
            },
        }

        dataset = object.__new__(EmixPolicyDataset)
        dataset.reader = ReaderStub()
        dataset.records = [record]
        dataset.policy_temporal_spec = PolicyTemporalSpec(n_obs_steps=1, action_steps=1)
        dataset.loader_cfg = EmixLoaderSettings(
            policy_type="emix_diffusion",
            mixup_image_mode="linear",
            mixup_visual_keys=["observation.images.head"],
        )
        dataset._sample_alpha = lambda: 0.25

        sample = dataset[0]

        torch.testing.assert_close(
            sample["observation.images.head"],
            torch.full((1, 3, 2, 2), 0.75),
        )
        self.assertAlmostEqual(float(sample["emix_alpha"]), 0.25, places=6)



class PatchLoaderTests(unittest.TestCase):
    def test_patch_mode_uses_annotation_mask_and_omits_alpha(self):
        from embodied_mixup.policies.utils.batch import PolicyTemporalSpec
        from embodied_mixup.policies.utils.emix_loader_registry_utils import (
            EmixLoaderSettings,
            EmixPolicyDataset,
            EmixTrainingSample,
        )

        class ReaderStub:
            def load_member_window(self, group_id, member_id):
                return windows[member_id]

        member_a = SimpleNamespace(source_frame_index=0)
        member_b = SimpleNamespace(source_frame_index=1)
        group = SimpleNamespace(group_id="group-0", members=[member_a, member_b])
        record = EmixTrainingSample(group=group, member_ids=["a", "b"], sample_id="sample-0")
        image_a = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)
        image_b = torch.full((1, 2, 2, 3), 255, dtype=torch.uint8)
        windows = {
            member_id: {
                "feature_payload": {
                    "observation.images.head": image.numpy(),
                    "observation.state": torch.zeros((1, 2)).numpy(),
                    "action": torch.zeros((1, 1)).numpy(),
                }
            }
            for member_id, image in (("a", image_a), ("b", image_b))
        }
        dataset = object.__new__(EmixPolicyDataset)
        dataset.reader = ReaderStub()
        dataset.records = [record]
        dataset.policy_temporal_spec = PolicyTemporalSpec(n_obs_steps=1, action_steps=1)
        dataset.loader_cfg = EmixLoaderSettings(
            policy_type="emix_diffusion",
            mixup_image_mode="patch",
            annotation_root="unused-in-this-unit-test",
            mixup_visual_keys=["observation.images.head"],
        )
        mask = np.array([[True, False], [False, False]])
        dataset.annotation_reader = SimpleNamespace(target_mask=lambda frame_index: mask)
        dataset._sample_alpha = lambda: self.fail("patch mode must not sample alpha")

        sample = dataset[0]

        self.assertNotIn("emix_alpha", sample)
        image = sample["observation.images.head"][0]
        torch.testing.assert_close(image[:, 0, 0], torch.ones(3))
        torch.testing.assert_close(image[:, 1, 1], torch.zeros(3))


class _NoiseScheduler:
    config = SimpleNamespace(num_train_timesteps=8)

    def add_noise(self, sample, noise, timesteps):
        return sample


class _DiffusionStub:
    def __init__(self):
        self.noise_scheduler = _NoiseScheduler()
        self.unet_calls = 0

    def _prepare_global_conditioning(self, batch):
        return None

    def unet(self, sample, timesteps, global_cond=None):
        self.unet_calls += 1
        value = 1.0 if self.unet_calls == 1 else 2.0
        return torch.full_like(sample, value)


class DiffusionPatchLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from embodied_mixup.policies.DP.emix_diffusion import EmixDiffusionPolicy
        except Exception as exc:
            raise unittest.SkipTest(f"EMIX diffusion policy import is unavailable: {exc}")
        cls.policy_class = EmixDiffusionPolicy

    def make_policy(self, image_mode):
        policy = object.__new__(self.policy_class)
        policy.config = SimpleNamespace(
            mixup_image_mode=image_mode,
            prediction_type="sample",
            mixup_loss_weight=1.0,
        )
        policy.diffusion = _DiffusionStub()
        return policy

    @staticmethod
    def make_batch(include_alpha):
        actions = torch.zeros((2, 2, 1), dtype=torch.float32)
        batch = {
            "observation.state": torch.zeros((2, 1)),
            "emix_actions": [actions, actions.clone()],
            "action_is_pad": torch.zeros((2, 2), dtype=torch.bool),
        }
        if include_alpha:
            batch["emix_alpha"] = torch.tensor([0.25, 0.75])
        return batch

    def test_patch_mode_sums_branch_means_without_alpha_or_dividing_by_two(self):
        policy = self.make_policy("patch")
        batch = self.make_batch(include_alpha=False)

        loss, _ = policy._compute_emix_loss(batch)

        # The stub predictions produce branch MSE means 1 and 4.
        self.assertAlmostEqual(float(loss), 5.0, places=6)

    def test_linear_mode_keeps_alpha_weighted_loss(self):
        policy = self.make_policy("linear")
        batch = self.make_batch(include_alpha=True)

        loss, _ = policy._compute_emix_loss(batch)

        # Per-sample losses are 1 and 4, weighted by alpha 0.25 and 0.75.
        self.assertAlmostEqual(float(loss), 2.5, places=6)


class PatchAnnotationReaderTests(unittest.TestCase):
    def test_resolves_all_foreground_instances_by_global_dataset_index(self):
        from embodied_mixup.policies.utils.patch_annotations import PatchAnnotationReader

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.instances.h5"
            labels = np.zeros((3, 2, 2), dtype=np.uint8)
            labels[1, 0, 1] = 1
            labels[1, 1, 0] = 2
            with h5py.File(path, "w") as handle:
                handle.attrs["schema_version"] = 1
                handle.attrs["objects"] = json.dumps(["background", "patch_target", "other_object"])
                handle.create_dataset("labels", data=labels, chunks=(1, 2, 2), compression="gzip")
            reader = PatchAnnotationReader(Path(directory), object(), ([0, 3], [3, 6]))
            mask = reader.target_mask(1)
            np.testing.assert_array_equal(mask, [[False, True], [True, False]])

    def test_scene_surfaces_and_robot_are_background(self):
        from embodied_mixup.policies.utils.patch_annotations import PatchAnnotationReader

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.instances.h5"
            labels = np.array([[[1, 2, 3], [4, 5, 0]]], dtype=np.uint8)
            objects = [
                "background", "patch_target", "scene_table_6",
                "scene_ground_1", "scene_link1_8", "scene_106_skillet_34",
            ]
            with h5py.File(path, "w") as handle:
                handle.attrs["schema_version"] = 1
                handle.attrs["objects"] = json.dumps(objects)
                handle.create_dataset("labels", data=labels)
            reader = PatchAnnotationReader(Path(directory), object(), ([0], [1]))
            np.testing.assert_array_equal(
                reader.target_mask(0), [[True, False, False], [False, True, False]]
            )

    def test_missing_episode_sidecar_fails(self):
        from embodied_mixup.policies.utils.patch_annotations import PatchAnnotationReader

        with tempfile.TemporaryDirectory() as directory:
            reader = PatchAnnotationReader(Path(directory), object(), ([0], [2]))
            with self.assertRaisesRegex(FileNotFoundError, "episode 0"):
                reader.target_mask(0)


class DetectOverlapTests(unittest.TestCase):
    @staticmethod
    def _record(sample_id, frame_a, frame_b):
        group = SimpleNamespace(
            group_id=f"group-{sample_id}",
            members=[
                SimpleNamespace(source_frame_index=frame_a),
                SimpleNamespace(source_frame_index=frame_b),
            ],
        )
        return SimpleNamespace(
            sample_id=sample_id,
            group=group,
            member_ids=[f"{sample_id}-a", f"{sample_id}-b"],
        )

    @staticmethod
    def _dataset(records, masks, refuse_ratio):
        from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixPolicyDataset

        dataset = object.__new__(EmixPolicyDataset)
        dataset.records = records
        dataset.loader_cfg = SimpleNamespace(detect_overlap_refuse_ratio=refuse_ratio)
        dataset.annotation_reader = SimpleNamespace(target_mask=lambda frame: masks[frame])
        return dataset

    def test_overlap_ratio_uses_intersection_over_union(self):
        mask_a = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        mask_b = np.array([[1, 0, 1], [0, 0, 0]], dtype=bool)
        self.assertAlmostEqual(mask_overlap_ratio(mask_a, mask_b), 1.0 / 3.0)
        self.assertEqual(mask_overlap_ratio(np.zeros((2, 2)), np.zeros((2, 2))), 0.0)

    def test_over_limit_pair_is_replaced_by_next_feasible_pair(self):
        from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixTrainingSample

        masks = {
            0: np.array([[1, 1, 0, 0]], dtype=bool),
            1: np.array([[1, 0, 0, 0]], dtype=bool),
            2: np.array([[1, 0, 0, 0]], dtype=bool),
            3: np.array([[0, 1, 0, 0]], dtype=bool),
        }
        records = [
            EmixTrainingSample(self._record("bad", 0, 1).group, ["bad-a", "bad-b"], "bad"),
            EmixTrainingSample(self._record("good", 2, 3).group, ["good-a", "good-b"], "good"),
        ]
        dataset = self._dataset(records, masks, refuse_ratio=0.2)
        selected, selected_mask, fallback = dataset._resolve_patch_pair(0, records[0])
        self.assertEqual(selected.sample_id, "good")
        self.assertIsNone(fallback)
        np.testing.assert_array_equal(selected_mask, masks[3])

    def test_all_over_limit_pairs_fallback_to_minimum_and_record_reason(self):
        from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixTrainingSample

        masks = {
            0: np.array([[1, 1, 0, 0]], dtype=bool),
            1: np.array([[1, 0, 0, 0]], dtype=bool),
            2: np.array([[1, 1, 0, 0]], dtype=bool),
            3: np.array([[1, 1, 0, 0]], dtype=bool),
            4: np.array([[1, 1, 0, 0]], dtype=bool),
            5: np.array([[1, 0, 1, 0]], dtype=bool),
        }
        specs = [("high", 0, 1), ("higher", 2, 3), ("lowest", 4, 5)]
        records = [
            EmixTrainingSample(
                self._record(name, frame_a, frame_b).group,
                [f"{name}-a", f"{name}-b"],
                name,
            )
            for name, frame_a, frame_b in specs
        ]
        dataset = self._dataset(records, masks, refuse_ratio=0.2)
        selected, _, fallback = dataset._resolve_patch_pair(0, records[0])
        self.assertEqual(selected.sample_id, "lowest")
        self.assertEqual(fallback["reason"], "all_candidate_pairs_exceed_refuse_ratio")
        self.assertAlmostEqual(fallback["selected_overlap_ratio"], 1.0 / 3.0)
        self.assertEqual(fallback["refuse_ratio"], 0.2)

    def test_pipeline_cleanup_preserves_fallback_log(self):
        import sys

        scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run import preserve_detect_overlap_fallback_log

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_output = root / "train"
            source = train_output / "checkpoints" / "detect_overlap_fallback.txt"
            source.parent.mkdir(parents=True)
            source.write_text("{\"train_step\":3}\n")
            destination = root / "logs" / "detect_overlap_fallback" / "task_patch.txt"
            preserve_detect_overlap_fallback_log(train_output, destination)
            self.assertEqual(destination.read_text(), "{\"train_step\":3}\n")

    def test_fallback_log_is_created_only_for_fallback_events(self):
        import json
        import tempfile
        from embodied_mixup.policies.train import _record_detect_overlap_fallbacks

        event = {
            "reason": "all_candidate_pairs_exceed_refuse_ratio",
            "refused_sample_id": "source",
            "refused_overlap_ratio": 0.75,
            "selected_sample_id": "selected",
            "selected_overlap_ratio": 0.5,
            "refuse_ratio": 0.1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _record_detect_overlap_fallbacks({"detect_overlap_fallback": [None]}, output_dir, 7)
            log_path = output_dir / "checkpoints" / "detect_overlap_fallback.txt"
            self.assertFalse(log_path.exists())
            _record_detect_overlap_fallbacks({"detect_overlap_fallback": [event]}, output_dir, 8)
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["train_step"], 8)
            self.assertEqual(rows[0]["refused_sample_id"], "source")


if __name__ == "__main__":
    unittest.main()
