"""Unit tests for EMIX collage training settings and scheduling."""

from __future__ import annotations

import unittest

import torch

from embodied_mixup.policies.utils.mix_scheduler_utils import (
    ConstantMixScheduler,
    LinearMixScheduler,
    MIX_SCHEDULER_REGISTRY,
    StepMixScheduler,
)
from embodied_mixup.policies.utils.mix_scheduler import make_mix_scheduler
from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixLoaderSettings
from embodied_mixup.policies import train as train_module


class TestConstantScheduler(unittest.TestCase):
    def test_returns_exact_prob(self):
        s = ConstantMixScheduler(emix_batch_prob=0.3)
        self.assertAlmostEqual(s.emix_batch_prob(1, 1000), 0.3)
        self.assertAlmostEqual(s.emix_batch_prob(999, 1000), 0.3)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            ConstantMixScheduler(emix_batch_prob=-0.1)

    def test_rejects_above_one(self):
        with self.assertRaises(ValueError):
            ConstantMixScheduler(emix_batch_prob=1.01)

    def test_boundary_zero(self):
        s = ConstantMixScheduler(emix_batch_prob=0.0)
        self.assertAlmostEqual(s.emix_batch_prob(1, 100), 0.0)

    def test_boundary_one(self):
        s = ConstantMixScheduler(emix_batch_prob=1.0)
        self.assertAlmostEqual(s.emix_batch_prob(1, 100), 1.0)


class TestLinearScheduler(unittest.TestCase):
    def test_start_prob(self):
        s = LinearMixScheduler(start_emix_batch_prob=0.5, end_emix_batch_prob=0.1)
        # step=1: progress=(1-1)/(1000-1)=0 -> start
        self.assertAlmostEqual(s.emix_batch_prob(1, 1000), 0.5)

    def test_end_prob(self):
        s = LinearMixScheduler(start_emix_batch_prob=0.5, end_emix_batch_prob=0.1)
        # step=total_steps: progress=(1000-1)/(1000-1)=1 -> end
        self.assertAlmostEqual(s.emix_batch_prob(1000, 1000), 0.1)

    def test_middle_prob(self):
        s = LinearMixScheduler(start_emix_batch_prob=0.4, end_emix_batch_prob=0.0)
        # step=501, total=1001: progress=500/1000=0.5 -> 0.4 + 0.5*(-0.4)=0.2
        self.assertAlmostEqual(s.emix_batch_prob(501, 1001), 0.2, places=5)

    def test_total_steps_one(self):
        s = LinearMixScheduler(start_emix_batch_prob=0.3, end_emix_batch_prob=0.1)
        # total_steps<=1: progress=1.0 -> end
        self.assertAlmostEqual(s.emix_batch_prob(1, 1), 0.1)

    def test_rejects_invalid_start(self):
        with self.assertRaises(ValueError):
            LinearMixScheduler(start_emix_batch_prob=1.5, end_emix_batch_prob=0.1)

    def test_rejects_invalid_end(self):
        with self.assertRaises(ValueError):
            LinearMixScheduler(start_emix_batch_prob=0.3, end_emix_batch_prob=-0.1)


class TestStepScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = StepMixScheduler(
            steps=[
                {"start_percentage": 0.0, "end_percentage": 0.25, "ratio": 0.4},
                {"start_percentage": 0.25, "end_percentage": 0.75, "ratio": 0.2},
                {"start_percentage": 0.75, "end_percentage": 1.0, "ratio": 0.0},
            ]
        )

    def test_returns_ratio_for_each_interval(self):
        self.assertAlmostEqual(self.scheduler.emix_batch_prob(1, 101), 0.4)
        self.assertAlmostEqual(self.scheduler.emix_batch_prob(26, 101), 0.2)
        self.assertAlmostEqual(self.scheduler.emix_batch_prob(76, 101), 0.0)

    def test_interval_boundary_uses_next_interval(self):
        self.assertAlmostEqual(self.scheduler.emix_batch_prob(26, 101), 0.2)

    def test_uncovered_percentage_returns_zero(self):
        scheduler = StepMixScheduler(
            steps=[{"start_percentage": 0.2, "end_percentage": 0.4, "ratio": 0.5}]
        )
        self.assertAlmostEqual(scheduler.emix_batch_prob(1, 101), 0.0)
        self.assertAlmostEqual(scheduler.emix_batch_prob(51, 101), 0.0)

    def test_single_interval_is_constant_scheduler(self):
        scheduler = StepMixScheduler(
            steps=[{"start_percentage": 0.0, "end_percentage": 1.0, "ratio": 0.3}]
        )
        self.assertAlmostEqual(scheduler.emix_batch_prob(1, 100), 0.3)
        self.assertAlmostEqual(scheduler.emix_batch_prob(100, 100), 0.3)

    def test_rejects_overlapping_or_unordered_intervals(self):
        with self.assertRaises(ValueError):
            StepMixScheduler(
                steps=[
                    {"start_percentage": 0.0, "end_percentage": 0.5, "ratio": 0.4},
                    {"start_percentage": 0.4, "end_percentage": 1.0, "ratio": 0.2},
                ]
            )

    def test_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            StepMixScheduler(
                steps=[{"start_percentage": 0.0, "end_percentage": 1.0, "ratio": 1.1}]
            )


class TestMakeScheduler(unittest.TestCase):
    def test_constant(self):
        s = make_mix_scheduler({"type": "constant", "batch_prob": 0.2})
        self.assertIsInstance(s, ConstantMixScheduler)
        self.assertAlmostEqual(s.emix_batch_prob(1, 100), 0.2)

    def test_linear(self):
        s = make_mix_scheduler({"type": "linear", "start_batch_prob": 0.4, "end_batch_prob": 0.1})
        self.assertIsInstance(s, LinearMixScheduler)

    def test_step(self):
        s = make_mix_scheduler(
            {
                "type": "step",
                "steps": [
                    {"start_percentage": 0.0, "end_percentage": 0.5, "ratio": 0.4},
                    {"start_percentage": 0.5, "end_percentage": 1.0, "ratio": 0.1},
                ],
            }
        )
        self.assertIsInstance(s, StepMixScheduler)
        self.assertAlmostEqual(s.emix_batch_prob(1, 100), 0.4)

    def test_step_requires_steps(self):
        with self.assertRaises(ValueError):
            make_mix_scheduler({"type": "step"})

    def test_unknown_type(self):
        with self.assertRaises(ValueError):
            make_mix_scheduler({"type": "cosine", "batch_prob": 0.2})

    def test_missing_type(self):
        with self.assertRaises(ValueError):
            make_mix_scheduler({"batch_prob": 0.2})


class TestCollageConfig(unittest.TestCase):
    def test_default_settings_are_alpha_free(self):
        cfg = EmixLoaderSettings(policy_type="emix_diffusion")
        self.assertEqual(cfg.collage_difference_threshold, 0.08)
        self.assertEqual(cfg.collage_dilation_kernel, 3)
        self.assertFalse(hasattr(cfg, "mixup_alpha_mode"))

    def test_collage_copies_only_changed_region_without_blending(self):
        from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixPolicyDataset

        dataset = object.__new__(EmixPolicyDataset)
        dataset.loader_cfg = EmixLoaderSettings(
            policy_type="emix_diffusion",
            collage_difference_threshold=0.1,
            collage_dilation_kernel=1,
        )
        image_0 = torch.zeros(3, 5, 5)
        image_1 = image_0.clone()
        image_1[:, 2, 2] = 1.0
        collage, copied = dataset._make_collage(image_0, image_1)
        self.assertTrue(bool(copied[2, 2]))
        self.assertEqual(float(collage[0, 2, 2]), 1.0)
        self.assertEqual(float(collage[0, 0, 0]), 0.0)
        self.assertNotEqual(float(collage[0, 2, 2]), 0.5)

    def test_loader_settings_no_mixup_prob(self):
        cfg = EmixLoaderSettings(policy_type="emix_diffusion")
        self.assertFalse(hasattr(cfg, "mixup_prob"))


class TestMixupProbRemoved(unittest.TestCase):
    def test_no_mixup_prob_in_loader_settings(self):
        import inspect
        from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixLoaderSettings
        fields = [f.name for f in EmixLoaderSettings.__dataclass_fields__.values()]
        self.assertNotIn("mixup_prob", fields)

    def test_no_mixup_prob_in_emix_diffusion_cfg(self):
        try:
            from embodied_mixup.policies.DP.cfg_emix_diffusion import EmixDiffusionConfig
        except Exception as exc:
            self.skipTest(f"EMIX diffusion config import is unavailable in this environment: {exc}")
        self.assertFalse(hasattr(EmixDiffusionConfig, "mixup_prob"))

    def test_no_mixup_prob_in_emix_smolvla_cfg(self):
        from embodied_mixup.policies.SmolVLA.cfg_emix_smolvla import EmixSmolVLAConfig
        self.assertFalse(hasattr(EmixSmolVLAConfig, "mixup_prob"))

    def test_no_mixup_prob_in_emix_act_cfg(self):
        try:
            from embodied_mixup.policies.ACT.cfg_emix_act import EmixACTConfig
        except Exception as exc:
            self.skipTest(f"EMIX ACT config import is unavailable in this environment: {exc}")
        self.assertFalse(hasattr(EmixACTConfig, "mixup_prob"))


class TestExplicitTrainingMode(unittest.TestCase):
    def test_default_without_emix_defaults_to_default(self):
        self.assertEqual(train_module._resolve_training_mode("diffusion", {}), "default")

    def test_emix_training_requires_explicit_mode(self):
        with self.assertRaises(ValueError):
            train_module._resolve_training_mode(
                "emix_diffusion",
                {"training": {"seed": 0}},
            )

    def test_mixed_mode_requires_emix_policy(self):
        with self.assertRaises(ValueError):
            train_module._resolve_training_mode(
                "diffusion",
                {"training": {"training_mode": "mixed", "seed": 0}},
            )

    def test_emix_mode_requires_emix_policy(self):
        with self.assertRaises(ValueError):
            train_module._resolve_training_mode(
                "diffusion",
                {"training": {"training_mode": "emix", "seed": 0}},
            )

    def test_mixed_mode_is_accepted_for_emix_policy(self):
        self.assertEqual(
            train_module._resolve_training_mode(
                "emix_diffusion",
                {"training": {"training_mode": "mixed", "seed": 0}},
            ),
            "mixed",
        )

    def test_emix_mode_is_accepted_for_emix_policy(self):
        self.assertEqual(
            train_module._resolve_training_mode(
                "emix_diffusion",
                {"training": {"training_mode": "emix", "seed": 0}},
            ),
            "emix",
        )


if __name__ == "__main__":
    unittest.main()
