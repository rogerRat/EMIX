"""Verify noise statistics, RNG isolation and training/evaluation boundaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from embodied_mixup.policies.utils.image_noise import ImageNoiseConfig, add_image_noise
from embodied_mixup.policies import train as train_module


KEY = "observation.images.head_camera"
ROOT = Path(__file__).resolve().parents[1]


class TestImageNoise(unittest.TestCase):
    def test_gaussian_statistics_and_independent_channels(self):
        image = torch.full((8, 1, 3, 128, 128), 0.5)
        noisy = add_image_noise(
            {KEY: image}, ImageNoiseConfig(True, 0.0025), seed=7, step=1,
        )[KEY]
        error = (noisy - image).squeeze(1).movedim(1, 0).reshape(3, -1)
        self.assertLess(error.mean(dim=1).abs().max().item(), 0.0005)
        torch.testing.assert_close(error.var(dim=1), torch.full((3,), 0.0025), rtol=0.02, atol=0)
        correlation = torch.corrcoef(error) - torch.eye(3)
        self.assertLess(correlation.abs().max().item(), 0.015)
        # Pixelwise samples, rather than a shared offset or whole-image scale.
        self.assertGreater(error[0].std().item(), 0.045)
        self.assertTrue(torch.equal(image, torch.full_like(image, 0.5)))

    def test_clipping_preserves_fields_and_handles_multiple_cameras(self):
        image = torch.stack([torch.zeros(3, 16, 16), torch.ones(3, 16, 16)])
        state = torch.ones(2, 7)
        action = torch.ones(2, 16, 7)
        pad = torch.zeros(2, dtype=torch.bool)
        batch = {KEY: image, "observation.images.wrist": image,
                 "observation.state": state, "action": action, f"{KEY}_is_pad": pad,
                 "emix_alpha": torch.tensor([0.2, 0.8]), "emix_source_images": image}
        noisy = add_image_noise(batch, ImageNoiseConfig(True, 0.1), seed=0, step=1)
        self.assertGreaterEqual(noisy[KEY].min().item(), 0)
        self.assertLessEqual(noisy[KEY].max().item(), 1)
        self.assertGreater(noisy[KEY][0].max().item(), 0)
        self.assertLess(noisy[KEY][1].min().item(), 1)
        self.assertFalse(torch.equal(noisy[KEY], noisy["observation.images.wrist"]))
        for key in ("observation.state", "action", f"{KEY}_is_pad", "emix_alpha", "emix_source_images"):
            self.assertIs(noisy[key], batch[key])

    def test_seed_step_reproducibility_and_global_rng_isolation(self):
        batch = {KEY: torch.full((2, 3, 16, 16), 0.5)}
        config = ImageNoiseConfig(True)
        state = torch.random.get_rng_state().clone()
        first = add_image_noise(batch, config, seed=3, step=20)[KEY]
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first, add_image_noise(batch, config, seed=3, step=20)[KEY]))
        self.assertFalse(torch.equal(first, add_image_noise(batch, config, seed=3, step=21)[KEY]))
        self.assertFalse(torch.equal(first, add_image_noise(batch, config, seed=4, step=20)[KEY]))

    def test_disabled_and_zero_variance_are_exact_noops(self):
        batch = {KEY: torch.rand(2, 3, 16, 16)}
        state = torch.random.get_rng_state().clone()
        for config in (ImageNoiseConfig(), ImageNoiseConfig(True, 0)):
            self.assertIs(add_image_noise(batch, config, seed=0, step=1), batch)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_invalid_config_and_image_contract(self):
        for payload in (False, {"enabled": "true"}, {"variance": -1},
                        {"variance": float("nan")}, {"variance": float("inf")},
                        {"variance": True}, {"std": 0.05}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ImageNoiseConfig.from_payload(payload)
        for image in (torch.zeros(3, 16, 16, dtype=torch.uint8), torch.zeros(2, 16, 16)):
            with self.assertRaises(ValueError):
                add_image_noise({KEY: image}, ImageNoiseConfig(True), seed=0, step=1)

    def test_training_config_and_checkpoint_record_noise(self):
        payload = {"policy": {"type": "diffusion", "device": "cpu", "n_obs_steps": 1},
                   "dataset": {"repo_id": "test", "root": "/unused"},
                   "train": {"output_dir": str(ROOT / "outputs/noise-test"),
                             "image_noise": {"enabled": True, "variance": 0.01}}}
        cfg = train_module._build_train_config(payload, ROOT / "unused.yaml")
        self.assertEqual(cfg.image_noise, ImageNoiseConfig(True, 0.01))
        checkpoint_cfg = train_module.SaveableTrainConfig(
            cfg.policy_cfg, cfg.dataset_root, cfg.dataset_repo_id, cfg.steps,
            None, None, cfg.training_mode, cfg.image_noise,
        )
        import json
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_cfg.save_pretrained(Path(directory))
            saved = json.loads((Path(directory) / "train_config.json").read_text())
        self.assertEqual(saved["image_noise"], {"enabled": True, "variance": 0.01})
        self.assertNotIn("image_noise", cfg.policy_cfg.__dict__)

    def test_flow_noise_protocols_share_training_and_separate_evaluation(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            spec = importlib.util.spec_from_file_location("noise_test_run", ROOT / "scripts/run.py")
            flow = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(flow)
        finally:
            sys.path.pop(0)
        config = flow.load_config(ROOT / "configs/scripts/run_v0_noise_tn_en.yaml")
        clean = flow.load_config(ROOT / "configs/scripts/run_v0_noise_tn_ec.yaml")
        self.assertEqual(len(config["tasks"]), 5)
        self.assertEqual(config["tasks"], clean["tasks"])
        self.assertEqual(set(config["tasks"]), {
            f"{task}_left_cen_arm_top_cam" for task in (
                "place_can_basket", "place_dual_shoes", "place_bread_basket",
                "stack_blocks_two", "blocks_ranking_rgb",
            )
        })
        self.assertEqual(config["protocol"]["execution_order"], ["v0-noise-tn-ec", "v0-noise-tn-en"])
        self.assertEqual(flow.selected_policies(config), [("baseline", False)])
        task = config["tasks"][0]
        for emix in (False, True):
            payload = flow.train_config(task, config, emix=emix)
            self.assertEqual(payload["image_noise"], {"enabled": True, "variance": 0.0025})
            noisy_eval = flow.eval_config(task, config, emix=emix)["evaluation"]
            clean_eval = flow.eval_config(task, clean, emix=emix)["evaluation"]
            self.assertTrue(noisy_eval["image_noise"]["enabled"])
            self.assertFalse(clean_eval["image_noise"]["enabled"])
            self.assertEqual(noisy_eval["checkpoint"], clean_eval["checkpoint"])
            self.assertNotEqual(noisy_eval["output_dir"], clean_eval["output_dir"])
            cfg = train_module._build_train_config(payload, ROOT / "unused.yaml")
            self.assertTrue(cfg.image_noise.enabled)
            self.assertEqual(cfg.protocol["family"], "v0-noise")
            payload["image_noise"]["variance"] = 1
            self.assertEqual(config["training"]["image_noise"]["variance"], 0.0025)
        del config["evaluation"]
        self.assertNotIn("image_noise", flow.eval_config(task, config, emix=False)["evaluation"])

    def test_evaluation_noise_is_after_pixel_scaling_and_explicit_only(self):
        import numpy as np
        from evaluation.evaluate import _make_policy_observation

        rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
        obs = {"observation": {"head_camera": {"rgb": rgb}},
               "joint_action": {"vector": np.arange(14, dtype=np.float32)}}
        clean = _make_policy_observation(obs, 16, 16)
        torch.testing.assert_close(clean[KEY], torch.full((1, 3, 16, 16), 128 / 255))
        noisy = _make_policy_observation(
            obs, 16, 16, image_noise=ImageNoiseConfig(True), noise_seed=10, noise_step=1,
        )
        self.assertFalse(torch.equal(clean[KEY], noisy[KEY]))
        torch.testing.assert_close(clean["observation.state"], noisy["observation.state"])
        self.assertTrue(np.array_equal(rgb, np.full_like(rgb, 128)))
        torch.testing.assert_close(_make_policy_observation(obs, 16, 16)[KEY], clean[KEY])


if __name__ == "__main__":
    unittest.main()
