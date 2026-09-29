"""Tests for bounded EMIX input snapshot export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from embodied_mixup.visualization.emix_debug import EmixDebugExporter, EmixDebugSettings


class TestEmixDebugExporter(unittest.TestCase):
    def test_exports_image_action_plots_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = EmixDebugSettings(
                enabled=True,
                output_dir=temp_dir,
                max_samples=1,
                every_n_steps=1,
                action_xyz_indices=(0, 1, 2),
            )
            exporter = EmixDebugExporter(settings)
            batch = {
                "observation.images.head_camera": torch.ones(1, 2, 3, 12, 16),
                "emix_actions": [
                    torch.tensor([[[0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]]),
                    torch.tensor([[[0.0, 0.1, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]]]),
                ],
                "emix_collage_masks": {
                    "observation.images.head_camera": torch.ones(1, 2, 12, 16),
                },
                "emix_mask": torch.tensor([[True, True]]),
                "sample_id": ["sample-0"],
                "group_id": ["group-0"],
                "member_ids": [["member-0", "member-1"]],
            }

            self.assertEqual(exporter.export_batch(batch, step=7), 1)
            self.assertTrue(exporter.finished)
            output_files = list(Path(temp_dir).glob("step_*.png"))
            self.assertEqual(len(output_files), 1)
            manifest = Path(temp_dir) / "manifest.jsonl"
            payload = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(payload["step"], 7)
            self.assertEqual(payload["collage_copied_fraction"], 1.0)
            self.assertEqual(payload["member_ids"], ["member-0", "member-1"])

    def test_disabled_exporter_is_a_noop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = EmixDebugSettings(enabled=False, output_dir=temp_dir)
            exporter = EmixDebugExporter(settings)
            self.assertEqual(exporter.export_batch({}, step=1), 0)
            self.assertEqual(list(Path(temp_dir).iterdir()), [])
