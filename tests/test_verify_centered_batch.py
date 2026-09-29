"""Regression tests for the centred-arm shared-space batch audit."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "verify_centered_batch.py"
SPEC = importlib.util.spec_from_file_location("verify_centered_batch", SCRIPT_PATH)
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class TestCenteredBatchAudit(unittest.TestCase):
    def _write_task(self, root: Path, family: str, *, rate: float = 0.05) -> None:
        name = VERIFY.task_name(family)
        collection = root / "data" / "robotwin" / name / name
        data = collection / "data"
        data.mkdir(parents=True)
        for episode in range(5):
            (data / f"episode{episode}.hdf5").touch()
        (collection / "collection_metrics.json").write_text(json.dumps({
            "accepted_episodes": 5,
            "attempted_seeds": 100,
            "collection_success_rate": rate,
        }))
        info = root / "data" / "lerobot" / name / "meta" / "info.json"
        info.parent.mkdir(parents=True)
        info.write_text(json.dumps({"features": {
            VERIFY.IMAGE_FEATURE: {"shape": [224, 224, 3]}
        }}))

    def test_workspace_contract_has_no_failures(self):
        self.assertEqual(VERIFY.validate_shared_workspace(), [])

    def test_task_artifacts_accept_exact_five_percent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_task(root, "blocks_ranking_rgb")
            self.assertEqual(VERIFY.validate_task_artifacts(root, "blocks_ranking_rgb"), [])

    def test_task_artifacts_reject_low_success_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_task(root, "blocks_ranking_rgb", rate=0.049)
            failures = VERIFY.validate_task_artifacts(root, "blocks_ranking_rgb")
            self.assertTrue(any("success rate" in failure for failure in failures))
