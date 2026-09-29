"""Audit the shared-space centred-arm collection batch.

This verifier is deliberately independent of the simulator.  It proves the
configuration contract for all active task families and verifies collection
artifacts produced by the rate-gated collector.  A batch passes only if every
task uses the exact common randomization envelope and has at least five
accepted episodes with an observed success rate of five percent or higher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
_ORIGINAL_CWD = Path.cwd()
os.chdir(ROBOTWIN_ROOT)
try:
    from envs.left.left_task_manifests import (  # noqa: E402
        CENTERED_WIDE_WORKSPACE,
        all_manifests,
        centered_wide_workspace_manifest,
    )
finally:
    os.chdir(_ORIGINAL_CWD)


ACTIVE_FAMILIES = (
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_dual_shoes",
    "stack_blocks_two",
    "stack_bowls_two",
)
TASK_SUFFIX = "_left_cen_arm_right_wide_cam"
MIN_EPISODES = 5
MAX_SEED_ATTEMPTS = 100
MIN_SUCCESS_RATE = 0.05
IMAGE_FEATURE = "observation.images.head_camera"


def task_name(family: str) -> str:
    return f"{family}{TASK_SUFFIX}"


def validate_shared_workspace(families: Iterable[str] = ACTIVE_FAMILIES) -> list[str]:
    """Return configuration-contract failures for the named active families."""
    failures: list[str] = []
    manifests = all_manifests()
    for family in families:
        manifest = centered_wide_workspace_manifest(manifests[family])
        for actor, spec in manifest.actor_specs.items():
            if spec.workspace != CENTERED_WIDE_WORKSPACE:
                failures.append(f"{family}/{actor}: workspace differs from shared envelope")
    return failures


def validate_task_artifacts(root: Path, family: str) -> list[str]:
    """Return collection, rate, and image-contract failures for one task."""
    name = task_name(family)
    collection = root / "data" / "robotwin" / name / name
    dataset = root / "data" / "lerobot" / name
    failures: list[str] = []

    metrics_path = collection / "collection_metrics.json"
    if not metrics_path.is_file():
        return [f"{family}: missing collection metrics: {metrics_path}"]
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        accepted = int(metrics["accepted_episodes"])
        attempts = int(metrics["attempted_seeds"])
        rate = float(metrics["collection_success_rate"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return [f"{family}: invalid collection metrics: {error}"]
    if accepted < MIN_EPISODES:
        failures.append(f"{family}: accepted {accepted}, need at least {MIN_EPISODES}")
    if not 0 < attempts <= MAX_SEED_ATTEMPTS:
        failures.append(f"{family}: attempted {attempts}, need 1..{MAX_SEED_ATTEMPTS}")
    if rate < MIN_SUCCESS_RATE:
        failures.append(f"{family}: success rate {rate:.2%}, need at least {MIN_SUCCESS_RATE:.2%}")
    if attempts and abs(rate - accepted / attempts) > 1e-9:
        failures.append(f"{family}: recorded rate does not equal accepted/attempted")

    hdf5_count = len(list((collection / "data").glob("episode*.hdf5")))
    if hdf5_count < MIN_EPISODES:
        failures.append(f"{family}: found {hdf5_count} HDF5 episodes, need at least {MIN_EPISODES}")
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        failures.append(f"{family}: missing LeRobot metadata: {info_path}")
        return failures
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        shape = info["features"][IMAGE_FEATURE]["shape"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        failures.append(f"{family}: invalid LeRobot metadata: {error}")
    else:
        if shape != [224, 224, 3]:
            failures.append(f"{family}: image shape {shape!r}, need [224, 224, 3]")
    return failures


def audit(root: Path) -> list[str]:
    """Return all failures; an empty list proves the complete batch contract."""
    failures = validate_shared_workspace()
    for family in ACTIVE_FAMILIES:
        failures.extend(validate_task_artifacts(root, family))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    failures = audit(args.project_root.resolve())
    if failures:
        print("Centered batch audit FAILED:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Centered batch audit PASSED: ten tasks share one workspace and all meet the 5% rate gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
