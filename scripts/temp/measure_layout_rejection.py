"""Measure rejection rates for candidate union ranges (pure numpy, no scene).

For each task, injects candidate workspace overrides into the manifest, then
runs ``sample_layout`` many times and reports how often the sampler locks
(SceneInitRejectError) and the mean attempts per successful layout.  Candidate
ranges that lock are not worth piloting.

Usage:
    python scripts/temp/measure_layout_rejection.py            # current ranges
    python scripts/temp/measure_layout_rejection.py --cand    # candidate ranges
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASKS = [
    "blocks_ranking_rgb_left_impl",
    "blocks_ranking_size_left_impl",
    "place_bread_basket_left_impl",
    "place_bread_skillet_left_impl",
    "place_burger_fries_left_impl",
    "place_can_basket_left_impl",
    "place_cans_plasticbox_left_impl",
    "place_dual_shoes_left_impl",
    "stack_blocks_two_left_impl",
    "stack_bowls_two_left_impl",
]

CLASSES = {
    "blocks_ranking_rgb_left_impl": "BlocksRankingRgbLeftImpl",
    "blocks_ranking_size_left_impl": "BlocksRankingSizeLeftImpl",
    "place_bread_basket_left_impl": "PlaceBreadBasketLeftImpl",
    "place_bread_skillet_left_impl": "PlaceBreadSkilletLeftImpl",
    "place_burger_fries_left_impl": "PlaceBurgerFriesLeftImpl",
    "place_can_basket_left_impl": "PlaceCanBasketLeftImpl",
    "place_cans_plasticbox_left_impl": "PlaceCansPlasticboxLeftImpl",
    "place_dual_shoes_left_impl": "PlaceDualShoesLeftImpl",
    "stack_blocks_two_left_impl": "StackBlocksTwoLeftImpl",
    "stack_bowls_two_left_impl": "StackBowlsTwoLeftImpl",
}

# Candidate v3 union ranges: {module: {actor: (xlo, xhi, ylo, yhi, clearance)}}
# (yaw ranges and base_quat are left untouched; clearance is the pairwise
#  centre-distance for footprint-aware rejection.)
CANDIDATES = {
    "place_bread_basket_left_impl": {
        "breadbasket": (-0.44, -0.12, -0.28, 0.04, 0.13),
        "bread0": (-0.44, -0.34, -0.18, 0.04, 0.10),
        "bread1": (-0.44, -0.34, -0.18, 0.04, 0.10),
    },
    "place_bread_skillet_left_impl": {
        "bread": (-0.44, -0.34, -0.18, 0.04, 0.10),
        "skillet": (-0.44, -0.12, -0.24, 0.14, 0.17),
    },
    "place_burger_fries_left_impl": {
        "tray": (-0.44, -0.08, -0.16, -0.04, 0.08),
        "hamburg": (-0.46, -0.34, 0.10, 0.16, 0.08),
        "frenchfries": (-0.46, -0.34, 0.10, 0.16, 0.08),
    },
    "place_can_basket_left_impl": {
        "basket": (-0.38, -0.12, -0.30, 0.00, 0.15),
        "can": (-0.25, -0.20, 0.0, 0.08, 0.15),
    },
    "place_cans_plasticbox_left_impl": {
        "plasticbox": (-0.42, -0.10, -0.28, -0.02, 0.13),
        "can1": (-0.42, -0.10, -0.15, -0.07, 0.08),
        "can2": (-0.42, -0.10, -0.15, -0.07, 0.08),
    },
    "place_dual_shoes_left_impl": {
        "shoe_box": (-0.46, -0.14, -0.18, -0.02, 0.13),
        "left_shoe": (-0.46, -0.14, 0.13, 0.14, 0.10),
        "right_shoe": (-0.46, -0.14, 0.13, 0.14, 0.10),
    },
}


def _inject(task, overrides: dict):
    from envs.left.left_task_base import XYYawRange
    specs = {}
    for name, spec in task.manifest.actor_specs.items():
        if name in overrides:
            xlo, xhi, ylo, yhi, clear = overrides[name]
            specs[name] = replace(
                spec,
                workspace=XYYawRange(x=(xlo, xhi), y=(ylo, yhi), yaw_rad=spec.workspace.yaw_rad),
                minimum_clearance_m=clear,
            )
        else:
            specs[name] = spec
    from envs.left.left_task_base import TaskManifest
    task.manifest = TaskManifest(
        semantic_task=task.manifest.semantic_task,
        actor_specs=specs,
        route_bin_count=task.manifest.route_bin_count,
        pilot_seed_count=task.manifest.pilot_seed_count,
        min_success_rate=task.manifest.min_success_rate,
        min_crossing_pair_rate=task.manifest.min_crossing_pair_rate,
    )


def measure(module_name: str, seeds: int = 300, overrides: dict | None = None) -> dict:
    module = importlib.import_module(f"envs.left.{module_name}")
    cls = getattr(module, CLASSES[module_name])
    failures = 0
    layouts_total = 0
    attempts = []
    for seed in range(seeds):
        np.random.seed(seed)
        task = cls()
        if overrides:
            _inject(task, overrides)
        try:
            layout = task.sample_layout()
        except Exception:  # noqa: BLE001
            failures += 1
            continue
        layouts_total += 1
        attempts.append(1)
    return {
        "task": module_name.replace("_left_impl", ""),
        "layouts_ok": layouts_total,
        "locked": failures,
        "resolve_rate": round(100.0 * layouts_total / seeds, 1),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand", action="store_true", help="use candidate ranges")
    parser.add_argument("--seeds", type=int, default=300)
    parser.add_argument("tasks", nargs="*")
    args = parser.parse_args(argv)
    targets = args.tasks or TASKS
    for name in targets:
        overrides = CANDIDATES.get(name) if args.cand else None
        try:
            result = measure(name, args.seeds, overrides)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: MEASURE ERROR {type(exc).__name__}: {exc}")
            continue
        tag = "CAND" if overrides else "CURR"
        print(f"[{tag}] {result['task']:20s} resolve={result['resolve_rate']}% "
              f"locked={result['locked']}/{args.seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
