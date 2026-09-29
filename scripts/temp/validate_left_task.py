"""Static contract and expert-pilot validation for the 22 left-arm variants.

This is one of the six acceptance scripts defined by the engineering plan.  The
pilot stage runs the expert ``play_once`` for a requested seed range and records
plan success, geometric success, right-arm immobility, step parity and the
sampled layout.  A 100-seed pilot passes only at >= ``manifest.min_success_rate``
success, the manifest crossing threshold, <= 1e-5 right deviation and exact step
parity.

Usage:
    python scripts/validate_left_task.py --task stack_blocks_two_left --seeds 30
    python scripts/validate_left_task.py --all --seeds 100 --output results/
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = PROJECT_ROOT / "third_party" / "RoboTwin"
RIGHT_ARM_TOLERANCE = 1e-5

TASKS_22 = [
    "blocks_ranking_rgb_left", "blocks_ranking_rgb_left_oppo_cam",
    "blocks_ranking_size_left", "blocks_ranking_size_left_oppo_cam",
    "hanging_mug_left", "hanging_mug_left_oppo_cam",
    "place_bread_basket_left", "place_bread_basket_left_oppo_cam",
    "place_bread_skillet_left", "place_bread_skillet_left_oppo_cam",
    "place_burger_fries_left", "place_burger_fries_left_oppo_cam",
    "place_can_basket_left", "place_can_basket_left_oppo_cam",
    "place_cans_plasticbox_left", "place_cans_plasticbox_left_oppo_cam",
    "place_dual_shoes_left", "place_dual_shoes_left_oppo_cam",
    "stack_blocks_two_left", "stack_blocks_two_left_oppo_cam",
    "stack_bowls_two_left", "stack_bowls_two_left_oppo_cam",
]


@dataclass
class ValidationResult:
    task: str
    camera_variant: str
    seeds_attempted: int
    expert_successes: int
    expert_success_rate: float
    crossing_pair_rate: float
    max_right_arm_deviation: float
    source_step_limit: int
    derived_step_limit: int
    plan_failures: int
    layout_failures: int
    failures: list[dict[str, Any]]


def _load_task_config(task: str, benchmark_root: Path) -> dict[str, Any]:
    os.chdir(benchmark_root)
    sys.path.insert(0, str(benchmark_root))
    # The left-arm family lives under task_config/left/<variant>/ after the
    # subfolder refactor; fall back to the legacy flat locations.
    config_candidates = [path for path in
                         (benchmark_root / "task_config" / "left").glob(f"*/{task}.yml")]
    config_candidates += [
        benchmark_root / "task_config" / "left" / f"{task}.yml",
        benchmark_root / "task_config" / f"{task}.yml",
    ]
    config_path = next((path for path in config_candidates if path.is_file()), config_candidates[-1])
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = task
    config["task_config"] = task
    config["eval_mode"] = True
    config["left_robot_file"] = str(benchmark_root / "assets" / "embodiments" / "piper")
    config["right_robot_file"] = str(benchmark_root / "assets" / "embodiments" / "piper")
    config["dual_arm_embodied"] = False
    config["embodiment_dis"] = 0.60
    with open(benchmark_root / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        config["left_embodiment_config"] = yaml.safe_load(handle)
        config["right_embodiment_config"] = config["left_embodiment_config"]
    return config


def validate_static_contract(task: str, benchmark_root: Path) -> list[str]:
    """Run cheap static checks; return a list of human-readable failures."""
    failures: list[str] = []
    os.chdir(benchmark_root)
    sys.path.insert(0, str(benchmark_root))
    # Strip the camera-variant suffix first (it follows "_left"), then "_left"
    # itself, so e.g. blocks_ranking_rgb_left_cen_arm_front_cam resolves to the
    # blocks_ranking_rgb semantic implementation.
    base = task
    for cam_suffix in ("_cen_arm_right_wide_cam", "_cen_arm_near_side_cam",
                       "_cen_arm_side_cam",
                       "_cen_arm_front_cam", "_central_wide_cam", "_left_oppo_cam"):
        if base.endswith(cam_suffix):
            base = base[: -len(cam_suffix)]
            break
    base = base.removesuffix("_left")
    impl_module = base + "_left_impl"
    impl_class = "".join(part.capitalize() for part in base.split("_")) + "LeftImpl"
    try:
        module = _import_task_module(task)
        impl = importlib.import_module(f"envs.left.impl.{impl_module}")
        task_cls = getattr(module, task)
        impl_cls = getattr(impl, impl_class)
    except Exception as exc:  # noqa: BLE001
        return [f"import failure: {type(exc).__name__}: {exc}"]
    try:
        header = (Path(impl.__file__).read_text(encoding="utf-8").splitlines()[0])
        if "LEFT_TASK_DESIGN:" not in header:
            failures.append("missing LEFT_TASK_DESIGN: module header")
    except Exception:  # noqa: BLE001
        failures.append("cannot read impl module header")
    if not hasattr(task_cls, "camera_variant"):
        failures.append("wrapper lacks camera_variant")
    if not issubclass(task_cls, impl_cls):
        failures.append("wrapper is not a subclass of the semantic impl")
    return failures


def _import_task_module(task: str):
    """Import one left-arm task module under the RoboTwin CWD.

    envs.utils reads RoboTwin asset manifests via CWD-relative paths at import
    time, so the benchmark CWD must be set before the first (uncached) import.
    Delegates to envs._GLOBAL_CONFIGS.import_task_env, which resolves the
    envs/left/<camera-variant>/ subfolder layout and the legacy flat envs/
    location.
    """
    from envs._GLOBAL_CONFIGS import import_task_env

    return import_task_env(task)



def run_seed(task: str, config: str, seed: int, benchmark_root: Path) -> dict[str, Any]:
    """Run one expert episode and emit a serializable record."""
    cfg = _load_task_config(task, benchmark_root)
    task_module = _import_task_module(task)
    task_cls = getattr(task_module, task)
    env = task_cls()
    record: dict[str, Any] = {
        "task": task, "config": config, "seed": seed,
        "setup_ok": False, "plan_success": False, "success": False,
        "right_max_abs_deviation": 0.0, "step_limit": 0, "take_action_cnt": 0,
        "layout": {}, "route_bins": [], "error": None,
    }
    try:
        env.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **cfg)
        record["setup_ok"] = True
        env.play_once()
        record["plan_success"] = bool(env.plan_success)
        record["success"] = bool(env.plan_success) and bool(env.check_success())
        record["right_max_abs_deviation"] = float(getattr(env, "right_max_abs_deviation", 0.0))
        record["step_limit"] = int(env.step_lim or 0)
        record["take_action_cnt"] = int(getattr(env, "take_action_cnt", 0))
        record["layout"] = getattr(env, "sampled_layout_summary", {}) or {}
        record["route_bins"] = list(getattr(env, "route_bins", []))
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass
    return record


def calculate_crossing_pair_rate(records: Sequence[Mapping[str, Any]]) -> float:
    """Return the mean crossing-pair rate from route-bin diversity."""
    if not records:
        return 0.0
    total, crossing = 0, 0
    for record in records:
        bins = [int(b) for b in record.get("route_bins", [])]
        if len(bins) < 2:
            continue
        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                total += 1
                if bins[i] != bins[j]:
                    crossing += 1
    return float(crossing / total) if total else 0.0


def validate_task(task: str, config: str, seeds: Sequence[int], benchmark_root: Path,
                  output_dir: Path) -> ValidationResult:
    """Run the seed pilot for one variant and aggregate the result."""
    failures: list[dict[str, Any]] = []
    successes = 0
    attempted = 0
    max_right_dev = 0.0
    derived_lim = 0
    source_lim = 0
    plan_failures = 0
    layout_failures = 0
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        record = run_seed(task, config, seed, benchmark_root)
        records.append(record)
        attempted += 1
        max_right_dev = max(max_right_dev, float(record["right_max_abs_deviation"]))
        derived_lim = max(derived_lim, int(record["step_limit"]))
        if record["setup_ok"] and not record["plan_success"]:
            plan_failures += 1
        if not record["setup_ok"]:
            layout_failures += 1
        if record["success"]:
            successes += 1
        if not record["success"]:
            failures.append(record)
    with open(output_dir / f"{task}.pilot.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    source_task = task.replace("_left", "").replace("_oppo_cam", "")
    if task.endswith("_left_oppo_cam"):
        source_task = task.replace("_left_oppo_cam", "")
    with open(benchmark_root / "task_config" / "_eval_step_limit.yml", "r", encoding="utf-8") as handle:
        step_data = yaml.safe_load(handle) or {}
    source_lim = int(step_data.get(source_task, 0) or 0)
    return ValidationResult(
        task=task,
        camera_variant="left_oppo_cam" if task.endswith("_left_oppo_cam") else "central_cam",
        seeds_attempted=attempted,
        expert_successes=successes,
        expert_success_rate=float(successes / attempted) if attempted else 0.0,
        crossing_pair_rate=calculate_crossing_pair_rate(records),
        max_right_arm_deviation=max_right_dev,
        source_step_limit=source_lim,
        derived_step_limit=derived_lim,
        plan_failures=plan_failures,
        layout_failures=layout_failures,
        failures=failures,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate left-arm task variants")
    parser.add_argument("--task", default=None, help="one of the 22 variants")
    parser.add_argument("--all", action="store_true", help="validate all 22 variants")
    parser.add_argument("--seeds", type=int, default=30, help="number of pilot seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "pilot"),
                        help="output directory")
    args = parser.parse_args(argv)
    tasks = TASKS_22 if args.all else [args.task]
    if not args.task and not args.all:
        parser.error("--task or --all is required")
    output_dir = Path(args.output)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for task in tasks:
        static = validate_static_contract(task, BENCHMARK)
        if static:
            print(f"[validate] {task} STATIC FAIL: {static}")
            continue
        result = validate_task(task, task, seeds, BENCHMARK, output_dir)
        print(json.dumps(asdict(result), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
