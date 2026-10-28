"""Collect native RoboTwin demonstrations.

RoboTwin resolves both its task and configuration relative to its own working
directory.  This wrapper deliberately runs the native collector in that
directory, then moves the resulting collection into the repository's ``data``
directory so that it can be consumed by the conversion tools.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence


# Keep this list explicit.  A typo should fail before importing RoboTwin (and
# consequently before starting a simulator process).  Add a task here only
# when its implementation is present in third_party/RoboTwin/envs.
ALLOWED_TASKS = frozenset(
    {
        # Existing reference implementations (kept unchanged).
        "click_bell_left",
        "place_object_scale_left",
        "place_object_scale_left_oppo_cam",
        # Eleven left-arm semantic families, three camera variants each.
        # hanging_mug is DEPRECATED / NotFullyImplemented: it remains listed
        # only to keep the collection contract stable, but HangingMugLeftImpl
        # warns on instantiation and it must not be collected.
        "blocks_ranking_rgb_left",
        "blocks_ranking_rgb_left_oppo_cam",
        "blocks_ranking_rgb_left_central_wide_cam",
        "blocks_ranking_size_left",
        "blocks_ranking_size_left_oppo_cam",
        "blocks_ranking_size_left_central_wide_cam",
        "hanging_mug_left",
        "hanging_mug_left_oppo_cam",
        "hanging_mug_left_central_wide_cam",
        "place_bread_basket_left",
        "place_bread_basket_left_oppo_cam",
        "place_bread_basket_left_central_wide_cam",
        "place_bread_skillet_left",
        "place_bread_skillet_left_oppo_cam",
        "place_bread_skillet_left_central_wide_cam",
        "place_burger_fries_left",
        "place_burger_fries_left_oppo_cam",
        "place_burger_fries_left_central_wide_cam",
        "place_can_basket_left",
        "place_can_basket_left_oppo_cam",
        "place_can_basket_left_central_wide_cam",
        "place_cans_plasticbox_left",
        "place_cans_plasticbox_left_oppo_cam",
        "place_cans_plasticbox_left_central_wide_cam",
        "place_dual_shoes_left",
        "place_dual_shoes_left_oppo_cam",
        "place_dual_shoes_left_central_wide_cam",
        "stack_blocks_two_left",
        "stack_blocks_two_left_oppo_cam",
        "stack_blocks_two_left_central_wide_cam",
        "stack_bowls_two_left",
        "stack_bowls_two_left_oppo_cam",
        "stack_bowls_two_left_central_wide_cam",
        "blocks_ranking_rgb_left_cen_arm_right_wide_cam",
        "blocks_ranking_size_left_cen_arm_right_wide_cam",
        "place_bread_basket_left_cen_arm_right_wide_cam",
        "place_bread_skillet_left_cen_arm_right_wide_cam",
        "place_burger_fries_left_cen_arm_right_wide_cam",
        "place_can_basket_left_cen_arm_right_wide_cam",
        "place_cans_plasticbox_left_cen_arm_right_wide_cam",
        "place_dual_shoes_left_cen_arm_right_wide_cam",
        "stack_blocks_two_left_cen_arm_right_wide_cam",
        "stack_bowls_two_left_cen_arm_right_wide_cam",
        "blocks_ranking_rgb_left_cen_arm_near_side_cam",
        "blocks_ranking_size_left_cen_arm_near_side_cam",
        "place_bread_basket_left_cen_arm_near_side_cam",
        "place_bread_skillet_left_cen_arm_near_side_cam",
        "place_burger_fries_left_cen_arm_near_side_cam",
        "place_can_basket_left_cen_arm_near_side_cam",
        "place_cans_plasticbox_left_cen_arm_near_side_cam",
        "place_dual_shoes_left_cen_arm_near_side_cam",
        "stack_blocks_two_left_cen_arm_near_side_cam",
        "stack_bowls_two_left_cen_arm_near_side_cam",
        "blocks_ranking_rgb_left_cen_arm_side_cam",
        "blocks_ranking_size_left_cen_arm_side_cam",
        "place_bread_basket_left_cen_arm_side_cam",
        "place_bread_skillet_left_cen_arm_side_cam",
        "place_burger_fries_left_cen_arm_side_cam",
        "place_can_basket_left_cen_arm_side_cam",
        "place_cans_plasticbox_left_cen_arm_side_cam",
        "place_dual_shoes_left_cen_arm_side_cam",
        "stack_blocks_two_left_cen_arm_side_cam",
        "stack_bowls_two_left_cen_arm_side_cam",
        "blocks_ranking_rgb_left_cen_arm_front_cam",
        "blocks_ranking_size_left_cen_arm_front_cam",
        "place_bread_basket_left_cen_arm_front_cam",
        "place_bread_skillet_left_cen_arm_front_cam",
        "place_burger_fries_left_cen_arm_front_cam",
        "place_can_basket_left_cen_arm_front_cam",
        "place_cans_plasticbox_left_cen_arm_front_cam",
        "place_dual_shoes_left_cen_arm_front_cam",
        "stack_blocks_two_left_cen_arm_front_cam",
        "stack_bowls_two_left_cen_arm_front_cam",
        "blocks_ranking_rgb_left_cen_arm_top_cam",
        "blocks_ranking_size_left_cen_arm_top_cam",
        "place_bread_basket_left_cen_arm_top_cam",
        "place_bread_skillet_left_cen_arm_top_cam",
        "place_burger_fries_left_cen_arm_top_cam",
        "place_can_basket_left_cen_arm_top_cam",
        "place_cans_plasticbox_left_cen_arm_top_cam",
        "place_dual_shoes_left_cen_arm_top_cam",
        "stack_blocks_two_left_cen_arm_top_cam",
        "stack_bowls_two_left_cen_arm_top_cam",
    }
)


def _paths(project_root: Path, task: str, config: str) -> tuple[Path, Path]:
    """Return RoboTwin's collection path and this project's destination."""
    source = project_root / "third_party" / "RoboTwin" / "data" / task / config
    destination = project_root / "data" / "robotwin" / task / config
    return source, destination


def _set_episode_num(config_file: Path, episodes: int) -> None:
    """Rewrite the task config's ``episode_num`` line in place.

    Same rule as ``scripts/dataflow.sh --num``: the generated cen_arm configs
    tolerate a runtime episode-count override (the asset generator's check
    mode normalizes it back), so drivers can pin the collection target without
    regenerating the source asset.
    """
    text = config_file.read_text(encoding="utf-8")
    updated, count = re.subn(r"^episode_num:[^\n]*", f"episode_num: {episodes}", text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected exactly one 'episode_num:' line in {config_file}, found {count}")
    if updated != text:
        config_file.write_text(updated, encoding="utf-8")
        print(f"[RoboTwin] episode_num -> {episodes} in {config_file}")


def collect(
    task: str,
    *,
    task_config: str | None = None,
    episodes: int = 300,
    project_root: Path | None = None,
    python: str | None = None,
    cuda_visible_devices: str | None = None,
    output_root: Path | None = None,
    dry_run: bool = False,
    instance_annotations: bool = False,
) -> Path:
    """Run RoboTwin collection and return the moved collection directory."""
    if task not in ALLOWED_TASKS:
        choices = ", ".join(sorted(ALLOWED_TASKS))
        raise ValueError(f"Unsupported RoboTwin task {task!r}. Allowed tasks: {choices}")

    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    benchmark_root = root / "third_party" / "RoboTwin"
    config = task_config or task
    config_candidates = tuple(
        benchmark_root / "task_config" / "left" / folder / f"{config}.yml"
        # Mirrors the envs/left layout: one subfolder per camera variant, base
        # holds the default central-cam configs, "" catches top-level utility
        # tasks (place_object_scale_left*). Keep in sync with
        # envs/_GLOBAL_CONFIGS.py task_config_yml_path and evaluation/evaluate.py.
        for folder in ("cen_arm_right_wide_cam", "cen_arm_near_side_cam",
                       "cen_arm_side_cam", "cen_arm_front_cam", "cen_arm_top_cam",
                       "central_wide_cam", "left_oppo_cam",
                       "base", "")
    ) + (benchmark_root / "task_config" / f"{config}.yml",)
    config_file = next((path for path in config_candidates if path.is_file()), config_candidates[-1])
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"RoboTwin directory does not exist: {benchmark_root}")
    if not config_file.is_file():
        raise FileNotFoundError(
            "RoboTwin task config does not exist. Checked: "
            + ", ".join(str(path) for path in config_candidates)
        )

    source, destination = _paths(root, task, config)
    if output_root is not None:
        destination = output_root.resolve() / task / config
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing collection: {destination}. "
            "Move or remove it explicitly before collecting again."
        )

    command = [python or sys.executable, "script/collect_data.py", task, config]
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if instance_annotations:
        env["EMIX_INSTANCE_ANNOTATIONS"] = "1"
    print(f"[RoboTwin] {' '.join(command)}")
    print(f"[RoboTwin] cwd: {benchmark_root}")
    print(f"[RoboTwin] output: {destination}")
    if dry_run:
        return destination
    if episodes is not None:
        if episodes < 1:
            raise ValueError("episodes must be a positive integer")
        _set_episode_num(config_file, episodes)

    subprocess.run(command, cwd=benchmark_root, env=env, check=True)
    if not source.is_dir():
        raise FileNotFoundError(
            f"RoboTwin exited successfully but produced no collection: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    seed_path = destination / "seed.txt"
    if seed_path.is_file():
        seeds = [int(seed) for seed in seed_path.read_text(encoding="utf-8").split()]
        attempts = max(seeds) + 1 if seeds else 0
        metrics = {
            "accepted_episodes": len(seeds),
            "attempted_seeds": attempts,
            "collection_success_rate": len(seeds) / attempts if attempts else 0.0,
        }
        (destination / "collection_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (destination / ".complete").write_text("ok\n", encoding="ascii")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(ALLOWED_TASKS))
    parser.add_argument(
        "--task-config",
        help="RoboTwin config basename (default: same as TASK, without .yml)",
    )
    parser.add_argument(
        "--episodes", type=int, default=300,
        help="Collection target; rewrites episode_num in the task config "
             "(same rule as scripts/dataflow.sh --num)",
    )
    parser.add_argument("--python", help="Python executable used for RoboTwin")
    parser.add_argument("--cuda", help="Value for CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Collection root (default: <project>/data/robotwin)",
    )
    parser.add_argument("--instance-annotations", action="store_true", help="Capture lossless raw actor IDs and object tables for patch training.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = collect(
            args.task,
            task_config=args.task_config,
            episodes=args.episodes,
            python=args.python,
            cuda_visible_devices=args.cuda,
            output_root=args.output_root,
            dry_run=args.dry_run,
            instance_annotations=args.instance_annotations,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"[RoboTwin] error: {error}", file=sys.stderr)
        return 2
    if not args.dry_run:
        print(f"[RoboTwin] moved collection to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
