"""Backfill visualization HTML+PNG for every task with a complete EmixIndex.

Runs the flow's own ``run_vis_task`` stage (same commands, same output root)
without acquiring the exclusive flow lock, which a long training run can hold
for hours.  Renders are CPU-only and serial, and always apply the flow's
performance guardrails (``visualization.max_points`` point cap plus
``skip_click_images``), so one task renders in minutes instead of tens.

This is the tool for the "an EmixIndex already exists" branch of the
visualization guarantee: it renders whatever is complete right now and leaves
every index untouched.  A task is skipped when its HTML/PNG already exist
(resume-safe), and tasks without a parseable ``emix_index.json`` are reported
and passed over.

Usage:
  python scripts/backfill_vis.py                    # default run.yaml tasks
  python scripts/backfill_vis.py \
      --config configs/scripts/run_cen_arm_top_cam_500ep_timeratio.yaml \
      --output-root outputs/intersections_timeratio
  python scripts/backfill_vis.py --config a.yaml --config b.yaml --max-points 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# scripts/run.py imports oidn_env from its own directory, which Python only
# puts on sys.path when run.py is executed as a script -- add it explicitly
# so the module import below works from anywhere.
sys.path.insert(0, str(ROOT / "scripts"))

from run import (  # noqa: E402  (scripts/run.py imported as a module)
    DEFAULT_LOG_ROTATE_BYTES,
    FLOW_CONFIG,
    RotatingLog,
    completed_index,
    completed_vis,
    load_config,
    log_rotate_bytes,
    run_vis_task,
    task_paths,
    task_priority,
    vis_output_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render HTML+PNG visualizations for existing EmixIndexes (lock-free).",
    )
    parser.add_argument(
        "--config", type=Path, action="append", default=None,
        help="Flow runbook whose tasks/index/vis roots to render; repeatable "
             "(default: configs/scripts/run.yaml).",
    )
    parser.add_argument(
        "--index-root", type=Path, default=None,
        help="Override each config's paths.index_root ({task} template), e.g. an "
             "archived index tree: archive/<batch>/data/emix_index/{task}. The "
             "index files are only read, never modified.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Override each config's paths.dataset_root ({task} template), e.g. "
             "an archived LeRobot tree when the live data/ root is emptied: "
             "archive/<batch>/data/lerobot/{task}. Only read, never modified.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Override each config's visualization.output_root, e.g. "
             "outputs/intersections_timeratio.",
    )
    parser.add_argument(
        "--max-points", type=int, default=None,
        help="Override visualization.max_points (default: the config value, "
             "normally 10000).",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated subset of task names to render (default: every "
             "task listed in the runbook).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = args.config or [FLOW_CONFIG]
    only_tasks = {name.strip() for name in args.tasks.split(",")} if args.tasks else None

    log_directory = ROOT / "logs" / "vis-backfill"
    log = RotatingLog(log_directory, 0, DEFAULT_LOG_ROTATE_BYTES)

    failed: list[str] = []
    rendered = reused = without_index = 0
    for config_path in config_paths:
        config = load_config(config_path.resolve())
        if args.dataset_root is not None:
            config["paths"]["dataset_root"] = str(args.dataset_root)
        if args.index_root is not None:
            config["paths"]["index_root"] = str(args.index_root)
        if args.output_root is not None:
            # normalize_config already copied output_root into paths, so keep
            # both views in sync.
            config["visualization"]["output_root"] = str(args.output_root)
            config["paths"]["visualization_root"] = str(args.output_root)
        if args.max_points is not None:
            config["visualization"]["max_points"] = args.max_points
        tasks = [task for task in config["tasks"] if only_tasks is None or task in only_tasks]
        tasks.sort(key=lambda task: task_priority(task, config))

        message = (
            f"[vis-backfill] {config_path.name}: {len(tasks)} tasks -> "
            f"{config['paths']['visualization_root']} "
            f"(max-points {config['visualization']['max_points']}, "
            f"skip-click-images {config['visualization'].get('skip_click_images', True)})\n"
        )
        log.write(message)
        print(message, end="", flush=True)

        for task_index, task in enumerate(tasks, start=1):
            _, dataset, index = task_paths(task, config)
            if not completed_index(index):
                without_index += 1
                message = f"[vis-backfill] skipping {task}: no complete EmixIndex at {index}\n"
                log.write(message)
                print(message, end="", flush=True)
                continue
            html, png = vis_output_paths(task, config)
            require_assets = not bool(config["visualization"].get("skip_click_images", True))
            if completed_vis(html, png, task, require_assets=require_assets):
                reused += 1
                message = f"[vis-backfill] reusing complete visualization: {html}\n"
                log.write(message)
                print(message, end="", flush=True)
                continue
            try:
                run_vis_task(task, None, config, task_index=task_index, task_total=len(tasks), log=log)
                rendered += 1
            except BaseException:
                failed.append(task)
                message = f"[vis-backfill] FAILED on {task}; continuing with next task\n"
                log.write(message)
                print(message, end="", flush=True)

    log.write("[vis-backfill] finished\n")
    log.close()
    summary = (
        f"[vis-backfill] finished: {rendered} rendered, {reused} reused, "
        f"{without_index} without index, {len(failed)} failed"
    )
    print(summary, flush=True)
    if failed:
        raise SystemExit(f"{summary}: {', '.join(failed)}")


if __name__ == "__main__":
    main()
