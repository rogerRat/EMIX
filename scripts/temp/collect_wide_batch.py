#!/usr/bin/env python
"""Collect 10 episodes for the 9 remaining central_wide_cam variants.

Runs at most ``--parallel`` collections concurrently, round-robin assigning
two GPUs.  Each task's output is redirected to logs/collect_<task>.log.
Exit 0 only if every collection succeeded; skips tasks whose collection
directory already exists.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/rogerspyke/miniforge3/envs/emix-twin/bin/python"
LOG_DIR = PROJECT_ROOT / "logs"

TASKS = [
    "blocks_ranking_size_left_central_wide_cam",
    "place_bread_basket_left_central_wide_cam",
    "place_bread_skillet_left_central_wide_cam",
    "place_burger_fries_left_central_wide_cam",
    "place_can_basket_left_central_wide_cam",
    "place_cans_plasticbox_left_central_wide_cam",
    "place_dual_shoes_left_central_wide_cam",
    "stack_blocks_two_left_central_wide_cam",
    "stack_bowls_two_left_central_wide_cam",
]


def collect_one(task: str, gpu: int) -> int:
    cmd = [
        PYTHON, "data_collection/robotwin.py", task,
        "--task-config", f"{task}_10ep", "--cuda", str(gpu), "--episodes", "10",
    ]
    logf = (LOG_DIR / f"collect_{task}.log").open("w")
    print(f"[collect] {task} on cuda:{gpu}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT)
    logf.close()
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--gpu0", type=int, default=0)
    parser.add_argument("--gpu1", type=int, default=1)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue = []
    for task in TASKS:
        collected = PROJECT_ROOT / "data" / "robotwin" / task / f"{task}_10ep"
        if collected.is_dir():
            print(f"SKIP {task}: collection already exists", flush=True)
            continue
        queue.append(task)

    gpus = [args.gpu0, args.gpu1]
    jobs = {task: gpus[i % 2] for i, task in enumerate(queue)}

    failures = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(collect_one, task, gpu): task for task, gpu in jobs.items()}
        for future in as_completed(futures):
            task = futures[future]
            rc = future.result()
            print(f"[result] {task}: exit {rc}", flush=True)
            if rc != 0:
                failures.append(task)

    if failures:
        print(f"FAILED: {failures}", flush=True)
        return 1
    print("All remaining collections done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
