"""Temporary diagnostic: run setup/play/check/close for one left task variant.

Not part of the deliverable.  Usage:
    python scripts/temp/diag_one.py <task_name> <seed>
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)


def build(task_name: str):
    sys.path.insert(0, str(BENCHMARK))
    envs = importlib.import_module(f"envs.{task_name}")
    task = getattr(envs, task_name)()
    with open(BENCHMARK / "task_config" / f"{task_name}.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = task_name
    config["task_config"] = task_name
    config["eval_mode"] = True
    config["left_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["right_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["dual_arm_embodied"] = False
    config["embodiment_dis"] = 0.60
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    return task, config


def main() -> int:
    task_name, seed = sys.argv[1], int(sys.argv[2])
    task, config = build(task_name)
    try:
        task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
        if "--trace" in sys.argv:
            orig_move = task.move
            calls = []

            def traced(*a, **k):
                result = orig_move(*a, **k)
                calls.append((len(calls), bool(task.plan_success)))
                return result

            task.move = traced
        info = task.play_once()
        plan = bool(task.plan_success)
        success = plan and bool(task.check_success())
        print(f"[diag] {task_name} seed={seed} plan={plan} success={success}")
        print(f"[diag] info={info.get('info')}")
        if "--trace" in sys.argv and "calls" in dir():
            print(f"[diag] move_trace={calls}")
        if hasattr(task, "camera_variant"):
            print(f"[diag] camera_variant={task.camera_variant}")
        if hasattr(task, "right_max_abs_deviation"):
            print(f"[diag] right_max_abs_deviation={task.right_max_abs_deviation}")
        return 0 if success else 1
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
