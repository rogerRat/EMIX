"""Temporary diagnostic: source hanging_mug across seeds, find a completed hang.
Not part of the deliverable.  Usage:
    python scripts/temp/diag_src_seeds.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import sapien
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)


def build(task_name: str):
    sys.path.insert(0, str(BENCHMARK))
    envs = importlib.import_module(f"envs.{task_name}")
    task = getattr(envs, task_name)()
    with open(BENCHMARK / "task_config" / "hanging_mug_left.yml", "r", encoding="utf-8") as handle:
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
    task_name = "hanging_mug"
    for seed in range(30):
        task, config = build(task_name)
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
            mug_start = task.mug.get_pose().p
            rack_start = task.rack.get_pose().p
            task.play_once()
            plan = bool(task.plan_success)
            mug_fp = task.mug.get_functional_point(0)[:3]
            rack_p = task.rack.get_pose().p
            rack_fp = task.rack.get_functional_point(0)[:3]
            mid = (rack_p + rack_fp) / 2
            mug_q = task.mug.get_pose().q
            succ = task.check_success()
            print(f"[srcs] {seed:2d} plan={plan} mug_start=({mug_start[0]:+.2f},{mug_start[1]:+.2f}) "
                  f"rack=({rack_start[0]:+.2f},{rack_start[1]:+.2f}) "
                  f"mug_fp=({mug_fp[0]:+.3f},{mug_fp[1]:+.3f},{mug_fp[2]:+.3f}) "
                  f"mid=({mid[0]:+.3f},{mid[1]:+.3f}) succ={succ} mug_q={np.round(mug_q,3)}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[srcs] {seed:2d} EXC {type(exc).__name__}: {exc}", flush=True)
        finally:
            try:
                task.close_env()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
