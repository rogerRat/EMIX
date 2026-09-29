"""Temporary experiment: fix mug/rack layout to test reachability of z-lift.

Not part of the deliverable.  Usage:
    python scripts/temp/diag_layout.py <task_name> <mug_x> <mug_y> <rack_x> <rack_y>
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
    task_name, mx, my, rx, ry = (sys.argv[1], float(sys.argv[2]), float(sys.argv[3]),
                                 float(sys.argv[4]), float(sys.argv[5]))
    qpos = sys.argv[6] if len(sys.argv) > 6 else "707"
    task, config = build(task_name)
    if len(sys.argv) > 7:
        tx, ty, tz = (float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]))
        task.transit_pos = [tx, ty, tz, 1, 0, 0, 0]
        print(f"[layout] transit override=({tx},{ty},{tz})", file=sys.stderr)

    if qpos == "707":
        mug_q = np.array([0.707, 0.707, 0.0, 0.0])
    else:
        mug_q = np.array([0.5, 0.5, 0.5, 0.5])

    def fixed_layout():
        mug_pose = sapien.Pose(np.array([mx, my, 0.74]), mug_q)
        rack_pose = sapien.Pose(np.array([rx, ry, 0.74]), np.array([0.707, 0.707, 0.0, 0.0]))
        return {"mug": mug_pose, "rack": rack_pose}

    task.sample_layout = fixed_layout
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        orig_move = task.move
        calls = []

        def traced(*a, **k):
            result = orig_move(*a, **k)
            calls.append((len(calls), bool(task.plan_success)))
            return result

        task.move = traced
        task.play_once()
        print(f"[layout] mug=({mx},{my}) rack=({rx},{ry}) plan={task.plan_success} trace={calls}")
        return 0 if task.plan_success else 1
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
