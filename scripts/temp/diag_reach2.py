"""Temporary diagnostic: map the true left-arm reach region.
Not a deliverable.  Usage: python scripts/temp/diag_reach2.py
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
    task, config = build("place_can_basket_left")
    from envs.utils.action import ArmTag
    task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    print(f"[r2] left base origin: {np.round(np.asarray(task.robot.left_entity_origion_pose.p), 3)}", flush=True)
    print(f"[r2] home ee: {np.round(np.asarray(task.robot.get_left_ee_pose()), 3)}", flush=True)
    arm = ArmTag("left")
    quat = [0.579, -0.404, 0.403, 0.583]  # a typical grasp-facing quat
    reach = []
    # Sweep x from -0.45 to 0.05, y from -0.10 to 0.15 at z=0.85 (table-ish)
    for x in [-0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05]:
        for y in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]:
            task.plan_success = True
            task.move(task.move_to_pose(arm, np.array([x, y, 0.85, *quat])))
            ok = bool(task.plan_success)
            print(f"[r2] x={x:.2f} y={y:.2f} z=0.85 -> {'OK ' if ok else 'FAIL'}", flush=True)
            if ok:
                reach.append((x, y))
    print(f"[r2] reachable at z=0.85: {reach}", flush=True)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
