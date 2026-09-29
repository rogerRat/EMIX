"""Temporary diagnostic: dump step-4 place poses and their reachability.

Not part of the deliverable.  Usage:
    python scripts/temp/diag_place.py hanging_mug_left <mug_x> <mug_y> <rack_x> <rack_y> [transit_x transit_y transit_z]
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
    task_name = sys.argv[1]
    mx, my = float(sys.argv[2]), float(sys.argv[3])
    rx, ry = float(sys.argv[4]), float(sys.argv[5])
    task, config = build(task_name)
    if len(sys.argv) > 6:
        tx, ty, tz = (float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
        task.transit_pos = [tx, ty, tz, 1, 0, 0, 0]
        print(f"[place] transit override=({tx},{ty},{tz})", file=sys.stderr)

    def fixed_layout():
        mug_pose = sapien.Pose(np.array([mx, my, 0.74]), np.array([0.707, 0.707, 0.0, 0.0]))
        rack_pose = sapien.Pose(np.array([rx, ry, 0.74]), np.array([0.707, 0.707, 0.0, 0.0]))
        return {"mug": mug_pose, "rack": rack_pose}

    task.sample_layout = fixed_layout
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        task.play_once()
        plan = bool(task.plan_success)
        ee = np.array(task.robot.get_left_ee_pose())
        mug_pose = task.mug.get_pose()
        mug_fp = np.asarray(task.mug.get_functional_point(0))
        rack_pose = task.rack.get_pose()
        rack_fp = np.asarray(task.rack.get_functional_point(0))
        print(f"[place] mug_fp0 = {np.round(mug_fp[:3], 3)} rack_fp0 = {np.round(rack_fp[:3], 3)}")
        print(f"[place] rack_pose z={np.round(rack_pose.p[2], 3)} ee={np.round(ee[:3], 3)}")
        print(f"[place] after play_once: plan={plan}")

        if not plan:
            task.plan_success = True
        pre = task.get_place_pose(
            task.mug, "left", target_pose=list(rack_fp), functional_point_id=0,
            constrain="align", pre_dis=0.05, pre_dis_axis="fp",
        )
        place = task.get_place_pose(
            task.mug, "left", target_pose=list(rack_fp), functional_point_id=0,
            constrain="align", pre_dis=-0.05, pre_dis_axis="fp",
        )
        print(f"[place] place_pre_pose = {np.round(pre, 3)}")
        print(f"[place] place_pose     = {np.round(place, 3)}")
        for label, pose in (("place_pre", pre), ("place_pose", place)):
            task.plan_success = True
            result = task.robot.left_plan_path(list(pose))
            print(f"[place] {label} plan status = {result['status']}")
        return 0 if plan else 1
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
