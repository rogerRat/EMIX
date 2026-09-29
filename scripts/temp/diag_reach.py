"""Temporary diagnostic: map left-arm reachability for the align-place target.

Not part of the deliverable.  Usage:
    python scripts/temp/diag_reach.py <rack_x> <rack_y>
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
    task_name = "hanging_mug_left"
    mx, my = -0.30, 0.0
    rx, ry = float(sys.argv[1]), float(sys.argv[2])
    task, config = build(task_name)
    task.transit_pos = [-0.30, -0.20, 0.72, 1, 0, 0, 0]

    def fixed_layout():
        mug_pose = sapien.Pose(np.array([mx, my, 0.74]), np.array([0.707, 0.707, 0.0, 0.0]))
        rack_pose = sapien.Pose(np.array([rx, ry, 0.74]), np.array([0.707, 0.707, 0.0, 0.0]))
        return {"mug": mug_pose, "rack": rack_pose}

    task.sample_layout = fixed_layout
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        rack_fp = np.asarray(task.rack.get_functional_point(0))
        print(f"[reach] rack_fp0 = {np.round(rack_fp[:3], 3)}")
        from envs.utils import ArmTag  # noqa: PLC0415

        arm_tag = ArmTag("left")
        task.move(task.grasp_actor(task.mug, arm_tag=arm_tag, pre_grasp_dis=0.05))
        task.move(task.move_by_displacement(arm_tag=arm_tag, z=0.08))
        task.move(task.place_actor(task.mug, arm_tag=arm_tag,
                                   target_pose=task.transit_pos,
                                   pre_dis=0.05, dis=0.0, constrain="free"))
        task.move(task.move_by_displacement(arm_tag=arm_tag, z=0.1))
        before = bool(task.plan_success)
        current = np.asarray(task.robot.get_left_ee_pose())
        print(f"[reach] after steps0-3 ee={np.round(current[:3], 3)} q={np.round(current[3:], 3)}")
        place = task.get_place_pose(
            task.mug, "left", target_pose=list(task.rack.get_functional_point(0)),
            functional_point_id=0, constrain="align", pre_dis=-0.05, pre_dis_axis="fp",
        )
        print(f"[reach] align place target = {np.round(place, 3)}")
        # Probe position reachability using the current EE orientation.
        px, py, pz = place[0], place[1], place[2]
        res = task.robot.left_plan_path([px, py, pz] + list(current[3:]))
        print(f"[reach] target pos + current orient: {res['status']}")
        res = task.robot.left_plan_path(list(place))
        print(f"[reach] target pos + target orient: {res['status']}")
        # Probe offset targets: raise the mug COM by lowering/raising EE z.
        for dz in (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20):
            res = task.robot.left_plan_path([px, py, pz + dz] + list(place[3:]))
            print(f"[reach]   z+{dz:+.2f}: {res['status']}")
        task.move(task.place_actor(task.mug, arm_tag=arm_tag,
                                   target_pose=list(task.rack.get_functional_point(0)),
                                   functional_point_id=0, constrain="align",
                                   pre_dis=0.05, dis=-0.05, pre_dis_axis="fp"))
        print(f"[reach] align_place plan={bool(task.plan_success)}")
        return 0 if task.plan_success else 1
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
