"""Temporary diagnostic: where is the align orientation reachable?

Not part of the deliverable.  Usage:
    python scripts/temp/diag_alignreach.py
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
    task, config = build(task_name)
    task.transit_pos = [-0.30, -0.20, 0.72, 1, 0, 0, 0]

    def fixed_layout():
        mug_pose = sapien.Pose(np.array([-0.30, 0.0, 0.74]), np.array([0.707, 0.707, 0.0, 0.0]))
        rack_pose = sapien.Pose(np.array([-0.20, 0.08, 0.74]), np.array([0.707, 0.707, 0.0, 0.0]))
        return {"mug": mug_pose, "rack": rack_pose}

    task.sample_layout = fixed_layout
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        from envs.utils import ArmTag  # noqa: PLC0415

        arm_tag = ArmTag("left")
        task.move(task.grasp_actor(task.mug, arm_tag=arm_tag, pre_grasp_dis=0.05))
        task.move(task.move_by_displacement(arm_tag=arm_tag, z=0.08))
        task.move(task.place_actor(task.mug, arm_tag=arm_tag,
                                   target_pose=task.transit_pos,
                                   pre_dis=0.05, dis=0.0, constrain="free"))
        task.move(task.move_by_displacement(arm_tag=arm_tag, z=0.1))
        print(f"[ar] steps0-3 plan={bool(task.plan_success)}", flush=True)
        task.plan_success = True
        align = np.asarray(task.get_place_pose(
            task.mug, "left", target_pose=list(task.rack.get_functional_point(0)),
            functional_point_id=0, constrain="align", pre_dis=-0.05, pre_dis_axis="fp",
        ))
        print(f"[ar] align q = {np.round(align[3:], 3)}", flush=True)
        n_ok = 0
        for zz in (0.70, 0.80, 0.90, 1.00, 1.10):
            for xx in (-0.40, -0.32, -0.24, -0.16):
                row = []
                for yy in (-0.30, -0.20, -0.10, 0.00, 0.10, 0.20):
                    res = task.robot.left_plan_path([xx, yy, zz] + list(align[3:]))
                    ok = res["status"] == "Success"
                    n_ok += int(ok)
                    row.append("S" if ok else ".")
                print(f"[ar] z={zz:.2f} x={xx:+.2f}: " + "".join(row) + "  (y=-0.30..0.20)", flush=True)
        print(f"[ar] total successes with align orientation = {n_ok}", flush=True)
        return 0
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
