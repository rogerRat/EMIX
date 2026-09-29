"""Temporary diagnostic: which rack yaws give a reachable align orientation for
the left arm?  Not part of the deliverable.

Usage:
    python scripts/temp/diag_yawgrid.py
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d
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


def quat_mat(q):
    return t3d.quaternions.quat2mat(np.asarray(q, dtype=float))


def mat_quat(m):
    return t3d.quaternions.mat2quat(m)


def main() -> int:
    task_name = "hanging_mug_left"
    task, config = build(task_name)
    task.transit_pos = [-0.30, -0.20, 0.72, 1, 0, 0, 0]
    mug_pos = np.array([-0.30, 0.0, 0.74])
    rack_pos = np.array([-0.20, 0.08, 0.74])
    rack_q0 = np.array([0.707, 0.707, 0.0, 0.0])

    def fixed_layout():
        return {
            "mug": sapien.Pose(mug_pos, np.array([0.707, 0.707, 0.0, 0.0])),
            "rack": sapien.Pose(rack_pos, rack_q0),
        }

    task.sample_layout = fixed_layout
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        from envs.utils import ArmTag  # noqa: PLC0415

        arm_tag = ArmTag("left")
        chosen = task.choose_grasp_pose(task.mug, arm_tag=arm_tag, pre_dis=0.05, target_dis=0.0)
        grasp_pose = chosen[1]
        r_ee = quat_mat(grasp_pose[3:])
        r_fp = quat_mat(task.mug.get_functional_point(0, "pose").q)
        r_ee_in_fp = r_fp.T @ r_ee
        r0 = quat_mat(task.rack.get_functional_point(0, "pose").q)
        print(f"[yg] r_ee_in_fp = {np.round(mat_quat(r_ee_in_fp), 3)}", flush=True)
        center = [-0.25, 0.0, 0.85]
        for deg in range(-180, 181, 15):
            rz = quat_mat([math.cos(math.radians(deg) / 2), 0, 0, math.sin(math.radians(deg) / 2)])
            r_target = rz @ r0
            final_r = r_target @ r_ee_in_fp
            final_q = mat_quat(final_r)
            final_x = final_r[:, 0]
            res = task.robot.left_plan_path(center + list(final_q))
            ok = res["status"] == "Success"
            print(f"[yg] deg={deg:+4d} align_x=({final_x[0]:+.2f},{final_x[1]:+.2f},{final_x[2]:+.2f}) "
                  f"q={np.round(final_q,3)} center={'S' if ok else 'F'}",
                  flush=True)
        return 0
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
