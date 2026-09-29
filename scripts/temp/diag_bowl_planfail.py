"""Temporary diagnostic: capture Curobo plan failures in stack_bowls_two_left.

Wraps left_planner.plan_path and left_plan_multi_path to log target/goal poses
and statuses, plus the current EE pose before each move.  Not a deliverable.
Usage:
    python scripts/temp/diag_bowl_planfail.py
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
    with open(BENCHMARK / "task_config" / "stack_bowls_two_left.yml", "r", encoding="utf-8") as handle:
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
    import torch
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import JointState as CuroboJointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    task_name = "stack_bowls_two_left"
    for seed in range(4):
        task, config = build(task_name)
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        except Exception as exc:  # noqa: BLE001
            print(f"[pf] seed={seed} SETUP EXC {type(exc).__name__}: {exc}", flush=True)
            try:
                task.close_env()
            except Exception:  # noqa: BLE001
                pass
            continue
        planner = task.robot.left_planner
        orig_plan = planner.plan_path

        def traced_plan(curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
            tp = np.concatenate([target_gripper_pose.p, target_gripper_pose.q])
            # Curobo result object introspection
            joint_indices = [planner.all_joints.index(name) for name in planner.active_joints_name
                             if name in planner.all_joints]
            joint_angles = [round(float(curr_joint_pos[idx]), 5) for idx in joint_indices]
            raw = planner.motion_gen.plan_single(
                CuroboJointState.from_position(
                    torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1),
                    joint_names=planner.active_joints_name,
                ),
                CuroboPose.from_list(list(tp)),
                MotionGenPlanConfig(max_attempts=10),
            )
            res = orig_plan(curr_joint_pos, target_gripper_pose, constraint_pose=constraint_pose, arms_tag=arms_tag)
            tag = res["status"]
            if tag != "Success":
                ee = task.robot.get_left_ee_pose()
                print(f"[pf] seed={seed} FAIL goal=({np.round(tp,3)}) "
                      f"ee=({np.round(np.array(ee[:3]),3)}) raw_success={raw.success.item()} "
                      f"raw_err={getattr(raw, 'error', None)}", flush=True)
            return res

        planner.plan_path = traced_plan
        try:
            task.play_once()
            print(f"[pf] seed={seed} final plan={task.plan_success}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[pf] seed={seed} EXC {type(exc).__name__}: {exc}", flush=True)
        finally:
            try:
                task.close_env()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
