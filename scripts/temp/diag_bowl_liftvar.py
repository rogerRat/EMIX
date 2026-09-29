"""Temporary diagnostic: test lift-plan variants at the failure point.

At the exact goal Curobo rejects, test (A) full SE(3), (B) position-only
constraint, (C) collision-check of the start config.  Not a deliverable.
Usage:
    python scripts/temp/diag_bowl_liftvar.py
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
    captured = {}

    def make_fixed_layout():
        bowl_pose = sapien.Pose(np.array([-0.25, -0.1, 0.74]), np.array([0.5, 0.5, 0.5, 0.5]))
        bowl_pose2 = sapien.Pose(np.array([-0.25, 0.1, 0.74]), np.array([0.5, 0.5, 0.5, 0.5]))
        return {"bowl1": bowl_pose, "bowl2": bowl_pose2}

    task, config = build(task_name)
    task.sample_layout = make_fixed_layout
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        planner = task.robot.left_planner
        pre_play_joints = np.array(task.robot.get_left_arm_jointState(), dtype=float).copy()
        orig_plan = planner.plan_path

        def traced_plan(curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
            res = orig_plan(curr_joint_pos, target_gripper_pose, constraint_pose=constraint_pose, arms_tag=arms_tag)
            if res["status"] != "Success":
                captured["goal"] = np.concatenate([target_gripper_pose.p, target_gripper_pose.q])
                captured["qpos"] = np.array(curr_joint_pos, dtype=float).copy()
            return res

        planner.plan_path = traced_plan
        task.play_once()
        print(f"[lv] final plan={task.plan_success} captured={'yes' if captured else 'no'}", flush=True)
        if not captured:
            return 0

        goal = captured["goal"]
        qpos = captured["qpos"]
        joint_indices = [planner.all_joints.index(name) for name in planner.active_joints_name
                         if name in planner.all_joints]
        joint_angles = [round(float(qpos[idx]), 5) for idx in joint_indices]
        start = CuroboJointState.from_position(
            torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1),
            joint_names=planner.active_joints_name,
        )

        def try_goal(g, constraint=None):
            try:
                pc = MotionGenPlanConfig(max_attempts=10)
                if constraint is not None:
                    pc.pose_cost_metric = (
                        planner.motion_gen.tensor_args.to_device(
                            torch.tensor(constraint, dtype=torch.float32).cuda()))
                r = planner.motion_gen.plan_single(start, CuroboPose.from_list(list(g)), pc)
                return bool(r.success.item())
            except Exception as exc:  # noqa: BLE001
                return f"EXC {type(exc).__name__}: {exc}"

        print(f"[lv] active_joints={planner.active_joints_name}", flush=True)
        print(f"[lv] goal=({np.round(goal,3)})", flush=True)
        print(f"[lv] A full SE(3):            {try_goal(goal)}", flush=True)
        for dz in (0.02, 0.05, 0.15, 0.2):
            g2 = goal.copy()
            g2[2] += dz
            print(f"[lv] A z+{dz}: {try_goal(g2)}", flush=True)
        # plan back to the arm's original pose from the failing start
        home = list(task.robot.left_original_pose)
        print(f"[lv] A to-home: {try_goal(home)}", flush=True)
        # same goal from the PRE-PLAY start (known good, gripper open)
        good_angles = [round(float(pre_play_joints[i]), 5) for i in range(len(planner.active_joints_name))]
        good_start = CuroboJointState.from_position(
            torch.tensor(good_angles, dtype=torch.float32).cuda().reshape(1, -1),
            joint_names=planner.active_joints_name,
        )
        try:
            pc = MotionGenPlanConfig(max_attempts=10)
            r = planner.motion_gen.plan_single(good_start, CuroboPose.from_list(list(goal)), pc)
            print(f"[lv] same-goal-from-pre-play: {bool(r.success.item())}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[lv] same-goal-from-pre-play EXC {type(exc).__name__}: {exc}", flush=True)
        # joint limits from the curobo yml
        import yaml as _yaml
        with open(planner.yml_path, "r", encoding="utf-8") as handle:
            yml = _yaml.safe_load(handle)
        limits = yml.get("robot_cfg", {}).get("kinematics", {}).get("joint_limits", {})
        bad = []
        for name, ang in zip(planner.active_joints_name, joint_angles):
            lim = limits.get(name)
            if lim is None:
                lim = limits.get("joint_" + name)
            if lim is not None:
                lo, hi = lim["lower"], lim["upper"]
                if ang < lo or ang > hi:
                    bad.append((name, ang, lo, hi))
        print(f"[lv] fail-start angles={np.round(joint_angles,3)}", flush=True)
        print(f"[lv] pre-play angles={np.round(good_angles,3)}", flush=True)
        print(f"[lv] joint-limit violations: {bad}", flush=True)
        return 0
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
