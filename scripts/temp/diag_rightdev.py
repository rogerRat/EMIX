"""Temporary diagnostic: measure the per-joint right-arm deviation over a
successful expert play.  Not a deliverable.
Usage:
    python scripts/temp/diag_rightdev.py
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
    task, config = build("stack_bowls_two_left")
    from envs.utils.action import ArmTag
    layout = {
        "bowl1": sapien.Pose(np.array([-0.25, -0.1, 0.74]), np.array([0.5, 0.5, 0.5, 0.5])),
        "bowl2": sapien.Pose(np.array([-0.25, 0.1, 0.74]), np.array([0.5, 0.5, 0.5, 0.5])),
    }
    task.sample_layout = lambda: layout

    def move_bowl_arm(self, actor, target_pose):
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(actor, arm_tag=arm_tag, contact_point_id=0, pre_grasp_dis=0.1))
        if not self.plan_success:
            return False
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        if not self.plan_success:
            return False
        target_pose_7 = target_pose.tolist() + self.quat_of_target_pose
        self.move(self.place_actor(actor, target_pose=target_pose_7, arm_tag=arm_tag,
                                   functional_point_id=0, pre_dis=0.09, dis=0.0,
                                   constrain="align"))
        if not self.plan_success:
            return False
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.09, move_axis="arm"))
        return self.plan_success

    try:
        task.move_bowl = move_bowl_arm.__get__(task, type(task))
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        home = np.array(task.right_home_real_qpos, dtype=float)
        print(f"[rd] right_home_real={np.round(home,6)}", flush=True)
        # capture deviation during play via assert override
        orig_assert = type(task).assert_right_arm_stationary

        def traced_assert(self):
            actual = np.asarray(self.robot.get_right_arm_real_jointState(), dtype=np.float32)
            dev = np.abs(actual - self.right_home_real_qpos)
            print(f"[rd] per-joint dev={np.round(dev.astype(float), 8)}", flush=True)
            return orig_assert(self)

        task.assert_right_arm_stationary = traced_assert.__get__(task, type(task))
        task.play_once()
        print(f"[rd] final plan={task.plan_success} succ={bool(task.check_success())}", flush=True)
        return 0
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
