"""Temporary: reproduce the impl flow (validate path) with the exact seed-0
layout and localize why grasp fails there while the sweep harness succeeds.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "hanging_mug_left"
MUG_P = sapien.Pose([-0.4224, 0.0078, 0.741], [0.707, 0.707, 0.0, 0.0])
RACK_P = sapien.Pose([-0.29, -0.02, 0.741], [-0.11906, -0.28744, 0.3626, 0.8754])


def main() -> int:
    from envs.hanging_mug_left import hanging_mug_left
    from envs.utils.action import ArmTag
    task = hanging_mug_left()
    with open(BENCHMARK / "task_config" / f"{TASK_NAME}.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = TASK_NAME
    config["task_config"] = TASK_NAME
    config["eval_mode"] = True
    config["left_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["right_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["dual_arm_embodied"] = False
    config["embodiment_dis"] = 0.60
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml", "r", encoding="utf-8") as handle:
        ec = yaml.safe_load(handle)
    config["left_embodiment_config"] = ec
    config["right_embodiment_config"] = ec

    task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    try:
        print(f"[I] mug pos = {np.round(np.asarray(task.mug.get_pose().p),3)}", flush=True)
        print(f"[I] rack pos = {np.round(np.asarray(task.rack.get_pose().p),3)}", flush=True)
        print(f"[I] plan_success before = {task.plan_success}", flush=True)
        print(f"[I] layout = {task.sampled_layout_summary}", flush=True)
        # step-by-step hang_mug_left
        from envs.utils.action import ArmTag
        arm = ArmTag("left")
        task.plan_success = True
        task.move(task.grasp_actor(task.mug, arm_tag=arm, pre_grasp_dis=0.05))
        print(f"[I] after grasp plan_success={task.plan_success}", flush=True)
        if not task.plan_success:
            return 0
        rack_mid = (np.asarray(task.rack.get_pose().p)
                    + np.asarray(task.rack.get_functional_point(0))[:3]) / 2
        fp_off = (np.asarray(task.mug.get_functional_point(0))[:3]
                  - np.asarray(task.mug.get_pose().p))
        tgt = [float(rack_mid[0] - fp_off[0]), float(rack_mid[1] - fp_off[1]), 0.77]
        print(f"[I] place target={np.round(tgt,3)} rack_mid={np.round(rack_mid,3)}", flush=True)
        task.plan_success = True
        task.move(task.place_actor(task.mug, arm_tag=arm, target_pose=tgt,
                                   pre_dis=0.05, dis=0.0, constrain="free"))
        print(f"[I] after place plan_success={task.plan_success}", flush=True)
        if not task.plan_success:
            return 0
        task.move(task.open_gripper(arm))
        for _ in range(120):
            task.scene.step()
        mfp = np.asarray(task.mug.get_functional_point(0))[:3]
        mp = np.asarray(task.mug.get_pose().p)
        print(f"[I] settled mug={np.round(mp,3)} fp={np.round(mfp,3)} "
              f"check={task.check_success()}", flush=True)
    finally:
        task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
