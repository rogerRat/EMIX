"""Temporary diagnostic: standing-can grasp across candidate spawn positions.
Not a deliverable.  Usage:
    python scripts/temp/diag_place7.py <seed> <can_x> <can_y> [standing]
Spawns the can at a fixed x,y (0.741 table) and reports grasp plan status.
Pass standing=1 for the source upright quat, else identity (lying).
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import sapien
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "place_can_basket_left"


def build():
    envs = importlib.import_module(f"envs.{TASK_NAME}")
    task = getattr(envs, TASK_NAME)()
    with open(BENCHMARK / "task_config" / f"{TASK_NAME}.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = TASK_NAME
    config["task_config"] = TASK_NAME
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
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    can_x = float(sys.argv[2]) if len(sys.argv) > 2 else -0.35
    can_y = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    standing = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    task, config = build()
    from dataclasses import replace
    from envs.utils.action import ArmTag
    from envs.place_can_basket_left_impl import _actor_aabb
    try:
        man = task.manifest
        can_spec = man.actor_specs["can"]
        q = (0.707225, 0.706849, -0.0100455, -0.00982061) if standing else (1.0, 0.0, 0.0, 0.0)
        new_specs = dict(man.actor_specs)
        new_specs["can"] = replace(can_spec, base_quat=q)
        task.manifest = replace(man, actor_specs=new_specs)
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        # Override the can pose to the requested spawn.
        pose = task.can.get_pose()
        task.can.actor.set_pose(sapien.Pose([can_x, can_y, 0.741], pose.q))
        # Rest on the table bottom if lying.
        lo, _ = _actor_aabb(task.can)
        if lo < 0.741 - 0.001:
            p = list(task.can.get_pose().p)
            p[2] += 0.741 - lo
            task.can.actor.set_pose(sapien.Pose(p, task.can.get_pose().q))
        for _ in range(2000):
            task.scene.step()
        task.scene.update_render()
        can_p = np.asarray(task.can.get_pose().p)
        qq = np.asarray(task.can.get_pose().q)
        lo, hi = _actor_aabb(task.can)
        print(f"[p7] seed={seed} can={np.round(can_p, 3)} q={np.round(qq, 2)} "
              f"aabb=[{lo:.3f},{hi:.3f}] model={task.can_id}", flush=True)
        task.plan_success = True
        arm_tag = ArmTag("left")
        task.move(task.grasp_actor(task.can, arm_tag=arm_tag, pre_grasp_dis=0.05))
        print(f"[p7] grasp plan={task.plan_success}", flush=True)
        if task.plan_success:
            ee = np.asarray(task.robot.get_left_ee_pose())
            off = np.asarray(task.can.get_pose().p[:3]) - ee[:3]
            print(f"[p7]   ee={np.round(ee[:3], 3)} off={np.round(off, 3)}", flush=True)
            # Try a straight lift.
            lift = ee.copy()
            lift[2] = 1.05
            task.move(task.move_to_pose(arm_tag, lift))
            print(f"[p7]   lift plan={task.plan_success} ee={np.round(np.asarray(task.robot.get_left_ee_pose())[:3], 3)}", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
