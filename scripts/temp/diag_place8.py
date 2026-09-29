"""Temporary diagnostic: mirror the source place_can_basket recipe for the left arm.
Not a deliverable.  Usage:
    python scripts/temp/diag_place8.py <seed> [mode]
mode: 0 = source standing can + place_actor
      1 = lying can + place_actor
      2 = source standing can + hover-drop
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
    mode = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    task, config = build()
    from dataclasses import replace
    from envs.utils.action import ArmTag
    from envs.place_can_basket_left_impl import _actor_aabb
    try:
        man = task.manifest
        standing_q = (0.707225, 0.706849, -0.0100455, -0.00982061)
        identity_q = (1.0, 0.0, 0.0, 0.0)
        q = standing_q if mode in (0, 2) else identity_q
        new_specs = dict(man.actor_specs)
        new_specs["can"] = replace(new_specs["can"], base_quat=q)
        task.manifest = replace(man, actor_specs=new_specs)
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        print(f"[p8] seed={seed} mode={mode} can_model={task.can_id} basket={task.basket_id}", flush=True)
        # Override to source positions.
        basket_p = task.basket.get_pose()
        task.basket.actor.set_pose(sapien.Pose([0.02, -0.065, basket_p.p[2]], basket_p.q))
        can_p = task.can.get_pose()
        task.can.actor.set_pose(sapien.Pose([-0.22, 0.05, 0.741], can_p.q))
        lo, _ = _actor_aabb(task.can)
        if lo < 0.741 - 0.001:
            p = list(task.can.get_pose().p)
            p[2] += 0.741 - lo
            task.can.actor.set_pose(sapien.Pose(p, task.can.get_pose().q))
        for _ in range(2000):
            task.scene.step()
        task.scene.update_render()
        can_p = np.asarray(task.can.get_pose().p)
        print(f"[p8] can settle={np.round(can_p, 3)} q={np.round(task.can.get_pose().q, 2)}", flush=True)
        task.plan_success = True
        arm_tag = ArmTag("left")
        task.move(task.grasp_actor(task.can, arm_tag=arm_tag, pre_grasp_dis=0.05))
        print(f"[p8] grasp plan={task.plan_success}", flush=True)
        if not task.plan_success:
            return 0
        ee = np.asarray(task.robot.get_left_ee_pose())
        off = np.asarray(task.can.get_pose().p[:3]) - ee[:3]
        print(f"[p8]   ee={np.round(ee[:3], 3)} off={np.round(off, 3)}", flush=True)
        if mode in (0, 1):
            # place_actor to the nearer functional point.
            f0 = task.basket.get_functional_point(0)
            f1 = task.basket.get_functional_point(1)
            tp = np.array(f0 if np.linalg.norm(f0[:2]-ee[:2]) < np.linalg.norm(f1[:2]-ee[:2]) else f1)
            tp[3:] = (-1, 0, 0, 0)
            task.move(task.place_actor(task.can, arm_tag=arm_tag, target_pose=tp,
                                       dis=0.02, is_open=False, constrain="free"))
            print(f"[p8] place_actor plan={task.plan_success}", flush=True)
            if not task.plan_success:
                # source fallback branch
                task.plan_success = True
                tp2 = tp.copy()
                tp2[0] += -0.15
                tp2[2] += 0.15
                task.move(task.move_to_pose(arm_tag, tp2))
                task.move(task.move_by_displacement(arm_tag, z=-0.1))
                task.move(task.open_gripper(arm_tag))
                print(f"[p8] fallback plan={task.plan_success} can={np.round(task.can.get_pose().p, 3)}", flush=True)
            else:
                task.move(task.open_gripper(arm_tag, pos=0.7))
                task.move(task.open_gripper(arm_tag, pos=0.7))
        else:
            # hover-drop (source standing can)
            hover = ee.copy()
            basket_p2 = np.asarray(task.basket.get_pose().p)
            hover[:2] = basket_p2[:2] - off[:2]
            hover[2] = max(ee[2] + 0.20, float(task.basket_rim_top) + 0.17)
            task.move(task.move_to_pose(arm_tag, hover))
            print(f"[p8] hover plan={task.plan_success}", flush=True)
            if not task.plan_success:
                return 0
            task.move(task.open_gripper(arm_tag, pos=0.7))
            task.move(task.open_gripper(arm_tag, pos=0.7))
        task.left_return_home()
        task.move(task.open_gripper(arm_tag, pos=1.0))
        can_f = np.asarray(task.can.get_pose().p)
        lo, hi = _actor_aabb(task.can)
        print(f"[p8] final can={np.round(can_f, 3)} aabb=[{lo:.3f},{hi:.3f}]", flush=True)
        print(f"[p8] contact_basket={task.check_actors_contact(task.can_name, task.basket_name)} "
              f"table={task.check_actors_contact(task.can_name, 'table')}", flush=True)
        print(f"[p8] check_success={task.check_success()}", flush=True)
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
