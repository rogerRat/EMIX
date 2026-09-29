"""Temporary: map grasp+lift feasibility over the left workspace.

Single scene, replace shoe actor each point, test grasp + lift z=0.15.
Prints OK set so we can design two non-overlapping spawn bands.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "place_dual_shoes_left"


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
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml", "r", encoding="utf-8") as handle:
        ec = yaml.safe_load(handle)
    config["left_embodiment_config"] = ec
    config["right_embodiment_config"] = ec
    return task, config


def main() -> int:
    from envs.utils.action import ArmTag
    from envs.utils.create_actor import create_actor
    task, config = build()
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[M] setup raised {type(exc).__name__}: {exc}", flush=True)
        task.close_env()
        return 1
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)

    q = np.asarray([0.707, 0.707, 0.0, 0.0])
    results = {}
    for py in [0.11, 0.12, 0.13, 0.14, 0.15]:
        for px in np.arange(-0.46, -0.29, 0.02):
            act = create_actor(scene=task, pose=sapien.Pose([float(px), py, 0.741], list(q)),
                               modelname="041_shoe", convex=True, model_id=4)
            task.plan_success = True
            pp, gp = task.choose_grasp_pose(act, arm_tag=ArmTag("left"), pre_dis=0.1)
            ok_g = pp is not None and gp is not None
            if ok_g:
                task.plan_success = True
                task.move(task.grasp_actor(act, arm_tag=ArmTag("left"), pre_grasp_dis=0.1))
                ok_g = bool(task.plan_success)
            ok_l = False
            if ok_g:
                task.plan_success = True
                task.move_by_displacement(arm_tag=ArmTag("left"), z=0.15)
                ok_l = bool(task.plan_success)
            results[(float(px), py)] = (ok_g, ok_l)
            task.plan_success = True
            task.left_return_home()
            task.scene.remove_actor(act.actor)
    for py in [0.11, 0.12, 0.13, 0.14, 0.15]:
        row = "".join("O" if results[(float(px), py)][1] else
                      ("g" if results[(float(px), py)][0] else ".")
                      for px in np.arange(-0.46, -0.29, 0.02))
        print(f"[M] y={py}: {row}", flush=True)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
