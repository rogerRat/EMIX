"""Temporary: planning-only grasp+lift feasibility map (no execution).

For each candidate spawn point: choose_grasp_pose from home (plan-only), then
plan a pure z-lift from the planned grasp pose.  No arm motion, no state
carry-over, deterministic.
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
        print(f"[P] setup raised {type(exc).__name__}: {exc}", flush=True)
        task.close_env()
        return 1
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)

    q = np.asarray([0.707, 0.707, 0.0, 0.0])
    for py in [0.14, 0.15, 0.16, 0.17, 0.18]:
        row = []
        for px in np.arange(-0.46, -0.29, 0.02):
            act = create_actor(scene=task, pose=sapien.Pose([float(px), py, 0.741], list(q)),
                               modelname="041_shoe", convex=True, model_id=4)
            task.plan_success = True
            r = task.choose_grasp_pose(act, arm_tag=ArmTag("left"), pre_dis=0.1)
            ok_g = bool(r is not None and r[0] is not None and r[1] is not None)
            if ok_g:
                # Two-step: plan to grasp pose, then a pure z-lift from the
                # final grasp joint state.
                grasp_pose = np.asarray(r[1], dtype=float)
                home_q = np.asarray(task.robot.left_entity.get_qpos(), dtype=np.float32)
                res_g = task.robot.left_plan_path(grasp_pose, last_qpos=home_q)
                ok_g2 = bool(isinstance(res_g, dict) and res_g.get("status") == "Success")
                if ok_g2:
                    grasp_q = np.asarray(res_g["position"], dtype=np.float32)[-1]
                    lift_pose = grasp_pose.copy()
                    lift_pose[:3] += np.array([0.0, 0.0, 0.15])
                    res_l = task.robot.left_plan_path(lift_pose, last_qpos=grasp_q)
                    ok_l = bool(isinstance(res_l, dict) and res_l.get("status") == "Success")
                else:
                    ok_l = False
            else:
                ok_l = False
            row.append("O" if ok_g and ok_l else ("g" if ok_g else "."))
            task.scene.remove_actor(act.actor)
        print(f"[P] y={py}: {''.join(row)}", flush=True)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
