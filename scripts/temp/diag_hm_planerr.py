"""Temporary: inspect left_plan_path failure detail after grasping mug."""
from __future__ import annotations
import importlib, json, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "hanging_mug_left"
RACK_Q0 = np.asarray([-0.22, -0.22, 0.67, 0.67])
MUG_POS = np.asarray([-0.35, 0.02, 0.741])


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


def quat_yaw(q, yaw):
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qy = np.asarray([0.0, 0.0, sy, cy])
    qw, qx, qy_, qz = q
    rw, rx, ry, rz = qy
    return np.asarray([
        rw * qw - rx * qx - ry * qy_ - rz * qz,
        rw * qx + rx * qw + ry * qz - rz * qy_,
        rw * qy_ - rx * qz + ry * qw + rz * qx,
        rw * qz + rx * qy_ - ry * qx + rz * qw,
    ])


def main() -> int:
    from envs.utils.action import ArmTag
    from envs.utils.create_actor import create_actor
    task, config = build()
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[E] setup raised {type(exc).__name__}: {exc}", flush=True)
        task.close_env()
        return 1
    task.scene.remove_actor(task.mug.actor)
    task.scene.remove_actor(task.rack.actor)

    mug_p = sapien.Pose(list(MUG_POS), [0.707, 0.707, 0.0, 0.0])
    task.mug = create_actor(scene=task, pose=mug_p, modelname="039_mug",
                            convex=True, model_id=0)
    task.mug.set_mass(0.05)
    for _ in range(60):
        task.scene.step()

    arm = ArmTag("left")
    task.plan_success = True
    task.move(task.grasp_actor(task.mug, arm_tag=arm, pre_grasp_dis=0.05))
    task.plan_success = True
    task.move_by_displacement(arm_tag=arm, z=0.08)
    print(f"[E] EE now = {np.round(np.asarray(task.robot.get_left_ee_pose()),3)}", flush=True)

    rq = quat_yaw(RACK_Q0, 0.0)
    rack_pose = sapien.Pose([-0.30, 0.00, 0.741], list(rq))
    task.rack = create_actor(scene=task, pose=rack_pose,
                             modelname="040_rack", is_static=True, convex=True)
    fp = np.asarray(task.rack.get_functional_point(0))
    task.plan_success = True
    _, actions = task.place_actor(task.mug, arm_tag=arm, target_pose=fp,
                                  functional_point_id=0, constrain="align",
                                  pre_dis=0.05, dis=-0.05, pre_dis_axis="fp")
    tgt = np.asarray(actions[1].target_pose)
    print(f"[E] target = {np.round(tgt,3)}", flush=True)
    task.plan_success = True
    res = task.robot.left_plan_path(list(tgt))
    if res is None:
        print("[E] left_plan_path returned None", flush=True)
    else:
        for k, v in res.items():
            if k == "interpolated_traj":
                continue
            print(f"[E]   {k} = {v}", flush=True)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
