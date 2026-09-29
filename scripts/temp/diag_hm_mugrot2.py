"""Temporary: scan mug x-rotation, full scene per iteration.

Source task spawns mug with rotate_lim=[0,1.57] about x.  Grasp is adaptive
so the mug's world orientation after grasp tracks the spawn rotation, and the
align quat (which must align the mug x-axis to the rack fp0 z-axis) varies
continuously.  Find a rotx where the align place target is reachable by the
left arm while the rack stays hook-up (fp0 z high).  Fresh scene per
iteration for a clean state.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "hanging_mug_left"
RACK_Q0 = np.asarray([-0.22, -0.22, 0.67, 0.67])
MUG_POS = np.asarray([-0.35, 0.02, 0.741])
BASE_Q = np.asarray([0.707, 0.707, 0.0, 0.0])


def build(seed):
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
    try:
        task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        task.close_env()
        return None
    task.scene.remove_actor(task.mug.actor)
    task.scene.remove_actor(task.rack.actor)
    return task


def quat_mult(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_rotx(theta):
    return np.asarray([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])


def main() -> int:
    from envs.utils.action import ArmTag
    from envs.utils.create_actor import create_actor
    for theta in np.arange(0.0, 1.57 + 0.001, 0.157):
        task = build(0)
        if task is None:
            print(f"[R] rotx={theta:4.2f} build failed", flush=True)
            continue
        rack_pose = sapien.Pose([-0.20, 0.05, 0.741], list(RACK_Q0))
        task.rack = create_actor(scene=task, pose=rack_pose,
                                 modelname="040_rack", is_static=True, convex=True)
        fp = np.asarray(task.rack.get_functional_point(0))
        mq = quat_mult(BASE_Q, quat_rotx(float(theta)))
        mug_p = sapien.Pose(list(MUG_POS), list(mq))
        task.mug = create_actor(scene=task, pose=mug_p, modelname="039_mug",
                                convex=True, model_id=0)
        task.mug.set_mass(0.05)
        for _ in range(60):
            task.scene.step()

        arm = ArmTag("left")
        task.plan_success = True
        task.move(task.grasp_actor(task.mug, arm_tag=arm, pre_grasp_dis=0.05))
        grasp_ok = task.plan_success
        if not grasp_ok:
            print(f"[R] rotx={theta:4.2f} grasp=.", flush=True)
            task.close_env()
            continue
        task.plan_success = True
        task.move_by_displacement(arm_tag=arm, z=0.08)
        lift_ok = task.plan_success
        fp_pose = task.mug.get_functional_point(0, "pose")
        fp_pos = np.asarray(fp_pose.p)
        mx = np.asarray(fp_pose.to_transformation_matrix()[:3, 0])
        align_ok, tgt = False, None
        if lift_ok:
            task.plan_success = True
            _, actions = task.place_actor(task.mug, arm_tag=arm, target_pose=fp,
                                          functional_point_id=0, constrain="align",
                                          pre_dis=0.05, dis=-0.05, pre_dis_axis="fp")
            if len(actions) >= 2:
                tgt = np.asarray(actions[1].target_pose)
                task.plan_success = True
                r1 = task.robot.left_plan_path(list(actions[0].target_pose))
                task.plan_success = True
                r2 = task.robot.left_plan_path(list(actions[1].target_pose))
                align_ok = (r1 is not None and r1.get("status") == "Success"
                            and r2 is not None and r2.get("status") == "Success")
        if tgt is not None:
            print(f"[R] rotx={theta:4.2f} grasp={'O' if grasp_ok else '.'} "
                  f"lift={'O' if lift_ok else '.'} align={'O' if align_ok else '.'} "
                  f"mugfp=({fp_pos[0]:+.2f},{fp_pos[1]:+.2f},{fp_pos[2]:+.2f}) "
                  f"mx=({mx[0]:+.2f},{mx[1]:+.2f},{mx[2]:+.2f}) "
                  f"tgt=({tgt[0]:+.2f},{tgt[1]:+.2f},{tgt[2]:+.2f}) "
                  f"q=({tgt[3]:+.2f},{tgt[4]:+.2f},{tgt[5]:+.2f},{tgt[6]:+.2f})", flush=True)
        else:
            print(f"[R] rotx={theta:4.2f} grasp={'O' if grasp_ok else '.'} "
                  f"lift={'O' if lift_ok else '.'} align=.", flush=True)
        task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
