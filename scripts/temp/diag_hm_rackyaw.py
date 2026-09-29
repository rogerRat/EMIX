"""Temporary: is the rack hook height the real blocker?

Sweep the rack yaw (z-rotation added onto the source base_quat) and record the
world hook height (functional point z), the rack top z, and where the rack
middle lands.  We want a hook z in the reachable window [0.80, 0.87] with a
left-arm-reachable rack x band so a top-down free placement could reach it.
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


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rz(a):
    return np.asarray([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


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
    task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    task.scene.remove_actor(task.mug.actor)
    task.scene.remove_actor(task.rack.actor)
    return task


def main() -> int:
    from envs.utils.create_actor import create_actor
    base_q = np.asarray([-0.22, -0.22, 0.67, 0.67])
    for rx in [-0.30, -0.34, -0.26]:
        for ry in [-0.04, 0.00]:
            for yaw_deg in range(0, 360, 20):
                q = qmul(rz(np.deg2rad(yaw_deg)), base_q)
                task = build()
                try:
                    rack_pose = sapien.Pose([rx, ry, 0.741], list(q))
                    task.rack = create_actor(scene=task, pose=rack_pose,
                                             modelname="040_rack", is_static=True, convex=True)
                    for _ in range(10):
                        task.scene.step()
                    fp = np.asarray(task.rack.get_functional_point(0))[:3]
                    rp = np.asarray(task.rack.get_pose().p)
                    mid = (rp + fp) / 2
                    # reachable window: mid z within [0.80, 0.87], mid xy near the
                    # grasp-posture reachable band x in [-0.42,-0.30], |y|<=0.05
                    if 0.80 <= mid[2] <= 0.87 and -0.42 <= mid[0] <= -0.30 and abs(mid[1]) <= 0.05:
                        print(f"[R] rack=({rx:+.2f},{ry:+.2f}) yaw={yaw_deg:3d} "
                              f"fp0=({fp[0]:+.2f},{fp[1]:+.2f},{fp[2]:+.2f}) "
                              f"mid=({mid[0]:+.2f},{mid[1]:+.2f},{mid[2]:+.2f}) "
                              f"q=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}) WINDOW", flush=True)
                    elif abs(mid[2] - 0.86) <= 0.03 and abs(mid[0]) <= 0.02:
                        print(f"[R] rack=({rx:+.2f},{ry:+.2f}) yaw={yaw_deg:3d} "
                              f"fp0=({fp[0]:+.2f},{fp[1]:+.2f},{fp[2]:+.2f}) "
                              f"mid=({mid[0]:+.2f},{mid[1]:+.2f},{mid[2]:+.2f}) NEAR-mid-x", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[R] rack=({rx:+.2f},{ry:+.2f}) yaw={yaw_deg} EXC {type(exc).__name__}", flush=True)
                finally:
                    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
