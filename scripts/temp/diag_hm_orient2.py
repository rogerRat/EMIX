"""Temporary: scan mug tip orientation (Ry then Rz) for a reachable LEFT grasp
with a SMALL EE-mug vertical gap.

gap_z = EE_z - mug_base_z after grasp.  A small gap lets the EE lift the mug
high enough to reach the rack top (~0.98).  The source x+90 orientation gives
gap_z=+0.23 (EE above mug); rotating the mug about y lowers the contact point
toward the side (gap_z -> ~0).

For each orientation we also record the mug's functional-point world offset so
we can reason about where the mug lands after a placement.
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
MUG_POS = [-0.42, 0.00, 0.741]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def ry(a):
    return np.asarray([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])


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
    from envs.utils.action import ArmTag
    from envs.utils.create_actor import create_actor
    for alpha_deg in range(0, 181, 30):
        for beta_deg in range(0, 360, 60):
            q = qmul(rz(np.deg2rad(beta_deg)), ry(np.deg2rad(alpha_deg)))
            task = build()
            try:
                task.mug = create_actor(scene=task, pose=sapien.Pose(MUG_POS, list(q)),
                                        modelname="039_mug", convex=True, model_id=0)
                task.mug.set_mass(0.05)
                for _ in range(60):
                    task.scene.step()
                mp = np.asarray(task.mug.get_pose().p)
                if mp[2] < 0.70 or abs(mp[0] - MUG_POS[0]) > 0.03 or abs(mp[1]) > 0.03:
                    print(f"[O] a={alpha_deg:3d} b={beta_deg:3d} toppled mp=({mp[0]:+.2f},{mp[1]:+.2f},{mp[2]:+.2f})", flush=True)
                    continue
                mq = np.asarray(task.mug.get_pose().q)
                cp = np.asarray(task.mug.get_contact_point(0, "pose"))[:3]
                cp_off = cp - mp
                arm = ArmTag("left")
                task.plan_success = True
                task.move(task.grasp_actor(task.mug, arm_tag=arm, pre_grasp_dis=0.05))
                g = task.plan_success
                if not g:
                    print(f"[O] a={alpha_deg:3d} b={beta_deg:3d} grasp=.", flush=True)
                    continue
                ee = np.asarray(task.robot.get_left_ee_pose())
                gap = float(ee[2] - mp[2])
                fp = np.asarray(task.mug.get_functional_point(0))[:3] - mp
                task.plan_success = True
                task.move_by_displacement(arm_tag=arm, z=0.05)
                l05 = task.plan_success
                print(f"[O] a={alpha_deg:3d} b={beta_deg:3d} grasp=O gap={gap:+.2f} "
                      f"ee_z={ee[2]:.2f} cp_off=({cp_off[0]:+.2f},{cp_off[1]:+.2f},{cp_off[2]:+.2f}) "
                      f"fp_off=({fp[0]:+.2f},{fp[1]:+.2f},{fp[2]:+.2f}) lift05={'O' if l05 else '.'}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[O] a={alpha_deg:3d} b={beta_deg:3d} EXC {type(exc).__name__}: {exc}", flush=True)
            finally:
                task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
