"""Temporary: parameter sweep for the tilted-rack hanging_mug design.

Rolls the rack about x (roll=45 deg) so the tilted pillar catches the mug.
Scans mug x-range and target z to find a band where grasp, place and the
"mug rides the pillar" physics all succeed at >0.2.

Layout: mug x in [mx_lo, mx_hi], y in [-0.01, 0.01]; rack fixed at (rx, ry)
with roll=45; target z = tz.
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


def rx(a):
    return np.asarray([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])


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


def run_seed(task, mug_p, rack_p, qr, tz):
    from envs.utils.action import ArmTag
    from envs.utils.create_actor import create_actor
    task.mug = create_actor(scene=task, pose=mug_p, modelname="039_mug",
                            convex=True, model_id=0)
    task.mug.set_mass(0.05)
    task.rack = create_actor(scene=task, pose=rack_p,
                             modelname="040_rack", is_static=True, convex=True)
    for _ in range(60):
        task.scene.step()
    fp = np.asarray(task.rack.get_functional_point(0))[:3]
    rp = np.asarray(task.rack.get_pose().p)
    mid = (rp + fp) / 2
    arm = ArmTag("left")
    task.plan_success = True
    task.move(task.grasp_actor(task.mug, arm_tag=arm, pre_grasp_dis=0.05))
    if not task.plan_success:
        return "grasp"
    fp_off = np.asarray(task.mug.get_functional_point(0))[:3] - np.asarray(task.mug.get_pose().p)
    tgt = [float(mid[0] - fp_off[0]), float(mid[1] - fp_off[1]), tz]
    task.plan_success = True
    res = task.place_actor(task.mug, arm_tag=arm, target_pose=tgt,
                           pre_dis=0.05, dis=0.0, constrain="free")
    task.move(res)
    if not task.plan_success:
        return "place"
    task.move(task.open_gripper(arm))
    for _ in range(200):
        task.scene.step()
    return "ok" if task.check_success() else "geo"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mx-lo", type=float, default=-0.42)
    parser.add_argument("--mx-hi", type=float, default=-0.38)
    parser.add_argument("--tz", type=float, default=0.77)
    parser.add_argument("--rx", type=float, default=-0.29)
    parser.add_argument("--ry", type=float, default=-0.02)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--roll", type=float, default=45.0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    qr = qmul(rx(np.deg2rad(args.roll)), np.asarray([-0.22, -0.22, 0.67, 0.67]))
    rng = np.random.RandomState(20260809)
    counts = {"grasp": 0, "place": 0, "geo": 0, "ok": 0}
    for seed in range(args.seeds):
        task = build()
        if task is None:
            counts["grasp"] += 1
            continue
        try:
            mx = rng.uniform(args.mx_lo, args.mx_hi)
            my = rng.uniform(-0.01, 0.01)
            mug_p = sapien.Pose([mx, my, 0.741], [0.707, 0.707, 0.0, 0.0])
            rack_p = sapien.Pose([args.rx, args.ry, 0.741], list(qr))
            st = run_seed(task, mug_p, rack_p, qr, args.tz)
            counts[st] += 1
        except Exception:  # noqa: BLE001
            counts["grasp"] += 1
        finally:
            task.close_env()
    rate = counts["ok"] / args.seeds
    flag = "  <-- OK" if rate > 0.2 else ""
    print(f"[S] {args.tag or 'cfg'} mx=[{args.mx_lo:+.2f},{args.mx_hi:+.2f}] "
          f"tz={args.tz:.2f} rack=({args.rx:+.2f},{args.ry:+.2f}) "
          f"ok={counts['ok']} grasp={counts['grasp']} place={counts['place']} "
          f"geo={counts['geo']} rate={rate:.2f}{flag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
