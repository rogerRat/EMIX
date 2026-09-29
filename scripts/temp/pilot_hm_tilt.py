"""Temporary: pilot the tilted-rack hanging_mug design under random layout.

Rack is rolled about x by 45 deg so the hook (fp0) world z ~ 0.85-0.87 sits in
the left-arm reachable band.  Mug keeps the source x+90 orientation.  Sequence:
grasp -> free place (mug base toward rack-middle xy, z ~ 0.77) -> open -> settle.

Reports success rate over seeds; used to decide whether to bake this into the
impl (roll fixed in sample_layout, rack x/y band moved to the reachable zone).
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
TZ = 0.75
ROLL = 45.0
RACK_X = (-0.30, -0.26)
RACK_Y = (-0.04, 0.00)
MUG_X = (-0.42, -0.34)
MUG_Y = (-0.02, 0.02)


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


def main() -> int:
    import argparse
    from envs.utils.action import ArmTag
    from envs.utils.create_actor import create_actor
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--tz", type=float, default=TZ)
    parser.add_argument("--roll", type=float, default=ROLL)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    qr = qmul(rx(np.deg2rad(args.roll)), np.asarray([-0.22, -0.22, 0.67, 0.67]))
    rng = np.random.RandomState(20260809)
    rows = []
    for seed in range(args.seeds):
        task = build()
        if task is None:
            rows.append((seed, "buildfail", 0.0, "", ""))
            continue
        try:
            mx = rng.uniform(*MUG_X)
            my = rng.uniform(*MUG_Y)
            rx_ = rng.uniform(*RACK_X)
            ry_ = rng.uniform(*RACK_Y)
            task.mug = create_actor(scene=task, pose=sapien.Pose([mx, my, 0.741],
                                                                [0.707, 0.707, 0.0, 0.0]),
                                    modelname="039_mug", convex=True, model_id=0)
            task.mug.set_mass(0.05)
            task.rack = create_actor(scene=task, pose=sapien.Pose([rx_, ry_, 0.741], list(qr)),
                                     modelname="040_rack", is_static=True, convex=True)
            for _ in range(60):
                task.scene.step()
            fp = np.asarray(task.rack.get_functional_point(0))[:3]
            rp = np.asarray(task.rack.get_pose().p)
            mid = (rp + fp) / 2
            arm = ArmTag("left")
            task.plan_success = True
            task.move(task.grasp_actor(task.mug, arm_tag=arm, pre_grasp_dis=0.05))
            g_ok = task.plan_success
            if not g_ok:
                rows.append((seed, "grasp", 0.0,
                             f"mug=({mx:+.2f},{my:+.2f}) rack=({rx_:+.2f},{ry_:+.2f})",
                             "grasp fail"))
                continue
            fp_off = np.asarray(task.mug.get_functional_point(0))[:3] - np.asarray(task.mug.get_pose().p)
            tgt = [float(mid[0] - fp_off[0]), float(mid[1] - fp_off[1]), args.tz]
            task.plan_success = True
            res = task.place_actor(task.mug, arm_tag=arm, target_pose=tgt,
                                   pre_dis=0.05, dis=0.0, constrain="free")
            task.move(res)
            pl_ok = task.plan_success
            if not pl_ok:
                rows.append((seed, "place", 0.0,
                             f"mug=({mx:+.2f},{my:+.2f}) rack=({rx_:+.2f},{ry_:+.2f})",
                             f"place fail mid=({mid[0]:+.2f},{mid[1]:+.2f},{mid[2]:+.2f})"))
                continue
            task.move(task.open_gripper(arm))
            for _ in range(200):
                task.scene.step()
            ok = task.check_success()
            mfp = np.asarray(task.mug.get_functional_point(0))[:3]
            mp = np.asarray(task.mug.get_pose().p)
            rows.append((seed, "ok" if ok else "geo", float(ok),
                         f"mug=({mx:+.2f},{my:+.2f}) rack=({rx_:+.2f},{ry_:+.2f})",
                         f"mug=({mp[0]:+.2f},{mp[1]:+.2f},{mp[2]:+.2f}) "
                         f"fp=({mfp[0]:+.2f},{mfp[1]:+.2f},{mfp[2]:+.2f}) mid=({mid[0]:+.2f},{mid[1]:+.2f})"))
        except Exception as exc:  # noqa: BLE001
            rows.append((seed, "exc", 0.0, "", f"{type(exc).__name__}: {exc}"))
        finally:
            task.close_env()

    n_ok = sum(1 for _, _, s, _, _ in rows if s > 0)
    for seed, stage, s, cfg, note in rows:
        print(f"[P] seed={seed:2d} {stage:9s} {'OK' if s > 0 else '--'} {cfg} {note}", flush=True)
    print(f"[P] RESULT {n_ok}/{args.seeds} = {n_ok / max(args.seeds, 1):.2f} "
          f"(tz={args.tz} roll={args.roll})", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(f"seeds={args.seeds} ok={n_ok} rate={n_ok / max(args.seeds, 1):.2f}\n")
            for row in rows:
                handle.write(f"{row}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
