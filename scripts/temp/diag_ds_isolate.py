"""Temporary: isolate which object makes seed-0 shoes unstable."""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

from envs.utils.create_actor import create_actor
from envs.utils.transforms import cal_quat_dis


def build():
    envs = importlib.import_module("envs.place_dual_shoes_left")
    task = getattr(envs, "place_dual_shoes_left")()
    with open(BENCHMARK / "task_config" / "place_dual_shoes_left.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = "place_dual_shoes_left"
    config["task_config"] = "place_dual_shoes_left"
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


def cal_change(p1, p2):
    return float(np.abs(cal_quat_dis(p1.q, p2.q)) * 180)


def settle_stable(task, actors, steps=2500, check=500):
    for _ in range(steps):
        task.scene.step()
    task.scene.update_render()
    poses = []
    for _ in range(check):
        task.scene.step()
        poses.append([a.get_pose() for a in actors])
    result = []
    for ai, a in enumerate(actors):
        final = poses[-1][ai]
        worst = 0.0
        for p in poses[-200:]:
            worst = max(worst, cal_change(final, p[ai]))
        result.append((a.get_name(), worst <= 3.0, worst))
    return result


def main() -> int:
    task, config = build()
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[iso] setup raised {type(exc).__name__}", flush=True)
    for _ in range(200):
        task.scene.step()
    task.scene.update_render()

    ee_l = np.round(np.asarray(task.robot.get_left_ee_pose()[:3]), 3)
    ee_r = np.round(np.asarray(task.robot.get_right_ee_pose()[:3]), 3)
    print(f"[iso] left_ee={ee_l} right_ee={ee_r}", flush=True)

    seed0 = {
        "shoe_box": ([-0.30, -0.13, 0.741], [0.5, 0.5, -0.5, -0.5]),
        "left_shoe": ([-0.410, 0.039, 0.741], [0.697, 0.697, -0.117, 0.117]),
        "right_shoe": ([-0.133, 0.031, 0.741], [0.700, 0.700, -0.099, 0.099]),
    }

    def spawn(name, model_id):
        p, q = seed0[name]
        return create_actor(scene=task, pose=sapien.Pose(p, q),
                            modelname="007_shoe-box" if name == "shoe_box" else "041_shoe",
                            convex=True, is_static=(name == "shoe_box"), model_id=model_id)

    # remove the real actors
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)
    task.scene.remove_actor(task.shoe_box.actor)

    # test A: left shoe alone (no box)
    a = spawn("left_shoe", 7)
    r = settle_stable(task, [a])
    print(f"[iso] A left alone no box: {r}", flush=True)
    task.scene.remove_actor(a.actor)

    # test B: right shoe alone (no box)
    a = spawn("right_shoe", 7)
    r = settle_stable(task, [a])
    print(f"[iso] B right alone no box: {r}", flush=True)
    task.scene.remove_actor(a.actor)

    # test C: both shoes, no box
    l = spawn("left_shoe", 7)
    rr = spawn("right_shoe", 7)
    r = settle_stable(task, [l, rr])
    print(f"[iso] C both no box: {r}", flush=True)
    task.scene.remove_actor(l.actor)
    task.scene.remove_actor(rr.actor)

    # test D: left shoe alone WITH box
    l = spawn("left_shoe", 7)
    b = spawn("shoe_box", 0)
    r = settle_stable(task, [l, b])
    print(f"[iso] D left + box: {r}", flush=True)
    task.scene.remove_actor(l.actor)
    task.scene.remove_actor(b.actor)

    # test E: right shoe alone WITH box
    rr = spawn("right_shoe", 7)
    b = spawn("shoe_box", 0)
    r = settle_stable(task, [rr, b])
    print(f"[iso] E right + box: {r}", flush=True)
    task.scene.remove_actor(rr.actor)
    task.scene.remove_actor(b.actor)

    # test F: both shoes WITH box (full replicate)
    l = spawn("left_shoe", 7)
    rr = spawn("right_shoe", 7)
    b = spawn("shoe_box", 0)
    r = settle_stable(task, [l, rr, b])
    print(f"[iso] F full replicate: {r}", flush=True)
    task.scene.remove_actor(l.actor)
    task.scene.remove_actor(rr.actor)
    task.scene.remove_actor(b.actor)

    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
