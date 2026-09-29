"""Temporary: which shoe models are stable at seed-0 positions (with box)."""
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
    worst_list = []
    for ai, a in enumerate(actors):
        final = poses[-1][ai]
        worst = 0.0
        for p in poses[-200:]:
            worst = max(worst, cal_change(final, p[ai]))
        worst_list.append(worst)
    return worst_list


def main() -> int:
    task, config = build()
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[m] setup raised {type(exc).__name__}", flush=True)
    for _ in range(200):
        task.scene.step()
    task.scene.update_render()
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)
    task.scene.remove_actor(task.shoe_box.actor)

    seed0 = {
        "left_shoe": ([-0.410, 0.039, 0.741], [0.697, 0.697, -0.117, 0.117]),
        "right_shoe": ([-0.133, 0.031, 0.741], [0.700, 0.700, -0.099, 0.099]),
    }
    box = create_actor(scene=task, pose=sapien.Pose([-0.30, -0.13, 0.741], [0.5, 0.5, -0.5, -0.5]),
                       modelname="007_shoe-box", convex=True, is_static=True)

    for model_id in range(10):
        actors = []
        for name in ("left_shoe", "right_shoe"):
            p, q = seed0[name]
            a = create_actor(scene=task, pose=sapien.Pose(p, q),
                             modelname="041_shoe", convex=True, model_id=model_id)
            actors.append(a)
        wl = settle_stable(task, actors)
        flag = "OK " if all(w <= 3.0 for w in wl) else "BAD"
        print(f"[m] m={model_id}: {flag} L={wl[0]:.2f} R={wl[1]:.2f}", flush=True)
        for a in actors:
            task.scene.remove_actor(a.actor)
    task.scene.remove_actor(box.actor)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
