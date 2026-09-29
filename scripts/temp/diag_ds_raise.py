"""Temporary: does raising spawn z stabilize seed-0 shoes at exact poses."""
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
        result.append((a.get_name(), worst <= 3.0, worst, np.round(np.asarray(final.p), 3)))
    return result


def main() -> int:
    task, config = build()
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[raise] setup raised {type(exc).__name__}", flush=True)
    for _ in range(200):
        task.scene.step()
    task.scene.update_render()
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)
    task.scene.remove_actor(task.shoe_box.actor)

    seed0 = {
        "left_shoe": ([-0.410, 0.039], [0.697, 0.697, -0.117, 0.117]),
        "right_shoe": ([-0.133, 0.031], [0.700, 0.700, -0.099, 0.099]),
    }

    for raise_z in [0.0, 0.03, 0.05, 0.08]:
        for model_id in [0, 7]:
            actors = []
            for name in ("left_shoe", "right_shoe"):
                p, q = seed0[name]
                z = 0.741 + raise_z
                a = create_actor(scene=task, pose=sapien.Pose([p[0], p[1], z], q),
                                 modelname="041_shoe", convex=True, model_id=model_id)
                actors.append(a)
            r = settle_stable(task, actors)
            names = ["L", "R"]
            desc = " ".join(f"{names[i]}={'OK' if r[i][1] else 'BAD'}({r[i][2]:.1f}deg @{r[i][3][:2]})" for i in range(2))
            print(f"[raise] raise={raise_z} m={model_id}: {desc}", flush=True)
            for a in actors:
                task.scene.remove_actor(a.actor)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
