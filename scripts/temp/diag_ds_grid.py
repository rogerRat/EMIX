"""Temporary: robustness grid for shoe models 0/4 across manifest bands."""
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


def settle_worst(task, actor, steps=2500, check=500):
    for _ in range(steps):
        task.scene.step()
    task.scene.update_render()
    poses = []
    for _ in range(check):
        task.scene.step()
        poses.append(actor.get_pose())
    final = poses[-1]
    worst = 0.0
    for p in poses[-200:]:
        worst = max(worst, cal_change(final, p))
    return worst, final


def apply_yaw(quat, yaw):
    q = np.asarray(quat, dtype=float)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    yq = np.array([cy, 0.0, 0.0, sy])
    qq = np.array([
        q[0] * yq[0] - q[1] * yq[1] - q[2] * yq[2] - q[3] * yq[3],
        q[0] * yq[1] + q[1] * yq[0] + q[2] * yq[3] - q[3] * yq[2],
        q[0] * yq[2] - q[1] * yq[3] + q[2] * yq[0] + q[3] * yq[1],
        q[0] * yq[3] + q[1] * yq[2] - q[2] * yq[1] + q[3] * yq[0],
    ])
    n = np.linalg.norm(qq)
    return qq / n


def main() -> int:
    task, config = build()
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[g] setup raised {type(exc).__name__}", flush=True)
    for _ in range(200):
        task.scene.step()
    task.scene.update_render()
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)
    task.scene.remove_actor(task.shoe_box.actor)

    left_grid = [(-0.44, -0.08), (-0.44, 0.04), (-0.38, -0.08), (-0.38, 0.04), (-0.41, 0.0)]
    right_grid = [(-0.17, -0.08), (-0.17, 0.04), (-0.13, -0.08), (-0.13, 0.04), (-0.15, 0.0)]

    for model_id in [0, 4]:
        for band, grid in (("L", left_grid), ("R", right_grid)):
            for yaw in [0.0, 0.25, 0.5]:
                fails = []
                for (px, py) in grid:
                    q = apply_yaw((0.707, 0.707, 0.0, 0.0), yaw)
                    a = create_actor(scene=task, pose=sapien.Pose([px, py, 0.741], list(q)),
                                     modelname="041_shoe", convex=True, model_id=model_id)
                    worst, final = settle_worst(task, a)
                    if worst > 3.0:
                        fails.append((px, py, round(worst, 1)))
                    task.scene.remove_actor(a.actor)
                print(f"[g] m={model_id} band={band} yaw={yaw}: fails={fails if fails else 'NONE'}", flush=True)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
