"""Temporary: shoe settle stability vs spawn raise for models 0-9."""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

from envs.place_can_basket_left_impl import _actor_aabb
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


def cal_quat_change(p1, p2):
    return float(np.abs(cal_quat_dis(p1.q, p2.q)) * 180)


def main() -> int:
    task, config = build()
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[diag] setup raised {type(exc).__name__}", flush=True)
    for _ in range(200):
        task.scene.step()
    task.scene.update_render()
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)
    task.scene.remove_actor(task.shoe_box.actor)

    table_z = 0.7405
    quats = [(0.707, 0.707, 0.0, 0.0), (0.7071, 0.7071, 0.05, -0.05),
             (0.699, 0.699, 0.08, -0.08)]
    for raise_z in [0.0, 0.02, 0.04, 0.06]:
        for model_id in [0, 2, 4, 6, 8]:
            for qi, quat in enumerate(quats):
                act = create_actor(scene=task, pose=sapien.Pose([-0.40, -0.10, table_z + raise_z], list(quat)),
                                   modelname="041_shoe", convex=True, model_id=model_id)
                for _ in range(2000):
                    task.scene.step()
                task.scene.update_render()
                poses = []
                for _ in range(500):
                    task.scene.step()
                    poses.append(act.get_pose())
                final = poses[-1]
                worst = 0.0
                for p in poses[-200:]:
                    worst = max(worst, cal_quat_change(final, p))
                stable = worst <= 3.0
                pf = np.round(np.asarray(act.get_pose().p), 3)
                print(f"[ss] raise={raise_z} m={model_id} q{qi}={quat[0]:.3f} STABLE={stable} "
                      f"worst={worst:.2f}deg rest=({pf[0]:.2f},{pf[1]:.2f},{pf[2]:.3f})", flush=True)
                task.scene.remove_actor(act.actor)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
