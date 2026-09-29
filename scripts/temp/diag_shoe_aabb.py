"""Temporary: measure 041_shoe hull AABB at spawn quats for models 0-9."""
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


def main() -> int:
    task, config = build()
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        for _ in range(200):
            task.scene.step()
        task.scene.update_render()
    except Exception as exc:  # noqa: BLE001
        print(f"[diag] setup raised {type(exc).__name__}: {exc}", flush=True)
    task.scene.update_render()
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)
    task.scene.remove_actor(task.shoe_box.actor)

    table_z = 0.74
    for model_id in range(10):
        for quat, label in [((0.707, 0.707, 0.0, 0.0), "lying"),
                            ((0.5, 0.5, -0.5, -0.5), "boxquat")]:
            act = create_actor(scene=task, pose=sapien.Pose([-0.30, 0.0, table_z], list(quat)),
                               modelname="041_shoe", convex=True, model_id=model_id)
            lo, hi = _actor_aabb(act)
            h = float(hi - lo)
            print(f"[shoe] m={model_id} {label}: z-lo={lo:.4f} z-hi={hi:.4f} h={h:.4f}", flush=True)
            task.scene.remove_actor(act.actor)

    # shoe box AABB
    box = create_actor(scene=task, pose=sapien.Pose([-0.30, -0.13, 0.74], [0.5, 0.5, -0.5, -0.5]),
                       modelname="007_shoe-box", convex=True, is_static=True)
    blo, bhi = _actor_aabb(box)
    print(f"[box] z-lo={blo:.4f} z-hi={bhi:.4f} h={bhi-blo:.4f}", flush=True)
    for i in range(2):
        fp = box.get_functional_point(i)
        print(f"[box] fp{i}={np.round(np.asarray(fp), 4)}", flush=True)
    task.scene.remove_actor(box.actor)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
