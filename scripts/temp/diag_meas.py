"""Temporary: AABB of hamburg/fries models in candidate spawn quats."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

import importlib
from envs.place_can_basket_left_impl import _actor_aabb
from envs.utils.create_actor import create_actor

def main() -> int:
    task = getattr(importlib.import_module("envs.place_burger_fries_left"), "place_burger_fries_left")()
    with open(BENCHMARK / "task_config" / "place_burger_fries_left.yml", "r", encoding="utf-8") as h:
        config = yaml.safe_load(h)
    config["task_name"] = "place_burger_fries_left"
    config["task_config"] = "place_burger_fries_left"
    config["eval_mode"] = True
    config["left_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["right_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["dual_arm_embodied"] = False
    config["embodiment_dis"] = 0.60
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml", "r", encoding="utf-8") as h:
        ec = yaml.safe_load(h)
    config["left_embodiment_config"] = ec
    config["right_embodiment_config"] = ec
    task.setup_demo(now_ep_num=9, seed=9, is_test=True, **config)
    for _ in range(500):
        task.scene.step()
    task.scene.update_render()
    table_z = 0.741
    for name, n_models in (("006_hamburg", 6), ("005_french-fries", 2)):
        for model_id in range(n_models):
            for quat, qname in (((1.0, 0.0, 0.0, 0.0), "identity"), ((0.5, 0.5, 0.5, 0.5), "0.5q")):
                act = create_actor(scene=task, pose=sapien.Pose([-0.40, 0.0, table_z], list(quat)),
                                   modelname=name, convex=True, model_id=model_id)
                lo, hi = _actor_aabb(act)
                w = np.max(np.asarray(act.actor.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent).collision_shapes[0].get_scale()))
                print(f"[m] {name} m={model_id} {qname}: bot={lo:.3f} top={hi:.3f} h={hi-lo:.3f} (table {table_z})", flush=True)
                task.scene.remove_actor(act.actor)
    task.close_env()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
