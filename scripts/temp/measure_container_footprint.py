"""Measure container-collision XY footprints for the left-arm container tasks.

Loads each container actor at its canonical quat and reports the world-frame
half-extents (relative to the actor origin).  Used to size footprint-aware
rejection for the union-range randomization iteration.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import sapien
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

from envs.utils.create_actor import create_actor  # noqa: E402


def aabb_xyz(actor):
    entity = actor.actor
    comp = None
    for name in ("PhysxRigidStaticComponent", "PhysxRigidDynamicComponent"):
        comp = entity.find_component_by_type(getattr(sapien.physx, name))
        if comp is not None:
            break
    if comp is None:
        return None, None
    world = actor.get_pose().to_transformation_matrix()
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for shape in comp.collision_shapes:
        local = shape.get_local_pose().to_transformation_matrix()
        verts = np.asarray(shape.get_vertices(), dtype=np.float64)
        verts = verts * np.asarray(shape.get_scale(), dtype=np.float64)
        verts = verts @ local[:3, :3].T + local[:3, 3]
        verts = verts @ world[:3, :3].T + world[:3, 3]
        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))
    return lo, hi


def build(task_name: str):
    try:
        envs = importlib.import_module(f"envs.left.{task_name}")
        task = getattr(envs, task_name)()
    except ModuleNotFoundError:
        envs = importlib.import_module(f"envs.{task_name}")
        task = getattr(envs, task_name)()
    config_path = BENCHMARK / "task_config" / "left" / f"{task_name}.yml"
    if not config_path.is_file():
        config_path = BENCHMARK / "task_config" / f"{task_name}.yml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = task_name
    config["task_config"] = task_name
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


# (task wrapper, actor name, modelname, model_id, scale, is_static, base_quat, raised_z)
FOODS = [
    ("place_burger_fries_left", "hamburg", "006_hamburg", 0, None, False,
     (0.5, 0.5, 0.5, 0.5), 0.04),
    ("place_burger_fries_left", "frenchfries", "005_french-fries", 0, None, False,
     (1.0, 0.0, 0.0, 0.0), 0.04),
    ("place_bread_basket_left", "bread", "075_bread", 0, None, False,
     (0.707, 0.707, 0.0, 0.0), 0.0),
    ("place_cans_plasticbox_left", "can", "071_can", 0, None, False,
     (0.5, 0.5, 0.5, 0.5), 0.0),
    ("place_dual_shoes_left", "shoe", "041_shoe", 4, None, False,
     (0.707, 0.707, 0.0, 0.0), 0.0),
]

CONTAINERS = [
    ("place_bread_basket_left", "breadbasket", "076_breadbasket", 0, None, True,
     (0.5, 0.5, 0.5, 0.5), 0.0),
    ("place_bread_skillet_left", "skillet", "106_skillet", 0, None, True,
     (0.0, 0.0, 0.707, 0.707), 0.025),
    ("place_burger_fries_left", "tray", "008_tray", 0, (2.0, 2.0, 2.0), True,
     (0.706527, 0.706483, -0.0291356, -0.0291767), 0.0),
    ("place_can_basket_left", "basket", "110_basket", 0, None, True,
     (0.5, 0.5, 0.5, 0.5), 0.0),
    ("place_cans_plasticbox_left", "plasticbox", "062_plasticbox", 3, None, True,
     (0.5, 0.5, 0.5, 0.5), 0.0),
    ("place_dual_shoes_left", "shoe_box", "007_shoe-box", 0, None, True,
     (0.707, 0.707, 0.0, 0.0), 0.0),
]


def main() -> int:
    for wrapper, name, model, model_id, scale, static, quat, raised in CONTAINERS + FOODS:
        task, config = build(wrapper)
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        try:
            actor = create_actor(
                scene=task, pose=sapien.Pose(np.array([0.0, 0.0, 0.74 + raised]), quat),
                modelname=model, model_id=model_id, convex=True, is_static=static,
                scale=scale,
            )
            lo, hi = aabb_xyz(actor)
            if lo is None:
                print(f"{name:12s} no collision component")
                continue
            half = (hi - lo) / 2.0
            centre = (hi + lo) / 2.0
            print(f"{name:12s} {model:18s} quat={[round(float(q),3) for q in quat]} "
                  f"half=({half[0]:.3f},{half[1]:.3f}) zlo={lo[2]:.3f} zhi={hi[2]:.3f} "
                  f"origin_xy=({centre[0]:+.3f},{centre[1]:+.3f})")
        finally:
            try:
                task.close_env()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
