"""Temporary: validate box quat stand90 - AABB, fp, shoe settle."""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

from envs.utils.create_actor import create_actor


def aabb_xyz(actor):
    entity = actor.actor
    comp = None
    for name in ("PhysxRigidStaticComponent", "PhysxRigidDynamicComponent"):
        comp = entity.find_component_by_type(getattr(sapien.physx, name))
        if comp is not None:
            break
    if comp is None:
        return None
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
    except Exception as exc:  # noqa: BLE001
        print(f"[s] setup raised {type(exc).__name__}: {exc}", flush=True)
    for _ in range(100):
        task.scene.step()
    task.scene.update_render()
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)
    task.scene.remove_actor(task.shoe_box.actor)

    for name, q in (("src_120", [0.5, 0.5, -0.5, -0.5]),
                    ("stand90", [0.707, 0.707, 0.0, 0.0])):
        qq = np.asarray(q, dtype=float)
        qq = qq / np.linalg.norm(qq)
        box = create_actor(scene=task, pose=sapien.Pose([-0.20, -0.13, 0.74], list(qq)),
                           modelname="007_shoe-box", convex=True, is_static=True)
        r = aabb_xyz(box)
        if r:
            lo, hi = r
            print(f"[s] {name}: box x=[{lo[0]:.3f},{hi[0]:.3f}] y=[{lo[1]:.3f},{hi[1]:.3f}] "
                  f"z=[{lo[2]:.3f},{hi[2]:.3f}]", flush=True)
        fp0 = box.get_functional_point(0)
        fp1 = box.get_functional_point(1)
        print(f"[s] {name}: fp0={np.round(np.asarray(fp0),3)}", flush=True)
        print(f"[s] {name}: fp1={np.round(np.asarray(fp1),3)}", flush=True)
        task.scene.remove_actor(box.actor)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
