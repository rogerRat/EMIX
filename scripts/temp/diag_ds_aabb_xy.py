"""Temporary: full xy AABB of shoe-box and shoe models."""
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
        print(f"[xy] setup raised {type(exc).__name__}", flush=True)
    for _ in range(100):
        task.scene.step()
    task.scene.update_render()
    task.scene.remove_actor(task.left_shoe.actor)
    task.scene.remove_actor(task.right_shoe.actor)
    task.scene.remove_actor(task.shoe_box.actor)

    box = create_actor(scene=task, pose=sapien.Pose([-0.30, -0.13, 0.74], [0.5, 0.5, -0.5, -0.5]),
                       modelname="007_shoe-box", convex=True, is_static=True)
    r = aabb_xyz(box)
    if r:
        lo, hi = r
        print(f"[xy] box: x=[{lo[0]:.3f},{hi[0]:.3f}] y=[{lo[1]:.3f},{hi[1]:.3f}] "
              f"z=[{lo[2]:.3f},{hi[2]:.3f}]", flush=True)
    task.scene.remove_actor(box.actor)

    for model_id in [4]:
        for yaw in [0.0, 0.5]:
            q = np.asarray([0.707, 0.707, 0.0, 0.0])
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            yq = np.array([cy, 0.0, 0.0, sy])
            qq = np.array([
                q[0]*yq[0]-q[1]*yq[1]-q[2]*yq[2]-q[3]*yq[3],
                q[0]*yq[1]+q[1]*yq[0]+q[2]*yq[3]-q[3]*yq[2],
                q[0]*yq[2]-q[1]*yq[3]+q[2]*yq[0]+q[3]*yq[1],
                q[0]*yq[3]+q[1]*yq[2]-q[2]*yq[1]+q[3]*yq[0],
            ])
            qq = qq / np.linalg.norm(qq)
            act = create_actor(scene=task, pose=sapien.Pose([-0.30, 0.0, 0.741], list(qq)),
                               modelname="041_shoe", convex=True, model_id=model_id)
            r = aabb_xyz(act)
            if r:
                lo, hi = r
                print(f"[xy] shoe m={model_id} yaw={yaw}: x=[{lo[0]:.3f},{hi[0]:.3f}] "
                      f"y=[{lo[1]:.3f},{hi[1]:.3f}] z=[{lo[2]:.3f},{hi[2]:.3f}]", flush=True)
            task.scene.remove_actor(act.actor)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
