"""Temporary diagnostic: dump basket collision pieces to find floor plates.
Usage: python scripts/temp/diag_basket_pieces.py <seed>
For each collision shape: center, world AABB z, and the fraction of its top
area that is 'flat' (near-horizontal).  Floor plates show z extent < 0.05.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

def build(task_name: str):
    envs = importlib.import_module(f"envs.{task_name}")
    task = getattr(envs, task_name)()
    with open(BENCHMARK / "task_config" / f"{task_name}.yml", "r", encoding="utf-8") as handle:
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

def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task, config = build("place_can_basket_left")
    task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
    bp = np.asarray(task.basket.get_pose().p)
    print(f"[bp] seed={seed} basket_model={task.basket_id} pose={np.round(bp,3)}", flush=True)
    entity = task.basket.actor
    static = entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
    world = task.basket.get_pose().to_transformation_matrix()
    pieces = []
    for i, shape in enumerate(static.collision_shapes):
        local = shape.get_local_pose().to_transformation_matrix()
        scale = np.asarray(shape.get_scale(), dtype=np.float64)
        verts = np.asarray(shape.get_vertices(), dtype=np.float64) * scale
        verts_world = verts @ local[:3,:3].T + local[:3,3]
        verts_world = verts_world @ world[:3,:3].T + world[:3,3]
        lo = verts_world.min(axis=0); hi = verts_world.max(axis=0)
        cz = float((lo[2]+hi[2])/2)
        dz = float(hi[2]-lo[2])
        cx = float((lo[0]+hi[0])/2); cy = float((lo[1]+hi[1])/2)
        # flat score: top vertices close to same z
        top = verts_world[verts_world[:,2] > lo[2]+0.5*dz]
        flat = float(top[:,2].std()) if len(top) > 3 else 99.0
        pieces.append((i, cx, cy, cz, dz, flat, len(verts_world)))
    # print sorted by z
    for i, cx, cy, cz, dz, flat, n in sorted(pieces, key=lambda p: p[3]):
        tag = "FLOOR?" if (dz < 0.08 and 0.80 < cz < 0.90) else ("LOW?" if cz < 0.76 else "")
        print(f"[bp] piece {i:2d} center=({cx:+.3f},{cy:+.3f},{cz:+.3f}) zspan={dz:.3f} flat_std={flat:.3f} nverts={n} {tag}", flush=True)
    task.close_env()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
