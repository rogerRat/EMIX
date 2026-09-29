"""Temporary diagnostic: solid drop spots per basket model (full reset per spot).
Usage: python scripts/temp/diag_drop_spots.py
Drops a lying can from z=0.92 at each relative spot with pose+quat+velocity
reset.  SOLID means rest z>0.80, basket contact, no table contact.
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

def reset_can(task, x, y, z=0.92):
    pose = task.can.get_pose()
    entity = task.can.actor
    task.can.actor.set_pose(sapien.Pose([x, y, z], [1.0, 0.0, 0.0, 0.0]))
    rb = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    rb.set_linear_velocity([0, 0, 0])
    rb.set_angular_velocity([0, 0, 0])

def main() -> int:
    task, config = build("place_can_basket_left")
    # fine grid over the promising +x region
    spots = []
    for dx in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
        for dy in [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04]:
            spots.append((dx, dy))
    results = {}
    for bid in (0, 1):
        results[bid] = {}
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        # force basket model
        bp = np.asarray(task.basket.get_pose().p)
        for dx, dy in spots:
            reset_can(task, bp[0]+dx, bp[1]+dy)
            for _ in range(1500):
                task.scene.step()
            task.scene.update_render()
            p = np.asarray(task.can.get_pose().p)
            cb = task.check_actors_contact(task.can_name, task.basket_name)
            ct = task.check_actors_contact(task.can_name, "table")
            results[bid][(dx,dy)] = (round(float(p[2]),2), bool(cb), bool(ct))
        task.close_env()
    for bid in results:
        print(f"[ds] basket_model={bid}", flush=True)
        solid = []
        for dx in sorted({s[0] for s in results[bid]}):
            row = []
            for dy in sorted({s[1] for s in results[bid]}, reverse=True):
                z, cb, ct = results[bid][(dx,dy)]
                good = (z > 0.80 and cb and not ct)
                if good: solid.append((dx,dy,z))
                row.append(f"{dy:+.2f}:{z:.2f}{'/'+('B' if cb else '-')+'/'+('T' if ct else '-')}" + ("*" if good else ""))
            print(f"[ds] dx={dx:+.2f}  " + "  ".join(row), flush=True)
        print(f"[ds] SOLID: {solid}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
