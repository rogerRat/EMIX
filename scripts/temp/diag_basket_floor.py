"""Temporary diagnostic: map solid floor spots in the basket from collision pieces.
Usage: python scripts/temp/diag_basket_floor.py <seed>
Drops a can from above at a grid of x,y over the basket and records rest z + contact.
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
    from envs.place_can_basket_left_impl import _actor_aabb
    task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
    # Basket collision shape dump
    print(f"[bf] basket_id={task.basket_id} pose={np.round(np.asarray(task.basket.get_pose().p),3)}", flush=True)
    bp = np.asarray(task.basket.get_pose().p)
    # grid over basket footprint: x from bp[0]-0.12 to +0.12, y -0.12..+0.12
    print(f"[bf] grid x=[{bp[0]-0.10},{bp[0]+0.10}] y=[{bp[1]-0.10},{bp[1]+0.10}]", flush=True)
    for dx in np.arange(-0.10, 0.11, 0.025):
        row = []
        for dy in np.arange(-0.10, 0.11, 0.025):
            x = float(bp[0] + dx); y = float(bp[1] + dy)
            # reset can above the spot
            can_pose = task.can.get_pose()
            task.can.actor.set_pose(sapien.Pose([x, y, 0.90], can_pose.q))
            for _ in range(1200):
                task.scene.step()
            task.scene.update_render()
            p = np.asarray(task.can.get_pose().p)
            lo, hi = _actor_aabb(task.can)
            contact = task.check_actors_contact(task.can_name, task.basket_name)
            row.append(f"({dx:+.2f},{dy:+.2f}):z={p[2]:.2f}c={1 if contact else 0}")
        print("[bf] " + " ".join(row), flush=True)
    task.close_env()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
