"""Temporary: settle stability of candidate spawn configs.

Candidate A: hamburg flat (0.5q) models 0,1 across hamburg band.
Candidate B: french-fries lying (identity) across fries band incl. y=-0.21/-0.20.
Measure settle displacement across the full manifest bands for every model.
"""
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
    envs = importlib.import_module("envs.place_burger_fries_left")
    task = getattr(envs, "place_burger_fries_left")()
    with open(BENCHMARK / "task_config" / "place_burger_fries_left.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = "place_burger_fries_left"
    config["task_config"] = "place_burger_fries_left"
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
    table_z = 0.741
    task, config = build()
    task.setup_demo(now_ep_num=9, seed=9, is_test=True, **config)
    for _ in range(500):
        task.scene.step()
    task.scene.update_render()
    task.scene.remove_actor(task.hamburg.actor)
    task.scene.remove_actor(task.frenchfries.actor)

    def test(name, model_ids, quat, grid):
        for model_id in model_ids:
            act = create_actor(scene=task, pose=sapien.Pose([-0.40, 0.0, table_z], list(quat)),
                               modelname=name, convex=True, model_id=model_id)
            lo, hi = _actor_aabb(act)
            worst = 0.0
            nroll = 0
            for (cx, cy) in grid:
                p = list(act.get_pose().p)
                p[0], p[1] = cx, cy
                p[2] = table_z + (table_z - lo)
                act.actor.set_pose(sapien.Pose(p, act.get_pose().q))
                for _ in range(60):
                    task.scene.step()
                for _ in range(1500):
                    task.scene.step()
                task.scene.update_render()
                x1 = np.asarray(act.get_pose().p)
                disp = float(np.hypot(x1[0] - cx, x1[1] - cy))
                if disp > 0.02:
                    nroll += 1
                    print(f"[s2] {name} m={model_id} spawn=({cx:.2f},{cy:.2f}) DISP={disp*100:.1f}cm "
                          f"rest=({x1[0]:.2f},{x1[1]:.2f},{x1[2]:.2f})", flush=True)
                worst = max(worst, disp)
            print(f"[s2] {name} m={model_id} WORST={worst*100:.2f}cm roll_spawns={nroll}/{len(grid)}", flush=True)
            task.scene.remove_actor(act.actor)

    hamburg_grid = [(-0.44, 0.04), (-0.44, 0.06), (-0.44, 0.08),
                    (-0.40, 0.04), (-0.40, 0.06), (-0.40, 0.08),
                    (-0.36, 0.04), (-0.36, 0.06), (-0.36, 0.08)]
    fries_grid = [(-0.44, -0.22), (-0.44, -0.21), (-0.44, -0.20), (-0.44, -0.19), (-0.44, -0.16),
                  (-0.40, -0.22), (-0.40, -0.21), (-0.40, -0.20), (-0.40, -0.19), (-0.40, -0.16),
                  (-0.36, -0.22), (-0.36, -0.21), (-0.36, -0.20), (-0.36, -0.19), (-0.36, -0.16)]
    test("006_hamburg", [0, 1], (0.5, 0.5, 0.5, 0.5), hamburg_grid)
    test("005_french-fries", [0, 1], (1.0, 0.0, 0.0, 0.0), fries_grid)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
