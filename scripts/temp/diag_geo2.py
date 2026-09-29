"""Temporary diagnostic: inspect basket/can collision geometry and AABBs.
Not a deliverable.  Usage:
    python scripts/temp/diag_geo2.py <seed>
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

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
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    return task, config


def shape_bounds(actor) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return per-collision-shape AABB (min, max) for an actor in world frame."""
    out = []
    entity = actor.actor
    shapes = getattr(entity, "collision_shapes", [])
    for _i, shape in enumerate(shapes):
        pose = shape.get_pose()
        shape_min, shape_max = shape.local_pose_min, shape.local_pose_max
        if shape_min is None:
            continue
        R = pose.to_transformation_matrix()[:3, :3]
        corners = np.array(
            [[shape_min[0], shape_min[1], shape_min[2]],
             [shape_min[0], shape_min[1], shape_max[2]],
             [shape_min[0], shape_max[1], shape_min[2]],
             [shape_min[0], shape_max[1], shape_max[2]],
             [shape_max[0], shape_min[1], shape_min[2]],
             [shape_max[0], shape_min[1], shape_max[2]],
             [shape_max[0], shape_max[1], shape_min[2]],
             [shape_max[0], shape_max[1], shape_max[2]]]
        )
        world = corners @ R.T + pose.p
        out.append((world.min(axis=0), world.max(axis=0)))
    return out


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task, config = build("place_can_basket_left")
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        basket_p = task.basket.get_pose().p
        can_p = task.can.get_pose().p
        print(f"[g2] basket p={np.round(basket_p, 3)} "
              f"q={np.round(task.basket.get_pose().q, 3)}", flush=True)
        print(f"[g2] can p={np.round(can_p, 3)} "
              f"q={np.round(task.can.get_pose().q, 3)}", flush=True)
        basket_bounds = shape_bounds(task.basket)
        can_bounds = shape_bounds(task.can)
        print(f"[g2] basket shapes={len(basket_bounds)}", flush=True)
        for i, (mn, mx) in enumerate(basket_bounds):
            print(f"[g2] basket shape {i}: min={np.round(mn, 3)} max={np.round(mx, 3)} "
                  f"h={mx[2]-mn[2]:.3f}", flush=True)
        for i, (mn, mx) in enumerate(can_bounds):
            print(f"[g2] can shape {i}: min={np.round(mn, 3)} max={np.round(mx, 3)} "
                  f"h={mx[2]-mn[2]:.3f}", flush=True)
        can_half = (can_bounds[0][1][2] - can_bounds[0][0][2]) / 2.0 if can_bounds else 0.058
        print(f"[g2] can_half={can_half:.3f} can_bottom={can_p[2]-can_half:.3f} "
              f"basket_floor={min(mn[2] for mn, _ in basket_bounds):.3f} "
              f"basket_rim={max(mx[2] for mx, _ in basket_bounds):.3f}", flush=True)
        for fp in range(2):
            print(f"[g2] basket fp{fp}={np.round(task.basket.get_functional_point(fp), 3)}",
                  flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
