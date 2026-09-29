"""Temporary diagnostic: measure left-arm reach at hover heights.
Usage: python scripts/temp/diag_reach_z.py <z>
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml

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
    z = float(sys.argv[1]) if len(sys.argv) > 1 else 1.085
    task, config = build("place_can_basket_left")
    from envs.utils.action import ArmTag
    task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    arm = ArmTag("left")
    quat = [0.579, -0.404, 0.403, 0.583]
    reach = []
    for x in [-0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10]:
        for y in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]:
            task.plan_success = True
            task.move(task.move_to_pose(arm, np.array([x, y, z, *quat])))
            ok = bool(task.plan_success)
            if ok:
                reach.append((round(x,2), round(y,2)))
    print(f"[rz] z={z:.3f} reachable:", reach)
    task.close_env()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
