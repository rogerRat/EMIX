"""Temporary diagnostic: source hanging_mug baseline geometry.
Not part of the deliverable.  Usage:
    python scripts/temp/diag_src_hang.py
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


def build(task_name: str):
    sys.path.insert(0, str(BENCHMARK))
    envs = importlib.import_module(f"envs.{task_name}")
    task = getattr(envs, task_name)()
    with open(BENCHMARK / "task_config" / "hanging_mug_left.yml", "r", encoding="utf-8") as handle:
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


def main() -> int:
    task_name = "hanging_mug"
    task, config = build(task_name)

    def fixed_layout():
        mug_pose = sapien.Pose(np.array([-0.15, 0.0, 0.74]), np.array([0.707, 0.707, 0.0, 0.0]))
        rack_pose = sapien.Pose(np.array([0.2, 0.15, 0.74]), np.array([-0.22, -0.22, 0.67, 0.67]))
        return {"mug": mug_pose, "rack": rack_pose}

    task.sample_layout = fixed_layout
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        task.play_once()
        mug_fp = task.mug.get_functional_point(0)[:3]
        rack_p = task.rack.get_pose().p
        rack_fp = task.rack.get_functional_point(0)[:3]
        mid = (rack_p + rack_fp) / 2
        succ_immediate = task.check_success()
        print(f"[src] immediate mug_fp=({mug_fp[0]:+.3f},{mug_fp[1]:+.3f},{mug_fp[2]:+.3f}) "
              f"mid=({mid[0]:+.3f},{mid[1]:+.3f},{mid[2]:+.3f}) "
              f"rack_p=({rack_p[0]:+.3f},{rack_p[1]:+.3f},{rack_p[2]:+.3f}) "
              f"rack_fp=({rack_fp[0]:+.3f},{rack_fp[1]:+.3f},{rack_fp[2]:+.3f}) "
              f"succ={bool(succ_immediate)}", flush=True)
        for n in (10, 60):
            for _ in range(n if n == 10 else 50):
                task.scene.step()
            mug_fp2 = task.mug.get_functional_point(0)[:3]
            print(f"[src] after {n} steps mug_fp=({mug_fp2[0]:+.3f},{mug_fp2[1]:+.3f},{mug_fp2[2]:+.3f}) "
                  f"succ={bool(task.check_success())}", flush=True)
        return 0
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
