"""Temporary diagnostic: sweep french-fries spawn quat and check settle stability.
Not a deliverable.  Usage:
    python scripts/temp/diag_ff_settle.py
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import sapien
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


def main() -> int:
    task, config = build("place_burger_fries_left")
    quats = {
        "upright": [1.0, 0.0, 0.0, 0.0],
        "yaw45": [0.924, 0.0, 0.0, 0.383],
        "yaw90": [0.707, 0.0, 0.0, 0.707],
        "x90": [0.707, 0.707, 0.0, 0.0],
    }
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        ff = task.frenchfries
        for name, quat in quats.items():
            ff.actor.set_pose(sapien.Pose([-0.38, -0.26, 0.80], quat))
            # advance physics to settle
            for _ in range(120):
                task.scene.step()
            p = np.round(np.asarray(ff.get_pose().p), 3)
            q = np.round(np.asarray(ff.get_pose().q), 3)
            stable = p[0] > -1.0 and p[1] > -1.0 and p[2] < 1.0
            print(f"[ff] quat={name:8s} -> p={p} q={q} stable={stable}", flush=True)
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
