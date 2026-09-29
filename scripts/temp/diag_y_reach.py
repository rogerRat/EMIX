"""Temporary diagnostic: sweep actor y position and report grasp plan status.
Not a deliverable.  Usage:
    python scripts/temp/diag_y_reach.py <task> [seed] [x]
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
    task_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    x = float(sys.argv[3]) if len(sys.argv) > 3 else -0.40
    task, config = build(task_name)
    from envs.utils.action import ArmTag
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        fries = task.frenchfries
        quat = [1.0, 0.0, 0.0, 0.0]
        for y in (-0.30, -0.26, -0.22, -0.18, -0.14, -0.10, -0.06):
            fries.actor.set_pose(sapien.Pose([x, y, 0.78], quat))
            for _ in range(40):
                task.scene.step()
            task.plan_success = True
            grip = task.grasp_actor(fries, arm_tag=ArmTag("left"), pre_grasp_dis=0.1)
            ok = isinstance(grip, tuple) and len(grip) == 2 and grip[1] and \
                getattr(grip[1][0], "action", None) == "move"
            p = np.round(np.asarray(fries.get_pose().p), 3)
            print(f"[yr] x={x:.2f} y={y:.2f} settled_p={p} grasp={'OK' if ok else 'FAIL'}", flush=True)
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
