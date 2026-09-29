"""Temporary diagnostic: run the SOURCE place_can_basket for left vs right arm.
Not a deliverable.  Usage: python scripts/temp/diag_src_can.py <seed> <arm>
arm: left | right
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

TASK_NAME = "place_can_basket"


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    arm = sys.argv[2] if len(sys.argv) > 2 else "left"
    envs = importlib.import_module(f"envs.{TASK_NAME}")
    task = getattr(envs, TASK_NAME)()
    with open(BENCHMARK / "task_config" / "place_can_basket_left.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = TASK_NAME
    config["task_config"] = TASK_NAME
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
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        # Force the arm tag.
        from envs.utils.action import ArmTag
        task.arm_tag = ArmTag(arm)
        task.plan_success = True
        basket_p = np.asarray(task.basket.get_pose().p)
        can_p = np.asarray(task.can.get_pose().p)
        print(f"[src] seed={seed} arm={arm} basket={np.round(basket_p, 3)} can={np.round(can_p, 3)}", flush=True)
        task.play_once()
        print(f"[src] after play plan={task.plan_success}", flush=True)
        can_f = np.asarray(task.can.get_pose().p)
        print(f"[src] final can={np.round(can_f, 3)}", flush=True)
        print(f"[src] check_success={task.check_success()}", flush=True)
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
