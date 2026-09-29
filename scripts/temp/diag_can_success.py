"""Temporary diagnostic: print every check_success predicate for a seed.
Not a deliverable.  Usage:
    python scripts/temp/diag_can_success.py <seed>
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


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task, config = build("place_can_basket_left")
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        for a in task.scene.get_all_actors():
            n = a.get_name()
            if "table" in n.lower() or "floor" in n.lower():
                print(f"[cs] actor {n} p={np.round(a.get_pose().p, 3)}", flush=True)
        task.play_once()
        can_p = np.asarray(task.can.get_pose().p)
        basket_p = np.asarray(task.basket.get_pose().p)
        print(f"[cs] plan={task.plan_success}", flush=True)
        print(f"[cs] can={np.round(can_p, 3)} basket={np.round(basket_p, 3)}", flush=True)
        print(f"[cs] dist={np.sum(np.sqrt(np.power(can_p - basket_p, 2))):.3f}", flush=True)
        print(f"[cs] basket_h_delta={abs(basket_p[2]-task.start_height):.4f}", flush=True)
        axis = (task.basket.get_pose().to_transformation_matrix()[:3, :3]
                @ np.array([[0, 1, 0]]).T).reshape(3)
        print(f"[cs] upright_dot={np.dot(axis, [0,0,1]):.3f}", flush=True)
        print(f"[cs] can_table_contact={task.check_actors_contact(task.can_name, 'table')}", flush=True)
        print(f"[cs] can_basket_contact={task.check_actors_contact(task.can_name, task.basket_name)}", flush=True)
        print(f"[cs] left_open={task.is_left_gripper_open()}", flush=True)
        print(f"[cs] right_static={task.right_arm_stationary()}", flush=True)
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
