"""Temporary diagnostic: measure grasp offset and place accuracy for boxes.
Usage:
    python scripts/temp/diag_block_place.py <task> <seed> <block_name>
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
    task_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    block_name = sys.argv[3] if len(sys.argv) > 3 else "block2"
    task, config = build(task_name)
    from envs.utils.action import ArmTag
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        block = getattr(task, block_name)
        target = getattr(task, f"{block_name}_target_pose")
        print(f"[bp] {block_name} init p={np.round(np.array(block.get_pose().p), 4)}", flush=True)
        print(f"[bp] target={np.round(np.array(target[:3]), 4)}", flush=True)
        task.move(task.grasp_actor(block, arm_tag=ArmTag("left"), pre_grasp_dis=0.09))
        print(f"[bp] grasp plan={task.plan_success}", flush=True)
        if task.plan_success:
            ee = np.asarray(task.robot.get_left_ee_pose(), dtype=np.float64)
            bp = np.asarray(block.get_pose().p, dtype=np.float64)
            print(f"[bp] ee={np.round(ee[:3], 4)} block={np.round(bp, 4)}", flush=True)
            print(f"[bp] offset block-ee={np.round(bp - ee[:3], 4)}", flush=True)
            task.move_by_displacement(arm_tag=ArmTag("left"), z=0.07)
            ee2 = np.asarray(task.robot.get_left_ee_pose(), dtype=np.float64)
            bp2 = np.asarray(block.get_pose().p, dtype=np.float64)
            print(f"[bp] after lift ee={np.round(ee2[:3], 4)} block={np.round(bp2, 4)}", flush=True)
            print(f"[bp] offset after lift={np.round(bp2 - ee2[:3], 4)}", flush=True)
            place = task.place_actor(block, arm_tag=ArmTag("left"), target_pose=target,
                                     pre_dis=0.09, dis=0.02, constrain="align")
            task.move(place)
            print(f"[bp] place plan={task.plan_success}", flush=True)
            ee3 = np.asarray(task.robot.get_left_ee_pose(), dtype=np.float64)
            bp3 = np.asarray(block.get_pose().p, dtype=np.float64)
            print(f"[bp] after place ee={np.round(ee3[:3], 4)} block={np.round(bp3, 4)}", flush=True)
            print(f"[bp] block-target del={np.round(bp3 - np.asarray(target[:3]), 4)}", flush=True)
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
