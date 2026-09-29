"""Temporary diagnostic: does the SOURCE stack_bowls_two succeed the left-arm
grasp+lift on a fixed left-workspace layout?  Not a deliverable.
Usage:
    python scripts/temp/diag_src_bowl_lift.py
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
    with open(BENCHMARK / "task_config" / f"{task_name}.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = task_name
    config["task_config"] = task_name
    config["eval_mode"] = True
    config["left_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["right_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["dual_arm_embodied"] = True
    config["embodiment_dis"] = 0.60
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    return task, config


def main() -> int:
    task_name = "stack_bowls_two"

    def make_fixed_layout():
        bowl_pose = sapien.Pose(np.array([-0.25, -0.1, 0.74]), np.array([0.5, 0.5, 0.5, 0.5]))
        bowl_pose2 = sapien.Pose(np.array([-0.25, 0.1, 0.74]), np.array([0.5, 0.5, 0.5, 0.5]))
        return {"bowl1": bowl_pose, "bowl2": bowl_pose2}

    task, config = build(task_name)
    task.sample_layout = make_fixed_layout
    orig_move = task.move

    def traced_move(actions_by_arm1, actions_by_arm2=None, save_freq=-1):
        tag = actions_by_arm1[0] if isinstance(actions_by_arm1, tuple) else "?"
        acts = actions_by_arm1[1] if isinstance(actions_by_arm1, tuple) else actions_by_arm1
        descs = []
        for a in acts or []:
            if getattr(a, "action", None) == "move":
                tp = getattr(a, "target_pose", None)
                tp = np.round(np.array(tp), 3).tolist() if tp is not None else None
                descs.append(f"move{tp}")
            else:
                descs.append(getattr(a, "action", "?"))
        before = bool(task.plan_success)
        result = orig_move(actions_by_arm1, actions_by_arm2, save_freq=save_freq)
        after = bool(task.plan_success)
        print(f"[sb] tag={tag} acts={descs} plan {before}->{after}", flush=True)
        return result

    task.move = traced_move
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        print(f"[sb] bowl1=({np.round(task.bowl1.get_pose().p,3)}) "
              f"bowl2=({np.round(task.bowl2.get_pose().p,3)})", flush=True)
        task.play_once()
        print(f"[sb] final plan={task.plan_success} succ={bool(task.check_success())}", flush=True)
        return 0
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
