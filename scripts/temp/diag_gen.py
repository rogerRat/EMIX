"""Temporary generic diagnostic: after play_once, print every actor pose and
the task's own check_success.  Not a deliverable.
Usage:
    python scripts/temp/diag_gen.py <task> <seed>
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
    task, config = build(task_name)
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        print(f"[dg] seed={seed} setup ok", flush=True)
        print(f"[dg] init layout={getattr(task, 'sampled_layout_summary', {})}", flush=True)
        task.play_once()
        print(f"[dg] final plan={task.plan_success} check={bool(task.check_success())}", flush=True)
        print(f"[dg] -- actors after play_once:", flush=True)
        for actor in task.scene.get_all_actors():
            name = actor.get_name()
            pose = actor.get_pose()
            p = np.round(np.array(pose.p), 3)
            q = np.round(np.array(pose.q), 3)
            print(f"[dg]   {name:20s} p={p} q={q}", flush=True)
        try:
            print(f"[dg] left open={task.is_left_gripper_open()} "
                  f"right_static={task.right_arm_stationary()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[dg] helper exc: {exc}", flush=True)
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
