"""Temporary diagnostic: after play_once, print check_success components.
Usage:
    python scripts/temp/diag_succ.py <task> <seed>
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
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    task, config = build(task_name)
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        print(f"[su] layout={getattr(task, 'sampled_layout_summary', {})}", flush=True)
        task.play_once()
        print(f"[su] final plan={task.plan_success}", flush=True)
        fp = task.skillet.get_functional_point(0)
        bq_full = task.bread.get_pose().q
        bq = task.bread.get_pose()
        bp = bq.p
        print(f"[su] bread_id={getattr(task, 'bread_id', '?')} skillet_id={getattr(task, 'skillet_id', '?')}", flush=True)
        print(f"[su] bread quat={np.round(np.array(bq_full), 3)}", flush=True)
        print(f"[su] skillet center={np.round(np.array(task.skillet.get_pose().p), 3)} "
              f"quat={np.round(np.array(task.skillet.get_pose().q), 3)}", flush=True)
        try:
            bb = task.bread.get_aabb()
            print(f"[su] bread aabb lo={np.round(bb.lo, 3)} hi={np.round(bb.hi, 3)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[su] aabb exc {exc}", flush=True)
        print(f"[su] fp={np.round(fp, 3)}", flush=True)
        print(f"[su] bread={np.round(bp, 3)}", flush=True)
        print(f"[su] |fp[:2]-bp[:2]|={np.round(np.abs(fp[:2]-bp[:2]), 4)} "
              f"(<0.035={bool(np.all(np.abs(fp[:2]-bp[:2]) < 0.035))})", flush=True)
        print(f"[su] fp[2]={fp[2]:.4f} >0.76+{task.table_z_bias:.4f}? {bool(fp[2] > 0.76 + task.table_z_bias)}", flush=True)
        print(f"[su] bp[2]={bp[2]:.4f} >0.76+{task.table_z_bias:.4f}? {bool(bp[2] > 0.76 + task.table_z_bias)}", flush=True)
        print(f"[su] left open={task.is_left_gripper_open()} right_static={task.right_arm_stationary()}", flush=True)
        print(f"[su] check_success={bool(task.check_success())}", flush=True)
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
