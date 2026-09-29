"""Temporary diagnostic: full traceback for blocks_ranking_rgb_left.
Not a deliverable.  Usage:
    python scripts/temp/diag_block_err.py
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)


def main() -> int:
    task_name = "blocks_ranking_rgb_left"
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
    config["dual_arm_embodied"] = False
    config["embodiment_dis"] = 0.60
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        task.play_once()
        print("[be] plan_success=", task.plan_success, "succ=", bool(task.check_success()))
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
