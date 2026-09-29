"""Temporary: reproduce validate play_once path with full traceback."""
from __future__ import annotations
import importlib, os, sys, traceback
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "place_burger_fries_left"


def build():
    envs = importlib.import_module(f"envs.{TASK_NAME}")
    task = getattr(envs, TASK_NAME)()
    with open(BENCHMARK / "task_config" / f"{TASK_NAME}.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = TASK_NAME
    config["task_config"] = TASK_NAME
    config["eval_mode"] = True
    config["left_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["right_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
    config["dual_arm_embodied"] = False
    config["embodiment_dis"] = 0.60
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml", "r", encoding="utf-8") as handle:
        ec = yaml.safe_load(handle)
    config["left_embodiment_config"] = ec
    config["right_embodiment_config"] = ec
    return task, config


def main() -> int:
    import numpy as np
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task, config = build()
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        print(f"[diag] setup done, plan_success={task.plan_success}", flush=True)
        task.play_once()
        print(f"[diag] play_once returned, plan_success={task.plan_success}", flush=True)
        print(f"[diag] success={task.check_success()}", flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        try:
            task.close_env()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
