"""Temporary: print dual_shoes layouts for seeds 0-9, catching UnStableError."""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "place_dual_shoes_left"


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
    task, config = build()
    for s in range(10):
        try:
            task.setup_demo(now_ep_num=s, seed=s, is_test=True, **config)
            status = "OK"
        except Exception as exc:  # noqa: BLE001
            status = f"{type(exc).__name__}: {exc}"
        lay = getattr(task, "sampled_layout_summary", {}) or {}
        shoe_id = getattr(task, "shoe_id", None)
        print(f"[ds] seed={s} status={status} shoe_id={shoe_id}", flush=True)
        for name in ("shoe_box", "left_shoe", "right_shoe"):
            if name in lay:
                p = lay[name][:3]
                q = lay[name][3:]
                print(f"[ds]   {name}: p=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) q=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})", flush=True)
        try:
            task.close_env()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
