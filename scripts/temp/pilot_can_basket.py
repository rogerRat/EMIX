"""Temporary pilot: place_can_basket_left success rate over seeds 0..N-1.
Not a deliverable.  Usage:
    python scripts/temp/pilot_can_basket.py [count] [start]
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

TASK_NAME = "place_can_basket_left"


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
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    return task, config


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    task, config = build()
    per_model = {}
    ok = 0
    for seed in range(start, start + count):
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
            can_id = int(getattr(task, "can_id", -1))
            basket_id = int(getattr(task, "basket_id", -1))
            task.play_once()
            succ = bool(task.check_success())
            can_p = np.round(np.asarray(task.can.get_pose().p, dtype=np.float64)[:3], 3)
            print(f"[p] seed={seed} can_model={can_id} basket={basket_id} "
                  f"succ={succ} can_p={can_p}", flush=True)
            if succ:
                ok += 1
            key = f"can{can_id}"
            per_model.setdefault(key, [0, 0])
            per_model[key][0] += int(succ)
            per_model[key][1] += 1
        except Exception:  # noqa: BLE001
            print(f"[p] seed={seed} EXC", flush=True)
            traceback.print_exc()
        finally:
            try:
                task.close_env()
            except Exception:  # noqa: BLE001
                pass
    print(f"[p] TOTAL {ok}/{count} = {ok / count:.2f}", flush=True)
    for k, (n, total) in sorted(per_model.items()):
        print(f"[p]   {k}: {n}/{total} = {n / total:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
