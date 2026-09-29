"""Temporary diagnostic: sweep seeds, report layout stability for
place_bread_skillet_left.  Not a deliverable.
Usage:
    python scripts/temp/diag_skillet_stability.py [n_seeds]
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

task_name = "place_bread_skillet_left"
envs = importlib.import_module(f"envs.{task_name}")
task_cls = getattr(envs, task_name)
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

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
ok = 0
for seed in range(n):
    task = task_cls()
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        lay = getattr(task, "sampled_layout_summary", {})
        bp = lay.get("bread", {})
        sp = lay.get("skillet", {})
        bq = bp.get("q", "?")
        print(f"[st] seed={seed:3d} OK   bread=({bp.get('p')}) q={bq} "
              f"skillet=({sp.get('p')})", flush=True)
        ok += 1
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        first = _tb.format_exc().splitlines()[-8:]
        print(f"[st] seed={seed:3d} FAIL {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        for line in first:
            print(f"[st]     | {line.strip()}", flush=True)
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass
print(f"[st] stable {ok}/{n}", flush=True)
