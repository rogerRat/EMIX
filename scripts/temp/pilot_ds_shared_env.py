"""Temporary: seed-sweep success-rate pilot for place_dual_shoes (shared envelope).

Runs the 2026-08-22 version-4 shared-envelope sampling (shoe x and the whole
shoe_box pose from the one shared batch envelope; shoe y anchored to the two
liftable rows) across N seeds and reports per-seed plan/success plus the
aggregate rate.  Only place_dual_shoes is tested.

Usage:
    python scripts/temp/pilot_ds_shared_env.py [num_seeds] [task_name]
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = sys.argv[2] if len(sys.argv) > 2 else "place_dual_shoes_left_cen_arm_near_side_cam"
CONFIG_FOLDER = {
    "place_dual_shoes_left_cen_arm_near_side_cam": "cen_arm_near_side_cam",
    "place_dual_shoes_left_cen_arm_side_cam": "cen_arm_side_cam",
}.get(TASK_NAME, "")
NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = ROOT / "results" / "pilot" / f"{TASK_NAME}.shared_env.pilot.jsonl"


def build():
    envs = importlib.import_module(f"envs.left.{CONFIG_FOLDER}.{TASK_NAME}")
    task = getattr(envs, TASK_NAME)()
    cfg_path = (BENCHMARK / "task_config" / "left" / CONFIG_FOLDER / f"{TASK_NAME}.yml")
    with open(cfg_path, "r", encoding="utf-8") as handle:
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
    rows = []
    for seed in range(NUM_SEEDS):
        task, config = build()
        row = {"seed": seed}
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
            row["layout"] = task.sampled_layout_summary
            task.play_once()
            row["plan_success"] = bool(task.plan_success)
            row["success"] = bool(task.check_success())
            row["actions"] = int(task.take_action_cnt)
        except Exception as exc:  # noqa: BLE001
            row["plan_success"] = False
            row["success"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        finally:
            try:
                task.close_env()
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)
        ok = sum(r["success"] for r in rows)
        plan = sum(r["plan_success"] for r in rows)
        print(f"[pilot] seed {seed:3d} plan={row['plan_success']} success={row['success']} "
              f"| running success {ok}/{len(rows)} plan {plan}/{len(rows)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(__import__("json").dumps(r) + "\n")
    ok = sum(r["success"] for r in rows)
    plan = sum(r["plan_success"] for r in rows)
    print(f"[pilot] FINAL: success {ok}/{len(rows)} = {ok/len(rows):.2f}, "
          f"plan {plan}/{len(rows)} = {plan/len(rows):.2f} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
