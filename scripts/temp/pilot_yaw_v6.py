"""Temporary pilot: 200-seed acceptance check for the 2026-08-23 fixes.

Tests stack_bowls_two (place pre_dis ladder) and place_cans_plasticbox (box
yaw + derived can envelope) on the cen_arm_near_side_cam wrappers.  Records
per-seed layout, stack target (stack_bowls) and success so target-coverage
bias can be audited, not just the rate.

Usage:
    python scripts/temp/pilot_yaw_v6.py <task_name> [num_seeds] [start_seed]

Writes results/pilot/yaw_v6/<task>.pilot.jsonl.
"""
from __future__ import annotations
import importlib, json, os, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = sys.argv[1] if len(sys.argv) > 1 else "place_cans_plasticbox_left_cen_arm_near_side_cam"
NUM_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
START = int(sys.argv[3]) if len(sys.argv) > 3 else 0
CONFIG_FOLDER = "cen_arm_near_side_cam"
OUT = ROOT / "results" / "pilot" / "yaw_v6" / f"{TASK_NAME}.pilot.jsonl"


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
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        ec = yaml.safe_load(handle)
    config["left_embodiment_config"] = ec
    config["right_embodiment_config"] = ec
    return task, config


def main() -> int:
    rows = []
    for seed in range(START, START + NUM_SEEDS):
        task, config = build()
        row = {"task": TASK_NAME, "seed": seed}
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
            row["layout"] = task.sampled_layout_summary
            if hasattr(task, "stack_target_xy"):
                row["stack_target"] = [float(v) for v in task.stack_target_xy]
            task.play_once()
            row["plan_success"] = bool(task.plan_success)
            row["success"] = bool(task.check_success())
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
        print(f"[pilot] seed {seed:4d} plan={row['plan_success']} success={row['success']} "
              f"| running success {ok}/{len(rows)} plan {plan}/{len(rows)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    ok = sum(r["success"] for r in rows)
    plan = sum(r["plan_success"] for r in rows)
    print(f"[pilot] FINAL: success {ok}/{len(rows)} = {ok/len(rows):.2f}, "
          f"plan {plan}/{len(rows)} = {plan/len(rows):.2f} -> {OUT}", flush=True)
    if rows and all("stack_target" in r for r in rows if r.get("success")):
        tgt = np.array([r["stack_target"] for r in rows if r["success"]])
        if len(tgt):
            print(f"[pilot] accepted stack targets: x[{tgt[:,0].min():.2f},{tgt[:,0].max():.2f}] "
                  f"y[{tgt[:,1].min():.2f},{tgt[:,1].max():.2f}] mean=({tgt[:,0].mean():.3f},{tgt[:,1].mean():.3f})",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
