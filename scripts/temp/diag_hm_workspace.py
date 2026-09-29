"""Temporary: map left-arm IK workspace at several heights.

From home state, probe a grid of (x, y) at given z with identity quat to see
where the left arm can actually reach.  Informs where the rack fp0 may live.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "hanging_mug_left"


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
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[W] setup raised {type(exc).__name__}: {exc}", flush=True)
        task.close_env()
        return 1

    print(f"[W] home EE = {np.round(np.asarray(task.robot.get_left_ee_pose()),3)}", flush=True)
    # identity quat (pointing +z, palm down orientation baseline)
    q = np.asarray([0.707, 0.0, 0.0, 0.707])
    for z in [0.70, 0.80, 0.90, 1.00]:
        print(f"[W] z={z:.2f} (x rows -0.50..0.05, y cols -0.05..0.40):", flush=True)
        for x in np.arange(-0.50, 0.06, 0.05):
            row = []
            for y in np.arange(-0.05, 0.41, 0.05):
                task.plan_success = True
                res = task.robot.left_plan_path(list([float(x), float(y), z]) + list(q))
                ok = res is not None and res.get("status") == "Success"
                row.append("O" if ok else ".")
            print(f"[W]   x={x:+.2f}  {''.join(row)}", flush=True)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
