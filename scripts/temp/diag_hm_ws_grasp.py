"""Temporary: map left-arm reachability in the grasp posture.

After grasping the mug the left EEF posture is roughly the grasp quat
(0.55,-0.45,0.45,0.55).  Map which world positions are reachable in that
fixed posture, and also the reachability at identity quat, over the rack band
x in [-0.5, 0.05], y in [-0.15, 0.25], z in [0.75, 1.0].
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "hanging_mug_left"


def load_config():
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
    return config


def main() -> int:
    envs = importlib.import_module(f"envs.{TASK_NAME}")
    task_cls = getattr(envs, TASK_NAME)
    config = load_config()
    env = task_cls()
    env.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
    GRASP_Q = [0.549, -0.447, 0.445, 0.549]
    ID_Q = [0.707, 0.0, 0.0, 0.707]
    for qlabel, quat in [("grasp", GRASP_Q), ("id", ID_Q)]:
        print(f"[WS] posture={qlabel}", flush=True)
        for z in [0.80, 0.88, 0.95]:
            row = []
            for x in np.arange(-0.50, 0.06, 0.05):
                for y in np.arange(-0.15, 0.26, 0.05):
                    tgt = [float(x), float(y), float(z)] + quat
                    env.plan_success = True
                    res = env.robot.left_plan_path(list(tgt))
                    ok = res is not None and res.get("status") == "Success"
                    if ok:
                        row.append(f"({x:+.2f},{y:+.2f})")
            print(f"[WS]   z={z:.2f}: {', '.join(row) if row else '(none)'}", flush=True)
    env.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
