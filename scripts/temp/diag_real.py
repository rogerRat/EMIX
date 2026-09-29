"""Temporary diagnostic: real manifest flow (no overrides) for place_can_basket_left.
Usage: python scripts/temp/diag_real.py <seed0> <nseeds>
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml

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
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml", "r", encoding="utf-8") as handle:
        ec = yaml.safe_load(handle)
    config["left_embodiment_config"] = ec
    config["right_embodiment_config"] = ec
    return task, config

def main() -> int:
    seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    task, config = build()
    succ = 0
    for s in range(seed0, seed0 + n):
        try:
            task.setup_demo(now_ep_num=s, seed=s, is_test=True, **config)
            bp = np.asarray(task.basket.get_pose().p)
            cp = np.asarray(task.can.get_pose().p)
            task.plan_success = True
            task.play_once()
            can_f = np.asarray(task.can.get_pose().p)
            ok = bool(task.check_success())
            succ += ok
            print(f"[real] seed={s} plan={task.plan_success} basket={np.round(bp,3)} "
                  f"can_tgt=({cp[0]:.2f},{cp[1]:.2f}) can_f={np.round(can_f,3)} success={ok}", flush=True)
        except Exception:
            import traceback; traceback.print_exc()
            print(f"[real] seed={s} EXC", flush=True)
        finally:
            try: task.close_env()
            except Exception: pass
    print(f"[real] SUCCESS {succ}/{n}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
