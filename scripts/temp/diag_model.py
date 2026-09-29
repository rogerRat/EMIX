"""Temporary: check can model_id / basket_id correlation with setup stability."""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

def build(task_name: str):
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
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml", "r", encoding="utf-8") as handle:
        ec = yaml.safe_load(handle)
    config["left_embodiment_config"] = ec
    config["right_embodiment_config"] = ec
    return task, config

def main() -> int:
    task, config = build("place_can_basket_left")
    for s in range(16):
        try:
            task.setup_demo(now_ep_num=s, seed=s, is_test=True, **config)
            status = "ok"
            cp = np.round(np.asarray(task.can.get_pose().p), 3)
        except Exception as exc:
            status = f"EXC:{type(exc).__name__}"
            cp = np.round(np.asarray(task.can.get_pose().p) if hasattr(task, "can") else [0,0,0], 3)
        print(f"[model] seed={s} {status} can_id={task.can_id} basket_id={task.basket_id} "
              f"can={cp} basket={np.round(np.asarray(task.basket.get_pose().p),3) if hasattr(task,'basket') else '-'}", flush=True)
        try: task.close_env()
        except Exception: pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
