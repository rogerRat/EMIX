"""Temporary: report can spawn pos, setup stability, grasp outcome per seed."""
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    task, config = build("place_can_basket_left")
    from envs.utils.action import ArmTag
    for s in range(n):
        try:
            task.setup_demo(now_ep_num=s, seed=s, is_test=True, **config)
        except Exception as exc:
            cp = np.round(np.asarray(task.can.get_pose().p) if hasattr(task, "can") else [0,0,0], 3)
            print(f"[spawn] seed={s} SETUP_EXC {type(exc).__name__} can={cp}", flush=True)
            try: task.close_env()
            except Exception: pass
            continue
        cp = np.round(np.asarray(task.can.get_pose().p), 3)
        cq = np.round(np.asarray(task.can.get_pose().q), 2)
        arm = ArmTag("left")
        task.plan_success = True
        task.move(task.grasp_actor(task.can, arm_tag=arm, pre_grasp_dis=0.05))
        print(f"[spawn] seed={s} setup_ok can_tgt={cp} q={cq} grasp={task.plan_success}", flush=True)
        task.close_env()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
