"""Temporary: step-level plan trace for place_dual_shoes_left."""
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
    from envs.utils.action import ArmTag
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task, config = build()
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
    except Exception as exc:  # noqa: BLE001
        print(f"[t] setup raised {type(exc).__name__}: {exc}", flush=True)
        task.close_env()
        return 1
    ls = np.round(np.asarray(task.left_shoe.get_pose().p), 3)
    rs = np.round(np.asarray(task.right_shoe.get_pose().p), 3)
    print(f"[t] seed={seed} L={ls} R={rs} shoe_id={task.shoe_id}", flush=True)

    def step(name, pair):
        task.move(pair)
        print(f"[t]   {name}: plan={task.plan_success} take_cnt={getattr(task, 'take_action_cnt', None)}", flush=True)
        return bool(task.plan_success)

    def step_disp(name, **kw):
        task.move_by_displacement(arm_tag=ArmTag("left"), **kw)
        print(f"[t]   {name}: plan={task.plan_success} take_cnt={getattr(task, 'take_action_cnt', None)}", flush=True)
        return bool(task.plan_success)

    for label, shoe in (("left_shoe", task.left_shoe), ("right_shoe", task.right_shoe)):
        if not task.plan_success:
            break
        ok = step(f"grasp_{label}", task.grasp_actor(shoe, arm_tag=ArmTag("left"), pre_grasp_dis=0.1))
        if not ok:
            break
        ok = step_disp(f"lift_{label}", z=0.15)
        if not ok:
            break
        fp = 0 if label == "left_shoe" else 1
        target = task.shoe_box.get_functional_point(fp)
        print(f"[t]   fp{fp}={np.round(np.asarray(target), 3)}", flush=True)
        ok = step(f"place_{label}", task.place_actor(shoe, arm_tag=ArmTag("left"),
                                                      target_pose=target, functional_point_id=0,
                                                      pre_dis=0.07, dis=0.02, constrain="align"))
        if not ok:
            break
        ok = step_disp(f"retreat_{label}", z=0.1, move_axis="arm")
        if not ok:
            break
        task.left_return_home()
    ok = bool(task.check_success())
    print(f"[t] SUCCESS={ok}", flush=True)
    try:
        task.close_env()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
