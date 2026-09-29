"""Temporary: source recipe (standing can + place_actor at fp) + derived check_success."""
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
    seed0 = int(sys.argv[1]); n = int(sys.argv[2])
    task, config = build("place_can_basket_left")
    from dataclasses import replace
    from envs.utils.action import ArmTag
    from envs.place_can_basket_left_impl import _actor_aabb
    man = task.manifest
    spec = man.actor_specs["can"]
    man = replace(man, actor_specs={**man.actor_specs,
        "can": replace(spec, base_quat=(0.707225, 0.706849, -0.0100455, -0.00982061))})
    task.manifest = man
    succ = 0; setup_ok = 0
    for s in range(seed0, seed0 + n):
        try:
            task.setup_demo(now_ep_num=s, seed=s, is_test=True, **config)
            setup_ok += 1
        except Exception as exc:
            print(f"[src2] seed={s} SETUP_EXC {type(exc).__name__}", flush=True)
            try: task.close_env()
            except Exception: pass
            continue
        arm = ArmTag("left")
        task.plan_success = True
        task.move(task.grasp_actor(task.can, arm_tag=arm, pre_grasp_dis=0.05))
        if not task.plan_success:
            print(f"[src2] seed={s} GRASP_FAIL", flush=True)
            task.close_env(); continue
        place_pose = np.asarray(task.get_arm_pose(arm))
        f0 = np.array(task.basket.get_functional_point(0))
        f1 = np.array(task.basket.get_functional_point(1))
        if np.linalg.norm(f0[:2] - place_pose[:2]) < np.linalg.norm(f1[:2] - place_pose[:2]):
            place_pose = f0.copy(); place_pose[:2] = f0[:2]; place_pose[3:] = (-1, 0, 0, 0)
        else:
            place_pose = f1.copy(); place_pose[:2] = f1[:2]; place_pose[3:] = (-1, 0, 0, 0)
        task.move(task.place_actor(task.can, arm_tag=arm, target_pose=place_pose,
                                   dis=0.02, is_open=False, constrain="free"))
        if not task.plan_success:
            task.plan_success = True
            place_pose[0] += -0.15
            place_pose[2] += 0.15
            task.move(task.move_to_pose(arm, place_pose))
            task.move_by_displacement(arm, z=-0.1)
            task.move(task.open_gripper(arm))
            task.move(task.open_gripper(arm, pos=0.7))
        else:
            task.move(task.open_gripper(arm))
            task.move_by_displacement(arm, z=0.12)
            task.move(task.open_gripper(arm, pos=0.7))
        task.left_return_home()
        task.move(task.open_gripper(arm, pos=1.0))
        can_f = np.round(np.asarray(task.can.get_pose().p), 3)
        lo, hi = _actor_aabb(task.can)
        ok = bool(task.check_success())
        succ += ok
        bc = task.check_actors_contact(task.can_name, task.basket_name)
        tc = task.check_actors_contact(task.can_name, "table")
        print(f"[src2] seed={s} plan={task.plan_success} can_f={can_f} aabb=[{lo:.3f},{hi:.3f}] "
              f"basket_c={bc} table_c={tc} success={ok}", flush=True)
        task.close_env()
    print(f"[src2] SUCCESS {succ}/{n} (setup_ok {setup_ok})", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
