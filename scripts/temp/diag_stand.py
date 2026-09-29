"""Temporary: test standing-can spawn stability + grasp + full recipe.
Usage: python scripts/temp/diag_stand.py <seed0> <nseeds> <quat_index>
quat_index: 0=source standing, 1=90deg about world x, 2=90deg about world y
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

QUATS = [
    (0.707225, 0.706849, -0.0100455, -0.00982061),  # source
    (0.7071068, 0.7071068, 0.0, 0.0),               # 90 deg about (1,1,0)/sqrt2 ~ x
    (0.7071068, 0.0, 0.7071068, 0.0),               # 90 deg about y
]

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
    seed0 = int(sys.argv[1]); n = int(sys.argv[2]); qi = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    task, config = build("place_can_basket_left")
    from envs.utils.action import ArmTag
    from envs.place_can_basket_left_impl import _actor_aabb, _rotate_quat
    from dataclasses import replace
    man = task.manifest
    spec = man.actor_specs["can"]
    man = replace(man, actor_specs={**man.actor_specs, "can": replace(spec, base_quat=QUATS[qi])})
    task.manifest = man
    for s in range(seed0, seed0 + n):
        try:
            task.setup_demo(now_ep_num=s, seed=s, is_test=True, **config)
        except Exception as exc:
            cp = np.round(np.asarray(task.can.get_pose().p) if hasattr(task, "can") else [0,0,0], 3)
            print(f"[st] seed={s} SETUP_EXC {type(exc).__name__} can={cp}", flush=True)
            try: task.close_env()
            except Exception: pass
            continue
        cp = np.round(np.asarray(task.can.get_pose().p), 3)
        cq = np.round(np.asarray(task.can.get_pose().q), 2)
        lo, hi = _actor_aabb(task.can)
        arm = ArmTag("left")
        task.plan_success = True
        task.move(task.grasp_actor(task.can, arm_tag=arm, pre_grasp_dis=0.05))
        grasp = bool(task.plan_success)
        # full recipe (raise 0.10)
        if grasp:
            ee = np.asarray(task.get_arm_pose(arm))
            task.move(task.move_to_pose(arm, np.array([ee[0], ee[1], ee[2] + 0.06, *_rotate_quat(ee[3:])])))
            ee2 = np.asarray(task.get_arm_pose(arm))
            offset = np.asarray(task.can.get_pose().p[:3]) - ee2[:3]
            bp = np.asarray(task.basket.get_pose().p)
            plate = np.array([bp[0] + 0.05, bp[1] - 0.04])
            task.move(task.move_to_pose(arm, np.concatenate([np.array([plate[0], plate[1], 0.95]) - offset, ee2[3:]])))
            ee3 = np.asarray(task.get_arm_pose(arm))
            offset2 = np.asarray(task.can.get_pose().p[:3]) - ee3[:3]
            task.move(task.move_to_pose(arm, np.concatenate([np.array([plate[0], plate[1], 0.90]) - offset2, ee3[3:]])))
            desc = bool(task.plan_success)
            task.move(task.open_gripper(arm, pos=0.7)); task.move(task.open_gripper(arm, pos=0.7))
            task.move(task.open_gripper(arm, pos=1.0))
            ee4 = np.asarray(task.get_arm_pose(arm))
            task.plan_success = True
            task.move(task.move_to_pose(arm, np.array([ee4[0], ee4[1], ee4[2] + 0.10, *ee4[3:]])))
            task.left_return_home()
            task.move(task.open_gripper(arm, pos=1.0))
            ok = bool(task.check_success())
            cf = np.round(np.asarray(task.can.get_pose().p), 3)
            print(f"[st] seed={s} qi={qi} can_tgt={cp} q={cq} aabb=[{lo:.3f},{hi:.3f}] "
                  f"grasp={grasp} desc={desc} success={ok} can_f={cf}", flush=True)
        else:
            print(f"[st] seed={s} qi={qi} can_tgt={cp} q={cq} grasp={grasp}", flush=True)
        task.close_env()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
