"""Temporary: find which move kills plan_success in the real place_can_basket_left flow."""
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
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    task, config = build("place_can_basket_left")
    from envs.place_can_basket_left_impl import _rotate_quat
    from envs.utils.action import ArmTag
    task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
    arm = ArmTag("left")
    task.plan_success = True
    def step(name):
        print(f"[plan] {name}: plan={task.plan_success}", flush=True)
        return bool(task.plan_success)
    task.move(task.grasp_actor(task.can, arm_tag=arm, pre_grasp_dis=0.05)); step("grasp")
    if not task.plan_success: task.close_env(); return 0
    ee = np.asarray(task.get_arm_pose(arm))
    target = np.array([ee[0], ee[1], ee[2] + 0.06, *_rotate_quat(ee[3:])])
    task.move(task.move_to_pose(arm, target)); step("rotate+lift")
    if not task.plan_success: task.close_env(); return 0
    ee2 = np.asarray(task.get_arm_pose(arm))
    offset = np.asarray(task.can.get_pose().p[:3]) - ee2[:3]
    basket_p = np.asarray(task.basket.get_pose().p)
    plate_world = np.array([basket_p[0] + 0.05, basket_p[1] - 0.04])
    hover_ee = np.concatenate([np.array([plate_world[0], plate_world[1], 0.95]) - offset, ee2[3:]])
    task.move(task.move_to_pose(arm, hover_ee)); step("hover")
    if not task.plan_success: task.close_env(); return 0
    ee3 = np.asarray(task.get_arm_pose(arm))
    offset2 = np.asarray(task.can.get_pose().p[:3]) - ee3[:3]
    drop_ee = np.concatenate([np.array([plate_world[0], plate_world[1], 0.90]) - offset2, ee3[3:]])
    task.move(task.move_to_pose(arm, drop_ee)); step("descend")
    if not task.plan_success: task.close_env(); return 0
    task.move(task.open_gripper(arm, pos=0.7)); step("open0.7a")
    task.move(task.open_gripper(arm, pos=0.7)); step("open0.7b")
    task.move(task.open_gripper(arm, pos=1.0)); step("open1.0")
    ee4 = np.asarray(task.get_arm_pose(arm))
    task.move(task.move_to_pose(arm, np.array([ee4[0], ee4[1], ee4[2] + 0.15, *ee4[3:]]))); step("raise")
    ok = task.left_return_home(); step("return_home")
    task.move(task.open_gripper(arm, pos=1.0)); step("final_open")
    print(f"[plan] check_success={task.check_success()} can={np.round(task.can.get_pose().p,3)}", flush=True)
    task.close_env()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
