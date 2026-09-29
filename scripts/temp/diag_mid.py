"""Temporary: probe can rolling per model id at a fixed lying spawn."""
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien
ROOT = Path("/home/rogerspyke/projects/EMIX/Embodied-Mixup")
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))
envs = importlib.import_module("envs.place_can_basket_left")
task = getattr(envs, "place_can_basket_left")()
with open(BENCHMARK / "task_config" / "place_can_basket_left.yml", "r", encoding="utf-8") as h:
    config = yaml.safe_load(h)
config["task_name"] = "place_can_basket_left"
config["task_config"] = "place_can_basket_left"
config["eval_mode"] = True
config["left_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
config["right_robot_file"] = str(BENCHMARK / "assets" / "embodiments" / "piper")
config["dual_arm_embodied"] = False
config["embodiment_dis"] = 0.60
with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml", "r", encoding="utf-8") as h:
    config["left_embodiment_config"] = yaml.safe_load(h)
    config["right_embodiment_config"] = yaml.safe_load(h)
task.setup_demo(now_ep_num=7, seed=7, is_test=True, **config)
from envs.place_can_basket_left_impl import _actor_aabb
from envs.utils import create_actor
for mid in [0, 1, 2, 3, 5, 6]:
    act = create_actor(scene=task, pose=sapien.Pose([-0.22, 0.05, 0.741], [1.0,0,0,0]),
                       modelname="071_can", model_id=mid, convex=True)
    lo, _ = _actor_aabb(act)
    p = list(act.get_pose().p)
    if lo < 0.741 - 0.001:
        p[2] += 0.741 - lo
    act.actor.set_pose(sapien.Pose(p, [1.0,0,0,0]))
    for _ in range(60): task.scene.step()
    x0 = np.asarray(act.get_pose().p)
    for _ in range(2500): task.scene.step()
    task.scene.update_render()
    x1 = np.asarray(act.get_pose().p)
    d = np.linalg.norm(x1[:2]-x0[:2])
    print(f"[r] model={mid} moved={d:.4f} rest=({x1[0]:.2f},{x1[1]:.2f},{x1[2]:.3f}) {'ROLL' if d>0.02 else 'ok'}", flush=True)
    act.actor.set_pose(sapien.Pose([-0.22, 0.05, 0.741], [1.0,0,0,0]))
    for _ in range(60): task.scene.step()
task.close_env()
