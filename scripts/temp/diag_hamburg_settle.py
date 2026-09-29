"""Temporary: hamburg settle displacement per model and spawn z.

Reuse one environment; swap the hamburg actor model and teleport across the
manifest range to measure settle rolling.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))

TASK_NAME = "place_burger_fries_left"


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
    from envs.place_can_basket_left_impl import _actor_aabb
    from envs.utils.create_actor import create_actor
    task, config = build()
    task.setup_demo(now_ep_num=9, seed=9, is_test=True, **config)
    for _ in range(500):
        task.scene.step()
    task.scene.update_render()
    for raise_z in (0.0, 0.04):
        for model_id in (0, 1, 2, 3, 4, 5):
            # swap the hamburg actor for the target model, resting on the table
            task.close_env()
            task, config = build()
            task.setup_demo(now_ep_num=9, seed=9, is_test=True, **config)
            task.scene.remove_actor(task.hamburg.actor)
            task.hamburg = create_actor(scene=task, pose=sapien.Pose([-0.40, 0.05, 0.741 + raise_z], [0.5, 0.5, 0.5, 0.5]),
                                        modelname="006_hamburg", convex=True, model_id=model_id)
            task.hamburg.set_mass(0.05)
            worst = 0.0
            for cx in (-0.44, -0.40, -0.36):
                for cy in (0.02, 0.05, 0.08):
                    task.hamburg.actor.set_pose(sapien.Pose([cx, cy, 0.741 + raise_z], [0.5, 0.5, 0.5, 0.5]))
                    for _ in range(60):
                        task.scene.step()
                    x0 = np.asarray(task.hamburg.get_pose().p)
                    for _ in range(1200):
                        task.scene.step()
                    task.scene.update_render()
                    x1 = np.asarray(task.hamburg.get_pose().p)
                    disp = float(np.hypot(x1[0] - cx, x1[1] - cy))
                    worst = max(worst, disp)
                    print(f"[hs] raise={raise_z} model={model_id} spawn=({cx:.2f},{cy:.2f}) "
                          f"disp={disp*100:.1f}cm rest=({x1[0]:.2f},{x1[1]:.2f})", flush=True)
            print(f"[hs] raise={raise_z} model={model_id} WORST={worst*100:.1f}cm", flush=True)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
