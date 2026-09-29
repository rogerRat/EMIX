"""Temporary: per-model can rolling during settle across the spawn grid.

For each candidate model_id, spawn the lying can (identity quat, bottom rested)
at a grid of points and measure xy displacement + quat drift over 2500 steps.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

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
    from envs.place_can_basket_left_impl import _actor_aabb
    task, config = build()
    task.setup_demo(now_ep_num=9, seed=9, is_test=True, **config)  # stable init
    for _ in range(500):
        task.scene.step()
    task.scene.update_render()
    for model_id in (1, 2, 3, 5, 6):
        # recreate the can with a forced model id
        task.close_env()
        task, config = build()
        task.setup_demo(now_ep_num=9, seed=9, is_test=True, **config)
        can_pose = task.can.get_pose()
        task.can_name = "071_can"
        from envs.utils.create_actor import create_actor
        # remove old can actor then spawn a fresh one with the target model
        task.scene.remove_actor(task.can.actor)
        task.can = create_actor(scene=task, pose=can_pose, modelname="071_can",
                                model_id=model_id, convex=True)
        lo, _ = _actor_aabb(task.can)
        p = list(task.can.get_pose().p)
        if lo < 0.741 - 0.001:
            p[2] += 0.741 - lo
        task.can.actor.set_pose(sapien.Pose(p, [1.0, 0.0, 0.0, 0.0]))
        task.can.set_mass(0.01)
        for _ in range(60):
            task.scene.step()
        task.scene.update_render()
        worst = 0.0
        for cx in (-0.25, -0.235, -0.22, -0.205):
            for cy in (0.0, 0.03, 0.06, 0.09):
                task.can.actor.set_pose(sapien.Pose([cx, cy, 0.741], [1.0, 0.0, 0.0, 0.0]))
                lo, _ = _actor_aabb(task.can)
                p = list(task.can.get_pose().p)
                if lo < 0.741 - 0.001:
                    p[2] += 0.741 - lo
                task.can.actor.set_pose(sapien.Pose(p, [1.0, 0.0, 0.0, 0.0]))
                for _ in range(60):
                    task.scene.step()
                x0 = np.asarray(task.can.get_pose().p)
                qa = task.can.get_pose().q
                for _ in range(2500):
                    task.scene.step()
                task.scene.update_render()
                x1 = np.asarray(task.can.get_pose().p)
                qb = task.can.get_pose().q
                d = float(2 * np.arccos(min(1.0, abs(np.dot(qa, qb)))) * 180 / np.pi)
                disp = float(np.hypot(x1[0] - x0[0], x1[1] - x0[1]))
                worst = max(worst, d, disp * 100.0)
                flag = "ROLL" if disp > 0.02 or d > 3.0 else "ok"
                print(f"[cmg] model={model_id} spawn=({cx:.3f},{cy:.2f}) "
                      f"disp={disp*100:.1f}cm drift={d:.1f}deg rest=({x1[0]:.2f},{x1[1]:.2f}) {flag}",
                      flush=True)
        print(f"[cmg] model={model_id} WORST={worst:.2f}", flush=True)
    task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
