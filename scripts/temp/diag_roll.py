"""Temporary: probe can rolling during settle across the spawn range.
For a grid of spawn points, spawn the lying can (identity quat + bottom rest)
and measure total x/y displacement + quat drift over 2500 settle steps.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml, sapien

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
    from envs.place_can_basket_left_impl import _actor_aabb
    from envs.utils.action import ArmTag
    task.setup_demo(now_ep_num=7, seed=7, is_test=True, **config)  # any stable seed to init env
    bp = np.asarray(task.basket.get_pose().p)
    # settle the basket once
    for _ in range(500): task.scene.step()
    task.scene.update_render()
    print(f"[roll] basket at {np.round(bp,3)}", flush=True)
    for cx in [-0.25, -0.235, -0.22, -0.205]:
        for cy in [0.0, 0.03, 0.06, 0.09]:
            q0 = task.can.get_pose().q
            task.can.actor.set_pose(sapien.Pose([cx, cy, 0.741], [1.0,0.0,0.0,0.0]))
            lo, _ = _actor_aabb(task.can)
            p = list(task.can.get_pose().p)
            if lo < 0.741 - 0.001:
                p[2] += 0.741 - lo
            task.can.actor.set_pose(sapien.Pose(p, [1.0,0.0,0.0,0.0]))
            # give a clean start
            for _ in range(60): task.scene.step()
            x0 = np.asarray(task.can.get_pose().p)
            qa = task.can.get_pose().q
            for _ in range(2500): task.scene.step()
            task.scene.update_render()
            x1 = np.asarray(task.can.get_pose().p)
            qb = task.can.get_pose().q
            # quat drift in degrees
            d = float(2*np.arccos(min(1.0, abs(np.dot(qa, qb))))*180/np.pi)
            dx = x1[0]-x0[0]; dy = x1[1]-x0[1]
            flag = "ROLL" if dx*dx+dy*dy > 0.0004 or d > 3.0 else "ok"
            print(f"[roll] spawn=({cx:.3f},{cy:.2f}) dx={dx:+.3f} dy={dy:+.3f} dz={x1[2]-x0[2]:+.3f} "
                  f"quat_drift={d:.1f} -> {flag} rest=({x1[0]:.2f},{x1[1]:.2f})", flush=True)
            # reset can velocity: teleport and settle next iteration
    task.close_env()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
