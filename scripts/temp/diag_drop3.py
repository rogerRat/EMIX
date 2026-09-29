"""Temporary diagnostic: track can orientation through descent and release.
Not a deliverable.  Usage: python scripts/temp/diag_drop3.py <seed>
Prints can center + collision AABB at each phase to find when the lying can
flips to standing.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

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
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    return task, config


def state(task, tag: str) -> None:
    from envs.place_can_basket_left_impl import _actor_aabb
    p = np.asarray(task.can.get_pose().p)
    q = np.asarray(task.can.get_pose().q)
    lo, hi = _actor_aabb(task.can)
    print(f"[d3] {tag}: can={np.round(p, 3)} h={hi - lo:.3f} aabb=[{lo:.3f},{hi:.3f}] "
          f"q={np.round(q, 2)}", flush=True)


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task, config = build()
    from envs.utils.action import ArmTag
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        print(f"[d3] seed={seed} can_model={task.can_id} basket={task.basket_id}", flush=True)
        task.plan_success = True
        arm_tag = ArmTag("left")
        task.move(task.grasp_actor(task.can, arm_tag=arm_tag, pre_grasp_dis=0.05))
        state(task, "post-grasp")
        if not task.plan_success:
            print("[d3] grasp failed", flush=True)
            return 0
        ee = np.asarray(task.robot.get_left_ee_pose())
        basket_p = task.basket.get_pose().p
        rim = float(getattr(task, "basket_rim_top", basket_p[2] + 0.17))
        off = np.asarray(task.can.get_pose().p[:3]) - ee[:3]
        hover = ee.copy()
        hover[:2] = basket_p[:2] - off[:2]
        hover[2] = max(ee[2] + 0.20, rim + 0.17)
        task.move(task.move_to_pose(arm_tag, hover))
        state(task, "hover")
        print(f"[d3] hover plan={task.plan_success}", flush=True)
        if not task.plan_success:
            return 0
        # Descent target: can centre level with the interior floor (fp z 0.85).
        floor = 0.85
        low_z = floor + abs(off[2]) + 0.055  # compensate the observed ~0.055 detour shortfall
        low = hover.copy()
        low[2] = low_z
        task.move(task.move_to_pose(arm_tag, low))
        state(task, "descent-target")
        print(f"[d3] descend plan={task.plan_success} ee={np.round(np.asarray(task.robot.get_left_ee_pose())[:3], 3)}", flush=True)
        if not task.plan_success:
            return 0
        # Release and watch the orientation step by step.
        task.move(task.open_gripper(arm_tag, pos=0.7))
        state(task, "after-open-0.7")
        task.move(task.open_gripper(arm_tag, pos=0.7))
        state(task, "after-hold")
        for k in range(20):
            task.scene.step()
        state(task, "after-20-step")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
