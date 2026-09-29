"""Temporary diagnostic: print functional points and get_place_pose math for
one task/seed.  Not a deliverable.
Usage:
    python scripts/temp/diag_placemath.py <task> [seed]
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


def build(task_name: str):
    sys.path.insert(0, str(BENCHMARK))
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
    with open(BENCHMARK / "assets" / "embodiments" / "piper" / "config.yml",
              "r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    return task, config


def main() -> int:
    task_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    task, config = build(task_name)
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        print(f"[pm] layout={getattr(task, 'sampled_layout_summary', {})}", flush=True)
        target_holder = getattr(task, "skillet", None) or getattr(task, "breadbasket", None) \
            or getattr(task, "basket", None) or getattr(task, "plasticbox", None) \
            or getattr(task, "tray", None) or getattr(task, "shoe_box", None)
        movable = getattr(task, "bread", None)
        if isinstance(movable, list):
            movable = movable[0] if movable else None
        if movable is None:
            movable = getattr(task, "can", None)
        if movable is None:
            movable = getattr(task, "hamburg", None)
        print(f"[pm] target_holder={type(target_holder).__name__ if target_holder else None} "
              f"movable={type(movable).__name__ if movable else None}", flush=True)
        for obj, name in ((target_holder, "target"), (movable, "movable")):
            if obj is None:
                continue
            p = np.round(np.array(obj.get_pose().p), 3)
            q = np.round(np.array(obj.get_pose().q), 3)
            print(f"[pm] {name} pos={p} quat={q}", flush=True)
            try:
                fp = obj.get_functional_point(0)
                print(f"[pm] {name} fp0={np.round(np.array(fp), 3)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[pm] {name} fp0 EXC {type(exc).__name__}: {exc}", flush=True)
        # current left EE
        print(f"[pm] ee={np.round(np.array(task.robot.get_left_ee_pose()), 3)}", flush=True)
        from envs.utils.action import ArmTag
        # perform grasp + lift so the bread is held, then recompute place pose
        if movable is not None:
            task.move(task.grasp_actor(movable, arm_tag=ArmTag("left"), pre_grasp_dis=0.07))
            print(f"[pm] after grasp plan={task.plan_success} "
                  f"ee={np.round(np.array(task.robot.get_left_ee_pose()), 3)}", flush=True)
            print(f"[pm] bread pos={np.round(np.array(movable.get_pose().p), 3)} "
                  f"quat={np.round(np.array(movable.get_pose().q), 3)}", flush=True)
            task.move_by_displacement(arm_tag=ArmTag("left"), z=0.1)
            print(f"[pm] after lift plan={task.plan_success} "
                  f"ee={np.round(np.array(task.robot.get_left_ee_pose()), 3)}", flush=True)
            print(f"[pm] bread pos={np.round(np.array(movable.get_pose().p), 3)} "
                  f"quat={np.round(np.array(movable.get_pose().q), 3)}", flush=True)
        if movable is not None and target_holder is not None:
            fp = target_holder.get_functional_point(0)
            print(f"[pm] target fp0={np.round(np.array(fp), 3)}", flush=True)
            import transforms3d as t3d
            place = task.get_place_pose(movable, "left", fp, constrain="free",
                                        pre_dis=0.05)
            print(f"[pm] get_place_pose free = {np.round(np.array(place), 3)}", flush=True)
            # replicate the math to find the garbage source
            actor_matrix = movable.get_pose().to_transformation_matrix()
            place_start_pose = movable.get_pose()
            actor2world = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).T
            from envs.utils import transforms as tr
            pp = tr.get_place_pose(place_start_pose, fp, constrain="free", z_transform=True)
            pp_mat = tr._toPose(pp).to_transformation_matrix()
            ps_mat = place_start_pose.to_transformation_matrix()
            start2target = pp_mat[:3, :3] @ ps_mat[:3, :3].T
            print(f"[pm] start2target={np.round(start2target, 3)}", flush=True)
            ee = np.array(task.robot.get_left_ee_pose(), dtype=np.float64)
            ee_mat = t3d.quaternions.quat2mat(ee[3:])
            tgm = start2target @ ee_mat
            print(f"[pm] target_grasp_matrix rot={np.round(tgm[:3, :3], 3)}", flush=True)
            offset = actor_matrix[:3, 3] - ee[:3]
            bias_local = np.linalg.inv(ee_mat) @ offset
            grasp_bias = tgm @ bias_local
            print(f"[pm] offset={np.round(offset, 3)} bias_local={np.round(bias_local, 3)} "
                  f"grasp_bias={np.round(grasp_bias, 3)}", flush=True)
            dis_vec = grasp_bias / (np.linalg.norm(grasp_bias) + 1e-12)
            res_pos = (np.array(pp[:3]) - grasp_bias - 0.05 * dis_vec)
            print(f"[pm] pp_pos={np.round(np.array(pp[:3]), 3)} res_pos={np.round(res_pos, 3)}", flush=True)
            # try planning the place move with and without position-only constraint
            from envs.utils.action import Action
            for label, cpose in (("full", None),
                                 ("posonly", [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])):
                act = Action(ArmTag("left"), "move", target_pose=list(place),
                             constraint_pose=cpose)
                before = bool(task.plan_success)
                task.move((ArmTag("left"), [act]))
                print(f"[pm] place move {label} plan {before}->{task.plan_success}", flush=True)
                task.plan_success = before
            # reachability sweep from current lift config
            base = np.array(task.robot.get_left_ee_pose(), dtype=np.float64)
            for xx in (-0.35, -0.30, -0.25, -0.20, -0.15, -0.10):
                target = base.copy()
                target[0] = xx
                act = Action(ArmTag("left"), "move", target_pose=target.tolist(),
                             constraint_pose=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
                before = bool(task.plan_success)
                task.move((ArmTag("left"), [act]))
                print(f"[pm] reach x={xx} posonly plan {before}->{task.plan_success}", flush=True)
                task.plan_success = before
                if task.plan_success:
                    task.plan_success = before
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
