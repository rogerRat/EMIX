"""Temporary diagnostic: why do box grasp poses fail left_plan_multi_path
for blocks_ranking_rgb_left seed 0?  Not a deliverable.
Usage:
    python scripts/temp/diag_block_grasp.py
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


def main() -> int:
    task_name = "blocks_ranking_rgb_left"
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
    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **config)
        qpos = np.round(task.robot.left_entity.get_qpos(), 4)
        print(f"[bg] left_qpos={qpos}", flush=True)
        # comparison: a bowl at block1's position
        from envs.utils import create_actor
        from envs.utils.action import ArmTag
        bowl = create_actor(scene=task, pose=task.block1.get_pose(), modelname="002_bowl",
                            model_id=3, convex=True)
        task.scene.step()
        try:
            acts = task.grasp_actor(bowl, arm_tag=ArmTag("left"), contact_point_id=0,
                                    pre_grasp_dis=0.1)
            print(f"[bg] bowl-grasp@block1 acts={'OK' if acts else 'None'} "
                  f"plan={task.plan_success}", flush=True)
            import transforms3d as t3d
            bgp = task.get_grasp_pose(bowl, "left", pre_dis=0.1, contact_point_id=0)
            print(f"[bg] bowl cp0 pre_pose_pos={np.round(np.array(bgp[:3]),3)} "
                  f"quat={np.round(np.array(bgp[-4:]),3)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[bg] bowl-grasp@block1 EXC {type(exc).__name__}: {exc}", flush=True)
        blk = task.block1
        cx, cy = blk.get_pose().p[:2]
        # top-down grasp quat from GRASP_DIRECTION_DIC
        topdown = [-0.5, 0.5, -0.5, -0.5]
        for h in (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10):
            pos = [cx, cy, h]
            lst = task.robot.create_target_pose_list(pos + topdown, blk.get_pose(), "left")
            traj = task.robot.left_plan_multi_path(lst)
            ok = sum(1 for s in traj["status"] if s == "Success")
            print(f"[bg] manual-topdown h={h} pos={pos} plan_ok={ok}/{len(lst)}", flush=True)
        # box candidate quats for cp0 vs working top-down
        blk = task.block1
        import transforms3d as t3d
        cm = blk.get_contact_point(0, "matrix")
        gcm = cm @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        gq = gcm[:3, :3]
        gpos = gcm[:3, 3] + gq @ np.array([-0.12 - 0.09, 0, 0]).T
        res_pose = list(gpos) + list(t3d.quaternions.mat2quat(gq))
        lst = task.robot.create_target_pose_list(res_pose, blk.get_contact_point(0, "list"), "left")
        for j, cand in enumerate(lst):
            print(f"[bg] box cp0 cand{j} pos={np.round(np.array(cand[:3]),3)} "
                  f"quat={np.round(np.array(cand[-4:]),3)}", flush=True)
        print(f"[bg] box quat={np.round(np.array(blk.get_pose().q),3)}", flush=True)
        print(f"[bg] manual top_down quat={[-0.5,0.5,-0.5,-0.5]}", flush=True)
        for name in ("block1", "block2", "block3"):
            blk = getattr(task, name)
            p = np.round(blk.get_pose().p, 3)
            print(f"[bg] {name} pos={p}", flush=True)
            for i, _ in blk.iter_contact_points():
                import transforms3d as t3d
                contact_matrix = blk.get_contact_point(i, "matrix")
                if contact_matrix is None:
                    print(f"[bg]   cp={i} no contact matrix", flush=True)
                    continue
                gcm = contact_matrix @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0],
                                                 [0, -1, 0, 0], [0, 0, 0, 1]])
                gq = gcm[:3, :3]
                gpos = gcm[:3, 3] + gq @ np.array([-0.12 - 0.09, 0, 0]).T
                gquat = t3d.quaternions.mat2quat(gq)
                res_pose = list(gpos) + list(gquat)
                lst = task.robot.create_target_pose_list(res_pose, blk.get_contact_point(i, "list"), "left")
                print(f"[bg]   cp={i} pre_pose_pos={np.round(np.array(res_pose[:3]), 3)} "
                      f"quat={np.round(np.array(res_pose[-4:]), 3)}", flush=True)
                traj = task.robot.left_plan_multi_path(lst)
                print(f"[bg]   cp={i} plan_status={list(traj['status'])}", flush=True)
                for j, s in enumerate(traj["status"]):
                    if s == "Success":
                        print(f"[bg]   cp={i} cand{j} pos={np.round(np.array(lst[j][:3]), 3)} OK", flush=True)
                    else:
                        print(f"[bg]   cp={i} cand{j} pos={np.round(np.array(lst[j][:3]), 3)} {s}", flush=True)
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
