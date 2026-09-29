"""Temporary diagnostic: at the post-place retreat, try many retreat targets
from the exact post-place state and report which plan.  Not a deliverable.
Usage:
    python scripts/temp/diag_retreat.py <task> [seed]
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import transforms3d as t3d
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
    from envs.utils.action import ArmTag, Action

    orig_retreat = task.left_retreat

    def try_target(label: str, pose7, constraint=None):
        act = Action(ArmTag("left"), "move", target_pose=list(pose7),
                     constraint_pose=constraint) if constraint else \
            Action(ArmTag("left"), "move", target_pose=list(pose7))
        before = bool(task.plan_success)
        task.move((ArmTag("left"), [act]))
        ok = bool(task.plan_success)
        print(f"[rt] {label:32s} -> plan {before}->{ok}", flush=True)
        if ok:
            # move executed; restore plan marker for subsequent probes
            task.plan_success = True
        else:
            task.plan_success = before
        return ok

    def probe(tag: str, z: float, axis: str, pos_only: bool):
        pose = np.asarray(task.robot.get_left_ee_pose(), dtype=np.float64)
        target = pose.copy()
        if axis == "world":
            target[:3] += np.asarray([0.0, 0.0, z])
        else:
            d = t3d.quaternions.quat2mat(pose[3:])[:, 0]
            d /= np.linalg.norm(d)
            target[:3] += z * d
        constraint = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0] if pos_only else None
        label = f"{tag} z={z} {axis} posonly={int(pos_only)}"
        return try_target(label, target, constraint)

    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        print(f"[rt] seed={seed} layout={getattr(task, 'sampled_layout_summary', {})}", flush=True)
        # instrument: at every left_retreat call, run the battery once then
        # use the best working retreat and continue normally.
        def patched_retreat(arm_tag, z=0.12):
            print(f"[rt] == left_retreat called z={z}, current ee="
                  f"{np.round(task.robot.get_left_ee_pose(), 3)}", flush=True)
            task.plan_success = True
            results = {}
            for axis in ("world", "arm"):
                for zz in (0.15, 0.10, 0.06):
                    results[(axis, zz)] = probe("full", zz, axis, False)
            print(f"[rt] -- full-SE3 world/arm probes: {results}", flush=True)
            pos_res = {}
            for axis in ("world", "arm"):
                for zz in (0.15, 0.10, 0.06):
                    pos_res[(axis, zz)] = probe("pos", zz, axis, True)
            print(f"[rt] -- position-only world/arm probes: {pos_res}", flush=True)
            # back to origin
            try_target("back_to_origin", list(task.robot.left_original_pose))
            # pick a working option and commit
            best = next(((axis, zz) for axis in ("world", "arm")
                         for zz in (0.06, 0.10, 0.15) if results[(axis, zz)]), None)
            if best is None:
                best = next(((axis, zz) for axis in ("world", "arm")
                             for zz in (0.06, 0.10, 0.15) if pos_res[(axis, zz)]), None)
            print(f"[rt] == chosen retreat option: {best}", flush=True)
            if best is None:
                task.plan_success = False
            return task.plan_success

        task.left_retreat = patched_retreat
        task.play_once()
        print(f"[rt] final plan={task.plan_success} succ={bool(task.check_success())} "
              f"take_action={task.take_action_cnt}", flush=True)
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
