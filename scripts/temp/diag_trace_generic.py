"""Temporary diagnostic: trace every move for one task/seed, print failure point.
Not a deliverable.  Usage:
    python scripts/temp/diag_trace_generic.py <task> [seed]
"""

from __future__ import annotations

import importlib
import os
import sys
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


def describe(acts):
    descs = []
    for a in acts or []:
        if getattr(a, "action", None) == "move":
            tp = getattr(a, "target_pose", None)
            tp = np.round(np.array(tp), 3).tolist() if tp is not None else None
            descs.append(f"move{tp}")
        else:
            descs.append(getattr(a, "action", "?"))
    return descs


def main() -> int:
    task_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    task, config = build(task_name)
    orig_move = task.move

    def traced_move(g1, g2=None, save_freq=-1):
        tag = g1[0] if isinstance(g1, (tuple, list)) and len(g1) > 0 else "?"
        acts = g1[1] if isinstance(g1, (tuple, list)) and len(g1) > 1 else g1
        before = bool(task.plan_success)
        result = orig_move(g1, g2, save_freq=save_freq)
        after = bool(task.plan_success)
        line = f"[tr] seed={seed} tag={tag} acts={describe(acts)} " \
               f"plan {before}->{after}"
        try:
            ee = np.round(np.asarray(task.robot.get_left_ee_pose(), dtype=np.float64)[:3], 3)
            line += f" ee={ee}"
            blocks = []
            for bname in ("block1", "block2", "block3", "can1", "can2",
                          "bread", "hamburg", "frenchfries", "can", "mug",
                          "bowl1", "bowl2", "shoe1", "shoe2"):
                if hasattr(task, bname):
                    try:
                        b = getattr(task, bname)
                        if isinstance(b, list):
                            bb = [np.round(np.asarray(x.get_pose().p, dtype=np.float64)[:3], 3)
                                  for x in b]
                            blocks.append(f"{bname}={bb}")
                        else:
                            blocks.append(f"{bname}={np.round(np.asarray(b.get_pose().p, dtype=np.float64)[:3], 3)}")
                    except Exception:  # noqa: BLE001
                        pass
            if blocks:
                line += " " + " ".join(blocks)
        except Exception:  # noqa: BLE001
            pass
        print(line, flush=True)
        return result

    task.move = traced_move
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
        layout = getattr(task, "sampled_layout_summary", {})
        print(f"[tr] seed={seed} layout={layout}", flush=True)
        task.play_once()
        print(f"[tr] seed={seed} final plan={task.plan_success} "
              f"succ={bool(task.check_success())} take_action={task.take_action_cnt}", flush=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
