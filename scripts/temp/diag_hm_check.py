"""Temporary: for several seeds, run play_once and dump the settle-state
mug/rack geometry and each check_success sub-predicate, so we can see why the
geometric check fails on the validate path while the sweep said 0.60.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
os.chdir(BENCHMARK)
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(ROOT / "scripts"))

TASK = "hanging_mug_left"
SEEDS = [0, 1, 2, 3, 4, 7, 26]


def main() -> int:
    import validate_left_task as vlt
    cfg = vlt._load_task_config(TASK, BENCHMARK)
    envs = importlib.import_module(f"envs.{TASK}")
    task_cls = getattr(envs, TASK)
    for seed in SEEDS:
        env = task_cls()
        try:
            env.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **cfg)
            env.play_once()
            mug_fp = np.asarray(env.mug.get_functional_point(0))[:3]
            mug_p = np.asarray(env.mug.get_pose().p)
            rack_p = np.asarray(env.rack.get_pose().p)
            rack_fp = np.asarray(env.rack.get_functional_point(0))[:3]
            mid = (rack_p + rack_fp) / 2
            eps = 0.02
            dxy = np.abs((mug_fp - mid)[:2])
            ok_xy = bool(np.all(dxy < eps))
            ok_h = bool(mug_fp[2] > 0.86)
            ok_l = env.is_left_gripper_open()
            ok_r = env.right_arm_stationary()
            print(f"[D] seed={seed} ok={env.check_success()} "
                  f"mug={np.round(mug_p,3)} mugfp={np.round(mug_fp,3)} "
                  f"mid={np.round(mid,3)} dxy={np.round(dxy,3)} "
                  f"[xy={ok_xy} h={ok_h} l={ok_l} r={ok_r}]", flush=True)
        finally:
            env.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
