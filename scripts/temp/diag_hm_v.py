"""Temporary: replicate validate_left_task.run_seed for one seed and inspect
plan_success after setup and after play_once, to find why validate sees
plan_failures while the equivalent manual flow succeeds.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "third_party" / "RoboTwin"
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(ROOT / "scripts"))

TASK = "hanging_mug_left"
SEED = 0


def main() -> int:
    import validate_left_task as vlt
    cfg = vlt._load_task_config(TASK, BENCHMARK)
    envs = importlib.import_module(f"envs.{TASK}")
    task_cls = getattr(envs, TASK)
    env = task_cls()
    try:
        env.setup_demo(now_ep_num=SEED, seed=SEED, is_test=True, **cfg)
        print(f"[V] setup plan_success={env.plan_success}", flush=True)
        print(f"[V] layout={env.sampled_layout_summary}", flush=True)
        rng_before = int(np.random.randint(0, 2 ** 31 - 1))
        env.play_once()
        rng_after = int(np.random.randint(0, 2 ** 31 - 1))
        print(f"[V] play_once plan_success={env.plan_success} "
              f"check={env.check_success()} rng_shift={rng_before}-{rng_after}", flush=True)
    finally:
        env.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
