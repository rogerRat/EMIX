"""Temporary: for each validate seed, run the full impl flow with a FIXED
mug model_id so we can separate layout noise from model geometry.  We report
per-seed result for model 0 (the sweep-verified geometry) across 30 layouts.
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
MODEL_ID = 0


def main() -> int:
    import validate_left_task as vlt
    cfg = vlt._load_task_config(TASK, BENCHMARK)
    envs = importlib.import_module(f"envs.{TASK}")
    task_cls = getattr(envs, TASK)
    ok = 0
    for seed in range(30):
        env = task_cls()
        try:
            env.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **cfg)
            env.mug_id = MODEL_ID
            env.play_once()
            s = bool(env.plan_success) and bool(env.check_success())
            ok += int(s)
            mug_p = np.asarray(env.mug.get_pose().p)
            print(f"[F] seed={seed} ok={s} plan={env.plan_success} "
                  f"check={env.check_success()} mugz={mug_p[2]:.3f}", flush=True)
        finally:
            env.close_env()
    print(f"[F] model={MODEL_ID} rate={ok}/30 = {ok/30:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
