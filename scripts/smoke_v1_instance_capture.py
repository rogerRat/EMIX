"""Check V1 RGB/actor-ID capture and task-object naming for selected tasks."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "third_party" / "RoboTwin"


def check(task_name: str):
    module = importlib.import_module(f"envs.left.cen_arm_top_cam.{task_name}")
    task = getattr(module, task_name)()
    config = yaml.safe_load((BENCHMARK / "task_config/left/cen_arm_top_cam" / f"{task_name}.yml").read_text())
    robot = yaml.safe_load((BENCHMARK / "assets/embodiments/piper/config.yml").read_text())
    config.update(task_name=task_name, task_config=task_name, eval_mode=True,
                  left_robot_file=str(BENCHMARK / "assets/embodiments/piper"),
                  right_robot_file=str(BENCHMARK / "assets/embodiments/piper"),
                  left_embodiment_config=robot, right_embodiment_config=robot,
                  dual_arm_embodied=False, embodiment_dis=0.6)
    with tempfile.TemporaryDirectory() as temporary:
        config.update(save_data=True, save_path=temporary)
        for seed in range(9700, 9720):
            try:
                task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
                break
            except Exception:
                task.close_env()
        else:
            raise RuntimeError(f"No valid initial scene for {task_name}")
        try:
            task._take_picture()
            task._take_picture()
            task.merge_pkl_to_hdf5_video()
            with h5py.File(Path(temporary) / "data/episode0.hdf5") as handle:
                rgb = handle["observation/head_camera/rgb"]
                ids = handle["observation/head_camera/actor_ids"]
                if len(rgb) != len(ids) or ids.ndim != 3:
                    raise ValueError("RGB/actor-ID frame shape mismatch")
                frame_count = len(rgb)
            names = json.loads((Path(temporary) / "data/episode0.actor_names.json").read_text())
            task_names = sorted(name for name in names.values() if name.startswith("task:"))
            if set(task_names) != {f"task:{name}" for name in task.manifest.actor_specs}:
                raise ValueError(f"Task object map mismatch: {task_names}")
            print(f"[smoke] {task_name}: {frame_count} frames, {task_names}", flush=True)
        finally:
            task.close_env()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args()
    os.chdir(BENCHMARK)
    sys.path.insert(0, str(BENCHMARK))
    for task in args.tasks:
        check(task)


if __name__ == "__main__":
    main()
