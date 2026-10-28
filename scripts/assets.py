#!/usr/bin/env python3
"""Build RoboTwin task assets with GPU workers pulling from one shared queue.

The selectable stages are ``collect``, ``convert``, ``index``, and ``vis``.
Every GPU worker claims the next unclaimed task from a single shared work
list and runs it serially, so slow tasks (hours-long collections) never idle
another card the way a static round-robin split did.  One simulator still
runs per card at any moment, and the command stays safe to resume: a
complete stage is reused and incomplete stage output is removed before retry.

The ``vis`` stage writes its artifacts under ``--output-root`` (default
``outputs/assets_vis``, one subfolder per artifact type) so batch pilots stay
isolated from the flow's ``outputs/intersections`` visualizations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Iterable

import oidn_env


ROOT = Path(__file__).resolve().parents[1]
# Both environments are installed beneath the detected Conda root.
EMIX = oidn_env.CONDA_ENVS / "emix" / "bin" / "python"
# Canonical public-name suffix per centred-arm camera variant.
VARIANT_SUFFIXES = {
    "cen_arm_right_wide_cam": "_left_cen_arm_right_wide_cam",
    "cen_arm_near_side_cam": "_left_cen_arm_near_side_cam",
    "cen_arm_side_cam": "_left_cen_arm_side_cam",
    "cen_arm_front_cam": "_left_cen_arm_front_cam",
    "cen_arm_top_cam": "_left_cen_arm_top_cam",
}
ACTIVE_FAMILIES = (
    "blocks_ranking_rgb", "blocks_ranking_size", "place_bread_basket",
    "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_dual_shoes", "stack_blocks_two",
    "stack_bowls_two",
)
ALL_STAGES = ("collect", "convert", "index", "vis")


def task_name(family: str, task_suffix: str) -> str:
    return family if family.endswith(task_suffix) else f"{family}{task_suffix}"


def collection_path(name: str) -> Path:
    return ROOT / "data" / "robotwin" / name / name


def dataset_path(name: str) -> Path:
    return ROOT / "data" / "lerobot" / name


def index_path(name: str) -> Path:
    return ROOT / "data" / "emix_index" / name


def complete_collection(path: Path, episodes: int) -> bool:
    return all((path / "data" / f"episode{index}.hdf5").is_file() for index in range(episodes))


def complete_dataset(path: Path, episodes: int, width: int, height: int) -> bool:
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    image = info.get("features", {}).get("observation.images.head_camera", {})
    return (
        info.get("total_episodes") == episodes
        and image.get("shape") == [height, width, 3]
        and any((path / "data").glob("**/*.parquet"))
        and any((path / "videos").glob("**/*.mp4"))
    )


def stage_environment(gpu: str) -> dict[str, str]:
    # Blackwell cards use the shared emix-twin runtime, with stale cuBLASLt
    # stripped and the activate.d hook mirrored by scripts/oidn_env.py.
    return oidn_env.stage_environment(gpu)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[assets] $ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def remove_incomplete(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def process_task(name: str, gpu: str, stages: set[str], args: argparse.Namespace) -> None:
    collection = collection_path(name)
    dataset = dataset_path(name)
    index = index_path(name)
    gpu_env = stage_environment(gpu)
    if "collect" in stages and not complete_collection(collection, args.episodes):
        remove_incomplete(collection)
        twin = oidn_env.twin_python(gpu)
        run([
            str(twin), "data_collection/robotwin.py", name,
            "--task-config", name, "--episodes", str(args.episodes),
            "--python", str(twin), "--cuda", gpu,
        ], env=gpu_env)
        if not complete_collection(collection, args.episodes):
            raise RuntimeError(f"{name}: collection did not produce {args.episodes} episodes")
    if "convert" in stages and not complete_dataset(dataset, args.episodes, args.image_width, args.image_height):
        remove_incomplete(dataset)
        run([
            str(EMIX), "data_processing/convert_to_lerobot.py",
            "--src", str(collection), "--tgt", str(dataset), "--repo-id", name,
            "--image-width", str(args.image_width), "--image-height", str(args.image_height),
            "--fps", "17", "--left-qpos-dim", "7",
        ], env=gpu_env)
        if not complete_dataset(dataset, args.episodes, args.image_width, args.image_height):
            raise RuntimeError(f"{name}: conversion output is incomplete")
    if "index" in stages and not (index / "emix_index.json").is_file():
        remove_incomplete(index)
        run([str(EMIX), "-m", "emix", "build", "--root", str(dataset), "--repo_id", name, "--output_dir", str(index)])
    if "vis" in stages:
        # Batch pilots keep their visualizations under the assets output root
        # (default outputs/assets_vis) so they never mix with the flow's
        # outputs/intersections artifacts or other batches.  Each artifact
        # type gets its own subfolder, mirroring the per-script defaults.
        vis_root = ROOT / args.output_root
        output = vis_root / "intersections" / f"{name}.html"
        overlay = vis_root / "first_frame" / f"{name}_first_frame_range.png"
        pick_place = vis_root / "pick_place" / f"{name}_pick_place.html"
        pick_place_png = pick_place.with_suffix(".png")
        # Complete visualizations are reused so reruns do not rewrite 13 MB
        # HTML files that a dev server (Live Server) may be watching.
        vis_done = (
            output.is_file() and output.stat().st_size > 0
            and overlay.is_file()
            and pick_place.is_file() and pick_place_png.is_file()
        )
        if not vis_done:
            run([str(EMIX), "-m", "emix", "visualize", "--repo_id", name, "--root", str(dataset), "--index", str(index / "emix_index.json"), "--output", str(output), "--skip-click-images"])
            # Occupancy heatmap over every episode's first frame so the object
            # placement range is visible without opening the HTML.
            run([
                str(EMIX), "-m", "embodied_mixup.visualization.first_frame_average",
                "--root", str(dataset),
                "--output", str(overlay),
            ])
            # Gripper pick/place transition points over the 3D trajectories.
            run([
                str(EMIX), "-m", "embodied_mixup.visualization.plot_actions",
                "--root", str(dataset),
                "--output", str(pick_place),
            ])


def run_gpu_worker(gpu: str, work: list[str], lock: threading.Lock,
                   stages: set[str], args: argparse.Namespace) -> None:
    """Pull tasks off the shared work list until it is empty.

    A static round-robin split left one GPU idle whenever its assigned tasks
    finished early (task durations differ by 10x: place_cans_plasticbox took
    hours, blocks_ranking minutes), so workers now claim tasks dynamically.
    One simulator still runs per card at any moment because each worker is
    strictly serial between claims.
    """
    while True:
        with lock:
            if not work:
                return
            name = work.pop(0)
        print(f"[assets] gpu={gpu} claimed task={name} "
              f"(remaining={len(work)}) stages={sorted(stages)}", flush=True)
        process_task(name, gpu, stages, args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(ACTIVE_FAMILIES))
    parser.add_argument("--variant", choices=sorted(VARIANT_SUFFIXES),
                        default="cen_arm_right_wide_cam",
                        help="centred-arm camera variant selecting the task suffix")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES))
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--output-root", default="outputs/assets_vis",
                        help="root folder for this batch's vis artifacts "
                             "(each artifact type in its own subfolder)")
    args = parser.parse_args()
    if args.episodes < 1 or not args.gpus:
        raise ValueError("episodes must be positive and at least one GPU is required")
    task_suffix = VARIANT_SUFFIXES[args.variant]
    names = [task_name(family, task_suffix) for family in args.tasks]
    invalid = [name for name in names if name.removesuffix(task_suffix) not in ACTIVE_FAMILIES]
    if invalid:
        raise ValueError(f"unsupported tasks: {', '.join(invalid)}")
    work = list(names)
    lock = threading.Lock()
    print(f"[assets] shared queue over gpus={args.gpus}: {work}", flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(run_gpu_worker, gpu, work, lock,
                                   set(args.stages), args)
                   for gpu in args.gpus]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001
                failures.append(str(error))
    if failures:
        raise RuntimeError("assets pipeline failed:\n" + "\n".join(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
