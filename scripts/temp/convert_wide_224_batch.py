#!/usr/bin/env python
"""Convert all 10 collected central_wide_cam 10ep collections to LeRobot 224x224."""
from __future__ import annotations
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/rogerspyke/projects/EMIX/Embodied-Mixup")
PY = "/home/rogerspyke/miniforge3/envs/emix/bin/python"
CONV = ROOT / "data_processing/convert_to_lerobot.py"
LOG = ROOT / "logs"

TASKS = [
    "blocks_ranking_rgb_left_central_wide_cam",
    "blocks_ranking_size_left_central_wide_cam",
    "place_bread_basket_left_central_wide_cam",
    "place_bread_skillet_left_central_wide_cam",
    "place_burger_fries_left_central_wide_cam",
    "place_can_basket_left_central_wide_cam",
    "place_cans_plasticbox_left_central_wide_cam",
    "place_dual_shoes_left_central_wide_cam",
    "stack_blocks_two_left_central_wide_cam",
    "stack_bowls_two_left_central_wide_cam",
]

def convert(task: str) -> int:
    src = ROOT / "data/robotwin" / task / f"{task}_10ep"
    tgt = ROOT / "data/lerobot" / f"{task}_10ep_224"
    if (tgt / "meta/info.json").exists():
        print(f"SKIP {task}: already converted", flush=True)
        return 0
    cmd = [PY, str(CONV), "--src", str(src), "--tgt", str(tgt),
           "--repo-id", f"{task}_10ep_224", "--instruction-type", "seen",
           "--image-width", "224", "--image-height", "224",
           "--fps", "17", "--left-qpos-dim", "7"]
    logf = (LOG / f"convert_{task}_224.log").open("w")
    print(f"[convert] {task}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
    logf.close()
    return r.returncode

def main() -> int:
    LOG.mkdir(exist_ok=True)
    queue = [t for t in TASKS if not (ROOT/"data/lerobot"/f"{t}_10ep_224/meta/info.json").exists()]
    if not queue:
        print("All already converted.")
        return 0
    fails = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(convert, t): t for t in queue}
        for fut in as_completed(futs):
            t = futs[fut]
            rc = fut.result()
            print(f"[result] {t}: exit {rc}", flush=True)
            if rc != 0:
                fails.append(t)
    if fails:
        print(f"FAILED: {fails}", flush=True)
        return 1
    print("All conversions done.", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
