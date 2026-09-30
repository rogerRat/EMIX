"""Run the frozen V1 data, collage, overlay and noise experiment sequence.

Each stage is a normal scripts/run.py invocation and can be rerun safely.
Status and a readable supervisor log live under outputs/v1_experiment_queue.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v1_experiment_queue"
STATUS = OUTPUT / "status.json"
STAGES = (
    ("collect_convert_500ep_all10", "run_v1_cen_arm_top_cam_500ep.yaml", "prepare"),
    ("index_visualize_all10", "run_v1_cen_arm_top_cam_500ep.yaml", "index"),
    ("collage_s5", "run_v1_collage_s5.yaml", None),
    ("collage_all10", "run_v1_collage_all10.yaml", None),
    ("overlay_s5_control", "run_v1_overlay_s5.yaml", None),
    ("overlay_all10_control", "run_v1_overlay_all10.yaml", None),
    ("noise_s5_train_noisy_eval_clean", "run_v1_noise_s5_clean.yaml", None),
    ("noise_s5_train_noisy_eval_noisy", "run_v1_noise_s5_noisy.yaml", "train-eval"),
    ("noise_all10_train_noisy_eval_clean", "run_v1_noise_all10_clean.yaml", None),
    ("noise_all10_train_noisy_eval_noisy", "run_v1_noise_all10_noisy.yaml", "train-eval"),
)


def save(state):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {
        "started_at": now(), "stages": {}, "sequence": [stage[0] for stage in STAGES]
    }
    state["pid"] = os.getpid()
    state["status"] = "running"
    save(state)
    with (OUTPUT / "supervisor.log").open("a", encoding="utf-8") as log:
        for name, filename, phase in STAGES:
            if state["stages"].get(name, {}).get("status") == "complete":
                continue
            command = [sys.executable, "scripts/run.py", "--config",
                       f"configs/scripts/{filename}"]
            if phase:
                command.extend(("--phase", phase))
            stage = state["stages"].setdefault(name, {})
            for attempt in (1, 2):
                stage.update(status="running", started_at=now(), attempt=attempt,
                             command=command)
                state["current_stage"] = name
                save(state)
                log.write(f"\n[{now()}] {name} attempt {attempt}: {' '.join(command)}\n")
                log.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
                stage.update(exit_code=result.returncode, ended_at=now())
                if result.returncode == 0:
                    stage["status"] = "complete"
                    save(state)
                    break
                stage["status"] = "retrying" if attempt == 1 else "failed"
                save(state)
            if stage["status"] != "complete":
                state["status"] = "failed"
                save(state)
                raise SystemExit(f"V1 stage failed twice: {name}; see {OUTPUT / 'supervisor.log'}")
        state["status"] = "complete"
        state["ended_at"] = now()
        state.pop("current_stage", None)
        save(state)


if __name__ == "__main__":
    main()
