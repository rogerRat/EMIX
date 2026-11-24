"""Report seed overlap and paired outcomes for multi-v0 evaluations."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs/multi-v0/eval"


def records(arm: str, task: str):
    directory = ROOT / arm / task
    metrics = sorted(directory.glob("*/metrics.json")) if directory.is_dir() else []
    if metrics:
        payload = json.loads(metrics[-1].read_text(encoding="utf-8"))
        return payload["records"], True
    resume = directory / ".eval_resume.json"
    if resume.is_file():
        payload = json.loads(resume.read_text(encoding="utf-8"))
        return payload["records"], False
    return [], False


def main() -> None:
    baseline_root = ROOT / "smolvla-baseline"
    print("arm\ttask\tcomplete\tbaseline_n\tarm_n\tcommon_n\tbaseline_common_success\tarm_common_success\tarm_only_wins\tbaseline_only_wins")
    for task_root in sorted(baseline_root.iterdir()):
        if not task_root.is_dir():
            continue
        task = task_root.name
        baseline, _ = records("smolvla-baseline", task)
        base = {int(row["seed"]): bool(row["success"]) for row in baseline}
        for arm in ("emix-mix", "emix-patch"):
            tested, complete = records(arm, task)
            other = {int(row["seed"]): bool(row["success"]) for row in tested}
            common = set(base) & set(other)
            print("\t".join(map(str, (
                arm, task, complete, len(base), len(other), len(common),
                sum(base[seed] for seed in common),
                sum(other[seed] for seed in common),
                sum(other[seed] and not base[seed] for seed in common),
                sum(base[seed] and not other[seed] for seed in common),
            ))))


if __name__ == "__main__":
    main()
