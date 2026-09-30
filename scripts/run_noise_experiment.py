"""Run the approved noise-protocol conditions sequentially and maintain docs/_exp.md.

Supports the v0-noise batch (DP vanilla, default) and the emix-noise batch
(EMIX-only, selected via --status-prefix emix-noise). Each batch gets its own
docs/_exp.md status marker block, supervisor lock, and run config pair
(configs/scripts/run_<prefix with _ separators>_<condition>.yaml).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run as flow


ROOT = flow.ROOT
CONDITIONS = ("tn_ec", "tn_en")


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def protocol_snapshot(config: dict) -> dict:
    """Extract the protocol-relevant config fields for cross-checking.

    Field-level replacement for the former source-file sha256 gate (user
    decision 2026-09-30): the experiment contract is defined by parameter
    values, not by code content hashes.
    """
    training = flow.require_mapping(config, "training")
    pipeline = flow.require_mapping(config, "pipeline")
    evaluation = flow.require_mapping(config, "evaluation")
    return {
        "tasks": list(config["tasks"]),
        "policies": [kind for kind, _ in flow.selected_policies(config)],
        "episodes_per_task": pipeline["episodes_per_task"],
        "image_width": pipeline["image_width"],
        "image_height": pipeline["image_height"],
        "evaluation_episodes": pipeline["evaluation_episodes"],
        "evaluation_seed": pipeline["evaluation_seed"],
        "training": {key: training[key] for key in sorted(training)},
        "evaluation_image_noise": evaluation.get("image_noise"),
        "evaluation_image_noise_seed": evaluation.get("image_noise_seed"),
        "emix": config.get("emix"),
        "train_root": config["paths"]["train_root"],
        "eval_root": config["paths"]["eval_root"],
        "dataset_root": config["paths"]["dataset_root"],
        "index_root": config["paths"].get("index_root"),
    }


def dataset_summary(dataset: Path) -> dict:
    """Field-level dataset identity: counts and sizes, no content hashing."""
    files = [path for folder in ("meta", "data", "videos")
             for path in (dataset / folder).rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "episode_files": len([path for path in files if path.suffix == ".parquet"]),
    }


def collect_results(configs: dict, emix: bool) -> dict:
    results = {}
    for condition, config in configs.items():
        rows = {}
        for task in config["tasks"]:
            metrics = sorted(flow.eval_output_dir(task, config, emix=emix).glob("*/metrics.json"))
            result = None
            if metrics:
                try:
                    result = json.loads(metrics[-1].read_text())
                except json.JSONDecodeError:
                    pass  # The evaluator may still be writing its final metrics.
            rows[task] = {
                "trained": flow.completed_training(task, config, emix=emix),
                "metrics": str(metrics[-1]) if metrics else None,
                "result": result,
            }
        results[condition] = rows
    return results


def record(state: dict, configs: dict, output: Path, emix: bool,
           status_start: str, status_end: str, model_label: str) -> None:
    state["updated_at"] = timestamp()
    state["results"] = collect_results(configs, emix)
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    temporary.replace(output / "status.json")
    lines = [status_start, "### 自动运行状态", "",
             f"- 状态：`{state['status']}`；当前条件：`{state.get('condition', 'preflight')}`。",
             f"- 更新：{state['updated_at']}；开始：{state['started_at']}。",
             f"- 运行目录：`{output.relative_to(ROOT)}`；监督进程 PID：{state['pid']}。",
             f"- 状态与完整结果：`{(output / 'status.json').relative_to(ROOT)}`。",
             "- 按用户定稿顺序自动执行：单侧训练/评估 → 双侧复用训练/评估。", "",
             f"| 任务 | 训练完成 | 单侧 {model_label} SR | 双侧 {model_label} SR |",
             "|---|---|---:|---:|"]
    for task in configs["tn_ec"]["tasks"]:
        def score(condition: str) -> str:
            result = state["results"][condition][task]["result"]
            return f"{result['successes']}/{result['episodes']} ({100 * result['success_rate']:.1f}%)" if result else "待完成"
        trained = state["results"]["tn_ec"][task]["trained"]
        lines.append(f"| {task} | {'是' if trained else '否'} | {score('tn_ec')} | {score('tn_en')} |")
    if state.get("error"):
        lines += ["", f"- 异常：{state['error']}"]
    lines += [status_end]
    doc = ROOT / "docs/_exp.md"
    content = doc.read_text()
    section = "\n".join(lines)
    if status_start in content:
        start = content.index(status_start)
        end = content.index(status_end, start) + len(status_end)
        content = content[:start] + section + content[end:]
    else:
        content += "\n\n" + section + "\n"
    doc.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status-prefix", default="v0-noise",
        help="Protocol family prefix selecting the run config pair, the docs/_exp.md "
             "status marker block, and the supervisor lock. Default keeps the "
             "original v0-noise behavior; use 'emix-noise' for the EMIX batch.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_start = f"<!-- {args.status_prefix}-run-status:start -->"
    status_end = f"<!-- {args.status_prefix}-run-status:end -->"
    config_stem = "run_" + args.status_prefix.replace("-", "_")
    with (ROOT / "logs" / f".{args.status_prefix}-suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "status.json").exists():
            raise FileExistsError(f"Run status already exists: {output}; use a fresh supervisor directory")
        configs, snapshots = {}, {}
        for condition in CONDITIONS:
            source = ROOT / f"configs/scripts/{config_stem}_{condition}.yaml"
            target = output / source.name
            shutil.copy2(source, target)
            snapshots[condition] = target
            configs[condition] = flow.load_config(target)
        first, second = (configs[key] for key in CONDITIONS)
        assert first["tasks"] == second["tasks"] and len(first["tasks"]) == 5
        assert first["training"] == second["training"]
        assert first["paths"]["train_root"] == second["paths"]["train_root"]
        assert first["paths"]["eval_root"] != second["paths"]["eval_root"]
        policies = flow.selected_policies(first)
        assert len(policies) == 1 and policies == flow.selected_policies(second), (
            f"noise batch must select exactly one policy kind, got: {policies}"
        )
        emix = policies[0][1]
        model_label = "EMIX" if emix else "DP"
        assert not first["evaluation"]["image_noise"]["enabled"]
        assert second["evaluation"]["image_noise"]["enabled"]
        import os
        state = {"status": "preflight", "started_at": timestamp(), "pid": os.getpid(),
                 "protocol_prefix": args.status_prefix, "policies": [kind for kind, _ in policies],
                 "execution_order": list(CONDITIONS), "stages": []}
        record(state, configs, output, emix, status_start, status_end, model_label)
        try:
            sources = list((ROOT / "embodied_mixup").rglob("*.py")) + [
                ROOT / "evaluation/evaluate.py", ROOT / "scripts/run.py",
                ROOT / "scripts/oidn_env.py", ROOT / "scripts/run_noise_experiment.py",
            ]
            field_snapshots = {condition: protocol_snapshot(configs[condition]) for condition in CONDITIONS}
            provenance = {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "robotwin_commit": subprocess.check_output(["git", "-C", "third_party/RoboTwin", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "protocol_fields": field_snapshots, "python": sys.executable,
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "lerobot", "diffusers", "numpy")},
                "gpu_runtime": {},
                "datasets": {},
            }
            for task in first["tasks"]:
                _, dataset, _ = flow.task_paths(task, first)
                if not flow.completed_dataset(dataset, 500, 84, 84):
                    raise ValueError(f"Dataset does not meet the approved 500ep/84x84 contract: {dataset}")
                provenance["datasets"][task] = dataset_summary(dataset)
            for gpu in flow.configured_cuda(first):
                probe = "import json, torch; print(json.dumps({'name':torch.cuda.get_device_name(0), 'capability':torch.cuda.get_device_capability(0), 'torch':torch.__version__}))"
                detected = json.loads(subprocess.check_output(
                    [sys.executable, "-c", probe], cwd=ROOT,
                    env=flow.command_env(first, gpu), text=True,
                ))
                detected["training_environment"] = flow.per_gpu_environment(first, gpu, "training")
                detected["evaluation_environment"] = flow.per_gpu_environment(first, gpu, "evaluation")
                provenance["gpu_runtime"][gpu] = detected
            (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
            (output / "working-tree.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
            for path in sources:
                target = output / "source_snapshot" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            flow.validate(first, flow.configured_cuda(first))
            for condition in CONDITIONS:
                if protocol_snapshot(configs[condition]) != field_snapshots[condition]:
                    raise RuntimeError(f"Protocol fields changed during the experiment: {condition}")
                config = configs[condition]
                state.update(status="running", condition=condition)
                print(f"[{timestamp()}] Starting {condition}", flush=True)
                logfile = output / f"{condition}-train-eval.log"
                command = [sys.executable, "-u", str(ROOT / "scripts/run.py"), "--config", str(snapshots[condition]), "--phase", "train-eval"]
                with logfile.open("w") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    state["child_pid"] = process.pid
                    record(state, configs, output, emix, status_start, status_end, model_label)
                    while process.poll() is None:
                        time.sleep(30)
                        record(state, configs, output, emix, status_start, status_end, model_label)
                state["stages"].append({"condition": condition, "exit_code": process.returncode, "finished_at": timestamp(), "log": str(logfile)})
                if process.returncode:
                    raise RuntimeError(f"{condition} failed with exit {process.returncode}; see {logfile}")
                results = collect_results(configs, emix)[condition]
                for task, row in results.items():
                    result = row["result"]
                    if not row["trained"] or result is None or result["episodes"] != 100:
                        raise RuntimeError(f"Incomplete formal result: {condition}/{task}")
                    if result["image_noise"] != config["evaluation"]["image_noise"]:
                        raise RuntimeError(f"Noise metadata mismatch: {condition}/{task}")
                flow.write_results_summary(config)
                record(state, configs, output, emix, status_start, status_end, model_label)
                print(f"[{timestamp()}] Completed {condition}", flush=True)
            state.update(status="complete", finished_at=timestamp())
            record(state, configs, output, emix, status_start, status_end, model_label)
            print(f"[{timestamp()}] Experiment complete", flush=True)
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=timestamp())
            record(state, configs, output, emix, status_start, status_end, model_label)
            raise


if __name__ == "__main__":
    main()
