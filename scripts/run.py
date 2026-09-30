"""Run the fixed two-GPU RoboTwin left-arm DP/EMIX experiment matrix.

Phases (enforced global barriers, in order): prepare -> index -> vis ->
train-eval. The ``vis`` phase renders every task with a complete EmixIndex into
``outputs/intersections/<task>.html`` and compiles it to a crisp
``outputs/intersections/<task>.png`` via headless Chrome
(``scripts/render_vis_png.py``), so clustering can be
inspected before training starts.
The mandatory ``pre_vis`` stage saves the four V0 inspection views under
each task/policy train and evaluation root before train-eval. Only the
explicit ``--skip-pre-vis`` CLI option bypasses it.

Resume behavior (simplest form, no strict validation):
    The train-eval phase reuses finished work instead of blindly re-running
    every stage. For each task and each policy kind (baseline / emix):

      * a train stage is skipped when its final checkpoint exists
        (``outputs/train/<task>_dp_<kind>/checkpoints/<steps:06d>/pretrained_model/model.safetensors``);
      * an eval stage is skipped when ``metrics.json`` exists under
        ``outputs/eval/<task>_dp_<kind>`` AND that training checkpoint still
        exists (otherwise the metrics are stale and get re-run).

    An *incomplete* artifact is removed first so the stage can restart cleanly
    (``train.py`` refuses a non-empty output_dir with ``FileExistsError``).
    This makes re-running a crashed flow cheap: finished training/eval is not
    repeated, and only the missing stages run. It is intentionally a coarse
    "artifact present" check -- no hashing or content validation.

Configuration is read from ``configs/scripts/run.yaml`` by default; pass
``--config`` to point the flow at another runbook (e.g. a per-variant batch
config). Run a single phase over existing data (e.g. to backfill
visualizations after all datasets/indexes already exist) with:
    python scripts/run.py --phase vis
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import shutil
import subprocess
from typing import Any

import yaml

import oidn_env


ROOT = Path(__file__).resolve().parents[1]
FLOW_CONFIG = ROOT / "configs" / "scripts" / "run.yaml"
ACTIVE_COMMAND: subprocess.Popen[str] | None = None
DEFAULT_LOG_ROTATE_BYTES = 50 * 1024 * 1024
# The checked-in run.yaml only carries operator-tuned values; these defaults
# are filled in by normalize_config so every flow-driven run keeps the debug
# captures on and the policy image list stable.
DEFAULT_ENCODER_DEBUG = {"enabled": True, "interval": 10000, "max_samples": 4}
DEFAULT_EMIX_DEBUG = {
    "enabled": True, "max_samples": 32, "every_n_steps": 10000,
    "image_keys": None, "max_action_steps": 1, "action_xyz_indices": [0, 1, 2],
    "action_is_delta": True, "min_display_fraction": 0.12,
}
DEFAULT_EMIX_VISUAL_KEYS = ["observation.images.head_camera"]


def load_config(config_path: Path = FLOW_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()

    def load_one(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        if path in stack:
            raise ValueError(f"Cyclic flow config inheritance: {path}")
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Flow config must be a mapping: {path}")
        parent = payload.pop("extends", None)
        if parent is None:
            return payload
        if not isinstance(parent, str) or not parent:
            raise ValueError("extends must name one parent YAML")
        base = load_one((path.parent / parent).resolve(), (*stack, path))

        def merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
            result = copy.deepcopy(a)
            for key, value in b.items():
                result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
            return result

        return merge(base, payload)

    config = load_one(config_path, ())
    if not isinstance(config, dict):
        raise ValueError("configs/scripts/run.yaml must contain a mapping")
    return normalize_config(resolve_config_references(config))


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Adapt the checked-in run.yaml schema to the internal config shape.

    The file keeps only what an operator tunes (project roots, ``gpu_batches``
    device assignment, pipeline/training/emix/visualization numbers). Derived
    data/output paths and always-on debug toggles are filled in here so the
    rest of the module reads one uniform config.
    """
    project = require_mapping(config, "project")
    robotwin_root = str(project.get("robotwin_root", "third_party/RoboTwin"))
    paths = {
        "robotwin_root": robotwin_root,
        "task_config_root": f"{robotwin_root}/task_config/left",
        "runtime_config_root": str(project.get("runtime_config_root", "configs/scripts/run/{task}")),
        "log_root": str(project.get("log_root", "logs")),
        # Per-batch overrides let several ablation batches coexist without
        # trading artifacts through the shared default roots (data/robotwin
        # stays shared because collections are reused verbatim).
        "collection_root": str(project.get("collection_root", "data/robotwin/{task}")),
        "dataset_root": str(project.get("dataset_root", "data/lerobot/{task}")),
        "index_root": str(project.get("index_root", "data/emix_index/{task}")),
        "train_root": str(project.get("train_root", "outputs/train/{task}")),
        "eval_root": str(project.get("eval_root", "outputs/eval/{task}")),
        "visualization_root": str(require_mapping(config, "visualization").get("output_root", "outputs/intersections")),
    }
    config["run_label"] = str(project.get("run_label", "batch"))
    gpu_batches = require_mapping(config, "gpu_batches")
    tasks: list[str] = []
    cuda: list[str] = []
    for device, device_tasks in gpu_batches.items():
        device = str(device)
        if device in cuda:
            raise ValueError(f"gpu_batches lists CUDA device {device} more than once")
        if not isinstance(device_tasks, list) or not device_tasks:
            raise ValueError(f"gpu_batches.{device} must be a non-empty task list")
        cuda.append(device)
        for task in device_tasks:
            if not isinstance(task, str) or not task:
                raise ValueError(f"gpu_batches.{device} contains an invalid task name: {task!r}")
            if task in tasks:
                raise ValueError(f"task {task} appears in more than one gpu_batches entry")
            tasks.append(task)
    pipeline = require_mapping(config, "pipeline")
    pipeline.setdefault("evaluation_device", "cuda")
    pipeline.setdefault("evaluation_video_fps", 10)
    # Visualization performance guardrails: every flow-driven render caps the
    # drawn cross-point groups (an uncapped 500k-group index bloats the HTML
    # to hundreds of MB and lags) and skips the click-to-inspect image
    # precompute unless a runbook explicitly turns it back on.
    visualization = require_mapping(config, "visualization")
    visualization.setdefault("max_points", 10000)
    visualization.setdefault("skip_click_images", True)
    require_mapping(config, "training").setdefault("encoder_debug", dict(DEFAULT_ENCODER_DEBUG))
    emix_cfg = require_mapping(config, "emix")
    emix_cfg.setdefault("visual_keys", list(DEFAULT_EMIX_VISUAL_KEYS))
    emix_cfg.setdefault("debug", dict(DEFAULT_EMIX_DEBUG))
    # Extra CLI flags forwarded to `emix build` in the index phase, e.g.
    # ["--exclude_start_radius", "0.02"]. Filters must also be recorded in the
    # batch SUMMARY; the built index keeps them in its config snapshot.
    emix_cfg.setdefault("build_extra_args", [])
    if not isinstance(emix_cfg["build_extra_args"], list):
        raise ValueError("emix.build_extra_args must be a list of CLI strings")
    config["paths"] = paths
    config["tasks"] = tasks
    config["cuda"] = cuda
    return config


def resolve_config_references(config: dict[str, Any]) -> dict[str, Any]:
    """Expand ${section.key} references in YAML scalar values."""
    import re

    pattern = re.compile(r"\$\{([^}]+)\}")

    def lookup(path: str) -> Any:
        value: Any = config
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"Unknown YAML variable: ${{{path}}}")
            value = value[part]
        return value

    def expand(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, str):
            return value
        matches = list(pattern.finditer(value))
        if len(matches) == 1 and matches[0].group(0) == value:
            return lookup(matches[0].group(1))
        return pattern.sub(lambda match: str(lookup(match.group(1))), value)

    expanded = expand(config)
    if not isinstance(expanded, dict):
        raise ValueError("Expanded flow configuration must contain a mapping")
    return expanded


def require_mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def configured_path(config: dict[str, Any], name: str, task: str | None = None) -> Path:
    paths = require_mapping(config, "paths")
    value = paths.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"paths.{name} must be a non-empty path")
    if task is not None:
        value = value.format(task=task)
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def task_paths(task: str, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    collection = configured_path(config, "collection_root", task) / task
    dataset = configured_path(config, "dataset_root", task)
    index = configured_path(config, "index_root", task)
    return collection, dataset, index


def average_frames_per_episode(dataset: Path) -> float:
    """Mean frames per episode for a LeRobot dataset (meta/info.json).

    A missing or unparsable info.json (fresh run before collection) sorts the
    task last rather than aborting the flow.
    """
    info_path = dataset / "meta" / "info.json"
    try:
        with info_path.open(encoding="utf-8") as handle:
            info = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return float("inf")
    if not isinstance(info, dict):
        return float("inf")
    try:
        total_episodes = int(info.get("total_episodes") or 0)
        total_frames = int(info.get("total_frames") or 0)
    except (TypeError, ValueError):
        return float("inf")
    if total_episodes <= 0:
        return float("inf")
    return total_frames / total_episodes


def task_priority(task: str, config: dict[str, Any]) -> float:
    """Sort key for a task by its converted dataset's mean frames per episode.

    Fewer frames per episode -> runs earlier in the serial queues.
    """
    _, dataset, _ = task_paths(task, config)
    return average_frames_per_episode(dataset)


def completed_collection(collection: Path, episodes: int) -> bool:
    """Return whether a native RoboTwin collection has every expected episode."""
    data_directory = collection / "data"
    if not data_directory.is_dir():
        return False
    return all(
        (data_directory / f"episode{episode_index}.hdf5").is_file()
        for episode_index in range(episodes)
    )


def completed_dataset(dataset: Path, episodes: int, width: int, height: int) -> bool:
    """Return whether a converted LeRobot dataset matches the flow contract."""
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        return False
    try:
        with info_path.open(encoding="utf-8") as handle:
            info = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(info, dict):
        return False
    camera = (info.get("features") or {}).get("observation.images.head_camera") or {}
    return (
        info.get("total_episodes") == episodes
        and camera.get("shape") == [height, width, 3]
        and any((dataset / "data").glob("**/*.parquet"))
        and any((dataset / "videos").glob("**/*.mp4"))
    )


def completed_index(index: Path) -> bool:
    """Return whether an EmixIndex output is parseable and structurally complete."""
    index_path = index / "emix_index.json"
    if not index_path.is_file():
        return False
    try:
        with index_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("public"), dict)
        and payload["public"].get("schema_version") == 2
        and isinstance(payload["public"].get("source_datasets"), list)
        and isinstance(payload.get("content"), dict)
        and isinstance(payload["content"].get("cross_point_groups"), list)
    )


def vis_output_paths(task: str, config: dict[str, Any]) -> tuple[Path, Path]:
    """HTML and PNG artifact paths for one task's visualization."""
    visualization = require_mapping(config, "visualization")
    root = configured_path(config, "visualization_root")
    html = root / f"{task}.html"
    return html, html.with_suffix(".png")


def completed_vis(html: Path, png: Path, task: str, *, require_assets: bool = True) -> bool:
    """Return whether a task's visualization is complete (HTML, PNG, assets).

    The ``<task>_*_assets`` directory only exists when click-to-inspect images
    were precomputed; a render made with ``skip_click_images`` produces none,
    so callers pass ``require_assets=False`` for those.
    """
    return (
        html.is_file()
        and html.stat().st_size > 0
        and png.is_file()
        and png.stat().st_size > 1024
        and (not require_assets or any(html.parent.glob(f"{task}_*_assets")))
    )


def remove_artifact(path: Path, description: str, log: RotatingLog | None = None) -> None:
    """Remove one known task artifact left by a failed pipeline stage."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    message = f"[flow] removed incomplete {description}: {path}\n"
    if log is not None:
        log.write(message)
    print(message, end="", flush=True)


def per_gpu_environment(config: dict[str, Any], gpu: str | None, environment_key: str) -> str:
    """Resolve a Conda environment name for one GPU, honoring per-GPU overrides.

    ``environments.per_gpu.<gpu>.<key>`` wins over the shared
    ``environments.<key>`` default, so each GPU card can pin its own compatible
    runtime explicitly. Below that, the RoboTwin runtime keys (``robotwin`` /
    ``evaluation``) are auto-routed by scripts/oidn_env.py: a Blackwell GPU
    (sm_120, whose SAPIEN OIDN plugin only exists in the 5060 variant) gets
    ``emix-twin-5060`` without any GPU-index pinning, because CUDA enumeration
    order differs from nvidia-smi and is not stable across cards.
    """
    environments = require_mapping(config, "environments")
    per_gpu = environments.get("per_gpu")
    if isinstance(per_gpu, dict):
        gpu_overrides = per_gpu.get(str(gpu))
        if isinstance(gpu_overrides, dict):
            override = gpu_overrides.get(environment_key)
            if isinstance(override, str) and override:
                return override
    environment_name = environments.get(environment_key)
    if not isinstance(environment_name, str) or not environment_name:
        raise ValueError(f"environments.{environment_key} must name a Conda environment")
    if environment_key in ("robotwin", "evaluation") and gpu is not None and oidn_env.is_blackwell(gpu):
        return oidn_env.EMIX_TWIN_5060_NAME
    return environment_name


def command_env(config: dict[str, Any], gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    rt_denoiser = require_mapping(config, "pipeline").get("rt_denoiser")
    if rt_denoiser is not None:
        if rt_denoiser not in ("oidn", "optix", "none"):
            raise ValueError("pipeline.rt_denoiser must be oidn, optix or none")
        env["ROBOTWIN_RT_DENOISER"] = rt_denoiser
    if require_mapping(config, "pipeline").get("collect_instance_ids", False):
        env["ROBOTWIN_COLLECT_INSTANCE_IDS"] = "1"
    spatial_preset = (config.get("protocol") or {}).get("spatial_preset")
    if spatial_preset is not None:
        if spatial_preset not in ("legacy", "v0"):
            raise ValueError("protocol.spatial_preset must be legacy or v0")
        env["ROBOTWIN_CENTERED_PROTOCOL"] = spatial_preset
    # Blackwell GPUs: strip the stale /usr/local/cuda cuBLASLt that the
    # inherited LD_LIBRARY_PATH would otherwise shadow over torch's bundled
    # 12.8 (detected per card in scripts/oidn_env.py). Every stage on the
    # card needs this, in either Conda environment.
    if gpu and oidn_env.is_blackwell(gpu):
        oidn_env.strip_stale_cuda(env)
    return env


def conda_command(
    config: dict[str, Any], environment_key: str, gpu: str | None = None, *command: str
) -> list[str]:
    """Wrap one command in its configured, isolated Conda environment.

    ``gpu`` selects the per-GPU environment override from ``environments.per_gpu``
    when one is configured; pass ``None`` to use the shared default.
    """
    environments = require_mapping(config, "environments")
    conda_executable = str(environments.get("conda_executable", ""))
    environment_name = per_gpu_environment(config, gpu, environment_key)
    if command and command[0] == "python":
        command = ("python", "-u", *command[1:])
    runtime_environment: dict[str, str] = {}
    environments = require_mapping(config, "environments")
    per_gpu = environments.get("per_gpu")
    if isinstance(per_gpu, dict):
        gpu_overrides = per_gpu.get(str(gpu))
        if isinstance(gpu_overrides, dict):
            configured_runtime = gpu_overrides.get("runtime_env")
            if isinstance(configured_runtime, dict):
                environment_runtime = configured_runtime.get(environment_key)
                if isinstance(environment_runtime, dict):
                    runtime_environment = {
                        str(key): str(value) for key, value in environment_runtime.items()
                    }
    runtime_prefix = ["env"]
    runtime_prefix.extend(
        f"{key}={value}" for key, value in runtime_environment.items()
    )
    if runtime_environment:
        command = (*runtime_prefix, *command)
    return [conda_executable, "run", "--no-capture-output", "-n", environment_name, *command]


def all_environment_names(config: dict[str, Any]) -> set[str]:
    """Collect every Conda env referenced by the shared and per-GPU mappings."""
    environments = require_mapping(config, "environments")
    # Env split is deliberate and must be preserved: RoboTwin data collection
    # and policy evaluation run in emix-twin (the only env with the vendored
    # curobo/RoboTwin runtime); data preprocessing and model training run in
    # emix. Do not merge evaluation back into `training`.
    keys = ("robotwin", "preprocessing", "training", "evaluation")
    names = {per_gpu_environment(config, None, key) for key in keys}
    per_gpu = environments.get("per_gpu")
    if isinstance(per_gpu, dict):
        for gpu_overrides in per_gpu.values():
            if isinstance(gpu_overrides, dict):
                names.update(
                    gpu_overrides[key]
                    for key in keys
                    if isinstance(gpu_overrides.get(key), str) and gpu_overrides.get(key)
                )
    return names


def validate_conda_environments(config: dict[str, Any]) -> None:
    """Verify each configured Conda environment before creating GPU workers."""
    environments = require_mapping(config, "environments")
    conda_executable = str(environments.get("conda_executable", ""))
    for environment_name in sorted(all_environment_names(config)):
        command = [conda_executable, "run", "--no-capture-output", "-n", environment_name, "python", "--version"]
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Configured Conda environment {environment_name!r} is unavailable."
            ) from error


def validate_worker_gpus(config: dict[str, Any], batches: dict[str, list[str]]) -> None:
    """Fail before collection when emix-twin cannot execute on a configured GPU."""
    benchmark = configured_path(config, "robotwin_root")
    for gpu, tasks in batches.items():
        task_name = tasks[0]
        probe = (
        "import torch, toppra, envs; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "device=torch.device('cuda:0'); "
        "result=(torch.ones(1, device=device) + 1).item(); "
        f"task=envs.import_task_env({task_name!r}); "
        "print(torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device), result, task.__name__)"
        )
        command = conda_command(config, "robotwin", gpu, "python", "-c", probe)
        try:
            subprocess.run(command, cwd=benchmark, env=command_env(config, gpu), check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"GPU {gpu} cannot initialize the RoboTwin runtime in environments.robotwin "
                f"({require_mapping(config, 'environments')['robotwin']!r}). "
                "Check the PyTorch/Curobo GPU build and the NumPy/TOPPRA ABI."
            ) from error


def validate_evaluation_runtime(config: dict[str, Any]) -> None:
    """Fail before training when the evaluation policy stack cannot import.

    The evaluation stage (``evaluation/evaluate.py``) imports
    ``lerobot.policies.diffusion.modeling_diffusion``, which transitively pulls
    in ``diffusers`` and ``huggingface_hub``.  A version drift there (e.g. a
    ``huggingface_hub`` too old for ``diffusers``) only surfaces at eval time,
    after an entire training run has completed — historically as
    ``ImportError: cannot import name 'DDUFEntry' from 'huggingface_hub'``.
    Probe the exact import now so the flow fails fast instead of after hours.
    """
    probe = (
        "from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder; "
        "print(DiffusionRgbEncoder.__name__)"
    )
    command = conda_command(config, "evaluation", None, "python", "-c", probe)
    try:
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip().splitlines()
        detail = "\n".join(stderr[-5:]) if stderr else str(error)
        raise RuntimeError(
            f"The evaluation policy stack cannot import in environments.evaluation "
            f"({per_gpu_environment(config, None, 'evaluation')!r}). "
            "Check that lerobot/diffusers and huggingface_hub are mutually "
            "compatible (diffusers >= 0.34 requires huggingface_hub >= 0.27).\n"
            f"{detail}"
        ) from error


def log_rotate_bytes(config: dict[str, Any]) -> int:
    """Rollover size for one worker log file, from `logging.rotate_bytes`.

    Falls back to DEFAULT_LOG_ROTATE_BYTES when the key is absent or is not a
    dict; any present value must be a positive integer.
    """
    logging_config = config.get("logging")
    if isinstance(logging_config, dict) and "rotate_bytes" in logging_config:
        rotate_bytes = logging_config["rotate_bytes"]
        if not isinstance(rotate_bytes, int) or rotate_bytes <= 0:
            raise ValueError("logging.rotate_bytes must be a positive integer")
        return rotate_bytes
    return DEFAULT_LOG_ROTATE_BYTES


class RotatingLog:
    """Append-only worker log that rolls to a new numbered file past a size limit.

    Files are named ``{worker:03d}-{sequence:04d}.log`` (``000-0000.log``,
    ``000-0001.log``, ...). The worker prefix is 0-based and the sequence starts
    at 0, incrementing each time the active file exceeds ``rotate_bytes`` so a
    single file never grows unreadably long.
    """

    def __init__(self, directory: Path, worker_index: int, rotate_bytes: int) -> None:
        self.directory = directory
        self.worker_index = worker_index
        self.rotate_bytes = rotate_bytes
        self.sequence = 0
        self._handle: Any | None = None
        self._size = 0

    @property
    def name(self) -> str:
        """Basename of the currently active log file."""
        return f"{self.worker_index:03d}-{self.sequence:04d}.log"

    def _path(self) -> Path:
        return self.directory / self.name

    def _open(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path()
        self._handle = path.open("a", encoding="utf-8")
        self._size = path.stat().st_size

    def write(self, text: str) -> None:
        if self._handle is None:
            self._open()
        assert self._handle is not None
        self._handle.write(text)
        self._handle.flush()
        self._size += len(text.encode("utf-8"))
        if self._size >= self.rotate_bytes:
            self.rotate()

    def rotate(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.sequence += 1
        self._open()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def run(
    command: list[str],
    *,
    gpu: str,
    log: RotatingLog,
    config: dict[str, Any],
    environment: dict[str, str] | None = None,
) -> None:
    global ACTIVE_COMMAND
    print(f"[gpu {gpu}] {' '.join(command)}", flush=True)
    log.write("$ " + " ".join(command) + "\n")
    ACTIVE_COMMAND = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment or command_env(config, gpu),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    try:
        assert ACTIVE_COMMAND.stdout is not None
        for line in ACTIVE_COMMAND.stdout:
            log.write(line)
            print(f"[gpu {gpu} | {log.name}] {line}", end="", flush=True)
        return_code = ACTIVE_COMMAND.wait()
    finally:
        ACTIVE_COMMAND = None
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def stop_active_command() -> None:
    """Stop the current command and every process in its session."""
    if ACTIVE_COMMAND is None or ACTIVE_COMMAND.poll() is not None:
        return
    try:
        os.killpg(ACTIVE_COMMAND.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        ACTIVE_COMMAND.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(ACTIVE_COMMAND.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def index_command_env(config: dict[str, Any]) -> dict[str, str]:
    """Give an index build the whole host CPU/RAM budget managed by this flow."""
    environment = command_env(config, "")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(os.cpu_count() or 1),
            "OPENBLAS_NUM_THREADS": str(os.cpu_count() or 1),
            "MKL_NUM_THREADS": str(os.cpu_count() or 1),
            "NUMEXPR_NUM_THREADS": str(os.cpu_count() or 1),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


@contextmanager
def flow_lock(lock_path: Path) -> Any:
    """Serialize flow instances so two experiments cannot consume host RAM."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        print(f"[flow] waiting for exclusive flow lock: {lock_path}", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print("[flow] acquired exclusive flow lock", flush=True)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def worker_signal_handler(signum: int, _frame: Any) -> None:
    """Ensure a terminated GPU worker does not leave simulators/trainers alive."""
    stop_active_command()
    raise SystemExit(128 + signum)


def train_config(task: str, config: dict[str, Any], *, emix: bool) -> dict[str, Any]:
    training = require_mapping(config, "training")
    _, dataset, index = task_paths(task, config)
    kind = "emix" if emix else "baseline"
    payload: dict[str, Any] = {
        key: training[key] for key in ("batch_size", "num_workers", "steps", "save_freq", "log_freq")
    }
    payload["output_dir"] = str(configured_path(config, "train_root", f"{task}_dp_{kind}"))
    payload["policy"] = {
        "type": "emix_diffusion" if emix else "diffusion",
        "vision_backbone": training["vision_backbone"],
        "pretrained_backbone_weights": training["pretrained_backbone_weights"],
        "use_group_norm": False,
        "optimizer_lr": training["optimizer_lr"],
        "push_to_hub": False,
        "horizon": training["horizon"],
        "n_obs_steps": training["n_obs_steps"],
        "n_action_steps": training["n_action_steps"],
    }
    # Explicitly freeze Diffusion defaults for versioned experiment protocols.
    for key in (
        "crop_shape", "crop_is_random", "down_dims", "kernel_size", "n_groups",
        "diffusion_step_embed_dim", "num_train_timesteps", "num_inference_steps",
        "noise_scheduler_type", "beta_schedule", "prediction_type", "clip_sample",
        "optimizer_betas", "optimizer_eps", "optimizer_weight_decay",
        "scheduler_name", "scheduler_warmup_steps",
    ):
        if key in training:
            payload["policy"][key] = copy.deepcopy(training[key])
    payload["dataset"] = {"repo_id": task, "root": str(dataset)}
    if "protocol" in config:
        payload["protocol"] = copy.deepcopy(config["protocol"])
    if "image_noise" in training:
        payload["image_noise"] = copy.deepcopy(training["image_noise"])
    # Every flow-driven training run (baseline and emix) must capture encoder
    # debug images: the DiffusionRgbEncoder periodically writes input images and
    # intermediate feature maps so training health is inspectable.
    payload["encoder_debug"] = {
        **require_mapping(training, "encoder_debug"),
        "output_dir": str(configured_path(config, "train_root", f"{task}_dp_{kind}") / "encoder_debug"),
    }
    if emix:
        emix_cfg = require_mapping(config, "emix")
        payload["emix"] = {
            "index_path": str(index / "emix_index.json"),
            # EMIX has its own training-time snapshot debug (mixed inputs plus
            # EEF action direction plots). Must be on for every flow-driven
            # emix run; vis_n_obs_steps/max_obs_steps are CLI-only and must not
            # appear here.
            "debug": {
                **require_mapping(emix_cfg, "debug"),
                "output_dir": str(configured_path(config, "train_root", f"{task}_dp_{kind}") / "debug"),
            },
            "mixup": {
                "alpha_mode": emix_cfg["alpha_mode"], "beta": emix_cfg["beta"],
                "alpha_value": None, "loss_weight": emix_cfg["loss_weight"],
                "visual_keys": emix_cfg["visual_keys"], "state_mode": emix_cfg["state_mode"],
                "method": emix_cfg.get("method", "overlay"),
            },
            "training": {"enabled": True, "seed": training["seed"], "training_mode": "mixed", "scheduler": copy.deepcopy(emix_cfg["scheduler"])},
        }
    return payload


def eval_config(task: str, config: dict[str, Any], *, emix: bool) -> dict[str, Any]:
    pipeline = require_mapping(config, "pipeline")
    training = require_mapping(config, "training")
    _, dataset, _ = task_paths(task, config)
    kind = "emix" if emix else "baseline"
    payload = {
        "dataset": {"repo_id": task, "root": str(dataset)},
        "evaluation": {
            "checkpoint": str(configured_path(config, "train_root", f"{task}_dp_{kind}") / "checkpoints" / f"{int(training['steps']):06d}" / "pretrained_model"),
            "output_dir": str(configured_path(config, "eval_root", f"{task}_dp_{kind}")),
            "benchmark_root": str(configured_path(config, "robotwin_root")), "task_name": task, "task_config": task,
            "piper_distance": pipeline["piper_distance"], "episodes": pipeline["evaluation_episodes"],
            "seed": pipeline["evaluation_seed"], "device": pipeline["evaluation_device"],
            "save_video": pipeline["save_evaluation_video"], "video_fps": pipeline["evaluation_video_fps"],
        },
    }
    if "protocol" in config:
        payload["protocol"] = copy.deepcopy(config["protocol"])
    evaluation = config.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    for key in ("image_noise", "image_noise_seed"):
        if key in evaluation:
            payload["evaluation"][key] = copy.deepcopy(evaluation[key])
    return payload


def write_runtime_configs(task: str, config: dict[str, Any]) -> dict[str, Path]:
    root = configured_path(config, "runtime_config_root", task)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for kind, enabled in (("baseline", False), ("emix", True)):
        for stage, payload in (("train", train_config(task, config, emix=enabled)), ("eval", eval_config(task, config, emix=enabled))):
            path = root / f"{kind}_{stage}.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
            paths[f"{kind}_{stage}"] = path
    return paths


def run_prepare_task(
    task: str,
    gpu: str,
    config: dict[str, Any],
    *,
    task_index: int,
    task_total: int,
    log: RotatingLog,
) -> None:
    pipeline = require_mapping(config, "pipeline")
    collection, dataset, _ = task_paths(task, config)
    episodes = int(pipeline["episodes_per_task"])
    collection_ready = completed_collection(collection, episodes)
    if collection_ready and pipeline.get("collect_instance_ids", False):
        collection_ready = all(
            (collection / "data" / f"episode{index}.actor_names.json").is_file()
            for index in range(episodes)
        )
    if collection.exists() and not collection_ready:
        remove_artifact(collection, "RoboTwin collection")
    dataset_ready = completed_dataset(
        dataset, episodes, int(pipeline["image_width"]), int(pipeline["image_height"])
    )
    if dataset_ready and pipeline.get("collect_instance_ids", False):
        dataset_ready = all(
            (dataset / "annotations" / "head_camera" / f"episode{index}.hdf5").is_file()
            for index in range(episodes)
        )
    if dataset.exists() and not dataset_ready:
        remove_artifact(dataset, "LeRobot dataset")
    stages: list[tuple[str, list[str]]] = []
    # Collect only when the converted dataset is missing too: a complete
    # LeRobot dataset is the artifact every later stage consumes, so a reused
    # dataset (e.g. restored from an archive) must not trigger a re-collection
    # just because the raw hdf5 collections are not on disk.
    if not dataset_ready and not collection_ready:
        # The collector itself appends <task>/<task_config>; task_paths above
        # expects one final <task> after the configured collection root.
        stages.append(("collect", conda_command(config, "robotwin", gpu, "python", "data_collection/robotwin.py", task, "--task-config", task, "--cuda", gpu, "--output-root", str(configured_path(config, "collection_root", task).parent))))
    if not dataset_ready:
        convert_command = conda_command(config, "preprocessing", gpu, "python", "data_processing/convert_to_lerobot.py", "--src", str(collection), "--tgt", str(dataset), "--repo-id", task, "--image-width", str(pipeline["image_width"]), "--image-height", str(pipeline["image_height"]), "--fps", str(pipeline["collection_fps"]), "--left-qpos-dim", str(pipeline["left_qpos_dim"]))
        if pipeline.get("collect_instance_ids", False):
            convert_command.append("--instance-annotations")
        stages.append(("convert-84x84", convert_command))
    print(f"[flow] prepare / GPU {gpu}: task {task_index}/{task_total} started: {task}", flush=True)
    if collection_ready:
        message = f"[flow] prepare / GPU {gpu}: reusing complete collection: {collection}\n"
        log.write(message)
        print(message, end="", flush=True)
    if dataset_ready:
        message = f"[flow] prepare / GPU {gpu}: reusing complete 84x84 dataset: {dataset}\n"
        log.write(message)
        print(message, end="", flush=True)
    for stage_index, (stage_name, command) in enumerate(stages, start=1):
        print(f"[flow] GPU {gpu}: task {task_index}/{task_total} | stage {stage_index}/{len(stages)}: {stage_name}", flush=True)
        try:
            run(command, gpu=gpu, log=log, config=config)
        except BaseException:
            if stage_name == "collect":
                remove_artifact(collection, "failed RoboTwin collection", log)
            elif stage_name == "convert-84x84":
                remove_artifact(dataset, "failed LeRobot dataset", log)
            raise
    print(f"[flow] prepare / GPU {gpu}: task {task_index}/{task_total} completed: {task}", flush=True)


def run_index_task(
    task: str,
    gpu: str,
    config: dict[str, Any],
    *,
    task_index: int,
    task_total: int,
    log: RotatingLog,
) -> None:
    _, dataset, index = task_paths(task, config)
    if completed_index(index):
        print(f"[flow] index: reusing complete index: {index}", flush=True)
        return
    if index.exists():
        remove_artifact(index, "incomplete EmixIndex")
    print(f"[flow] index: task {task_index}/{task_total} started: {task}", flush=True)
    command = conda_command(
        config, "preprocessing", gpu, "python", "-m", "emix", "build", "--root", str(dataset),
        "--repo_id", task, "--output_dir", str(index),
        *require_mapping(config, "emix").get("build_extra_args", []),
    )
    try:
        run(command, gpu=gpu, log=log, config=config, environment=index_command_env(config))
    except BaseException:
        remove_artifact(index, "failed EmixIndex", log)
        raise
    print(f"[flow] index: task {task_index}/{task_total} completed: {task}", flush=True)


def run_vis_task(
    task: str,
    gpu: str,
    config: dict[str, Any],
    *,
    task_index: int,
    task_total: int,
    log: RotatingLog,
) -> None:
    """Render one task's visualization (HTML + crisp PNG) from its EmixIndex.

    Runs after the index phase so every task already has a complete
    ``emix_index.json``. ``emix visualize --index`` reuses that index instead
    of rebuilding it; the HTML is written to ``outputs/intersections/<task>.html``
    and a headless-Chrome pass (scripts/render_vis_png.py) compiles it into
    ``outputs/intersections/<task>.png``.
    """
    visualization = require_mapping(config, "visualization")
    skip_click_images = bool(visualization.get("skip_click_images", True))
    _, dataset, index = task_paths(task, config)
    html, png = vis_output_paths(task, config)
    if completed_vis(html, png, task, require_assets=not skip_click_images):
        print(f"[flow] vis: reusing complete visualization: {html}", flush=True)
        return
    for artifact in (html, png):
        remove_artifact(artifact, "incomplete visualization", log)
    for asset_dir in html.parent.glob(f"{task}_*_assets"):
        remove_artifact(asset_dir, "stale visualization assets", log)
    print(f"[flow] vis: task {task_index}/{task_total} started: {task}", flush=True)
    visualize_args = [
        "--repo_id", task, "--root", str(dataset),
        "--index", str(index / "emix_index.json"),
        "--output", str(html),
        "--camera", str(visualization["camera"]),
        # Performance guardrails (always on): cap the drawn cross-point
        # groups and skip the click-to-inspect image precompute.
        "--max-points", str(visualization["max_points"]),
    ]
    if skip_click_images:
        visualize_args.append("--skip-click-images")
    stages: list[tuple[str, list[str]]] = [
        (
            "visualize-html",
            conda_command(
                config, "preprocessing", gpu, "python", "-m", "emix", "visualize",
                *visualize_args,
            ),
        ),
        (
            "visualize-png",
            conda_command(
                config, "preprocessing", gpu, "python", "scripts/render_vis_png.py",
                str(html), str(png),
                "--width", str(visualization["window_width"]),
                "--height", str(visualization["window_height"]),
                "--device-scale-factor", str(visualization["device_scale_factor"]),
                "--virtual-time-budget", str(visualization["virtual_time_budget_ms"]),
                "--margin", str(visualization["png_margin"]),
                "--timeout", str(visualization["chrome_timeout_s"]),
            ),
        ),
    ]
    for stage_index, (stage_name, command) in enumerate(stages, start=1):
        print(f"[flow] GPU {gpu}: task {task_index}/{task_total} | stage {stage_index}/{len(stages)}: {stage_name}", flush=True)
        try:
            run(command, gpu=gpu, log=log, config=config, environment=index_command_env(config))
        except BaseException:
            for artifact in (html, png):
                remove_artifact(artifact, "failed visualization", log)
            for asset_dir in html.parent.glob(f"{task}_*_assets"):
                remove_artifact(asset_dir, "failed visualization assets", log)
            raise
    print(f"[flow] vis: task {task_index}/{task_total} completed: {task}", flush=True)


def train_output_dir(task: str, config: dict[str, Any], *, emix: bool) -> Path:
    """Training artifact root for one task/policy-kind pair."""
    kind = "emix" if emix else "baseline"
    return configured_path(config, "train_root", f"{task}_dp_{kind}")


def final_checkpoint_dir(task: str, config: dict[str, Any], *, emix: bool) -> Path:
    """Directory holding the final pretrained model, as the eval config points at."""
    steps = int(require_mapping(config, "training")["steps"])
    return train_output_dir(task, config, emix=emix) / "checkpoints" / f"{steps:06d}" / "pretrained_model"


def completed_training(task: str, config: dict[str, Any], *, emix: bool) -> bool:
    """Return whether training produced its final checkpoint (simplest check)."""
    return (final_checkpoint_dir(task, config, emix=emix) / "model.safetensors").is_file()


def eval_output_dir(task: str, config: dict[str, Any], *, emix: bool) -> Path:
    """Evaluation artifact root for one task/policy-kind pair."""
    kind = "emix" if emix else "baseline"
    return configured_path(config, "eval_root", f"{task}_dp_{kind}")


def run_pre_vis_task(task: str, gpu: str, config: dict[str, Any], *,
                     emix: bool, destination: Path, train_config_path: Path,
                     log: RotatingLog) -> None:
    """Require the four V0 views beside this policy's train/eval artifacts."""
    sample_config = yaml.safe_load(train_config_path.read_text())
    identity = {"dataset": sample_config.get("dataset"),
                "policy": sample_config.get("policy"),
                "image_noise": sample_config.get("image_noise"),
                "emix": sample_config.get("emix")}
    if (destination / "complete.json").is_file():
        required = ("input_samples.png", "intersections.png", "pick_place.png",
                    "first_frame_range.png")
        try:
            record = json.loads((destination / "complete.json").read_text())
        except (ValueError, OSError):
            record = {}
        if record.get("identity") == identity and all(
            (destination / name).is_file() and (destination / name).stat().st_size > 0
            for name in required
        ):
            return
    _, dataset, index = task_paths(task, config)
    if not dataset.is_dir() or not (index / "emix_index.json").is_file():
        raise FileNotFoundError(f"pre_vis requires dataset and EmixIndex for {task}")
    # The existing intersection renderer is index-backed. Reuse its finished
    # image, but copy it into each policy's artifact package.
    run_vis_task(task, gpu, config, task_index=1, task_total=1, log=log)
    intersection_html, intersection_png = vis_output_paths(task, config)
    destination.mkdir(parents=True, exist_ok=True)
    commands = [
        ("input-samples", conda_command(
            config, "preprocessing", gpu, "python", "scripts/pre_vis_sample.py",
            "--train-config", str(train_config_path), "--output", str(destination / "input_samples.png"))),
        ("first-frame-range", conda_command(
            config, "preprocessing", gpu, "python", "-m", "embodied_mixup.visualization.first_frame_average",
            "--root", str(dataset), "--output", str(destination / "first_frame_range.png"))),
        ("pick-place", conda_command(
            config, "preprocessing", gpu, "python", "-m", "embodied_mixup.visualization.plot_actions",
            "--root", str(dataset), "--output", str(destination / "pick_place.html"),
            "--png", str(destination / "pick_place.png"))),
    ]
    for label, command in commands:
        if label == "input-samples" or not (destination / ({"first-frame-range": "first_frame_range.png", "pick-place": "pick_place.png"}.get(label, "input_samples.png"))).is_file():
            run(command, gpu=gpu, log=log, config=config)
    shutil.copy2(intersection_png, destination / "intersections.png")
    shutil.copy2(intersection_html, destination / "intersections.html")
    required = ("input_samples.png", "intersections.png", "pick_place.png", "first_frame_range.png")
    if not all((destination / name).is_file() and (destination / name).stat().st_size > 0 for name in required):
        raise RuntimeError(f"pre_vis is incomplete: {destination}")
    (destination / "complete.json").write_text(json.dumps({"task": task,
        "policy": "emix" if emix else "baseline", "artifacts": list(required),
        "identity": identity}, indent=2))


def completed_evaluation(task: str, config: dict[str, Any], *, emix: bool) -> bool:
    """Return whether evaluation wrote a metrics.json (simplest check)."""
    return any(eval_output_dir(task, config, emix=emix).glob("*/metrics.json"))


def selected_policies(config: dict[str, Any]) -> list[tuple[str, bool]]:
    """Policy kinds to train/evaluate, from ``pipeline.policies``.

    Defaults to both (baseline + emix). An ablation batch that only re-runs
    the EMIX arm sets ``policies: [emix]`` and reuses the reference batch's
    baseline numbers instead of retraining them.

    @input: config [dict] normalized flow configuration.
    @output: list of (kind, is_emix) tuples in fixed baseline-first order.
    @scenario: EMIX-only ablation batches.
    """
    pipeline = require_mapping(config, "pipeline")
    policies = pipeline.get("policies", ["baseline", "emix"])
    if not isinstance(policies, list) or not policies or any(
        policy not in ("baseline", "emix") for policy in policies
    ):
        raise ValueError(
            "pipeline.policies must be a non-empty subset of ['baseline', 'emix'], "
            f"got: {policies!r}"
        )
    return [(kind, kind == "emix") for kind in ("baseline", "emix") if kind in policies]


def run_training_task(
    task: str,
    gpu: str,
    config: dict[str, Any],
    *,
    task_index: int,
    task_total: int,
    log: RotatingLog,
) -> None:
    configs = write_runtime_configs(task, config)
    # Constraint: the train stages run in `training` (emix); the eval stages
    # drive the RoboTwin benchmark environment and must run in `evaluation`
    # (emix-twin). Keep these two keys distinct in the config.
    #
    # Each stage is skipped when its artifact is already complete (see the
    # module docstring). Incomplete artifacts are removed first so the stage
    # restarts cleanly -- train.py refuses a non-empty output_dir.
    print(f"[flow] train-eval / GPU {gpu}: task {task_index}/{task_total} started: {task}", flush=True)
    stages: list[tuple[str, list[str]]] = []
    for kind, emix in selected_policies(config):
        if completed_training(task, config, emix=emix):
            message = f"[flow] train-eval / GPU {gpu}: reusing completed {kind} training: {final_checkpoint_dir(task, config, emix=emix)}\n"
            log.write(message)
            print(message, end="", flush=True)
        else:
            incomplete_train = train_output_dir(task, config, emix=emix)
            if incomplete_train.exists() and any(
                item.name != "pre_vis" for item in incomplete_train.iterdir()
            ):
                remove_artifact(incomplete_train, f"incomplete {kind} training", log)
            stages.append(
                (
                    f"{kind}-train",
                    conda_command(
                        config, "training", gpu, "python", "train/train.py",
                        "--config", str(configs[f"{kind}_train"]),
                        "--seed", str(require_mapping(config, "training")["seed"]),
                    ),
                )
            )
        # Evaluation output is only reusable while the checkpoint it evaluates
        # still exists; otherwise the metrics are stale and get re-run.
        if completed_training(task, config, emix=emix) and completed_evaluation(task, config, emix=emix):
            message = f"[flow] train-eval / GPU {gpu}: reusing completed {kind} evaluation: {eval_output_dir(task, config, emix=emix)}\n"
            log.write(message)
            print(message, end="", flush=True)
        else:
            incomplete_eval = eval_output_dir(task, config, emix=emix)
            if incomplete_eval.exists() and any(
                item.name != "pre_vis" for item in incomplete_eval.iterdir()
            ):
                remove_artifact(incomplete_eval, f"incomplete {kind} evaluation", log)
            stages.append(
                (
                    f"{kind}-eval",
                    conda_command(
                        config, "evaluation", gpu, "python", "evaluation/evaluate.py",
                        "--config", str(configs[f"{kind}_eval"]),
                    ),
                )
            )
    for stage_index, (stage_name, command) in enumerate(stages, start=1):
        print(f"[flow] GPU {gpu}: task {task_index}/{task_total} | stage {stage_index}/{len(stages)}: {stage_name}", flush=True)
        kind = stage_name.split("-", 1)[0]
        is_emix = kind == "emix"
        destination = (train_output_dir(task, config, emix=is_emix)
                       if stage_name.endswith("-train") else eval_output_dir(task, config, emix=is_emix))
        if not config.get("_skip_pre_vis", False):
            run_pre_vis_task(task, gpu, config, emix=is_emix,
                             destination=destination / "pre_vis",
                             train_config_path=configs[f"{kind}_train"], log=log)
        run(command, gpu=gpu, log=log, config=config)
    print(f"[flow] train-eval / GPU {gpu}: task {task_index}/{task_total} completed: {task}", flush=True)


def write_results_summary(config: dict[str, Any], log_directory: Path | None = None) -> Path:
    """Write the batch's auto-generated results summary (RESULTS.md).

    Contains run provenance (label, timestamp, commit, flow log dir), the
    per-task success-rate table for the selected policies, a TOTAL row, and a
    clearly reserved empty section for agents to fill in manually.  Written
    after the train-eval phase; safe to re-run (overwrites the auto file).

    @input: config [dict] normalized flow configuration; log_directory of the
    current flow invocation (optional).
    @output: the Path of the written RESULTS.md.
    @scenario: fixed end-of-batch bookkeeping required by SOP-06/SOP-05.
    """
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        commit = "unknown"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = config["run_label"]
    pipeline = require_mapping(config, "pipeline")
    training = require_mapping(config, "training")
    emix_cfg = require_mapping(config, "emix")
    summary_root = ROOT / "outputs" / f"run_{label}" / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    out_path = summary_root / "RESULTS.md"

    policies = [kind for kind, _ in selected_policies(config)]
    table: dict[str, dict[str, tuple[float, int]]] = {}
    for task in config["tasks"]:
        for kind in policies:
            runs = sorted(eval_output_dir(task, config, emix=kind == "emix").glob("*/metrics.json"))
            if not runs:
                continue
            payload = json.loads(runs[-1].read_text(encoding="utf-8"))
            s = payload.get("summary", payload)
            sr = s.get("success_rate", s)
            n = int(s.get("num_episodes", 0))
            table.setdefault(task, {})[kind] = (100.0 * float(sr), n)

    lines: list[str] = [
        f"# RESULTS — {label}",
        "",
        f"- 生成时间: {timestamp}",
        f"- commit: `{commit}`",
        f"- flow 日志: `{log_directory}`" if log_directory else "",
        f"- tasks: {len(config['tasks'])} | episodes/task: {pipeline['episodes_per_task']} | "
        f"eval: {pipeline['evaluation_episodes']}ep seed {pipeline['evaluation_seed']}",
        f"- training: steps {training['steps']}, batch {training['batch_size']}, lr {training['optimizer_lr']}, "
        f"seed {training['seed']}; policies: {policies}",
        f"- protocol: {config.get('protocol', {})}",
        f"- training.image_noise: {training.get('image_noise', {'enabled': False})}; "
        f"evaluation.image_noise: {(config.get('evaluation') or {}).get('image_noise', {'enabled': False})}",
        f"- emix 混合: alpha {emix_cfg['alpha_mode']}({emix_cfg['beta']}), state {emix_cfg['state_mode']}, "
        f"loss {emix_cfg['loss_weight']}, scheduler {emix_cfg['scheduler'].get('type')} "
        f"{emix_cfg['scheduler'].get('start_batch_prob')}→{emix_cfg['scheduler'].get('end_batch_prob')}",
        f"- emix.build_extra_args: {require_mapping(config, 'emix').get('build_extra_args', [])}",
        "",
        "## 结果(成功率 %)",
        "",
    ]
    header = "| 任务 | " + " | ".join(f"{k} (n)" for k in policies) + " |"
    lines += [header, "|" + "---|" * (len(policies) + 1)]
    totals = {kind: [0.0, 0] for kind in policies}
    for task, values in sorted(table.items()):
        row = [task]
        for kind in policies:
            if kind in values:
                sr, n = values[kind]
                row.append(f"{sr:.1f} ({n})")
                totals[kind][0] += sr * n
                totals[kind][1] += n
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    total_row = ["**TOTAL(加权)**"]
    for kind in policies:
        sr_sum, n_sum = totals[kind]
        total_row.append(f"**{sr_sum / n_sum:.1f}** ({n_sum})" if n_sum else "-")
    lines.append("| " + " | ".join(total_row) + " |")
    lines += [
        "",
        "## Agent 手动总结区(预留,请在此补充)",
        "",
        "<!-- agents: 把本批次的结论、与对照批次的差异、异常观察写在这里 -->",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[flow] summary: wrote {out_path}", flush=True)
    return out_path


def worker(
    phase: str,
    gpu: str,
    tasks: list[str],
    config: dict[str, Any],
    worker_index: int,
    log_directory: Path,
) -> None:
    signal.signal(signal.SIGINT, worker_signal_handler)
    signal.signal(signal.SIGTERM, worker_signal_handler)
    log = RotatingLog(log_directory, worker_index, log_rotate_bytes(config))
    try:
        log.write(f"[flow] phase={phase} worker={worker_index:03d} gpu={gpu} started\n")
        print(f"[flow] {phase}: worker {worker_index:03d} / GPU {gpu} started (0/{len(tasks)} tasks complete)", flush=True)
        for task_index, task in enumerate(tasks, start=1):
            if phase == "prepare":
                run_prepare_task(task, gpu, config, task_index=task_index, task_total=len(tasks), log=log)
            elif phase == "index":
                run_index_task(task, gpu, config, task_index=task_index, task_total=len(tasks), log=log)
            elif phase == "vis":
                run_vis_task(task, gpu, config, task_index=task_index, task_total=len(tasks), log=log)
            elif phase == "train-eval":
                run_training_task(task, gpu, config, task_index=task_index, task_total=len(tasks), log=log)
            elif phase == "summary":
                pass  # handled once in the parent, not per worker task
            else:
                raise ValueError(f"Unknown flow phase: {phase}")
            print(f"[flow] {phase} / GPU {gpu}: queue progress {task_index}/{len(tasks)} tasks complete", flush=True)
    finally:
        stop_active_command()
        log.close()


def stop_workers(processes: list[mp.Process]) -> None:
    """Terminate and reap all workers before the parent process exits."""
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=15)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join()


def run_phase(
    phase: str,
    batches: dict[str, list[str]],
    config: dict[str, Any],
    log_directory: Path,
) -> None:
    """Run one global phase and wait for every worker before continuing."""
    print(f"[flow] ===== phase started: {phase} =====", flush=True)
    if phase == "summary":
        write_results_summary(config, log_directory)
        print(f"[flow] ===== phase completed: {phase} =====", flush=True)
        return
    processes = [
        mp.Process(
            target=worker,
            args=(phase, gpu, tasks, config, worker_index, log_directory),
            name=f"flow-{phase}-gpu-{gpu}",
        )
        for worker_index, (gpu, tasks) in enumerate(batches.items(), start=0)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print(f"[flow] Ctrl+C received during {phase}; stopping workers...", flush=True)
        stop_workers(processes)
        raise SystemExit(130)
    failed = [process.name for process in processes if process.exitcode]
    if failed:
        stop_workers(processes)
        raise SystemExit(f"Experiment flow failed during {phase}: {', '.join(failed)}")
    print(f"[flow] ===== phase completed: {phase} =====", flush=True)


def configured_cuda(config: dict[str, Any], override: list[str] | None = None) -> list[str]:
    values: Any = override if override is not None else config.get("cuda")
    if isinstance(values, str):
        values = values.split(",")
    if isinstance(values, list):
        values = [item for value in values for item in str(value).split(",")]
    if not isinstance(values, list) or not values or any(not str(value) for value in values):
        raise ValueError("cuda must be a non-empty list of CUDA device identifiers")
    devices = [str(value) for value in values]
    if len(set(devices)) != len(devices):
        raise ValueError("cuda devices must be unique")
    return devices


def make_batches(tasks: list[str], cuda: list[str], config: dict[str, Any]) -> dict[str, list[str]]:
    gpu_batches = config.get("gpu_batches")
    if gpu_batches is not None:
        # Explicit per-device assignment from run.yaml: keep it verbatim and
        # only restrict to the devices selected on the command line.
        assigned = {
            task: str(device)
            for device, device_tasks in gpu_batches.items()
            for task in device_tasks
        }
        missing = [task for task in tasks if task not in assigned]
        if missing:
            raise ValueError(f"tasks missing from gpu_batches: {', '.join(missing)}")
        batches = {device: [] for device in cuda}
        for task in tasks:
            device = assigned[task]
            if device not in batches:
                raise ValueError(
                    f"task {task} is assigned to CUDA {device} in gpu_batches, "
                    f"which is not among the selected devices {cuda}"
                )
            batches[device].append(task)
    else:
        batches = {device: [] for device in cuda}
        for index, task in enumerate(tasks):
            batches[cuda[index % len(cuda)]].append(task)
    for batch in batches.values():
        batch.sort(key=lambda task: task_priority(task, config))
    return {device: tasks for device, tasks in batches.items() if tasks}


def validate(config: dict[str, Any], cuda: list[str]) -> dict[str, list[str]]:
    tasks = config.get("tasks")
    pipeline = require_mapping(config, "pipeline")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError("tasks must be a non-empty list of task names")
    if len(set(tasks)) != len(tasks):
        raise ValueError("tasks must contain unique task names")
    batches = make_batches(list(tasks), cuda, config)
    benchmark = configured_path(config, "robotwin_root")
    task_config_root = configured_path(config, "task_config_root")
    environments = require_mapping(config, "environments")
    conda_executable = str(environments.get("conda_executable", ""))
    conda_path = Path(conda_executable)
    if not conda_path.is_absolute() or not conda_path.is_file() or not os.access(conda_path, os.X_OK):
        raise FileNotFoundError(
            "environments.conda_executable must be an executable absolute path, "
            f"got: {conda_executable!r}"
        )
    for environment_key in ("robotwin", "preprocessing", "training", "evaluation"):
        if not isinstance(environments.get(environment_key), str) or not environments[environment_key]:
            raise ValueError(f"environments.{environment_key} must name a Conda environment")
    validate_conda_environments(config)
    validate_worker_gpus(config, batches)
    validate_evaluation_runtime(config)
    for task in tasks:
        # Task configs mirror the envs/left layout: one subfolder per camera
        # variant (base/ holds the default central-cam configs, "" catches
        # top-level utility tasks). Keep in sync with
        # envs/_GLOBAL_CONFIGS.py task_config_yml_path.
        candidates = tuple(
            task_config_root / folder / f"{task}.yml"
        for folder in ("cen_arm_right_wide_cam", "cen_arm_near_side_cam",
                       "cen_arm_side_cam", "cen_arm_front_cam", "cen_arm_top_cam",
                       "central_wide_cam", "left_oppo_cam",
                       "base", "")
        )
        task_config = next((path for path in candidates if path.is_file()), candidates[-1])
        if not task_config.is_file():
            raise FileNotFoundError(f"Missing left task config: {task}")
        with task_config.open(encoding="utf-8") as handle:
            collection_config = yaml.safe_load(handle) or {}
        if collection_config.get("episode_num") != pipeline["episodes_per_task"]:
            raise ValueError(
                f"{task_config} must set episode_num to pipeline.episodes_per_task "
                f"({pipeline['episodes_per_task']})"
            )
        camera = collection_config.get("camera") or {}
        if camera.get("head_camera_type") != "D435-W" or not camera.get("collect_head_camera"):
            raise ValueError(f"{task_config} must collect the central wide head camera")
    # Priority: tasks with fewer average frames per episode run first in each
    # GPU queue, so short collection/training tasks finish early.
    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the configured RoboTwin left-arm DP/EMIX experiment matrix."
    )
    parser.add_argument(
        "--phase", type=str, default=None,
        choices=["prepare", "index", "vis", "pre_vis", "train-eval", "summary"],
        help="Run only this single phase over all configured tasks and exit. "
             "Skips the environment/GPU validation so a stage can be backfilled "
             "over existing data, e.g. `python scripts/run.py --phase vis`.",
    )
    parser.add_argument(
        "--cuda", nargs="+", default=None, metavar="DEVICE",
        help="CUDA devices to use, overriding the gpu_batches devices in run.yaml; e.g. --cuda 0 1 or --cuda 0,1.",
    )
    parser.add_argument(
        "--config", type=Path, default=FLOW_CONFIG,
        help="Flow configuration YAML to run (default: configs/scripts/run.yaml).",
    )
    parser.add_argument(
        "--skip-pre-vis", action="store_true",
        help="Explicitly bypass mandatory pre_vis artifacts for this invocation.",
    )
    return parser.parse_args()


def run_single_phase(config: dict[str, Any], phase: str, log_directory: Path, cuda: list[str]) -> None:
    """Run one phase standalone over every configured task (no validation).

    ``index`` and ``vis`` are CPU/RAM-heavy serial queues and run on a single
    worker; ``prepare``/``train-eval`` keep the per-GPU batch split.
    """
    tasks = config["tasks"]
    batches = make_batches(list(tasks), cuda, config)
    if phase == "summary":
        run_phase("summary", {next(iter(batches)): []}, config, log_directory)
        return
    if phase == "pre_vis":
        log = RotatingLog(log_directory, 0, DEFAULT_LOG_ROTATE_BYTES)
        try:
            for gpu, work in batches.items():
                for task in work:
                    configs = write_runtime_configs(task, config)
                    if not config.get("_skip_pre_vis", False):
                        for kind, emix in selected_policies(config):
                            for destination in (train_output_dir(task, config, emix=emix),
                                                eval_output_dir(task, config, emix=emix)):
                                run_pre_vis_task(task, gpu, config, emix=emix,
                                    destination=destination / "pre_vis",
                                    train_config_path=configs[f"{kind}_train"], log=log)
        finally:
            log.close()
        return
    if phase in ("index", "vis"):
        all_tasks = sorted(
            (task for tasks in batches.values() for task in tasks),
            key=lambda task: task_priority(task, config),
        )
        run_phase(phase, {next(iter(batches)): all_tasks}, config, log_directory)
    else:
        run_phase(phase, batches, config, log_directory)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    config["_skip_pre_vis"] = args.skip_pre_vis
    cuda = configured_cuda(config, args.cuda)
    log_root = configured_path(config, "log_root")
    lock_name = ".flow-cuda-" + "-".join(cuda) + ".lock"
    with flow_lock(log_root / lock_name):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_directory = log_root / f"flow-{timestamp}-cuda-{'-'.join(cuda)}"
        log_directory.mkdir(parents=True, exist_ok=False)
        print(f"[flow] logs: {log_directory}", flush=True)
        if args.phase:
            run_single_phase(config, args.phase, log_directory, cuda)
            if args.phase == "index":
                # A freshly built EmixIndex is visualized immediately (same
                # guardrails as the full flow's vis phase) so clustering can
                # be inspected before any training starts; tasks whose vis is
                # already complete are skipped inside the phase.
                run_single_phase(config, "vis", log_directory, cuda)
            return
        batches = validate(config, cuda)
        run_phase("prepare", batches, config, log_directory)
        # The index and vis phases run one serial CPU/RAM-intensive queue on a
        # single worker; apply the same episode-length priority (fewer frames
        # first) to that global queue. Visualization consumes the EmixIndex, so
        # it must follow the index phase.
        all_tasks = sorted(
            (task for tasks in batches.values() for task in tasks),
            key=lambda task: task_priority(task, config),
        )
        run_phase("index", {next(iter(batches)): all_tasks}, config, log_directory)
        run_phase("vis", {next(iter(batches)): all_tasks}, config, log_directory)
        if not args.skip_pre_vis:
            run_single_phase(config, "pre_vis", log_directory, cuda)
        run_phase("train-eval", batches, config, log_directory)
        run_phase("summary", {next(iter(batches)): []}, config, log_directory)


if __name__ == "__main__":
    main()
