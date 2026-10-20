"""Evaluate a LeRobot Diffusion Policy in the RoboTwin benchmark."""

import argparse
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from embodied_mixup.datasets import require_lerobot_dataset_root
from embodied_mixup.configs.config import resolve_project_output_path
from embodied_mixup.policies.utils.bootstrap import bootstrap_lerobot_runtime
from embodied_mixup.policies.utils.batch import assert_single_observation_config
from embodied_mixup.policies.utils.image_noise import ImageNoiseConfig, add_image_noise

LOGGER = logging.getLogger(__name__)


def freeze_right_action(
    action: np.ndarray, home_qpos: np.ndarray, left_qpos_dim: int
) -> np.ndarray:
    """Return a copied qpos vector whose right-arm suffix is frozen to home.

    @input: left-only policy action (or native-width compatibility action),
        full-width right-home qpos vector, and left projection boundary.
    @output: native-width simulator action with the right suffix supplied from
        the captured home qpos; never mutates the policy output.
    @scenario: The policy emits only left-arm qpos. Restore the captured right
        home qpos only at the RoboTwin simulator boundary so the inactive arm
        is not a policy input or output and never moves.
    """
    arr = np.asarray(action)
    home = np.asarray(home_qpos, dtype=np.float32)
    native_width = 2 * left_qpos_dim
    if arr.ndim != 1 or home.ndim != 1:
        raise ValueError("freeze_right_action expects one-dimensional qpos vectors")
    if home.shape[0] != native_width:
        raise ValueError(
            f"home qpos width must be {native_width}, got {home.shape[0]}"
        )
    if arr.shape[0] == left_qpos_dim:
        # Policy output is left-only; restore right home values only at the
        # RoboTwin simulator boundary.
        return np.concatenate((arr.astype(np.float32), home[left_qpos_dim:]))
    if arr.shape[0] != native_width:
        raise ValueError(
            f"policy qpos width must be {left_qpos_dim} (left-only) or "
            f"{native_width} (native compatibility), got {arr.shape[0]}"
        )
    projected = arr.astype(np.float32).copy()
    projected[left_qpos_dim:] = home[left_qpos_dim:]
    return projected


def _load_policy(checkpoint, dataset_root, repo_id, device):
    """Load one local LeRobot policy checkpoint.

    @input: checkpoint path, dataset_root path, repo_id string, device string.
    @output: tuple of policy, preprocessor, postprocessor, dataset metadata.
    @scenario: Reuse an EMIX-trained or baseline Diffusion Policy for RoboTwin rollout.
    """
    bootstrap_lerobot_runtime()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pretrained import PreTrainedConfig

    root = str(require_lerobot_dataset_root(dataset_root))
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    assert_single_observation_config(config, context="evaluation checkpoint")
    config.pretrained_path = checkpoint
    policy = make_policy(config, dataset.meta).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        dataset_stats=dataset.meta.stats,
    )
    return policy, preprocessor, postprocessor, config


def _decode_head_camera(observation, width, height):
    """Convert RoboTwin RGB observation into a policy tensor image.

    @input: observation RoboTwin dict; target width and height positive integers.
    @output: torch.Tensor shaped (1, 3, H, W) in float32 [0, 1].
    @scenario: Map the only approved visual input, head_camera, at rollout time.
    """
    image = observation["observation"]["head_camera"]["rgb"]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def _make_policy_observation(
    observation, width, height, *, left_qpos_dim=7, image_noise=None, noise_seed=0, noise_step=0,
):
    """Map a RoboTwin observation to the LeRobot feature contract.

    @input: observation RoboTwin dict, image width/height.
    @output: dict containing head image and left-arm-only state batch tensors.
    @scenario: Keep policy inputs identical to the converted training dataset.
    """
    batch = {
        "observation.images.head_camera": _decode_head_camera(observation, width, height),
        "observation.state": torch.as_tensor(
            observation["joint_action"]["vector"][:left_qpos_dim], dtype=torch.float32
        ).unsqueeze(0),
    }
    return add_image_noise(
        batch, image_noise or ImageNoiseConfig(), seed=noise_seed, step=noise_step,
    )


def _right_arm_motion_error():
    """Return the RoboTwin ``RightArmMotionError`` class, or ``None``.

    @input: None; imports lazily because ``envs`` is only reachable after
        ``_build_task`` put the RoboTwin benchmark root on ``sys.path``.
    @output: the exception class, or ``None`` for non-left task families
        that do not define it.
    @scenario: Let the evaluator treat policy-driven contact with the idle
        right arm as one failed episode instead of a crashed run.
    """
    try:
        from envs.left.left_task_base import RightArmMotionError
        return RightArmMotionError
    except ImportError:
        return None


def _build_task(task_name, task_config, piper_distance, benchmark_root):
    """Construct a RoboTwin task and resolved task configuration.

    @input: task name/config strings, explicit Piper distance, RoboTwin root path.
    @output: tuple(task instance, config dict).
    @scenario: Mirror the RoboTwin native evaluator without importing ManiSkill/Gym.
    """
    sys.path.insert(0, str(benchmark_root))
    envs_package = importlib.import_module("envs")
    envs = envs_package.import_task_env(task_name)
    task = getattr(envs, task_name)()
    config_candidates = tuple(
        benchmark_root / "task_config" / "left" / folder / f"{task_config}.yml"
        # Mirrors the envs/left layout: one subfolder per camera variant, base
        # holds the default central-cam configs, "" catches top-level utility
        # tasks (place_object_scale_left*). Keep in sync with
        # envs/_GLOBAL_CONFIGS.py task_config_yml_path and data_collection/robotwin.py.
        for folder in ("cen_arm_right_wide_cam", "cen_arm_near_side_cam",
                       "cen_arm_side_cam", "cen_arm_front_cam", "cen_arm_top_cam",
                       "central_wide_cam", "left_oppo_cam",
                       "base", "")
    ) + (benchmark_root / "task_config" / f"{task_config}.yml",)
    config_path = next((path for path in config_candidates if path.is_file()), None)
    if config_path is None:
        raise FileNotFoundError(
            "RoboTwin task config does not exist. Checked: "
            + ", ".join(str(path) for path in config_candidates)
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["task_name"] = task_name
    config["task_config"] = task_config
    config["eval_mode"] = True
    config["left_robot_file"] = str(benchmark_root / "assets" / "embodiments" / "piper")
    config["right_robot_file"] = str(benchmark_root / "assets" / "embodiments" / "piper")
    config["dual_arm_embodied"] = False
    config["embodiment_dis"] = float(piper_distance)
    with open(benchmark_root / "assets" / "embodiments" / "piper" / "config.yml", "r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    return task, config


def _policy_action(policy, preprocessor, postprocessor, policy_obs, device):
    """Run one Diffusion Policy action selection step.

    @input: policy and processors, batched policy observation, torch device.
    @output: np.ndarray qpos target vector.
    @scenario: Produce one qpos target while LeRobot internally manages its action chunk cache.
    """
    batch = {key: value.to(device) for key, value in policy_obs.items()}
    with torch.inference_mode():
        action = postprocessor(policy.select_action(preprocessor(batch)))
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _start_video_writer(task, video_path, fps):
    """Start RoboTwin's head-camera MP4 writer for one rollout episode.

    @input: initialized RoboTwin task, output MP4 path, positive frame rate.
    @output: ffmpeg process attached to the task's native video hooks.
    @scenario: Record exactly the same head-camera frames used by the policy.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "Saving evaluation videos requires ffmpeg, but it was not found on PATH."
        )
    observation = task.get_obs()
    frame = np.asarray(observation["observation"]["head_camera"]["rgb"])
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an RGB head-camera frame, got shape {frame.shape}.")
    height, width = frame.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    task._set_eval_video_ffmpeg(ffmpeg)
    return ffmpeg


def _stop_video_writer(task):
    """Flush and close the task's native video writer if one is active."""
    if getattr(task, "eval_video_ffmpeg", None) is not None:
        task._del_eval_video_ffmpeg()


def _load_pipeline_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Unified pipeline config must contain a YAML mapping.")
    return payload


def _config_value(args, payload, section, name, cli_name=None, required=True):
    cli_name = cli_name or name
    value = getattr(args, cli_name, None)
    section_payload = payload.get(section) or {}
    if value is None:
        value = section_payload.get(name)
    if value is None and required:
        raise ValueError(f"Missing required configuration value: {section}.{name}")
    return value


def main(args):
    """Run the fixed 100-episode RoboTwin evaluation protocol.

    @input: argparse namespace for benchmark, dataset, checkpoint, and output settings.
    @output: None; writes metrics JSON and optional RoboTwin MP4 files.
    @scenario: Compare baseline and EMIX checkpoints under identical valid seeds.
    """
    payload = _load_pipeline_config(args.config)
    dataset_payload = payload.get("dataset") or {}
    checkpoint = str(Path(_config_value(args, payload, "evaluation", "checkpoint")).expanduser().resolve())
    dataset_root_value = args.dataset_root or dataset_payload.get("lerobot_root") or dataset_payload.get("root")
    repo_id = args.repo_id or dataset_payload.get("repo_id")
    if not dataset_root_value or not repo_id:
        raise ValueError("dataset.lerobot_root (or dataset.root) and dataset.repo_id are required")
    dataset_root = str(Path(dataset_root_value).expanduser().resolve())
    output_dir = resolve_project_output_path(
        _config_value(args, payload, "evaluation", "output_dir"),
        field="evaluation.output_dir",
    )
    benchmark_root = Path(_config_value(args, payload, "evaluation", "benchmark_root")).expanduser().resolve()
    task_name = _config_value(args, payload, "evaluation", "task_name")
    task_config = _config_value(args, payload, "evaluation", "task_config")
    piper_distance = float(_config_value(args, payload, "evaluation", "piper_distance"))
    episodes = int(_config_value(args, payload, "evaluation", "episodes"))
    seed_value = int(_config_value(args, payload, "evaluation", "seed"))
    device = _config_value(args, payload, "evaluation", "device")
    evaluation_payload = payload.get("evaluation") or {}
    image_noise = ImageNoiseConfig.from_payload(evaluation_payload.get("image_noise"))
    image_noise_seed = int(evaluation_payload.get("image_noise_seed", seed_value))
    LOGGER.info("Evaluation image_noise=%s seed=%d", image_noise.to_dict(), image_noise_seed)
    save_video = bool(evaluation_payload.get("save_video", True))
    video_fps = int(evaluation_payload.get("video_fps", 10))
    if video_fps <= 0:
        raise ValueError("evaluation.video_fps must be positive")
    os.chdir(benchmark_root)
    policy, preprocessor, postprocessor, config = _load_policy(checkpoint, dataset_root, repo_id, device)
    image_shape = config.input_features["observation.images.head_camera"].shape
    image_height, image_width = int(image_shape[1]), int(image_shape[2])
    output = output_dir / f"{task_name}_{Path(checkpoint).name}_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    successes, records, seed, accepted = 0, [], seed_value * 100000 + 100000, 0
    while accepted < episodes:
        task, task_cfg = _build_task(task_name, task_config, piper_distance, benchmark_root)
        try:
            task.setup_demo(now_ep_num=accepted, seed=seed, is_test=True, **task_cfg)
            task.play_once()
            valid = task.plan_success and task.check_success()
            task.close_env()
        except Exception as error:
            LOGGER.warning("Skipping invalid seed %d: %s", seed, error)
            try:
                task.close_env()
            except Exception:
                pass
            seed += 1
            continue
        if not valid:
            seed += 1
            continue
        task, task_cfg = _build_task(task_name, task_config, piper_distance, benchmark_root)
        episode_video_path = output / "videos" / f"episode_{accepted:04d}.mp4"
        if save_video:
            task_cfg["eval_video_save_dir"] = str(episode_video_path.parent)
        video_started = False
        right_arm_error = _right_arm_motion_error()
        note = None
        try:
            task.setup_demo(now_ep_num=accepted, seed=seed, is_test=True, **task_cfg)
            home_qpos = np.asarray(task.get_obs()["joint_action"]["vector"], dtype=np.float32)
            if save_video:
                _start_video_writer(task, episode_video_path, video_fps)
                video_started = True
            left_dim = len(task.robot.get_left_arm_jointState())
            policy.reset()
            success = False
            while task.take_action_cnt < task.step_lim:
                observation = task.get_obs()
                policy_obs = _make_policy_observation(
                    observation, image_width, image_height, left_qpos_dim=left_dim, image_noise=image_noise,
                    noise_seed=image_noise_seed + seed * 1_000_000,
                    noise_step=task.take_action_cnt + 1,
                )
                action = _policy_action(policy, preprocessor, postprocessor, policy_obs, device)
                if action.shape != (left_dim,):
                    raise ValueError(
                        f"Checkpoint action shape {action.shape} does not match "
                        f"left-arm-only qpos shape {(left_dim,)}."
                    )
                action = freeze_right_action(action, home_qpos, left_dim)
                task.take_action(action, action_type="qpos")
                if task.eval_success or task.check_success():
                    success = True
                    break
        except Exception as error:
            if right_arm_error is None or not isinstance(error, right_arm_error):
                raise
            LOGGER.warning("Episode %d failed: policy disturbed the idle right arm: %s", accepted, error)
            success = False
            note = "right_arm_motion"
        finally:
            if video_started:
                _stop_video_writer(task)
            task.close_env()
        successes += int(success)
        records.append({
            "episode": accepted,
            "seed": seed,
            "success": success,
            "note": note,
            "video": str(episode_video_path.relative_to(output)) if save_video else None,
        })
        accepted += 1
        seed += 1
        LOGGER.info("Episode %d/%d success=%s", accepted, episodes, success)
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"success_rate": successes / episodes, "successes": successes, "episodes": episodes,
                   "records": records, "image_noise": image_noise.to_dict(),
                   "image_noise_seed": image_noise_seed, "checkpoint": checkpoint,
                   "rt_denoiser_requested": os.environ.get("ROBOTWIN_RT_DENOISER", "oidn"),
                   "protocol": payload.get("protocol")}, handle, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[eval] [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot Diffusion Policy in RoboTwin")
    parser.add_argument("--benchmark-root", default="third_party/RoboTwin")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-name")
    parser.add_argument("--task-config")
    parser.add_argument("--piper-distance", type=float)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset-root")
    parser.add_argument("--repo-id")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    try:
        main(parser.parse_args())
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
