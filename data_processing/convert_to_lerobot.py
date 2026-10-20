"""Convert RoboTwin episode HDF5 files into one LeRobot dataset."""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from embodied_mixup.datasets import default_repo_id_from_dataset_root
from fix_stats import fix_lerobot_image_stats
from instance_annotations import InstanceAnnotations, transform_pair

LOGGER = logging.getLogger(__name__)


def _decode_jpeg_frames(encoded_frames):
    """Decode RoboTwin fixed-width JPEG byte arrays.

    @input: encoded_frames np.ndarray with one encoded JPEG byte string per frame.
    @output: np.ndarray shaped (T, H, W, 3) in RoboTwin RGB uint8 order.
    @scenario: Convert RoboTwin head_camera HDF5 media to LeRobot video frames.

    RoboTwin captures RGB arrays but serializes them with OpenCV's JPEG encoder
    without a BGR conversion.  ``imdecode`` therefore returns the original
    channel order; applying BGR->RGB here would swap red and blue.
    """
    frames = []
    for encoded in encoded_frames:
        image = cv2.imdecode(np.frombuffer(bytes(encoded).rstrip(b"\0"), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode a RoboTwin head_camera JPEG frame.")
        frames.append(image)
    return np.asarray(frames, dtype=np.uint8)


def _resize_rgb(image, width, height):
    """Resize one RGB image without padding.

    @input: image np.ndarray (H, W, 3), width int > 0, height int > 0.
    @output: np.ndarray (height, width, 3) uint8.
    @scenario: Make all RoboTwin samples match one explicit policy image size.
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _pose_wxyz_to_rotvec(pose):
    """Convert a SAPIEN pose to a 6D position-rotation-vector feature.

    @input: pose np.ndarray shaped (T, 7), ordered xyz + quaternion wxyz.
    @output: np.ndarray shaped (T, 6), float32.
    @scenario: Supply the existing EMIX index builder with a left EEF geometry key.
    """
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Expected left endpose shape (T, 7), got {pose.shape}.")
    quat_xyzw = np.concatenate([pose[:, 4:7], pose[:, 3:4]], axis=1)
    return np.concatenate([pose[:, :3], Rotation.from_quat(quat_xyzw).as_rotvec()], axis=1).astype(np.float32)


def _read_instruction(instruction_path, instruction_type, instruction_index):
    """Read one RoboTwin language instruction from an episode sidecar.

    @input: instruction_path Path to instructions/episodeN.json; instruction_type
        is ``seen`` or ``unseen``; instruction_index selects one generated prompt.
    @output: str prompt, or None when the sidecar has no usable prompt.
    @scenario: Preserve RoboTwin's per-episode language prompt in LeRobot's task field.
    """
    if not instruction_path.is_file():
        return None
    try:
        payload = json.loads(instruction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read RoboTwin instruction file {instruction_path}: {exc}") from exc

    prompts = payload.get(instruction_type, []) if isinstance(payload, dict) else payload
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list):
        raise ValueError(
            f"Expected {instruction_type!r} in {instruction_path} to be a string or list of strings."
        )
    prompts = [prompt.strip() for prompt in prompts if isinstance(prompt, str) and prompt.strip()]
    if not prompts:
        return None
    if instruction_index >= len(prompts):
        raise ValueError(
            f"Instruction index {instruction_index} is out of range for {instruction_path} "
            f"({len(prompts)} {instruction_type} prompts)."
        )
    return prompts[instruction_index]


def _resolve_instruction(path, instructions_root, instruction_type, instruction_index, fallback_task):
    """Resolve the task text for one episode, preferring RoboTwin's sidecar."""
    instruction_path = instructions_root / f"{path.stem}.json"
    prompt = _read_instruction(instruction_path, instruction_type, instruction_index)
    if prompt is not None:
        return prompt, True
    if fallback_task:
        return fallback_task, False
    raise ValueError(
        f"No usable RoboTwin instruction found for {path.name} at {instruction_path}. "
        "Regenerate the collection instructions or pass --task as an explicit legacy fallback."
    )


def _load_episode(path, image_width, image_height, left_qpos_dim, *, require_instance_annotations=False):
    """Load one native RoboTwin episode and align observations with actions.

    @input: path Path to episodeN.hdf5; image dimensions and left_qpos_dim positive integers.
    @output: dict of aligned image, state, action, and auxiliary left EEF arrays.
    @scenario: Use qpos at t as observation and qpos at t+1 as qpos action labels.
    """
    with h5py.File(path, "r") as handle:
        rgb = _decode_jpeg_frames(handle["observation/head_camera/rgb"][:])
        state = np.asarray(handle["joint_action/vector"][:], dtype=np.float32)
        if "endpose/left_endpose" not in handle:
            raise ValueError(f"{path} lacks endpose/left_endpose required for EMIX indexing.")
        left_eef = _pose_wxyz_to_rotvec(np.asarray(handle["endpose/left_endpose"][:], dtype=np.float32))
        instance_ids = np.asarray(handle["observation/head_camera/instance_id"][:]) if "observation/head_camera/instance_id" in handle else None
    length = min(len(rgb), len(state), len(left_eef))
    if length < 2:
        raise ValueError(f"{path} has fewer than two aligned frames.")
    if require_instance_annotations and (instance_ids is None or len(instance_ids) < length):
        raise ValueError(f"{path} lacks frame-aligned observation/head_camera/instance_id data.")
    rgb = np.asarray([_resize_rgb(frame, image_width, image_height) for frame in rgb[:length]])
    state = state[:length]
    left_eef = left_eef[:length]
    if state.ndim != 2 or state.shape[1] != 2 * left_qpos_dim:
        width = state.shape[1] if state.ndim == 2 else "invalid"
        raise ValueError(
            f"{path} has native qpos width {width}; expected {2 * left_qpos_dim}."
        )
    home_right_qpos = state[0, left_qpos_dim:].copy()
    right_deviation = float(np.max(np.abs(state[:, left_qpos_dim:] - home_right_qpos)))
    if right_deviation > 1e-3:
        raise ValueError(
            f"{path} right arm changed by {right_deviation:.6g}, above the 1e-3 fixed-arm tolerance."
        )
    # The policy contract is single-arm: validate right qpos, then omit it from
    # both observation.state and action.
    state = state[:, :left_qpos_dim].copy()
    action = state[1:].copy()
    return {
        "images": rgb[:-1],
        "state": state[:-1],
        "action": action,
        "left_eef": left_eef[:-1],
        # Keep the active gripper as a per-frame vector so it can be stored as
        # an explicit auxiliary feature in the LeRobot dataset.
        "left_gripper": state[:-1, left_qpos_dim - 1].astype(np.float32),
        "instance_ids": None if instance_ids is None else instance_ids[:length - 1],
    }


def _create_dataset(repo_id, root, fps, sample):
    """Create the target LeRobot dataset schema.

    @input: repo_id str, root str, fps int, sample aligned episode payload.
    @output: LeRobotDataset writer.
    @scenario: Persist exactly one policy image, proprioception vector, action, and index-only EEF feature.
    """
    state_dim = int(sample["state"].shape[1])
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        features={
            "observation.images.head_camera": {
                "dtype": "video",
                "shape": tuple(sample["images"].shape[1:]),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": [f"qpos_{i}" for i in range(state_dim)]},
            "observation.eef_pose": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
            # LeRobot maps shape=(1,) to a scalar HF Value but its frame
            # validator requires an array. Use a 1x1 array to satisfy both.
            "observation.left_gripper": {
                "dtype": "float32",
                "shape": (1, 1),
                "names": ["singleton", "left_gripper"],
            },
            "action": {"dtype": "float32", "shape": (state_dim,), "names": [f"qpos_target_{i}" for i in range(state_dim)]},
        },
        image_writer_threads=0,
    )


def main(args):
    """Convert all numbered HDF5 episodes from one RoboTwin collection directory.

    @input: argparse namespace containing source/target roots, instruction options,
        image size, and fps.
    @output: None; creates a new LeRobot dataset and fixes its image statistics.
    @scenario: Bridge native RoboTwin collection output into current EMIX training inputs.
    """
    started_at = time.perf_counter()
    source = Path(args.src).expanduser().resolve()
    episodes = sorted((source / "data").glob("episode*.hdf5"), key=lambda item: int(item.stem.removeprefix("episode")))
    if not episodes:
        raise ValueError(f"No episode*.hdf5 files found under {source / 'data'}.")
    LOGGER.info(
        "stage=convert-start episodes=%d image_size=%dx%d annotations=%s",
        len(episodes), args.image_width, args.image_height,
        bool(getattr(args, "instance_annotations", False)),
    )
    root = Path(args.tgt).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Target dataset root is not empty: {root}")
    if args.instruction_index < 0:
        raise ValueError("--instruction-index must be non-negative.")
    instructions_root = (
        Path(args.instructions_root).expanduser().resolve()
        if args.instructions_root
        else source / "instructions"
    )
    resolved_instructions = {}
    fallback_count = 0
    for path in episodes:
        task_text, from_robotwin = _resolve_instruction(
            path,
            instructions_root,
            args.instruction_type,
            args.instruction_index,
            args.task,
        )
        resolved_instructions[path] = task_text
        fallback_count += not from_robotwin
    if fallback_count:
        LOGGER.warning(
            "Using --task for %d/%d episodes because RoboTwin instruction sidecars are missing or empty.",
            fallback_count,
            len(episodes),
        )
    first = _load_episode(
        episodes[0], args.image_width, args.image_height, args.left_qpos_dim,
        require_instance_annotations=getattr(args, "instance_annotations", False),
    )
    repo_id = args.repo_id or default_repo_id_from_dataset_root(root)
    dataset = _create_dataset(repo_id, str(root), args.fps, first)
    annotations_root = root / "annotations"
    if getattr(args, "instance_annotations", False):
        annotations_root.mkdir(parents=True, exist_ok=True)
    for episode_index, path in enumerate(tqdm(episodes, desc="Converting RoboTwin episodes")):
        episode = _load_episode(
            path, args.image_width, args.image_height, args.left_qpos_dim,
            require_instance_annotations=getattr(args, "instance_annotations", False),
        )
        if getattr(args, "instance_annotations", False):
            object_path = source / "instance_objects" / f"{path.stem}.json"
            if not object_path.is_file():
                raise FileNotFoundError(f"Missing episode object table: {object_path}")
            renderer_objects = {int(key): value for key, value in json.loads(object_path.read_text(encoding="utf-8")).items()}
            frame_ids = np.arange(len(episode["instance_ids"]), dtype=np.int64)
            annotation = InstanceAnnotations.from_renderer(
                episode["instance_ids"], renderer_objects=renderer_objects,
                dataset_id=repo_id, episode_id=str(episode_index), camera="head_camera",
                frame_ids=frame_ids,
            )
            _, annotation = transform_pair(
                None, annotation,
                stream_key=(repo_id, str(episode_index), "head_camera"),
                frame_ids=frame_ids, indices=list(range(len(frame_ids))),
                size=(args.image_width, args.image_height),
            )
            if "patch_target" not in annotation.objects:
                raise ValueError(f"Episode {path.stem} object table has no patch_target role")
            annotation.save(annotations_root / f"episode_{episode_index}.instances.h5")
        for index in range(len(episode["action"])):
            dataset.add_frame({
                "observation.images.head_camera": episode["images"][index],
                "observation.state": episode["state"][index],
                "observation.eef_pose": episode["left_eef"][index],
                "observation.left_gripper": np.asarray(
                    [[episode["left_gripper"][index]]], dtype=np.float32
                ),
                "action": episode["action"][index],
                "task": resolved_instructions[path],
            })
        dataset.save_episode()
    dataset._wait_image_writer()
    dataset._close_writer()
    dataset.meta._close_writer()
    dataset.stop_image_writer()
    fix_lerobot_image_stats(root=str(root), repo_id=repo_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".complete").write_text("ok\n", encoding="ascii")
    LOGGER.info(
        "stage=convert-complete episodes=%d elapsed_s=%.1f output=%s",
        len(episodes), time.perf_counter() - started_at, root,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s-[convert_to_lerobot]-[run]-[%(message)s]", datefmt="%Y-%m-%dT%H:%M:%S")
    parser = argparse.ArgumentParser(description="Convert RoboTwin HDF5 episodes to LeRobot format")
    parser.add_argument("--src", required=True, help="RoboTwin collection source directory.")
    parser.add_argument("--tgt", required=True, help="Target LeRobot dataset directory.")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repository ID; defaults to the target directory name.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Legacy fallback task text used only when an episode instruction sidecar is unavailable.",
    )
    parser.add_argument(
        "--instructions-root",
        default=None,
        help="RoboTwin instruction sidecars; defaults to <src>/instructions.",
    )
    parser.add_argument(
        "--instruction-type",
        choices=("seen", "unseen"),
        default="seen",
        help="Which RoboTwin generated instruction split to store in LeRobot task.",
    )
    parser.add_argument(
        "--instruction-index",
        type=int,
        default=0,
        help="Prompt index within each episode sidecar (default: first prompt).",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=None,
        help="Deprecated compatibility option; ignored. All source episodes are converted.",
    )
    parser.add_argument("--image-width", type=int, default=84)
    parser.add_argument("--image-height", type=int, default=84)
    parser.add_argument("--fps", type=int, default=17, help="Approximate 250 Hz / save_freq 15 source rate.")
    parser.add_argument("--left-qpos-dim", type=int, default=7, help="Left Piper arm-plus-gripper dimensions.")
    parser.add_argument("--instance-annotations", action="store_true", help="Convert synchronized raw actor IDs into lossless annotation sidecars.")
    main(parser.parse_args())
