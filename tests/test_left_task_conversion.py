"""Tests for the RoboTwin-to-LeRobot conversion and task lifecycle.

The conversion half exercises data_processing/convert_to_lerobot.py against
real HDF5 fixtures with a fake LeRobotDataset.create, so it runs without a
simulator or GPU.  The lifecycle half starts real SAPIEN scenes for every one
of the 22 left-arm variants; it is skipped unless ROBOTWIN_INTEGRATION=1 and
must run from the RoboTwin working directory with CUDA available.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import h5py
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSING_DIR = PROJECT_ROOT / "data_processing"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
for _path in (PROJECT_ROOT, DATA_PROCESSING_DIR, SCRIPTS_DIR, ROBOTWIN_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import convert_to_lerobot as convert  # noqa: E402

ROBOTWIN_INTEGRATION_ENABLED = os.environ.get("ROBOTWIN_INTEGRATION", "") == "1"
INTEGRATION_SEEDS = (100, 200, 300)


def _integration_task_subset() -> set[str] | None:
    """Optional comma-separated task filter via ROBOTWIN_INTEGRATION_TASKS."""
    raw = os.environ.get("ROBOTWIN_INTEGRATION_TASKS", "")
    if not raw.strip():
        return None
    return {name.strip() for name in raw.split(",") if name.strip()}


# ------------------------------------------------------------ HDF5 fixtures


def _jpeg_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise AssertionError("cv2.imencode failed")
    return buffer.tobytes()


def _write_episode(path: Path, n_frames: int, *, height=64, width=64,
                   state_cols=14, left_dim=7, include_endpose=True,
                   seed=0) -> None:
    """Write one realistic RoboTwin episode HDF5 file."""
    rng = np.random.default_rng(seed)
    encoded = []
    for _ in range(n_frames):
        image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        encoded.append(_jpeg_bytes(image))
    max_len = max(len(frame) for frame in encoded)
    rgb = np.zeros((n_frames, max_len), dtype=np.uint8)
    for index, frame in enumerate(encoded):
        rgb[index, : len(frame)] = np.frombuffer(frame, dtype=np.uint8)
    state = rng.standard_normal((n_frames, state_cols)).astype(np.float32)
    left_eef = np.zeros((n_frames, 7), dtype=np.float32)
    left_eef[:, :3] = rng.standard_normal((n_frames, 3)).astype(np.float32)
    left_eef[:, 3] = 1.0  # identity-ish quaternion (wxyz)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observation/head_camera/rgb", data=rgb)
        handle.create_dataset("joint_action/vector", data=state)
        if include_endpose:
            handle.create_dataset("endpose/left_endpose", data=left_eef)


class FakeLeRobotDataset:
    """Minimal in-memory stand-in for LeRobotDataset used by the conversion."""

    def __init__(self, repo_id: str, root: str | Path, fps: int,
                 features: dict, image_writer_threads: int = 0) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        self.fps = fps
        self.features = features
        self.frames: list[dict] = []
        self.episodes_saved = 0
        self.meta = type("Meta", (), {"_close_writer": lambda self: None})()
        self._writer_closed = False
        self._image_writer_stopped = False

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def add_frame(self, frame: dict) -> None:
        self.frames.append(dict(frame))

    def save_episode(self) -> None:
        self.episodes_saved += 1

    def _wait_image_writer(self) -> None:
        return None

    def _close_writer(self) -> None:
        self._writer_closed = True

    def stop_image_writer(self) -> None:
        self._image_writer_stopped = True


# --------------------------------------------------------------- conversion


class TestDecodeJpegFrames(unittest.TestCase):
    def test_decodes_rgb_frames(self):
        original = np.zeros((48, 48, 3), dtype=np.uint8)
        original[..., 0] = 240
        original[..., 1] = 128
        original[..., 2] = 16
        frame = _jpeg_bytes(original)
        width = max(len(frame), 32)
        rgb = np.zeros((2, width), dtype=np.uint8)
        rgb[0, : len(frame)] = np.frombuffer(frame, dtype=np.uint8)
        rgb[1, : len(frame)] = np.frombuffer(frame, dtype=np.uint8)
        decoded = convert._decode_jpeg_frames(rgb)
        self.assertEqual(decoded.shape, (2, 48, 48, 3))
        self.assertEqual(decoded.dtype, np.uint8)
        np.testing.assert_allclose(decoded[0, 20:28, 20:28], original[20:28, 20:28], atol=4)

    def test_invalid_bytes_raise(self):
        # Non-empty garbage that is not a JPEG: imdecode returns None (or, on
        # empty buffers, OpenCV raises directly); both must surface as errors.
        garbage = np.full((1, 8), 7, dtype=np.uint8)
        with self.assertRaises((ValueError, cv2.error)):
            convert._decode_jpeg_frames(garbage)


class TestResizeRgb(unittest.TestCase):
    def test_resizes_without_padding(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = convert._resize_rgb(image, 224, 224)
        self.assertEqual(resized.shape, (224, 224, 3))
        self.assertEqual(resized.dtype, np.uint8)


class TestPoseConversion(unittest.TestCase):
    def test_wxyz_to_rotvec_shape_and_translation(self):
        pose = np.zeros((4, 7), dtype=np.float32)
        pose[:, :3] = [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 0.5, 2.5], [4.0, 4.0, 4.0]]
        pose[:, 3] = 1.0  # identity quat w=1
        out = convert._pose_wxyz_to_rotvec(pose)
        self.assertEqual(out.shape, (4, 6))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out[:, :3], pose[:, :3], atol=1e-6)
        np.testing.assert_allclose(out[:, 3:], 0.0, atol=1e-6)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            convert._pose_wxyz_to_rotvec(np.zeros((7,), dtype=np.float32))


class TestInstructionResolution(unittest.TestCase):
    def test_missing_file_returns_none(self):
        missing = Path(tempfile.mkdtemp()) / "missing.json"
        self.assertIsNone(convert._read_instruction(missing, "seen", 0))

    def test_selects_prompt_by_type_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep0.json"
            path.write_text(json.dumps({"seen": ["a", "b"], "unseen": ["u"]}),
                            encoding="utf-8")
            self.assertEqual(convert._read_instruction(path, "seen", 1), "b")
            self.assertEqual(convert._read_instruction(path, "unseen", 0), "u")

    def test_out_of_range_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep0.json"
            path.write_text(json.dumps({"seen": ["a"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                convert._read_instruction(path, "seen", 5)

    def test_plain_string_payload_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep0.json"
            path.write_text(json.dumps("just a string"), encoding="utf-8")
            self.assertEqual(convert._read_instruction(path, "seen", 0), "just a string")

    def test_resolve_fallback_ordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = root / "episode0.json"
            sidecar.write_text(json.dumps({"seen": ["from sidecar"]}), encoding="utf-8")
            # Sidecar present: no fallback used.
            text, from_robotwin = convert._resolve_instruction(
                root / "episode0.hdf5", root, "seen", 0, "fallback task")
            self.assertEqual(text, "from sidecar")
            self.assertTrue(from_robotwin)
            # Sidecar missing: fallback used.
            text, from_robotwin = convert._resolve_instruction(
                root / "episode9.hdf5", root, "seen", 0, "fallback task")
            self.assertEqual(text, "fallback task")
            self.assertFalse(from_robotwin)
            # Sidecar missing and no fallback: raises.
            with self.assertRaises(ValueError):
                convert._resolve_instruction(root / "episode8.hdf5", root, "seen", 0, None)


class TestLoadEpisode(unittest.TestCase):
    def test_84x84_args_forwarded_and_right_qpos_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode0.hdf5"
            _write_episode(path, n_frames=8, height=480, width=640)
            episode = convert._load_episode(path, 84, 84, 7)
            self.assertEqual(episode["images"].shape, (7, 84, 84, 3))
            self.assertEqual(episode["images"].dtype, np.uint8)
            self.assertEqual(episode["state"].shape, (7, 14))
            self.assertEqual(episode["action"].shape, (7, 14))
            # Right suffix frozen to the first-frame home, even though the
            # source state drifts.
            home = episode["state"][0, 7:]
            np.testing.assert_array_equal(
                episode["action"][:, 7:],
                np.broadcast_to(home, episode["action"][:, 7:].shape),
            )
            self.assertEqual(episode["left_gripper"].shape, (7,))
            self.assertEqual(episode["left_eef"].shape, (7, 6))

    def test_fewer_than_two_frames_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode0.hdf5"
            _write_episode(path, n_frames=1)
            with self.assertRaises(ValueError):
                convert._load_episode(path, 84, 84, 7)

    def test_missing_endpose_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode0.hdf5"
            _write_episode(path, n_frames=5, include_endpose=False)
            with self.assertRaises(ValueError):
                convert._load_episode(path, 84, 84, 7)


class TestCreateDataset(unittest.TestCase):
    def test_features_match_sample_and_call_args(self):
        sample = {
            "images": np.zeros((5, 84, 84, 3), dtype=np.uint8),
            "state": np.zeros((5, 14), dtype=np.float32),
            "left_eef": np.zeros((5, 6), dtype=np.float32),
            "left_gripper": np.zeros((5,), dtype=np.float32),
        }
        with mock.patch.object(convert.LeRobotDataset, "create",
                               wraps=FakeLeRobotDataset.create) as create:
            dataset = convert._create_dataset("repo_id", "/tmp/root", 17, sample)
            self.assertEqual(dataset.repo_id, "repo_id")
            self.assertEqual(dataset.fps, 17)
            create.assert_called_once()
            kwargs = create.call_args.kwargs
            self.assertEqual(kwargs["image_writer_threads"], 0)
            features = kwargs["features"]
            self.assertEqual(features["observation.images.head_camera"]["shape"],
                             (84, 84, 3))
            self.assertEqual(features["observation.state"]["shape"], (14,))
            self.assertEqual(features["action"]["shape"], (14,))
            self.assertEqual(features["observation.left_gripper"]["shape"], (1, 1))


class TestConvertMain(unittest.TestCase):
    def _run_main(self, *, include_instructions=True):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        source = tmp / "source"
        (source / "data").mkdir(parents=True)
        for index in range(2):
            _write_episode(source / "data" / f"episode{index}.hdf5",
                           n_frames=6, seed=index)
        if include_instructions:
            (source / "instructions").mkdir()
            for index in range(2):
                (source / "instructions" / f"episode{index}.json").write_text(
                    json.dumps({"seen": [f"seen prompt {index}"]}), encoding="utf-8")
        target = tmp / "dataset"  # does not exist yet

        fix_calls: list[tuple] = []
        instances: list[FakeLeRobotDataset] = []

        def _fake_create(**kwargs):
            instance = FakeLeRobotDataset.create(**kwargs)
            instances.append(instance)
            return instance

        def _fake_fix_stats(*, root, repo_id):
            fix_calls.append((str(root), repo_id))

        with mock.patch.object(convert.LeRobotDataset, "create",
                               side_effect=_fake_create), \
             mock.patch.object(convert, "fix_lerobot_image_stats",
                               side_effect=_fake_fix_stats):
            args = argparse.Namespace(
                src=str(source),
                tgt=str(target),
                repo_id="test-repo",
                task=None,
                instructions_root=None,
                instruction_type="seen",
                instruction_index=0,
                image_width=84,
                image_height=84,
                fps=17,
                left_qpos_dim=7,
            )
            convert.main(args)
            dataset = instances[0]
            return dataset, fix_calls

    def test_converts_all_episodes_with_sidecar_tasks(self):
        dataset, fix_calls = self._run_main()
        self.assertEqual(dataset.episodes_saved, 2)
        self.assertEqual(len(dataset.frames), 2 * 5)  # 2 episodes x 5 frames
        keys = {"observation.images.head_camera", "observation.state",
                "observation.eef_pose", "observation.left_gripper", "action",
                "task"}
        for frame in dataset.frames:
            self.assertEqual(set(frame.keys()), keys)
        tasks = {frame["task"] for frame in dataset.frames}
        self.assertEqual(tasks, {"seen prompt 0", "seen prompt 1"})
        # Right qpos frozen per episode to that episode's first-frame home.
        frames_per_episode = 5
        for episode_index in range(2):
            group = dataset.frames[
                episode_index * frames_per_episode:(episode_index + 1) * frames_per_episode
            ]
            home = group[0]["action"][7:]
            for frame in group:
                np.testing.assert_array_equal(frame["action"][7:], home)
        self.assertEqual(len(fix_calls), 1)
        self.assertEqual(fix_calls[0], (str(Path(dataset.root)), "test-repo"))
        self.assertTrue(dataset._writer_closed)
        self.assertTrue(dataset._image_writer_stopped)

    def test_missing_instructions_with_no_fallback_raises(self):
        with self.assertRaises(ValueError):
            self._run_main(include_instructions=False)


# -------------------------------------------------------------- integration


def _integration_args() -> list[str]:
    from generate_left_task_assets import load_registry, REGISTRY_PATH
    return [spec.task_name for spec in load_registry(REGISTRY_PATH)]


def _build_setup_args(task_name: str, config_name: str, save_path: Path) -> dict:
    """Assemble setup_demo kwargs exactly like RoboTwin script/collect_data.py."""
    with open(ROBOTWIN_ROOT / "task_config" / f"{config_name}.yml",
              encoding="utf-8") as handle:
        args = yaml.safe_load(handle)
    args["task_name"] = task_name
    with open(ROBOTWIN_ROOT / "task_config" / "_embodiment_config.yml",
              encoding="utf-8") as handle:
        embodiment_types = yaml.safe_load(handle)
    embodiment = args["embodiment"]
    if len(embodiment) == 3:
        args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False

    def _robot_config(robot_file: str) -> dict:
        with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    args["left_embodiment_config"] = _robot_config(args["left_robot_file"])
    args["right_embodiment_config"] = _robot_config(args["right_robot_file"])
    args["embodiment_name"] = "piper+piper"
    args["task_config"] = config_name
    args["save_path"] = str(save_path)
    args["need_plan"] = True
    # seed is passed explicitly to setup_demo (collect_data.py passes it as a
    # dedicated kwarg; the YAML config never carries a seed key).
    return args


@unittest.skipUnless(ROBOTWIN_INTEGRATION_ENABLED,
                     "set ROBOTWIN_INTEGRATION=1 to run SAPIEN lifecycle tests")
class TestTaskLifecycle(unittest.TestCase):
    """Setup -> play -> check -> close for every left-arm variant."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._original_cwd = Path.cwd()
        os.chdir(ROBOTWIN_ROOT)
        cls.task_names = _integration_args()
        subset = _integration_task_subset()
        if subset is not None:
            cls.task_names = [name for name in cls.task_names if name in subset]
        cls.tmp_root = Path(tempfile.mkdtemp(prefix="left-task-lifecycle-"))
        cls.instances = {}

    @classmethod
    def tearDownClass(cls) -> None:
        for env in cls.instances.values():
            try:
                env.close_env()
            except Exception:
                pass
        shutil.rmtree(cls.tmp_root, ignore_errors=True)
        os.chdir(cls._original_cwd)

    def _episode_lifecycle(self, task_name: str, seed: int) -> None:
        import importlib
        module = importlib.import_module(f"envs.{task_name}")
        env = getattr(module, task_name)()
        self.instances[task_name] = env
        save_path = self.tmp_root / task_name / str(seed)
        save_path.mkdir(parents=True, exist_ok=True)
        args = _build_setup_args(task_name, task_name, save_path)
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        info = env.play_once()
        self.assertIsInstance(env.plan_success, bool)
        success = env.check_success()
        self.assertIsInstance(success, bool)
        self.assertIsInstance(info, dict)
        env.close_env()
        self.instances.pop(task_name, None)

    def test_all_variants_setup_play_check_close(self):
        for task_name in self.task_names:
            for seed in INTEGRATION_SEEDS:
                with self.subTest(task=task_name, seed=seed):
                    self._episode_lifecycle(task_name, seed)


if __name__ == "__main__":
    unittest.main()
