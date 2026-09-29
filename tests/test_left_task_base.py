"""Unit tests for the shared left-arm-only task base contract.

These tests exercise LeftTaskBase without starting a SAPIEN scene: the move
rejection, right-arm projection, and camera configuration paths are all pure
with respect to robot control and run on plain objects.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "RoboTwin"
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

# envs.utils reads RoboTwin asset manifests via CWD-relative paths at import
# time.  Import under the RoboTwin working directory, then restore CWD so the
# rest of the suite (including non-RoboTwin tests) runs from the project root.
_ORIGINAL_CWD = Path.cwd()
os.chdir(ROBOTWIN_ROOT)
try:
    from envs.left.left_task_base import (  # noqa: E402
        CAMERA_SPECS,
        CEN_ARM_RIGHT_WIDE_CAM,
        CEN_ARM_NEAR_SIDE_CAM,
        CEN_ARM_SIDE_CAM,
        CEN_ARM_FRONT_CAM,
        CENTRAL_CAM,
        CENTRAL_WIDE_CAM,
        CENTERED_BATCH_ARM_SEPARATION_M,
        LEFT_OPPO_CAM,
        LEFT_QPOS_DIM,
        NATIVE_QPOS_DIM,
        RIGHT_ARM_HIDDEN_WORLD_X_M,
        RIGHT_ARM_TOLERANCE,
        ActorSamplingSpec,
        CameraVariant,
        LeftArmTaskError,
        LeftTaskBase,
        XYYawRange,
    )
    from envs.left.left_task_manifests import (  # noqa: E402
        CENTERED_WIDE_WORKSPACE,
        all_manifests,
        centered_wide_workspace_manifest,
        manifest_hash,
    )
    from envs.utils import Action, ArmTag  # noqa: E402
finally:
    os.chdir(_ORIGINAL_CWD)


class _DummyTask(LeftTaskBase):
    camera_variant: CameraVariant = "central_cam"


class _OppoCameraTask(LeftTaskBase):
    camera_variant: CameraVariant = "left_oppo_cam"


class _WideCameraTask(LeftTaskBase):
    camera_variant: CameraVariant = "central_wide_cam"


class _CenteredWideTask(LeftTaskBase):
    camera_variant: CameraVariant = "cen_arm_right_wide_cam"


def _left_move_action() -> Action:
    return Action(ArmTag("left"), "move", target_pose=[0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0])


def _right_move_action() -> Action:
    return Action(ArmTag("right"), "move", target_pose=[0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0])


class TestMoveRejectsRightArm(unittest.TestCase):
    def test_rejects_right_arm_move(self):
        task = _DummyTask()
        task.task_name = "dummy"
        with self.assertRaises(LeftArmTaskError):
            task.move((ArmTag("right"), [_right_move_action()]), None)

    def test_rejects_right_arm_in_second_group(self):
        task = _DummyTask()
        task.task_name = "dummy"
        with self.assertRaises(LeftArmTaskError):
            task.move((ArmTag("left"), [_left_move_action()]), (ArmTag("right"), [_right_move_action()]))

    def test_rejects_parallel_action_groups(self):
        task = _DummyTask()
        task.task_name = "dummy"
        with self.assertRaises(LeftArmTaskError):
            task.move(
                (ArmTag("left"), [_left_move_action()]),
                (ArmTag("left"), [_left_move_action()]),
            )

    def test_rejects_string_right_tag(self):
        task = _DummyTask()
        task.task_name = "dummy"
        with self.assertRaises(LeftArmTaskError):
            task.move(("right", [_right_move_action()]), None)

    def test_rejects_invalid_arm_tag_value(self):
        task = _DummyTask()
        task.task_name = "dummy"
        with self.assertRaises((LeftArmTaskError, ValueError)):
            task.move(("front", [_left_move_action()]), None)

    def test_accepts_left_move_before_delegation(self):
        # A left-only single group is legal; it proceeds to super().move which
        # is mocked out here so we only assert no early rejection.
        task = _DummyTask()
        task.task_name = "dummy"
        original = LeftTaskBase.move

        def fake_move(self_, actions_by_arm1, actions_by_arm2=None, save_freq=-1):
            return ("delegated", actions_by_arm1)

        LeftTaskBase.move = fake_move
        try:
            result = task.move((ArmTag("left"), [_left_move_action()]), None)
            self.assertEqual(result[0], "delegated")
        finally:
            LeftTaskBase.move = original


class TestRightArmProjection(unittest.TestCase):
    def _task_with_home(self) -> _DummyTask:
        task = _DummyTask()
        task.task_name = "dummy"
        task.right_home_qpos = np.arange(LEFT_QPOS_DIM, dtype=np.float32) + 0.5
        return task

    def test_projects_right_suffix_to_home(self):
        task = self._task_with_home()
        action = np.zeros(NATIVE_QPOS_DIM, dtype=np.float64)
        projected = task.project_right_home(action)
        np.testing.assert_array_equal(projected[LEFT_QPOS_DIM:], task.right_home_qpos)
        np.testing.assert_array_equal(projected[:LEFT_QPOS_DIM], np.zeros(LEFT_QPOS_DIM))

    def test_returns_copy_not_view(self):
        task = self._task_with_home()
        action = np.zeros(NATIVE_QPOS_DIM, dtype=np.float64)
        projected = task.project_right_home(action)
        projected[:] = 99.0
        # The caller's buffer must be untouched.
        np.testing.assert_array_equal(action, np.zeros(NATIVE_QPOS_DIM))

    def test_rejects_wrong_width(self):
        task = self._task_with_home()
        with self.assertRaises(LeftArmTaskError):
            task.project_right_home(np.zeros(NATIVE_QPOS_DIM - 1))
        with self.assertRaises(LeftArmTaskError):
            task.project_right_home(np.zeros(NATIVE_QPOS_DIM + 1))

    def test_rejects_2d_input(self):
        task = self._task_with_home()
        with self.assertRaises(LeftArmTaskError):
            task.project_right_home(np.zeros((2, NATIVE_QPOS_DIM)))

    def test_rejects_before_capture(self):
        task = _DummyTask()
        task.right_home_qpos = None
        with self.assertRaises(LeftArmTaskError):
            task.project_right_home(np.zeros(NATIVE_QPOS_DIM))

    def test_take_action_rejects_ee_control(self):
        task = self._task_with_home()
        with self.assertRaises(LeftArmTaskError):
            task.take_action(np.zeros(NATIVE_QPOS_DIM), action_type="ee")


class TestRightArmVisibility(unittest.TestCase):
    class _RightEntity:
        def __init__(self):
            from sapien.core import Pose

            self.pose = Pose([0.3, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

        def get_root_pose(self):
            return self.pose

        def set_root_pose(self, pose):
            self.pose = pose

    def test_moves_right_arm_to_hidden_world_position(self):
        task = _CenteredWideTask()
        task.robot = type("Robot", (), {"right_entity": self._RightEntity()})()
        task.move_right_arm_out_of_view()
        self.assertAlmostEqual(
            task.robot.right_entity.get_root_pose().p[0],
            RIGHT_ARM_HIDDEN_WORLD_X_M,
            places=4,
        )


class TestCameraConfiguration(unittest.TestCase):
    def _kwargs(self) -> dict:
        return {
            "left_embodiment_config": {
                "static_camera_list": [
                    {
                        "name": "head_camera",
                        "position": [1.0, 2.0, 3.0],
                        "forward": [0.0, 0.0, 1.0],
                        "left": [0.0, 1.0, 0.0],
                    }
                ]
            }
        }

    def test_central_cam_preserves_head_camera(self):
        task = _DummyTask()
        out = task.configure_camera(self._kwargs())
        camera = out["left_embodiment_config"]["static_camera_list"][0]
        self.assertEqual(camera["position"], [1.0, 2.0, 3.0])
        self.assertEqual(camera["forward"], [0.0, 0.0, 1.0])
        self.assertEqual(camera["left"], [0.0, 1.0, 0.0])

    def test_central_cam_embeds_variant_key(self):
        task = _DummyTask()
        out = task.configure_camera(self._kwargs())
        self.assertEqual(out["camera_variant"], "central_cam")

    def test_oppo_cam_overwrites_position_forward_left(self):
        task = _OppoCameraTask()
        out = task.configure_camera(self._kwargs())
        camera = out["left_embodiment_config"]["static_camera_list"][0]
        self.assertEqual(camera["position"], list(LEFT_OPPO_CAM.position))
        self.assertEqual(camera["forward"], list(LEFT_OPPO_CAM.forward))
        self.assertEqual(camera["left"], list(LEFT_OPPO_CAM.left))

    def test_oppo_cam_embeds_variant_key(self):
        task = _OppoCameraTask()
        out = task.configure_camera(self._kwargs())
        self.assertEqual(out["camera_variant"], "left_oppo_cam")

    def test_wide_cam_preserves_pose_and_selects_wide_type(self):
        task = _WideCameraTask()
        kw = self._kwargs()
        kw["camera"] = {"head_camera_type": "D435", "collect_head_camera": True}
        out = task.configure_camera(kw)
        camera = out["left_embodiment_config"]["static_camera_list"][0]
        self.assertEqual(camera["position"], list(CENTRAL_WIDE_CAM.position))
        self.assertEqual(camera["forward"], list(CENTRAL_WIDE_CAM.forward))
        self.assertEqual(camera["left"], list(CENTRAL_WIDE_CAM.left))
        self.assertEqual(out["camera"]["head_camera_type"], "D435-W")

    def test_wide_cam_requires_camera_config(self):
        task = _WideCameraTask()
        with self.assertRaises(LeftArmTaskError):
            task.configure_camera(self._kwargs())

    def test_wide_cam_embeds_variant_key(self):
        task = _WideCameraTask()
        kw = self._kwargs()
        kw["camera"] = {"head_camera_type": "D435"}
        out = task.configure_camera(kw)
        self.assertEqual(out["camera_variant"], "central_wide_cam")

    def test_centered_batch_keeps_left_center_and_hides_right_arm(self):
        task = _CenteredWideTask()
        task.manifest = all_manifests()["blocks_ranking_rgb"]
        kw = self._kwargs()
        kw["camera"] = {"head_camera_type": "D435", "collect_head_camera": True}
        kw["embodiment"] = ["piper", "piper", 0.60]
        kw["left_embodiment_config"]["robot_pose"] = [[0.0, -0.45, 0.75, 1, 0, 0, 1]]
        out = task.configure_camera(kw)
        self.assertEqual(out["embodiment"][2], CENTERED_BATCH_ARM_SEPARATION_M)
        self.assertEqual(
            out["left_embodiment_config"]["robot_pose"][0][0],
            CENTERED_BATCH_ARM_SEPARATION_M / 2.0,
        )
        camera = out["left_embodiment_config"]["static_camera_list"][0]
        self.assertEqual(camera["position"], list(CEN_ARM_RIGHT_WIDE_CAM.position))
        self.assertEqual(camera["forward"], list(CEN_ARM_RIGHT_WIDE_CAM.forward))
        self.assertEqual(out["camera"]["head_camera_type"], "D435-W")

    def test_centered_side_cam_shares_batch_contract(self):
        task = _CenteredWideTask()
        task.camera_variant = "cen_arm_side_cam"
        task.manifest = all_manifests()["blocks_ranking_rgb"]
        kw = self._kwargs()
        kw["camera"] = {"head_camera_type": "D435", "collect_head_camera": True}
        kw["embodiment"] = ["piper", "piper", 0.60]
        kw["left_embodiment_config"]["robot_pose"] = [[0.0, -0.45, 0.75, 1, 0, 0, 1]]
        out = task.configure_camera(kw)
        self.assertEqual(
            out["left_embodiment_config"]["robot_pose"][0][0],
            CENTERED_BATCH_ARM_SEPARATION_M / 2.0,
        )
        camera = out["left_embodiment_config"]["static_camera_list"][0]
        self.assertEqual(camera["position"], list(CEN_ARM_SIDE_CAM.position))
        self.assertEqual(camera["forward"], list(CEN_ARM_SIDE_CAM.forward))
        self.assertEqual(out["camera"]["head_camera_type"], "D435-W")


    def test_requires_exactly_one_head_camera(self):
        task = _DummyTask()
        kw = {
            "left_embodiment_config": {
                "static_camera_list": [
                    {"name": "wrist_camera"},
                    {"name": "wrist_camera"},
                ]
            }
        }
        with self.assertRaises(ValueError):
            task.configure_camera(kw)

    def test_rejects_disabled_head_camera(self):
        task = _DummyTask()
        kw = self._kwargs()
        kw["camera"] = {"collect_head_camera": False}
        with self.assertRaises(LeftArmTaskError):
            task.configure_camera(kw)

    def test_rejects_enabled_wrist_camera(self):
        task = _DummyTask()
        kw = self._kwargs()
        kw["camera"] = {"collect_wrist_camera": True}
        with self.assertRaises(LeftArmTaskError):
            task.configure_camera(kw)

    def test_missing_embodiment_config_rejected(self):
        task = _DummyTask()
        with self.assertRaises(LeftArmTaskError):
            task.configure_camera({})

    def test_unknown_camera_variant_rejected(self):
        task = _DummyTask()
        task.camera_variant = "bogus_cam"  # type: ignore[assignment]
        with self.assertRaises(LeftArmTaskError):
            task.configure_camera(self._kwargs())


class TestCenteredWorkspaceContract(unittest.TestCase):
    def test_all_active_families_share_calibrated_workspace(self):
        from envs.left.left_task_manifests import CENTERED_FAMILY_OVERRIDES
        for semantic_task, manifest in all_manifests().items():
            if semantic_task == "hanging_mug":
                continue
            calibrated = centered_wide_workspace_manifest(manifest)
            self.assertTrue(calibrated.actor_specs)
            override = CENTERED_FAMILY_OVERRIDES.get(semantic_task, {})
            for name, spec in calibrated.actor_specs.items():
                expected = override.get(name, CENTERED_WIDE_WORKSPACE)
                self.assertEqual(spec.workspace, expected)


class TestCameraSpecs(unittest.TestCase):
    def test_spec_contract_values(self):
        self.assertEqual(CENTRAL_CAM.position, None)
        self.assertEqual(CENTRAL_CAM.forward, None)
        self.assertEqual(CENTRAL_CAM.left, None)
        self.assertEqual(LEFT_OPPO_CAM.name, "left_oppo_cam")
        self.assertEqual(LEFT_OPPO_CAM.head_camera_type, None)
        self.assertEqual(CENTRAL_WIDE_CAM.name, "central_wide_cam")
        self.assertEqual(CENTRAL_WIDE_CAM.position, (0.268, -0.45, 1.35))
        self.assertEqual(CENTRAL_WIDE_CAM.forward, (0.0, 0.6, -0.8))
        self.assertEqual(CENTRAL_WIDE_CAM.left, (-1.0, 0.0, 0.0))
        self.assertEqual(CENTRAL_WIDE_CAM.head_camera_type, "D435-W")
        self.assertEqual(CEN_ARM_NEAR_SIDE_CAM.name, "cen_arm_near_side_cam")
        self.assertEqual(CEN_ARM_NEAR_SIDE_CAM.position, (0.20, -0.45, 1.35))
        self.assertEqual(CEN_ARM_SIDE_CAM.name, "cen_arm_side_cam")
        self.assertEqual(CEN_ARM_SIDE_CAM.position, (0.45, -0.45, 0.95))
        self.assertEqual(
            CEN_ARM_SIDE_CAM.forward, (-0.707107, 0.707107, 0.0))
        self.assertEqual(
            CEN_ARM_SIDE_CAM.left, (-0.707107, -0.707107, 0.0))
        self.assertEqual(CEN_ARM_FRONT_CAM.position, (0.0, -0.30, 0.8))
        self.assertEqual(
            CEN_ARM_FRONT_CAM.forward, (0.0, 0.866025, -0.5))
        self.assertEqual(CEN_ARM_FRONT_CAM.head_camera_type, "D435-W")
        self.assertEqual(CEN_ARM_SIDE_CAM.head_camera_type, "D435-W")

    def test_specs_cover_all_variants(self):
        self.assertEqual(set(CAMERA_SPECS.keys()),
                         {"central_cam", "left_oppo_cam", "central_wide_cam",
                          "cen_arm_right_wide_cam", "cen_arm_near_side_cam",
                          "cen_arm_side_cam", "cen_arm_front_cam",
                          "cen_arm_top_cam"})

    def test_centered_right_camera_is_shifted_default_wide(self):
        spec = CAMERA_SPECS["cen_arm_right_wide_cam"]
        self.assertEqual(spec.position, (0.268, -0.45, 1.35))
        self.assertEqual(spec.forward, (0.0, 0.6, -0.8))
        self.assertEqual(spec.left, (-1.0, 0.0, 0.0))
        self.assertEqual(spec.head_camera_type, "D435-W")

    def test_centered_near_side_camera_aims_at_axis_point_0_90(self):
        spec = CAMERA_SPECS["cen_arm_near_side_cam"]
        forward = np.asarray(spec.forward, dtype=float)
        forward /= np.linalg.norm(forward)
        # C shifted +X by 0.20 m and raised to z = 1.35 m.
        position = np.asarray(spec.position, dtype=float)
        self.assertEqual(tuple(position), (0.20, -0.45, 1.35))
        # Sight-line through the workspace-axis point (0, 0, 0.90) with a
        # 42.42-degree depression; the xoy azimuth still points at O.
        sight = (np.array([0.0, 0.0, 0.9]) - position)
        sight /= np.linalg.norm(sight)
        np.testing.assert_allclose(forward, sight, atol=1e-6)
        azimuth = forward[:2] / np.linalg.norm(forward[:2])
        np.testing.assert_allclose(azimuth, (-0.20, 0.45) / np.hypot(0.20, 0.45), atol=1e-6)
        left = np.asarray(spec.left, dtype=float)
        self.assertAlmostEqual(float(left[2]), 0.0)  # level horizon
        up = np.cross(forward, left)
        self.assertGreater(float(up[2]), 0.0)  # up keeps a +Z component

    def test_centered_side_camera_is_level_at_0_95(self):
        spec = CAMERA_SPECS["cen_arm_side_cam"]
        forward = np.asarray(spec.forward, dtype=float)
        forward /= np.linalg.norm(forward)
        # Diagonal corner at (0.45, -0.45, 0.95).
        position = np.asarray(spec.position, dtype=float)
        self.assertEqual(tuple(position), (0.45, -0.45, 0.95))
        # Level sight-line at the point (0, 0, 0.95): the azimuth toward
        # the workspace centre O in the xoy projection.
        self.assertAlmostEqual(float(forward[2]), 0.0, places=6)
        azimuth = forward[:2] / np.linalg.norm(forward[:2])
        np.testing.assert_allclose(
            azimuth, (-0.45, 0.45) / np.hypot(0.45, 0.45), atol=1e-6)
        left = np.asarray(spec.left, dtype=float)
        self.assertAlmostEqual(float(left[2]), 0.0)  # level horizon
        up = np.cross(forward, left)
        np.testing.assert_allclose(up / np.linalg.norm(up), (0.0, 0.0, 1.0), atol=1e-6)


class TestCenteredStaticPoseSampling(unittest.TestCase):
    def test_centered_static_actor_preserves_asset_orientation(self):
        task = _CenteredWideTask()
        task.camera_variant = "cen_arm_right_wide_cam"
        spec = ActorSamplingSpec(
            "container", XYYawRange((-0.2, 0.2), (-0.15, 0.15), (0.0, 0.5)),
            0.0, True, (1.0, 0.0, 0.0, 0.0),
        )
        poses = [task.sample_spec_pose(spec) for _ in range(4)]
        for pose in poses:
            np.testing.assert_allclose(pose.q, spec.base_quat)

    def test_centered_dynamic_actor_still_randomizes_yaw(self):
        task = _CenteredWideTask()
        task.camera_variant = "cen_arm_right_wide_cam"
        spec = ActorSamplingSpec(
            "pickup", XYYawRange((-0.2, 0.2), (-0.15, 0.15), (0.0, 0.5)),
            0.0, False, (1.0, 0.0, 0.0, 0.0),
        )
        poses = [task.sample_spec_pose(spec) for _ in range(12)]
        self.assertTrue(any(not np.allclose(pose.q, spec.base_quat) for pose in poses))


class TestManifestPairing(unittest.TestCase):
    def test_each_pair_shares_one_manifest_hash(self):
        manifests = all_manifests()
        self.assertEqual(len(manifests), 11)
        for semantic_task, manifest in manifests.items():
            self.assertEqual(manifest.semantic_task, semantic_task)
            self.assertIsInstance(manifest_hash(manifest), str)
            self.assertEqual(len(manifest_hash(manifest)), 16)

    def test_pair_hashes_are_stable_and_documented(self):
        # Expected hashes are frozen in each impl docstring Pilot evidence.
        expected = {
            "blocks_ranking_rgb": "3b292b9e1d535999",
            "blocks_ranking_size": "ed8fb45f24857693",
            "hanging_mug": "28ee798eb3dfcde4",
            "place_bread_basket": "6d79f86cb766c76c",
            "place_bread_skillet": "e0f8b156b7b9c742",
            "place_burger_fries": "176bcde62f9cb29c",
            "place_can_basket": "ec2e2b5aa098c862",
            "place_cans_plasticbox": "d13fd6141c9178fe",
            "place_dual_shoes": "ce583ff0cacdbd38",
            "stack_blocks_two": "2585a2c9d73c5073",
            "stack_bowls_two": "ace5714486a162e4",
        }
        manifests = all_manifests()
        for semantic_task, expected_hash in expected.items():
            self.assertEqual(manifest_hash(manifests[semantic_task]), expected_hash)


class TestRightArmTolerance(unittest.TestCase):
    def test_tolerance_is_positive_and_finite(self):
        self.assertGreater(RIGHT_ARM_TOLERANCE, 0.0)
        self.assertTrue(np.isfinite(RIGHT_ARM_TOLERANCE))


if __name__ == "__main__":
    unittest.main()
