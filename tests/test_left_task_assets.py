"""Unit tests for the left-task asset generator and its on-disk artifacts.

The generator (scripts/generate_left_task_assets.py) is the single source of
truth for the 73 RoboTwin wrapper assets.  These tests verify the registry
contract, the render functions, the exported wrapper modules, and that the
on-disk artifacts match freshly-rendered payloads (``--check`` parity).
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

from generate_left_task_assets import (  # noqa: E402
    ACCEPTANCE_EPISODES,
    HEAD_CAMERA_TYPE_BY_VARIANT,
    INSTRUCTION_DIR,
    REGISTRY_PATH,
    STEP_LIMIT_FILE,
    TASK_CONFIG_DIR,
    TRAIN_EPISODES,
    VariantSpec,
    load_registry,
    main,
    render_eval_config,
    render_instruction,
    render_task_config,
    render_train_config,
    validate_registry,
    CENTERED_BATCH_VARIANTS,
    _VARIANT_SUFFIXES,
)


def _camel(semantic: str) -> str:
    """Camel-case a snake semantic task name: blocks_ranking_rgb -> BlocksRankingRgb."""
    return "".join(part.capitalize() for part in semantic.split("_"))


def _import_env_module(task_name: str):
    """Import an envs module under the RoboTwin CWD (asset paths are relative).

    envs.utils reads RoboTwin asset manifests via CWD-relative paths at import
    time; ``importlib.import_module`` caches the result, so this only needs the
    correct CWD on the first (uncached) import of each module.
    """
    original = Path.cwd()
    os.chdir(ROBOTWIN_ROOT)
    try:
        from envs._GLOBAL_CONFIGS import import_task_env

        return import_task_env(task_name)
    finally:
        os.chdir(original)


def _ascii_only(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


class TestRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = load_registry(REGISTRY_PATH)

    def test_registry_has_83_variants(self):
        self.assertEqual(len(self.specs), 83)

    def test_validate_registry_passes(self):
        validate_registry(self.specs)  # raises on any violation

    def test_active_families_have_six_cameras_and_deprecated_mug_has_three(self):
        by_semantic: dict[str, set[str]] = {}
        for spec in self.specs:
            by_semantic.setdefault(spec.semantic_task, set()).add(spec.camera_variant)
        self.assertEqual(len(by_semantic), 11)
        self.assertEqual(by_semantic["hanging_mug"], {
            "central_cam", "left_oppo_cam", "central_wide_cam"
        })
        for semantic, variants in by_semantic.items():
            if semantic != "hanging_mug":
                self.assertEqual(variants, set(_VARIANT_SUFFIXES))

    def test_task_names_are_canonical(self):
        for spec in self.specs:
            canonical = f"{spec.semantic_task}{_VARIANT_SUFFIXES[spec.camera_variant]}"
            self.assertEqual(spec.task_name, canonical)

    def test_source_step_limits_positive(self):
        for spec in self.specs:
            self.assertGreater(spec.source_step_limit, 0)

    def test_step_limit_file_has_all_task_names(self):
        text = STEP_LIMIT_FILE.read_text(encoding="utf-8")
        for spec in self.specs:
            self.assertIn(f"{spec.task_name}:", text)


class TestRenderTaskConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = load_registry(REGISTRY_PATH)

    @property
    def centered_specs(self):
        return [spec for spec in self.specs if spec.camera_variant in CENTERED_BATCH_VARIANTS]

    def test_episode_counts(self):
        for spec in self.specs:
            self.assertEqual(render_task_config(spec, TRAIN_EPISODES)["episode_num"], 500)
            self.assertEqual(
                render_task_config(spec, ACCEPTANCE_EPISODES)["episode_num"], 10
            )

    def test_centered_batch_has_rate_gate(self):
        for spec in self.specs:
            config = render_task_config(spec, TRAIN_EPISODES)
            if spec.camera_variant in CENTERED_BATCH_VARIANTS:
                self.assertEqual(config["max_seed_attempts"], 100)
            else:
                self.assertNotIn("max_seed_attempts", config)

    def test_single_arm_piper_embodiment(self):
        for spec in self.specs:
            cfg = render_task_config(spec, TRAIN_EPISODES)
            self.assertEqual(cfg["embodiment"], ["piper", "piper", 0.60])

    def test_head_camera_type_follows_variant(self):
        for spec in self.specs:
            camera = render_task_config(spec, TRAIN_EPISODES)["camera"]
            self.assertEqual(
                camera["head_camera_type"],
                HEAD_CAMERA_TYPE_BY_VARIANT[spec.camera_variant],
            )
            self.assertTrue(camera["collect_head_camera"])
            self.assertEqual(camera["wrist_camera_type"], "D435")
            self.assertFalse(camera["collect_wrist_camera"])

    def test_wide_variants_select_wide_sensor(self):
        for spec in self.specs:
            if spec.camera_variant not in {"central_wide_cam", "cen_arm_right_wide_cam",
                                           "cen_arm_top_cam"}:
                continue
            camera = render_task_config(spec, TRAIN_EPISODES)["camera"]
            self.assertEqual(camera["head_camera_type"], "D435-W")
            self.assertTrue(spec.task_name.endswith(_VARIANT_SUFFIXES[spec.camera_variant]))

    def test_data_type_is_rgb_qpos_endpose_only(self):
        for spec in self.specs:
            data_type = render_task_config(spec, TRAIN_EPISODES)["data_type"]
            self.assertTrue(data_type["rgb"])
            self.assertTrue(data_type["qpos"])
            self.assertTrue(data_type["endpose"])
            self.assertFalse(data_type["pointcloud"])
            self.assertFalse(data_type["depth"])
            self.assertFalse(data_type["third_view"])

    def test_domain_randomization_disabled(self):
        for spec in self.specs:
            dr = render_task_config(spec, TRAIN_EPISODES)["domain_randomization"]
            self.assertFalse(dr["random_background"])
            self.assertFalse(dr["cluttered_table"])
            self.assertEqual(dr["clean_background_rate"], 1)
            self.assertEqual(dr["random_head_camera_dis"], 0)
            self.assertFalse(dr["random_light"])

    def test_on_disk_task_configs_match_render(self):
        for spec in self.centered_specs:
            path = TASK_CONFIG_DIR / f"{spec.task_name}.yml"
            self.assertTrue(path.is_file(), f"missing task config {path}")
            on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
            rendered = render_task_config(spec, TRAIN_EPISODES)
            if spec.camera_variant in CENTERED_BATCH_VARIANTS:
                # dataflow.sh changes only the requested collection count.
                self.assertIn(on_disk.pop("episode_num"), (1, 5, 500))
                rendered.pop("episode_num")
            self.assertEqual(on_disk, rendered)

    def test_acceptance_configs_match_render_when_present(self):
        for spec in self.specs:
            path = TASK_CONFIG_DIR / f"{spec.task_name}_acceptance_10ep.yml"
            if path.exists():
                self.assertEqual(
                    yaml.safe_load(path.read_text(encoding="utf-8")),
                    render_task_config(spec, ACCEPTANCE_EPISODES),
                )


class TestInstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = load_registry(REGISTRY_PATH)

    @property
    def centered_specs(self):
        return [spec for spec in self.specs if spec.camera_variant in CENTERED_BATCH_VARIANTS]

    def _all_instruction_text(self, payload: dict) -> list[str]:
        texts = [payload["full_description"], payload["schema"], payload["preference"]]
        texts.extend(payload["seen"])
        texts.extend(payload["unseen"])
        return texts

    def test_render_returns_semantic_template(self):
        by_semantic: dict[str, VariantSpec] = {}
        for spec in self.specs:
            by_semantic.setdefault(spec.semantic_task, spec)
        for semantic, spec in by_semantic.items():
            payload = render_instruction(spec)
            self.assertIn("seen", payload)
            self.assertIn("unseen", payload)
            self.assertIn("full_description", payload)

    def test_all_text_is_ascii(self):
        for spec in self.specs:
            for text in self._all_instruction_text(render_instruction(spec)):
                self.assertTrue(_ascii_only(text), f"non-ascii instruction: {text!r}")

    def test_no_right_arm_reference(self):
        # Only right-ARM phrases violate the left-arm-only contract; positional
        # adjectives like "rightmost" refer to the table layout, not a limb.
        banned = ("right arm", "right-arm", "right hand", "right-hand")
        for spec in self.specs:
            for text in self._all_instruction_text(render_instruction(spec)):
                lowered = text.lower()
                for phrase in banned:
                    self.assertNotIn(phrase, lowered)

    def test_left_arm_objective_present(self):
        for spec in self.specs:
            payload = render_instruction(spec)
            blob = " ".join(self._all_instruction_text(payload)).lower()
            self.assertIn("left arm", blob)
            self.assertTrue(
                any(marker in blob for marker in ("{a}", "one arm")),
                f"{spec.semantic_task} instruction lacks a left-arm marker",
            )

    def test_on_disk_instruction_matches_render(self):
        for spec in self.centered_specs:
            path = INSTRUCTION_DIR / f"{spec.task_name}.json"
            self.assertTrue(path.is_file(), f"missing instruction {path}")
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, render_instruction(spec))

    def test_pair_instructions_are_identical(self):
        rendered: dict[str, dict] = {}
        for spec in self.specs:
            rendered.setdefault(spec.semantic_task, render_instruction(spec))
        for spec in self.specs:
            self.assertEqual(render_instruction(spec), rendered[spec.semantic_task])


class TestTrainEvalConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = load_registry(REGISTRY_PATH)

    def test_train_policy_is_emix_diffusion(self):
        for spec in self.specs:
            cfg = render_train_config(spec)
            self.assertEqual(cfg["policy"]["type"], "emix_diffusion")
            self.assertEqual(cfg["policy"]["vision_backbone"], "resnet18")
            self.assertEqual(cfg["dataset"]["repo_id"], spec.task_name)
            self.assertIn(spec.task_name, cfg["output_dir"])

    def test_train_emix_mixup_visual_key_is_head_camera(self):
        for spec in self.specs:
            cfg = render_train_config(spec)
            self.assertEqual(
                cfg["emix"]["mixup"]["visual_keys"], ["observation.images.head_camera"]
            )

    def test_eval_points_at_matching_task(self):
        for spec in self.specs:
            cfg = render_eval_config(spec)
            self.assertEqual(cfg["evaluation"]["task_name"], spec.task_name)
            self.assertEqual(cfg["evaluation"]["piper_distance"], 0.60)
            self.assertEqual(cfg["evaluation"]["episodes"], 100)
            self.assertIn(spec.task_name, cfg["evaluation"]["checkpoint"])
            self.assertIn(spec.task_name, cfg["evaluation"]["output_dir"])


class TestWrapperModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = load_registry(REGISTRY_PATH)

    def test_every_wrapper_module_exports_exact_class(self):
        for spec in self.specs:
            module = _import_env_module(spec.task_name)
            cls = getattr(module, spec.task_name)
            self.assertTrue(
                isinstance(cls, type),
                f"{spec.task_name} does not export a class",
            )

    def test_wrapper_subclasses_matching_impl(self):
        for spec in self.specs:
            impl_module = _import_env_module(f"{spec.semantic_task}_left_impl")
            impl_cls = getattr(impl_module, f"{_camel(spec.semantic_task)}LeftImpl")
            wrapper_cls = getattr(_import_env_module(spec.task_name), spec.task_name)
            self.assertTrue(
                issubclass(wrapper_cls, impl_cls),
                f"{spec.task_name} is not a subclass of {impl_cls.__name__}",
            )

    def test_wrapper_camera_variant_matches_registry(self):
        for spec in self.specs:
            wrapper_cls = getattr(_import_env_module(spec.task_name), spec.task_name)
            self.assertEqual(wrapper_cls.camera_variant, spec.camera_variant)

    def test_all_variant_wrappers_share_impl(self):
        by_semantic: dict[str, dict[str, str]] = {}
        for spec in self.specs:
            by_semantic.setdefault(spec.semantic_task, {})[spec.camera_variant] = spec.task_name
        for semantic, variants in by_semantic.items():
            central_impl = _import_env_module(variants["central_cam"]) \
                .__dict__.get(variants["central_cam"]).__mro__[1]
            for variant_name, task_name in variants.items():
                impl = _import_env_module(task_name).__dict__.get(task_name).__mro__[1]
                self.assertEqual(
                    central_impl, impl,
                    f"{semantic} {variant_name} wrapper does not share the impl",
                )


class TestCheckMode(unittest.TestCase):
    def test_check_returns_zero(self):
        self.assertEqual(
            main(["--check", "--centered-batch-only", "--registry", str(REGISTRY_PATH)]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
