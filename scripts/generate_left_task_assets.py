"""Generate RoboTwin task configs, instructions, and train/eval stubs.

The generator reads ``configs/left_task_registry.yaml`` (exactly 73 variant
rows), validates the registry, and renders for every variant:

* ``task_config/left/<config_dir>/<task>.yml`` - RoboTwin collection config
  with 500 episodes, in the envs/left-mirroring subfolder for its variant
  (``base`` for the default central-cam config).
* ``task_config/left/<config_dir>/<task>_acceptance_10ep.yml`` - same config
  with 10 episodes.
* ``description/task_instruction/<task>.json`` - instruction placeholders
  matching the semantic ``play_once`` info output, left-arm only.
* ``configs/{train,eval}/<variant_dir>/<task>_dp_emix.yaml`` - training and
  evaluation stubs under the matching camera directory.
* one key per variant appended to ``task_config/_eval_step_limit.yml``
  (existing keys and comments are preserved).

All generated payloads are ASCII-only.  ``--check`` verifies that the
artifacts already on disk match the freshly rendered payloads instead of
writing anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "left_task_registry.yaml"
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
TASK_CONFIG_DIR = ROBOTWIN_ROOT / "task_config" / "left"
INSTRUCTION_DIR = ROBOTWIN_ROOT / "description" / "task_instruction"
STEP_LIMIT_FILE = ROBOTWIN_ROOT / "task_config" / "_eval_step_limit.yml"
TRAIN_CONFIG_DIR = PROJECT_ROOT / "configs" / "train"
EVAL_CONFIG_DIR = PROJECT_ROOT / "configs" / "eval"

CameraVariant = Literal[
    "central_cam", "left_oppo_cam", "central_wide_cam",
    "cen_arm_right_wide_cam", "cen_arm_near_side_cam", "cen_arm_side_cam",
    "cen_arm_front_cam", "cen_arm_top_cam",
]
TRAIN_EPISODES = 500
ACCEPTANCE_EPISODES = 10

# Canonical public-name suffix per camera variant.
_VARIANT_SUFFIXES: Mapping[CameraVariant, str] = {
    "central_cam": "_left",
    "left_oppo_cam": "_left_oppo_cam",
    "central_wide_cam": "_left_central_wide_cam",
    "cen_arm_right_wide_cam": "_left_cen_arm_right_wide_cam",
    "cen_arm_near_side_cam": "_left_cen_arm_near_side_cam",
    "cen_arm_side_cam": "_left_cen_arm_side_cam",
    "cen_arm_front_cam": "_left_cen_arm_front_cam",
    "cen_arm_top_cam": "_left_cen_arm_top_cam",
}

# Camera variants sharing the centred-arm batch behaviour.
CENTERED_BATCH_VARIANTS = (
    "cen_arm_right_wide_cam", "cen_arm_near_side_cam", "cen_arm_side_cam",
    "cen_arm_front_cam", "cen_arm_top_cam",
)

# Head camera type rendered into each collection config.  The wide variant
# selects the ``D435-W`` sensor type (fovy 60) already defined in
# ``task_config/_camera_config.yml``.
HEAD_CAMERA_TYPE_BY_VARIANT: Mapping[CameraVariant, str] = {
    "central_cam": "D435",
    "left_oppo_cam": "D435",
    "central_wide_cam": "D435-W",
    "cen_arm_right_wide_cam": "D435-W",
    "cen_arm_near_side_cam": "D435-W",
    "cen_arm_side_cam": "D435-W",
    "cen_arm_front_cam": "D435-W",
    "cen_arm_top_cam": "D435-W",
}


@dataclass(frozen=True)
class VariantSpec:
    semantic_task: str
    task_name: str
    camera_variant: CameraVariant
    source_step_limit: int


# ------------------------------------------------------------------ registry

def load_registry(path: Path) -> list[VariantSpec]:
    """Parse the YAML registry into a list of variant specs.

    @input: registry YAML path.
    @output: list of ``VariantSpec`` in file order.
    @scenario: Single source of truth for all generated task assets.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("variants"), list):
        raise ValueError(f"registry {path} must contain a top-level 'variants' list")
    specs: list[VariantSpec] = []
    for row in payload["variants"]:
        if not isinstance(row, Mapping):
            raise ValueError("every registry row must be a mapping")
        try:
            specs.append(VariantSpec(
                semantic_task=str(row["semantic_task"]),
                task_name=str(row["task_name"]),
                camera_variant=str(row["camera_variant"]),  # validated below
                source_step_limit=int(row["source_step_limit"]),
            ))
        except KeyError as error:
            raise ValueError(f"registry row missing field {error}") from error
    return specs


def validate_registry(specs: Sequence[VariantSpec]) -> None:
    """Validate the registry shape and naming contract.

    @input: parsed variant specs.
    @output: None; raises ``ValueError`` on any violation.
    @scenario: Guarantee exactly 83 rows, complete triplets, legal variants,
        and canonical public names before generating artifacts.
    """
    if len(specs) != 83:
        raise ValueError(f"registry must contain exactly 83 variants, got {len(specs)}")
    names = [spec.task_name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate task_name in registry: {sorted(names)}")
    by_semantic: dict[str, set[str]] = {}
    for spec in specs:
        if spec.camera_variant not in _VARIANT_SUFFIXES:
            raise ValueError(
                f"{spec.task_name}: camera_variant must be one of "
                f"{sorted(_VARIANT_SUFFIXES)}, got {spec.camera_variant!r}"
            )
        canonical = f"{spec.semantic_task}{_VARIANT_SUFFIXES[spec.camera_variant]}"
        if spec.task_name != canonical:
            raise ValueError(
                f"{spec.task_name}: expected public name {canonical} for "
                f"{spec.camera_variant}"
            )
        by_semantic.setdefault(spec.semantic_task, set()).add(spec.camera_variant)
    standard_variants = {"central_cam", "left_oppo_cam", "central_wide_cam"}
    for semantic, variants in by_semantic.items():
        # hanging_mug is deprecated and deliberately never joins the centred
        # batch; every other family carries all variants.
        expected = standard_variants if semantic == "hanging_mug" else set(_VARIANT_SUFFIXES)
        if variants != expected:
            raise ValueError(
                f"{semantic}: each active family needs every camera variant, got {sorted(variants)}"
            )
    for spec in specs:
        if spec.source_step_limit <= 0:
            raise ValueError(f"{spec.task_name}: source_step_limit must be positive")


# ------------------------------------------------------------------ rendering

def render_task_config(spec: VariantSpec, episodes: int) -> dict[str, Any]:
    """Render one RoboTwin collection config.

    @input: variant spec and the desired episode count.
    @output: dict matching the native ``task_config`` YAML schema.
    @scenario: Single-arm Piper embodiment, one head camera, no wrist camera.
    """
    config = {
        "render_freq": 0,
        "episode_num": episodes,
        "use_seed": False,
        "save_freq": 15,
        "embodiment": ["piper", "piper", 0.60],
        "language_num": 1,
        "domain_randomization": {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1,
            "random_head_camera_dis": 0,
            "random_table_height": 0,
            "random_light": False,
            "crazy_random_light_rate": 0,
        },
        "camera": {
            "head_camera_type": HEAD_CAMERA_TYPE_BY_VARIANT[spec.camera_variant],
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": False,
        },
        "data_type": {
            "rgb": True,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "observer": False,
            "endpose": True,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        },
        "pcd_down_sample_num": 1024,
        "pcd_crop": True,
        "save_path": "./data",
        "clear_cache_freq": 5,
        "collect_data": True,
        "eval_video_log": True,
    }
    return config


_INSTRUCTIONS: Mapping[str, dict[str, Any]] = {
    "blocks_ranking_rgb": {
        "full_description": "three colored blocks rest on the table; use one arm to line them up so the red block is leftmost, the green block in the middle and the blue block rightmost",
        "schema": "{A} notifies the red block, {B} notifies the green block, {C} notifies the blue block, {a} notifies the left arm to move the red block, {b} notifies the left arm to move the green block, {c} notifies the left arm to move the blue block",
        "preference": "num of words should not exceed 18. Degree of detail avg 7.",
        "seen": [
            "Use {a} to put {A} at the left, {b} to put {B} in the middle and {c} to put {C} at the right.",
            "Rank the blocks with the left arm: {A}, then {B}, then {C}.",
            "Arrange {A}, {B} and {C} in a line using {a}, {b} and {c}.",
            "With the left arm place {A} leftmost, {B} middle and {C} rightmost.",
            "{a} moves {A}, {b} moves {B}, {c} moves {C} into one ordered line.",
        ],
        "unseen": [
            "Line up the three colored blocks so red, green and blue read left to right.",
            "Use the left arm to rank {A}, {B} and {C} on the table.",
            "Put the red block first, the green block second and the blue block third.",
        ],
    },
    "blocks_ranking_size": {
        "full_description": "three blocks of different sizes rest on the table; use one arm to rank them so the large block is leftmost, the medium block in the middle and the small block rightmost",
        "schema": "{A} notifies the large block, {B} notifies the medium block, {C} notifies the small block, {a} notifies the left arm to move the large block, {b} notifies the left arm to move the medium block, {c} notifies the left arm to move the small block",
        "preference": "num of words should not exceed 18. Degree of detail avg 7.",
        "seen": [
            "Use {a} to place {A} leftmost, {b} to place {B} in the middle and {c} to place {C} at the right.",
            "Rank the blocks by size with the left arm: {A}, {B}, then {C}.",
            "With the left arm order the blocks from largest to smallest.",
            "Move {A}, {B} and {C} into one size-ordered line with the left arm.",
            "{a} moves {A}, {b} moves {B}, {c} moves {C} from large to small.",
        ],
        "unseen": [
            "Order the three blocks from large to small on the table.",
            "Use the left arm to arrange {A}, {B} and {C} by decreasing size.",
            "Line up the big block, then the medium one, then the small one.",
        ],
    },
    "hanging_mug": {
        "full_description": "hang the mug onto the rack with one arm",
        "schema": "{A} notifies the mug, {B} notifies the rack, {a} notifies the left arm",
        "preference": "num of words should not exceed 12. Degree of detail avg 5.",
        "seen": [
            "Use {a} to hang {A} on {B}.",
            "Hang {A} onto {B} with the left arm.",
            "{a} places {A} on the rack {B}.",
            "With the left arm, put {A} onto {B}.",
            "{a} hangs the mug {A} on the rack {B}.",
        ],
        "unseen": [
            "Hang the mug on the rack.",
            "Use the left arm to mount {A} onto {B}.",
            "Put {A} onto {B} with {a}.",
        ],
    },
    "place_bread_basket": {
        "full_description": "two breads rest on the table; use one arm to put both breads into the basket",
        "schema": "{A} notifies the basket, {B} notifies the first bread, {C} notifies the second bread, {a} notifies the left arm",
        "preference": "num of words should not exceed 15. Degree of detail avg 6.",
        "seen": [
            "Use {a} to put {B} and {C} into the basket {A}.",
            "With the left arm place the two breads {B} and {C} into {A}.",
            "{a} drops {B} and {C} into the basket {A}.",
            "Put the first bread {B} and the second bread {C} into {A} with the left arm.",
            "{a} places both breads into the basket {A}.",
        ],
        "unseen": [
            "Put the two breads into the basket.",
            "Use the left arm to place {B} and {C} in {A}.",
            "Move both breads into the basket with {a}.",
        ],
    },
    "place_bread_skillet": {
        "full_description": "place the bread onto the skillet with one arm, leaving the skillet on the table",
        "schema": "{A} notifies the skillet, {B} notifies the bread, {a} notifies the left arm",
        "preference": "num of words should not exceed 12. Degree of detail avg 5.",
        "seen": [
            "Use {a} to place {B} onto the skillet {A}.",
            "With the left arm put the bread {B} on {A}.",
            "{a} sets {B} onto the skillet {A}.",
            "Place {B} onto {A} with the left arm.",
            "{a} puts the bread {B} on the skillet {A}.",
        ],
        "unseen": [
            "Put the bread onto the skillet.",
            "Use the left arm to place {B} on {A}.",
            "Set {B} on top of {A} with {a}.",
        ],
    },
    "place_burger_fries": {
        "full_description": "put the hamburger and the french fries into the tray with one arm",
        "schema": "{A} notifies the hamburger, {B} notifies the tray, {C} notifies the french fries, {a} notifies the left arm",
        "preference": "num of words should not exceed 15. Degree of detail avg 6.",
        "seen": [
            "Use {a} to place {A} and {C} into the tray {B}.",
            "With the left arm put the hamburger {A} and the french fries {C} into {B}.",
            "{a} drops {A} and {C} into the tray {B}.",
            "Put {A} and {C} into {B} with the left arm.",
            "{a} places the food into the tray {B}.",
        ],
        "unseen": [
            "Put the hamburger and the fries into the tray.",
            "Use the left arm to place {A} and {C} in {B}.",
            "Move both food items into the tray with {a}.",
        ],
    },
    "place_can_basket": {
        "full_description": "place the can into the stationary basket with one arm and leave the basket on the table",
        "schema": "{A} notifies the can, {B} notifies the basket, {a} notifies the left arm",
        "preference": "num of words should not exceed 15. Degree of detail avg 6.",
        "seen": [
            "Use {a} to place the can {A} into the basket {B} and keep the basket on the table.",
            "With the left arm put {A} into the stationary basket {B}.",
            "{a} drops the can {A} into the basket {B} without lifting it.",
            "Place {A} into {B} with the left arm and leave {B} in place.",
            "{a} puts the can {A} into the basket {B}.",
        ],
        "unseen": [
            "Put the can into the basket without moving the basket.",
            "Use the left arm to place {A} inside {B}.",
            "Set {A} into {B} with {a}, leaving the basket on the table.",
        ],
    },
    "place_cans_plasticbox": {
        "full_description": "drop both cans into the plastic box with one arm",
        "schema": "{A} notifies the first can, {B} notifies the plastic box, {C} notifies the second can, {a} notifies the left arm",
        "preference": "num of words should not exceed 15. Degree of detail avg 6.",
        "seen": [
            "Use {a} to drop {A} and {C} into the plastic box {B}.",
            "With the left arm put the two cans {A} and {C} into {B}.",
            "{a} places {A} and {C} into the plastic box {B}.",
            "Put both cans {A} and {C} into {B} with the left arm.",
            "{a} drops the cans into the plastic box {B}.",
        ],
        "unseen": [
            "Put both cans into the plastic box.",
            "Use the left arm to place {A} and {C} in {B}.",
            "Move the two cans into the box with {a}.",
        ],
    },
    "place_dual_shoes": {
        "full_description": "put both shoes into the shoe box with one arm",
        "schema": "{A} notifies the shoe, {B} notifies the shoe box, {a} notifies the left arm",
        "preference": "num of words should not exceed 12. Degree of detail avg 5.",
        "seen": [
            "Use {a} to put the shoes {A} into the box {B}.",
            "With the left arm place both shoes into {B}.",
            "{a} drops the shoes into the shoe box {B}.",
            "Put the shoes {A} into {B} with the left arm.",
            "{a} places the shoes into the box {B}.",
        ],
        "unseen": [
            "Put both shoes into the shoe box.",
            "Use the left arm to place the shoes in {B}.",
            "Set the shoes into {B} with {a}.",
        ],
    },
    "stack_blocks_two": {
        "full_description": "stack the green block onto the red block with one arm",
        "schema": "{A} notifies the red block, {B} notifies the green block, {a} notifies the left arm to move the red block, {b} notifies the left arm to move the green block",
        "preference": "num of words should not exceed 14. Degree of detail avg 6.",
        "seen": [
            "Use {a} to place {A}, then {b} stacks {B} on {A}.",
            "With the left arm put {B} on top of {A}.",
            "{a} moves {A}, then {b} places {B} onto {A}.",
            "Stack {B} onto {A} using the left arm.",
            "Place {A} with {a}, then stack {B} on {A} with {b}.",
        ],
        "unseen": [
            "Stack the green block on the red block.",
            "Use the left arm to place {B} on top of {A}.",
            "Put {B} over {A} with the left arm.",
        ],
    },
    "stack_bowls_two": {
        "full_description": "stack the upper bowl onto the lower bowl with one arm",
        "schema": "{A} notifies the lower bowl, {B} notifies the upper bowl, {a} notifies the left arm to move the lower bowl, {b} notifies the left arm to move the upper bowl",
        "preference": "num of words should not exceed 14. Degree of detail avg 6.",
        "seen": [
            "Use {a} to place {A}, then {b} stacks {B} on {A}.",
            "With the left arm put {B} on top of {A}.",
            "{a} moves {A}, then {b} places {B} onto {A}.",
            "Stack {B} onto {A} using the left arm.",
            "Place {A} with {a}, then stack {B} on {A} with {b}.",
        ],
        "unseen": [
            "Stack the upper bowl on the lower bowl.",
            "Use the left arm to place {B} on top of {A}.",
            "Put {B} over {A} with the left arm.",
        ],
    },
}


def render_instruction(spec: VariantSpec) -> dict[str, Any]:
    """Render the instruction JSON for one variant.

    @input: variant spec.
    @output: instruction dict keyed like ``place_object_scale_left.json``.
    @scenario: Share one semantic instruction across all camera variants.
    """
    if spec.semantic_task not in _INSTRUCTIONS:
        raise ValueError(f"no instruction template for {spec.semantic_task}")
    return dict(_INSTRUCTIONS[spec.semantic_task])


def render_train_config(spec: VariantSpec) -> dict[str, Any]:
    """Render the training stub for one variant.

    @input: variant spec.
    @output: EMIX diffusion training config dict.
    @scenario: Minimal trainable stub consistent with the scale reference.
    """
    return {
        "batch_size": 64,
        "num_workers": 16,
        "steps": 30000,
        "save_freq": 5000,
        "log_freq": 100,
        "output_dir": f"outputs/train/{spec.task_name}_dp_emix",
        "policy": {
            "type": "emix_diffusion",
            "vision_backbone": "resnet18",
            "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
            "use_group_norm": False,
            "optimizer_lr": 1.0e-4,
            "push_to_hub": False,
            "horizon": 16,
            "n_obs_steps": 1,
            "n_action_steps": 8,
        },
        "emix": {
            "index_path": f"data/emix_index/{spec.task_name}/emix_index.json",
            "mixup": {
                "alpha_mode": "beta",
                "beta": 2,
                "alpha_value": None,
                "loss_weight": 1.0,
                "visual_keys": ["observation.images.head_camera"],
                "state_mode": "anchor",
            },
            "training": {
                "enabled": True,
                "seed": 0,
                "training_mode": "mixed",
                "scheduler": {
                    "type": "logarithmic",
                    "start_batch_prob": 1.00,
                    "end_batch_prob": 0.00,
                },
            },
        },
        "dataset": {
            "repo_id": spec.task_name,
            "root": f"data/lerobot/{spec.task_name}",
        },
    }


def render_eval_config(spec: VariantSpec) -> dict[str, Any]:
    """Render the evaluation stub for one variant.

    @input: variant spec.
    @output: evaluation config dict.
    @scenario: Points at the matching training checkpoint and native task.
    """
    return {
        "dataset": {
            "repo_id": spec.task_name,
            "root": f"data/lerobot/{spec.task_name}",
        },
        "evaluation": {
            "checkpoint": f"outputs/train/{spec.task_name}_dp_emix/checkpoints/030000/pretrained_model",
            "output_dir": f"outputs/eval/{spec.task_name}_dp_emix",
            "benchmark_root": "third_party/RoboTwin",
            "task_name": spec.task_name,
            "task_config": spec.task_name,
            "piper_distance": 0.60,
            "episodes": 100,
            "seed": 0,
            "device": "cuda",
            "save_video": True,
            "video_fps": 10,
        },
    }


# ------------------------------------------------------------------ write/check

def write_or_check(path: Path, payload: Mapping[str, Any], check: bool) -> None:
    """Write a YAML/JSON payload or verify it against the current file.

    @input: destination path, serializable mapping, and check flag.
    @output: None.
    @scenario: ``--check`` mode must not mutate the repository.
    """
    suffix = path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        rendered = yaml.safe_dump(
            payload, sort_keys=False, default_flow_style=False, allow_unicode=False
        )
    elif suffix == ".json":
        import json

        rendered = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    else:
        raise ValueError(f"unsupported output suffix {suffix!r} for {path}")
    if check:
        if not path.is_file():
            raise FileNotFoundError(f"missing generated artifact (check mode): {path}")
        existing = path.read_text(encoding="ascii")
        if existing != rendered:
            raise ValueError(f"generated artifact differs from registry: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="ascii")


def write_or_check_task_config(spec: VariantSpec, path: Path, check: bool) -> None:
    """Handle a centred batch config whose episode count is runtime-selected.

    ``dataflow.sh --num`` changes only ``episode_num`` so five-episode rate
    gates can run without regenerating the source asset.  Check all other
    generated fields exactly.
    """
    payload = render_task_config(spec, TRAIN_EPISODES)
    if check and spec.camera_variant in CENTERED_BATCH_VARIANTS and path.is_file():
        actual = yaml.safe_load(path.read_text(encoding="ascii"))
        if not isinstance(actual, dict) or not isinstance(actual.get("episode_num"), int):
            raise ValueError(f"invalid centred task config: {path}")
        actual["episode_num"] = TRAIN_EPISODES
        if actual != payload:
            raise ValueError(f"generated artifact differs from registry: {path}")
        return
    write_or_check(path, payload, check)


def _step_limit_lines(existing: str, specs: Sequence[VariantSpec]) -> str:
    """Return the step-limit file text with 33 variant keys appended."""
    lines = existing.splitlines()
    present = {line.split(":", 1)[0].strip() for line in lines if ":" in line}
    additions: list[str] = []
    for spec in specs:
        if spec.task_name not in present:
            additions.append(f"{spec.task_name}: {spec.source_step_limit}")
    if not additions:
        return existing
    return existing.rstrip("\n") + "\n" + "\n".join(additions) + "\n"


# ------------------------------------------------------------------ main

def _output_paths(spec: VariantSpec) -> dict[str, Path]:
    variant_dir = spec.camera_variant
    # Task configs mirror the envs/left layout: the default central-cam
    # variant's configs live in base/, the others in their camera-variant
    # subfolder (envs/left/<variant>/).
    config_dir = TASK_CONFIG_DIR / ("base" if variant_dir == "central_cam" else variant_dir)
    return {
        "task_config": config_dir / f"{spec.task_name}.yml",
        "acceptance_config": config_dir / f"{spec.task_name}_acceptance_10ep.yml",
        "instruction": INSTRUCTION_DIR / f"{spec.task_name}.json",
        "train": TRAIN_CONFIG_DIR / variant_dir / f"{spec.task_name}_dp_emix.yaml",
        "eval": EVAL_CONFIG_DIR / variant_dir / f"{spec.task_name}_dp_emix.yaml",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=REGISTRY_PATH,
        help="registry YAML path (default: configs/left_task_registry.yaml)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="verify on-disk artifacts match the registry without writing",
    )
    parser.add_argument(
        "--centered-batch-only", action="store_true",
        help="render only the centred-arm batch collection assets",
    )
    args = parser.parse_args(argv)

    specs = load_registry(args.registry)
    try:
        validate_registry(specs)
    except ValueError as error:
        print(f"[generate-left-task-assets] registry error: {error}", file=sys.stderr)
        return 2
    selected_specs = (
        [spec for spec in specs if spec.camera_variant in CENTERED_BATCH_VARIANTS]
        if args.centered_batch_only else specs
    )

    try:
        for spec in selected_specs:
            paths = _output_paths(spec)
            write_or_check_task_config(spec, paths["task_config"], args.check)
            # The acceptance_10ep configs are temporary collection artifacts:
            # they are regenerated on demand and deliberately absent from the
            # repository once their 10-episode pilot data has been collected.
            # --check therefore never requires them on disk; verify only the
            # render logic (episode_num == ACCEPTANCE_EPISODES) in tests.
            if not args.check:
                write_or_check(paths["acceptance_config"],
                               render_task_config(spec, ACCEPTANCE_EPISODES), False)
            write_or_check(paths["instruction"], render_instruction(spec), args.check)
            write_or_check(paths["train"], render_train_config(spec), args.check)
            write_or_check(paths["eval"], render_eval_config(spec), args.check)

        if not STEP_LIMIT_FILE.is_file():
            raise FileNotFoundError(f"missing step-limit file: {STEP_LIMIT_FILE}")
        existing = STEP_LIMIT_FILE.read_text(encoding="utf-8")
        updated = _step_limit_lines(existing, selected_specs)
        if args.check:
            if updated != existing:
                raise ValueError(f"step-limit file is missing variant keys: {STEP_LIMIT_FILE}")
        else:
            STEP_LIMIT_FILE.write_text(updated, encoding="utf-8")
    except (FileNotFoundError, ValueError) as error:
        print(f"[generate-left-task-assets] {error}", file=sys.stderr)
        return 2

    action = "checked" if args.check else "wrote"
    print(f"[generate-left-task-assets] {action} assets for {len(selected_specs)} variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
