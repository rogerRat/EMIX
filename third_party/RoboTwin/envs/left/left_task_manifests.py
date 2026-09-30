"""Immutable per-family task manifests for the eleven left-arm families.

Every ``central_cam``/``left_oppo_cam`` pair shares exactly one manifest.  A
pair may not have separate geometry tuning because camera is the only intended
variation.

Initial ranges are the source task ranges shifted into the left-arm workspace
(the active left Piper sits at x=-0.30 m) and kept inside the arm's reliable
reach.  Any range/yaw change must be accompanied by a pilot result and a
version increase.  Validation is field-level (exact x/y/yaw/clearance/static
values, see tests/test_left_task_base.py); content hashing is deliberately
not used anywhere in the protocol.

``scan_object`` is explicitly excluded from this project and has no manifest.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Mapping

from .left_task_base import ActorSamplingSpec, TaskManifest, XYYawRange


def _spec(name: str, x: tuple[float, float], y: tuple[float, float],
          yaw_rad: tuple[float, float], min_clearance_m: float,
          static: bool, base_quat: tuple[float, float, float, float]
          = (1.0, 0.0, 0.0, 0.0)) -> ActorSamplingSpec:
    return ActorSamplingSpec(
        name=name,
        workspace=XYYawRange(x=x, y=y, yaw_rad=yaw_rad),
        minimum_clearance_m=min_clearance_m,
        static=static,
        base_quat=base_quat,
    )


def _manifest(semantic_task: str, specs: list[ActorSamplingSpec], *,
              route_bin_count: int = 4, pilot_seed_count: int = 100,
              min_success_rate: float = 0.20,
              min_crossing_pair_rate: float = 0.0) -> TaskManifest:
    return TaskManifest(
        semantic_task=semantic_task,
        actor_specs={spec.name: spec for spec in specs},
        route_bin_count=route_bin_count,
        pilot_seed_count=pilot_seed_count,
        min_success_rate=min_success_rate,
        min_crossing_pair_rate=min_crossing_pair_rate,
    )


# Shared left-workspace constants.
_LEFT_CENTER = (-0.30, 0.0)
# Arm-base exclusion band: actors should not spawn right under the shoulder.
ARM_BASE_X_BAND = (-0.35, -0.25)

MANIFESTS: dict[str, TaskManifest] = {}

# Centred-arm batch workspace candidate. All ten active tasks use this one
# envelope unchanged during the calibration pilot. The candidate preserves the
# broad y extent needed by shoes and bowls, while removing table-edge x values
# and large yaw rotations that make a centre-mounted Piper unplannable.
CENTERED_WIDE_WORKSPACE = XYYawRange(
    x=(-0.20, 0.20), y=(-0.15, 0.15), yaw_rad=(0.0, 0.50)
)

# Per-family overrides for the centred-arm batch (arm base at x=0).  A family
# that cannot hold the full shared envelope keeps its own symmetric envelope
# here instead.  place_cans_plasticbox (2026-08-23 yaw iteration, task-specific
# enlarged envelope sanctioned for this family alone): the plasticbox centre
# sweeps x in [-0.10,0.10], y in [-0.28,-0.20] WITH yaw in [-0.60,0.60] rad
# (sampled by the impl's _sample_box_pose because sample_spec_pose locks yaw
# for static actors in the centred batch); both cans sample one shared
# standing band x in [-0.18,0.18], y in [-0.24,-0.02] with FIXED upright
# orientation (yaw 0).  The impl's rejection sampler derives each can's
# admissible region from the sampled box pose and geometry: a can footprint
# (radius 0.029) must clear the box's yaw-rotated collision rectangle
# (half-extents 0.078 x 0.097) by >= 0.05 m, and can pairs keep the 0.015 m
# per-axis AABB margin.
CENTERED_FAMILY_OVERRIDES: dict[str, dict[str, XYYawRange]] = {
    "place_cans_plasticbox": {
        "plasticbox": XYYawRange(x=(-0.10, 0.10), y=(-0.28, -0.20), yaw_rad=(-0.60, 0.60)),
        "can1": XYYawRange(x=(-0.18, 0.18), y=(-0.24, -0.02), yaw_rad=(0.0, 0.00)),
        "can2": XYYawRange(x=(-0.18, 0.18), y=(-0.24, -0.02), yaw_rad=(0.0, 0.00)),
    },
    # stack_bowls_two (2026-08-23 reachable-envelope iteration): a 200-seed
    # pilot of the full shared envelope (results/pilot/yaw_v6) showed the
    # place pose (fixed bowl quat mapped onto the grasp offset) is
    # IK-unreachable for targets beyond x~0.1 / y~0.0 — plan failure 147/200
    # and zero successes with target x>0.12 or y>0.05 — while spawns over the
    # same region stay graspable (the planner picks its own approach).  Both
    # bowls AND the stack target therefore sample one shared reachable band
    # (success ~0.5 throughout, per target-region bins of the same pilot), so
    # the collected stack-point distribution matches the bowl spawn
    # distribution instead of collapsing to the reachable corner of a wider
    # envelope.
    "stack_bowls_two": {
        "bowl1": XYYawRange(x=(-0.20, 0.05), y=(-0.15, 0.00), yaw_rad=(0.0, 0.00)),
        "bowl2": XYYawRange(x=(-0.20, 0.05), y=(-0.15, 0.00), yaw_rad=(0.0, 0.00)),
    },
}

# ---------------------------------------------------------------------------
# V0 centred-batch randomization protocol (base-relative, base-1 draft,
# 2026-09-30).
#
# COORDINATE SEMANTICS (user decision 2026-09-30): every CENTERED_V0_OVERRIDES
# entry is expressed RELATIVE TO THE ACTIVE PIPER BASE, not the table centre.
# This is exactly the historical batches' semantics (their spawn bands were
# symmetric about the arm position, e.g. blocks x(-0.45,-0.16) about x=-0.30,
# and plasticbox v4 was "mirror-symmetric about x=-0.26, the closest feasible
# analogue of table-centre symmetry"), and it is what makes large absolute
# ranges reachable: the workspace is a reach disc centred on the base, so
# base-relative bands stay inside the disc by construction.
#
#   x: symmetric about the base x (the centred batch base sits at x=0, so
#      base-relative x numerically equals the world x);
#   y: a FORWARD-DISTANCE BAND from the base, positive = in front of the
#      base.  y CANNOT be base-symmetric: the base stands at the table's near
#      edge (world y=-0.45, tabletop y>=-0.35), so the table only exists in
#      front of the base.  Absolute y = CENTERED_V0_BASE_XY[1] + y_rel.
#
# Reach model used for the base-1 drafts (from the 2026-09-30 pilot decay
# data): reliable success needs actor-base distance <= ~0.5-0.55 m; the
# near-arm strip (abs y in [-0.32,-0.15], i.e. 0.13-0.30 m in front of the
# base) is the highest-success zone and was wasted by the table-centre
# symmetric drafts.  Hard bounds: tabletop y>=-0.35 (y_rel >= ~0.10 plus
# footprint margin), top-camera frame abs y<=+0.34 / x<=+0.49, tabletop
# x in [-0.60,0.60].
#
# Acceptance rule (user decision 2026-09-29): a 100-seed pilot must reach
# success rate >= 0.20; below that the range converges, above it the range
# may be widened.  Pilot evidence lands in results/pilot/v0_ranges/ and
# docs/_exp.md before any value is promoted from draft.
#
# Known physically-locked exceptions (no v0 entry, impl hardcodes stay):
# - place_dual_shoes: shoes snap to lift-verified rows abs y in {0.13,0.14}
#   (grasp+lift probe diag_ds_fullmap.py); y is not free to randomize.
# - stack_bowls_two: keeps its historical absolute CENTERED_FAMILY_OVERRIDES
#   band (see the note at the end of this registry).
CENTERED_V0_BASE_XY = (0.0, -0.45)  # active Piper base, world frame
# (mirrors assets/embodiments/piper/config.yml robot_pose [[0,-0.45,0.75,...]]
#  and the centred-batch comments in left_task_base.py)

# Base-1 drafts (2026-09-30).  Previous table-centre-symmetric round results
# are kept in the per-task comments for comparison; all previous evidence is
# archived under results/pilot/v0_ranges/*.d*.jsonl.
CENTERED_V0_OVERRIDES: dict[str, dict[str, XYYawRange]] = {
    "blocks_ranking_rgb": {
        # Target line is derived from block1's spec workspace (span <=0.32 m
        # at <=15 deg).  Table-centre round final: x±0.25/y±0.20 @ SR 0.22.
        "block1": XYYawRange(x=(-0.32, 0.32), y=(0.15, 0.50), yaw_rad=(0.0, 0.75)),
        "block2": XYYawRange(x=(-0.32, 0.32), y=(0.15, 0.50), yaw_rad=(0.0, 0.75)),
        "block3": XYYawRange(x=(-0.32, 0.32), y=(0.15, 0.50), yaw_rad=(0.0, 0.75)),
    },
    "blocks_ranking_size": {
        # Table-centre round final: x±0.25/y±0.20 @ SR 0.21 (rgb twin 0.22).
        "block1": XYYawRange(x=(-0.30, 0.30), y=(0.16, 0.48), yaw_rad=(0.0, 0.75)),
        "block2": XYYawRange(x=(-0.30, 0.30), y=(0.16, 0.48), yaw_rad=(0.0, 0.75)),
        "block3": XYYawRange(x=(-0.30, 0.30), y=(0.16, 0.48), yaw_rad=(0.0, 0.75)),
    },
    "place_bread_basket": {
        # Basket and breads share one overlapping band (union randomization);
        # the 0.13 m clearance is footprint-aware and unchanged.
        # Table-centre round final: x±0.22/y±0.20 @ SR 0.20 (failures were
        # reach-dominated at actor-base distance >=0.55).
        "breadbasket": XYYawRange(x=(-0.30, 0.30), y=(0.22, 0.48), yaw_rad=(0.0, 0.50)),
        "bread0": XYYawRange(x=(-0.30, 0.30), y=(0.22, 0.48), yaw_rad=(0.0, 0.50)),
        "bread1": XYYawRange(x=(-0.30, 0.30), y=(0.22, 0.48), yaw_rad=(0.0, 0.50)),
    },
    "place_bread_skillet": {
        # Table-centre round final: x±0.25/y±0.18 @ SR 0.41 (best case:
        # widened with no rate cost).
        "bread": XYYawRange(x=(-0.32, 0.32), y=(0.26, 0.50), yaw_rad=(0.0, 0.50)),
        "skillet": XYYawRange(x=(-0.32, 0.32), y=(0.26, 0.50), yaw_rad=(0.0, 0.50)),
    },
    "place_burger_fries": {
        # Structural-ceiling family (table-centre round: 0.11->0.15->0.17->
        # 0.21 at progressively narrower bands; 09-11 collection baseline
        # 0.173).  Occlusion when both foods share the band is accepted per
        # v3; base-1 keeps a conservative box.
        "tray": XYYawRange(x=(-0.22, 0.22), y=(0.22, 0.40), yaw_rad=(0.0, 0.00)),
        "hamburg": XYYawRange(x=(-0.22, 0.22), y=(0.22, 0.40), yaw_rad=(0.0, 0.20)),
        "frenchfries": XYYawRange(x=(-0.22, 0.22), y=(0.22, 0.40), yaw_rad=(0.0, 0.20)),
    },
    "place_can_basket": {
        # Directionally risky family (transport distance vs near-ceiling
        # release).  Table-centre round final: x±0.22/y±0.15 @ SR 0.22.
        "basket": XYYawRange(x=(-0.22, 0.22), y=(0.24, 0.42), yaw_rad=(0.0, 0.00)),
        "can": XYYawRange(x=(-0.22, 0.22), y=(0.24, 0.42), yaw_rad=(0.0, 0.00)),
    },
    "place_cans_plasticbox": {
        # Box keeps its yaw freedom via the impl's _sample_box_pose.
        # Table-centre round: SR 0.09->0.13->0.19 with the success window
        # measured at box-base distance 0.35-0.5 m (near <0.35 arm
        # collision, far >=0.5 unreachable; historical asymmetric near-arm
        # band piloted 0.47).  Base-1 places the box directly in that
        # window; cans sweep a wider forward band.
        "plasticbox": XYYawRange(x=(-0.12, 0.12), y=(0.22, 0.38), yaw_rad=(-0.60, 0.60)),
        "can1": XYYawRange(x=(-0.18, 0.18), y=(0.15, 0.45), yaw_rad=(0.0, 0.00)),
        "can2": XYYawRange(x=(-0.18, 0.18), y=(0.15, 0.45), yaw_rad=(0.0, 0.00)),
    },
    "place_dual_shoes": {
        # Shoes keep impl-hardcoded lift rows (exception above); the box y
        # declared here equals the impl's centered-batch hardcode
        # (abs -0.15,-0.02 = base-relative 0.30,0.43) and is overwritten by
        # the impl anyway; only the box x band applies.
        # Table-centre round final: box x±0.20 @ SR 0.24 (x±0.25 gave 0.17,
        # UnStableError-bound).
        "shoe_box": XYYawRange(x=(-0.22, 0.22), y=(0.30, 0.43), yaw_rad=(0.0, 0.00)),
    },
    "stack_blocks_two": {
        # Table-centre round final: x±0.30/y±0.22 @ SR 0.35 (plateau across
        # four widening steps 0.35-0.42).  Base-1 pulls the band toward the
        # near-arm strip where the legacy control piloted 0.52.
        "block1": XYYawRange(x=(-0.40, 0.40), y=(0.15, 0.50), yaw_rad=(0.0, 0.60)),
        "block2": XYYawRange(x=(-0.40, 0.40), y=(0.15, 0.50), yaw_rad=(0.0, 0.60)),
    },
    # "stack_bowls_two": NO v0 entry — documented physical exception
    # (2026-09-30).  The measured place-IK reach bound (target world x>0.12
    # or y>0.05 unreachable; 2026-08-23 200-seed pilot) and the impl's
    # three-point 0.13 m clearance geometry leave no workable base-relative
    # band either: the reachable forward strip (y_rel ~0.30-0.50, world
    # -0.15..+0.05) is narrower than the ~0.113 m-minimum layout geometry in
    # y when combined with the x bound (table-centre draft-2 x±0.15/y±0.08
    # piloted 0.10).  The family keeps its historical absolute
    # CENTERED_FAMILY_OVERRIDES band (x(-0.20,0.05), y(-0.15,0.00)) in both
    # protocols; evidence archived under
    # results/pilot/v0_ranges/stack_bowls_two_left_cen_arm_top_cam.d*.jsonl.
}

# Optional ellipse (disc) sampling per actor, A/B against the rectangular
# bands above (user suggestion 2026-09-30: define the range as the
# reach-disc ∩ tabletop region directly).  Values are BASE-RELATIVE:
# (rx, ry, y_center_rel) — x radius, y radius, ellipse centre distance in
# front of the base.  Every ellipse is inscribed so that no sampled point
# falls off-table (footprint-aware lower bound) or beyond the reliable
# reach radius (worst-point base distance <= ~0.51 m), unlike the
# rectangle's far corners.  sample_spec_pose draws area-uniformly inside.
# Ellipse A/B verdict (2026-10-01, all 100-seed pilots archived under
# results/pilot/v0_ranges/*.ellipse.pilot.jsonl): the ellipse wins for
# place_bread_basket (0.58 vs rect 0.47), place_cans_plasticbox (0.39 vs
# 0.29) and place_can_basket (0.27 vs 0.23); the rectangle stays for
# stack_blocks_two (tie at the base-2 box, and the rectangle can be pushed
# wider: 0.78 at x±0.40), place_burger_fries (0.39 vs 0.29) and
# place_dual_shoes (0.29 vs 0.20 — the ellipse's centre-concentrated x
# margin puts the box closer to the shoe rows and raises UnStableError).
# For the remaining tasks no ellipse was configured (rectangle final).
CENTERED_V0_FINAL_ELLIPSE_TASKS = frozenset({
    "place_bread_basket",
    "place_cans_plasticbox",
    "place_can_basket",
})
# Optional override for future A/B pilots: ROBOTWIN_V0_ELLIPSE=1 forces
# ellipses on every configured task, =0 disables them; unset uses the final
# per-task verdict above.
_ELLIPSE_OVERRIDE = os.environ.get("ROBOTWIN_V0_ELLIPSE")

CENTERED_V0_ELLIPSES: dict[str, dict[str, tuple[float, float, float]]] = {
    "stack_blocks_two": {
        # Bounding box matches the base-2 rectangle (x±0.35, y_rel 0.15-0.50);
        # worst-point base distance 0.51 m vs the rectangle's 0.61 m corners.
        "block1": (0.35, 0.175, 0.325),
        "block2": (0.35, 0.175, 0.325),
    },
    "place_bread_basket": {
        # Bounding box matches the base-2 rectangle (x±0.30, y_rel 0.22-0.48);
        # worst-point base distance 0.49 m vs the rectangle's 0.57 m corners.
        # A/B pilot 2026-09-30: ellipse 0.58 vs rectangle 0.47 — ellipse wins.
        "breadbasket": (0.30, 0.13, 0.35),
        "bread0": (0.30, 0.13, 0.35),
        "bread1": (0.30, 0.13, 0.35),
    },
    "place_can_basket": {
        # Rescue-round ellipse for the borderline family (rectangle piloted
        # 0.23); worst-point base distance ~0.44 m vs rectangle corner 0.47.
        "basket": (0.22, 0.09, 0.33),
        "can": (0.22, 0.09, 0.33),
    },
    "place_burger_fries": {
        # Rescue-round ellipse (rectangle piloted 0.39); worst-point ~0.45 m.
        "tray": (0.22, 0.09, 0.31),
        "hamburg": (0.22, 0.09, 0.31),
        "frenchfries": (0.22, 0.09, 0.31),
    },
    "place_cans_plasticbox": {
        # Rescue-round ellipses (rectangle piloted 0.29); box kept in its
        # measured 0.35-0.5 m distance window, cans sweep wider.
        "plasticbox": (0.12, 0.08, 0.30),
        "can1": (0.18, 0.15, 0.30),
        "can2": (0.18, 0.15, 0.30),
    },
    "place_dual_shoes": {
        # shoe_box only; the impl overwrites y with its hardcoded band, so
        # the ellipse effectively shapes the x distribution (rectangle
        # piloted 0.29).
        "shoe_box": (0.22, 0.065, 0.365),
    },
}

# Active centred-batch randomization protocol.  "v0" selects
# CENTERED_V0_OVERRIDES (base-relative, footprint-aware, per-task rectangle
# or ellipse), "legacy" keeps the historical shared envelope / family
# overrides.  Default is "v0" since 2026-10-01 (TrEv evaluation completed
# 2026-10-01; flip recorded in docs/_exp.md history).  Set
# ROBOTWIN_CENTERED_PROTOCOL=legacy to reproduce legacy-layout runs.
# Task-implementation revision.  The env switch below is a spatial preset
# name inherited from the V0 range pilots; it is not the experiment version.
TASK_IMPLEMENTATION_VERSION = "v1"
CENTERED_ACTIVE_PROTOCOL = os.environ.get("ROBOTWIN_CENTERED_PROTOCOL", "v0")


def _register(manifest: TaskManifest) -> None:
    MANIFESTS[manifest.semantic_task] = manifest


_register(_manifest(
    "blocks_ranking_rgb",
    [
        # Wide randomization (iteration 2026-08-13): all three blocks share one
        # overlapping pickup band so routes to the ordered target line start from
        # many directions and cross. x extends right toward the validated
        # place_object_scale_left bound (x <= -0.16 inside the reachable region),
        # y overlaps the target-line y band so short and long routes coexist, and
        # yaw matches the source task. The 0.10 m pairwise clearance keeps a
        # gripper-wide minimum gap so the expert can always operate between them.
        _spec("block1", (-0.45, -0.16), (-0.18, 0.08), (0.0, 0.75), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block2", (-0.45, -0.16), (-0.18, 0.08), (0.0, 0.75), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block3", (-0.45, -0.16), (-0.18, 0.08), (0.0, 0.75), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "blocks_ranking_size",
    [
        # Wide randomization (iteration 2026-08-13): all three blocks share one
        # overlapping pickup band so routes to the ordered target line start from
        # many directions and cross. x extends right to the validated
        # place_object_scale_left bound (x <= -0.16 inside the reachable
        # region), y overlaps the target-line y band so short and long routes
        # coexist, and yaw matches the blocks_ranking_rgb family. The 0.10 m
        # pairwise clearance keeps a gripper-wide minimum gap so the expert can
        # always operate between the larger size blocks.
        _spec("block1", (-0.45, -0.16), (-0.18, 0.08), (0.0, 0.75), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block2", (-0.45, -0.16), (-0.18, 0.08), (0.0, 0.75), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block3", (-0.45, -0.16), (-0.18, 0.08), (0.0, 0.75), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

# DEPRECATED (2026-08-09): the only reliable hanging geometry uses a rack
# tilted ~58 deg about x.  An upright, yaw-only rack (the only orientation
# permitted going forward) cannot physically hold the 039_mug, so the task is
# frozen and must not be collected.  Kept in the registry to preserve the
# manifest hash 28ee798eb3dfcde4 and the wrapper/test contract.
_register(_manifest(
    "hanging_mug",
    [
        # Single-arm pilot (iteration 4): the source x+90 mug orientation is
        # the only one that settles stably on the table, and the left EEF
        # ceiling (~1.03 m) plus the ~0.23 m grasp offset caps the mug
        # functional point near z=0.87.  To bring the hook into reach the rack
        # is rolled about x by 45 deg (fp0 world z ~0.85-0.87) and pinned to a
        # single left-reachable spot; the mug band narrows to the grasp-stable
        # strip that still leaves the "mug rides the tilted pillar" success
        # case above the 0.20 pilot threshold (30-seed rate 0.60).
        _spec("mug", (-0.44, -0.40), (-0.01, 0.01), (0.0, 0.0), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("rack", (-0.29, -0.29), (-0.02, -0.02), (0.0, 0.0), 0.10, True,
              base_quat=(-0.11906, -0.28744, 0.3626, 0.8754)),
    ],
))

_register(_manifest(
    "place_bread_basket",
    [
        # Union randomization (iteration 2026-08-14): the breadbasket now
        # sweeps the whole union of the family's appearance ranges
        # (x in [-0.44,-0.12], y in [-0.28,0.04]) so it can appear anywhere the
        # breads can, overlapping the bread pickup band maximally.  The basket
        # clearance is footprint-aware: the 076_breadbasket collision hull has
        # a ~0.097 m y half-extent, so the 0.13 m pairwise clearance keeps any
        # bread centre outside the basket hull (bread half-extent ~0.039 m).
        # The two breads keep the verified 075_bread downward-grasp band
        # (x in [-0.44,-0.34], y in [-0.18,0.04]) and a 0.10 m mutual gap.
        _spec("breadbasket", (-0.44, -0.12), (-0.28, 0.04), (0.0, 0.50), 0.13, True,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("bread0", (-0.44, -0.34), (-0.18, 0.04), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("bread1", (-0.44, -0.34), (-0.18, 0.04), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "place_bread_skillet",
    [
        # Union randomization (iteration 2026-08-14): the skillet now sweeps
        # the whole union of the family's appearance ranges (x in [-0.44,-0.12],
        # y in [-0.24,0.14]) so it can appear anywhere the bread can,
        # overlapping the bread pickup band maximally.  The skillet clearance
        # is footprint-aware: the 106_skillet collision hull has a ~0.135 m
        # half-extent, so the 0.17 m pairwise clearance keeps the bread centre
        # outside the skillet hull (bread half-extent ~0.039 m).  The bread
        # keeps the verified 075_bread downward-grasp band.
        _spec("bread", (-0.44, -0.34), (-0.18, 0.04), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("skillet", (-0.44, -0.12), (-0.24, 0.14), (0.0, 0.50), 0.17, True,
              base_quat=(0.0, 0.0, 0.707, 0.707)),
    ],
))

_register(_manifest(
    "place_burger_fries",
    [
        # Union randomization (iteration 2026-08-14): hamburg and frenchfries
        # now share one identical band (x in [-0.46,-0.34], y in [0.10,0.16])
        # so the two foods overlap maximally in appearance, both spawning
        # strictly above the tray footprint.  The tray (0.31 x 0.21 m footprint
        # at scale 2.0, collision z in [0.737,0.769]) sweeps the lower
        # workspace; its footprint top is at most y=0.061 (tray centre y-max
        # -0.04 plus the 0.101 m y half-extent), so both foods at centre
        # y >= 0.10 keep their hulls clear of the tray rim and never inter-
        # penetrate (hamburg half ~0.034 m, frenchfries hull ~0.048 m).  The
        # 0.08 m pairwise clearance keeps the two foods apart at spawn.
        _spec("tray", (-0.44, -0.08), (-0.16, -0.04), (0.0, 0.00), 0.08, True,
              base_quat=(0.706527, 0.706483, -0.0291356, -0.0291767)),
        _spec("hamburg", (-0.46, -0.34), (-0.35, -0.27), (0.0, 0.20), 0.08, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("frenchfries", (-0.46, -0.34), (-0.35, -0.27), (0.0, 0.20), 0.08, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "place_can_basket",
    [
        # Union randomization (iteration 2026-08-14): the basket sweeps
        # x in [-0.38,-0.12], y in [-0.30,0.00], widened upward so its
        # appearance band touches the can band (y in [0,0.08]) at y=0, giving
        # the two actors overlapping possible regions.  The quat (0.5,0.5,0.5,
        # 0.5) with zero yaw stays deterministic so the plate offset
        # (+0.05,-0.04) used by place_can_basket_left_impl remains valid.  The
        # 0.15 m pairwise clearance (can half-extent ~0.025 m, basket ~0.115 m)
        # keeps the lying can outside the basket hull whenever the bands
        # approach.  The can band itself is pinned by the lying-can downward-
        # grasp IK reach (y capped at 0.08, x in [-0.25,-0.20]).
        _spec("basket", (-0.38, -0.12), (-0.30, 0.00), (0.0, 0.00), 0.15, True,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        # The can spawns on its side (identity quat) near the left arm base
        # (x in [-0.25, -0.20], y in [0, 0.08]), matching the source layout
        # that keeps every downward grasp IK-reachable.  y is capped at 0.08:
        # cans at y ~ 0.10 sit at the edge of the left grasp zone and fail
        # choose_best_pose.  The lying can is transported and lowered onto the
        # basket plate rather than dropped.
        _spec("can", (-0.25, -0.20), (0.0, 0.08), (0.0, 0.00), 0.15, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "place_cans_plasticbox",
    [
        # Symmetric flanking randomization (version 4, 2026-08-22): the
        # version 3 union envelope (box and cans all sweeping x in
        # [-0.42,-0.10]) pilots at 0.28 but realizes only ~0.05 in collection
        # (3479 seeds for 184 episodes, ~20% of failures UnStableError from
        # cans spawning against the box hull).  Version 4 restores the
        # version 0 flanking structure and makes it exactly mirror-symmetric
        # about the left-workspace centre x=-0.26 (the closest feasible
        # analogue of table-centre symmetry for a left-arm-only task):
        #   - plasticbox centre band x in [-0.37,-0.15] (self-symmetric about
        #     -0.26), y in [-0.28,-0.23];
        #   - can1 x in [-0.42,-0.36] and can2 x in [-0.16,-0.10], mirror
        #     images (mirror(x) = -0.52 - x), y in [-0.15,-0.07].
        # The x bands alone no longer guarantee separation, so the impl's
        # rejection sampler uses per-axis footprint clearance (see
        # place_cans_plasticbox_left_impl): box half-extents (0.078, 0.097),
        # can half-extent 0.029, 0.01 m margin.
        _spec("plasticbox", (-0.37, -0.15), (-0.28, -0.23), (0.0, 0.00), 0.13, True,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("can1", (-0.42, -0.36), (-0.15, -0.07), (0.0, 0.00), 0.08, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("can2", (-0.16, -0.10), (-0.15, -0.07), (0.0, 0.00), 0.08, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
    ],
))

_register(_manifest(
    "place_dual_shoes",
    [
        # The shoe box (0.18 x 0.26 m footprint under the stand90 quat) is a
        # tall static hull; a shoe that spawns geometrically inside that
        # footprint interpenetrates the box rim and the settle phase ejects it,
        # raising UnStableError.  Both shoes therefore spawn above the box
        # (y >= 0.13).  The box sweeps x in [-0.46, -0.14] and y in [-0.18,
        # -0.02]; its y half-extent is 0.13, so the footprint top sits at most
        # at y=0.11, still below the shoe rows' y=0.13.  Union randomization
        # (iteration 2026-08-14): the box x band widens to the full shoe x
        # band, so the container's appearance range overlaps the shoes'
        # appearance range in x.
        # Physical-reset grasp+lift probes (diag_ds_fullmap.py, 0.02 m grid,
        # model 4, stand90 quat) showed the left arm can grasp AND lift a shoe
        # only on the y=0.13 and y=0.14 rows, with the lift-OK x band depending
        # on the row:
        #   y=0.13: "gOOOOgggggggggOOOOg."  -> OK in x[-0.46,-0.40] (left
        #           island) or x[-0.20,-0.14] (right island); the middle
        #           x[-0.38,-0.22] is grasp-only and cannot lift.
        #   y=0.14: ".gOOOOOOOOOOOOOOOg.g"  -> OK in the single continuous band
        #           x[-0.44,-0.18]; only x=-0.46 (grasp-only) and x<=-0.16 fail.
        # The impl snaps y to one verified row and samples x inside that row's
        # lift-OK set, then enforces the pairwise/basket clearance so the two
        # shoes never overlap.  The box quat is stand90, matching the shoe base
        # quat: the align place constraint then keeps the shoe standing (a
        # small wrist rotation) instead of requiring a lateral 120-degree roll
        # that the left arm cannot reach.
        _spec("shoe_box", (-0.46, -0.14), (-0.18, -0.02), (0.0, 0.00), 0.13, True,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("left_shoe", (-0.46, -0.14), (0.13, 0.14), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("right_shoe", (-0.46, -0.14), (0.13, 0.14), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "stack_blocks_two",
    [
        _spec("block1", (-0.45, -0.10), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block2", (-0.45, -0.10), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "stack_bowls_two",
    [
        _spec("bowl1", (-0.50, -0.05), (-0.15, 0.15), (0.0, 0.00), 0.13, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("bowl2", (-0.50, -0.05), (-0.15, 0.15), (0.0, 0.00), 0.13, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
    ],
))


def get_manifest(semantic_task: str) -> TaskManifest:
    """Return the immutable manifest for one semantic task name.

    @input: a source semantic task name such as ``place_burger_fries``.
    @output: the shared ``TaskManifest``.
    @scenario: Keep one source of geometry truth per family.
    """
    if semantic_task not in MANIFESTS:
        raise KeyError(f"Unknown left-arm semantic task: {semantic_task!r}")
    return MANIFESTS[semantic_task]


def centered_wide_workspace_manifest(manifest: TaskManifest) -> TaskManifest:
    """Return the new-batch manifest with the active protocol's spawn envelopes.

    Resolution order per actor: CENTERED_V0_OVERRIDES (only when
    CENTERED_ACTIVE_PROTOCOL == "v0"; entries are BASE-RELATIVE, see the
    registry header, and are translated to world coordinates here), then
    the historical CENTERED_FAMILY_OVERRIDES (absolute), then the
    historical CENTERED_WIDE_WORKSPACE (absolute). The deprecated
    hanging-mug family is deliberately never transformed or collected.
    Per-actor orientation, collision footprint and static/dynamic identity
    remain the same; only the initial randomization envelope changes.
    """
    if manifest.semantic_task == "hanging_mug":
        raise ValueError("deprecated hanging_mug cannot use the centred-wide batch")
    base_x, base_y = CENTERED_V0_BASE_XY
    v0_override = (
        CENTERED_V0_OVERRIDES.get(manifest.semantic_task, {})
        if CENTERED_ACTIVE_PROTOCOL == "v0"
        else {}
    )
    override = CENTERED_FAMILY_OVERRIDES.get(manifest.semantic_task, {})

    def _resolve(name: str, spec: ActorSamplingSpec):
        rel = v0_override.get(name)
        if rel is not None:
            workspace = XYYawRange(
                x=(base_x + rel.x[0], base_x + rel.x[1]),
                y=(base_y + rel.y[0], base_y + rel.y[1]),
                yaw_rad=rel.yaw_rad,
            )
            ellipse = None
            use_ellipse = (
                manifest.semantic_task in CENTERED_V0_FINAL_ELLIPSE_TASKS
                if _ELLIPSE_OVERRIDE is None
                else _ELLIPSE_OVERRIDE == "1"
            )
            if use_ellipse:
                ell = CENTERED_V0_ELLIPSES.get(manifest.semantic_task, {}).get(name)
                if ell is not None:
                    rx, ry, yc_rel = (float(v) for v in ell)
                    ellipse = (base_x, base_y + yc_rel, rx, ry)
            return replace(spec, workspace=workspace, ellipse=ellipse)
        workspace = override.get(name, CENTERED_WIDE_WORKSPACE)
        return replace(spec, workspace=workspace)

    return replace(manifest, actor_specs={
        name: _resolve(name, spec) for name, spec in manifest.actor_specs.items()
    })


def all_manifests() -> Mapping[str, TaskManifest]:
    """Return the immutable registry of all semantic manifests."""
    return dict(MANIFESTS)
