# Engineering Plan: Eleven Left-Arm RoboTwin Task Families

## Objective

Implement two named head-camera configurations for every derived task. `central_cam` is the existing central, scene-fixed `head_camera` from the left Piper embodiment configuration; it must be deep-copied unchanged and is bound to the public `*_left` task name. `left_oppo_cam` is bound to public `*_left_oppo_cam`; it uses the same robot, scene layout, expert trajectory, seeds, and success predicate, but replaces only the `head_camera` pose with `position=[-0.30, 0.45, 1.35]`, `forward=[0, -0.6, -0.8]`, and `left=[1, 0, 0]`. Both variants are left-arm-only, keep the right arm at home qpos, collect only the named head camera, disable wrist cameras, and convert all learning images to 84x84.

| ID | Source task | `central_cam` public variant | `left_oppo_cam` public variant | Source evaluation limit |
| --- | --- | --- | --- | ---: |
| T01 | `blocks_ranking_rgb` | `blocks_ranking_rgb_left` | `blocks_ranking_rgb_left_oppo_cam` | 1200 |
| T02 | `blocks_ranking_size` | `blocks_ranking_size_left` | `blocks_ranking_size_left_oppo_cam` | 1200 |
| T03 | `hanging_mug` **(DEPRECATED)** | `hanging_mug_left` | `hanging_mug_left_oppo_cam` | 900 |
| T04 | `place_bread_basket` | `place_bread_basket_left` | `place_bread_basket_left_oppo_cam` | 700 |
| T05 | `place_bread_skillet` | `place_bread_skillet_left` | `place_bread_skillet_left_oppo_cam` | 500 |
| T06 | `place_burger_fries` | `place_burger_fries_left` | `place_burger_fries_left_oppo_cam` | 500 |
| T07 | `place_can_basket` | `place_can_basket_left` | `place_can_basket_left_oppo_cam` | 700 |
| T08 | `place_cans_plasticbox` | `place_cans_plasticbox_left` | `place_cans_plasticbox_left_oppo_cam` | 800 |
| T09 | `place_dual_shoes` | `place_dual_shoes_left` | `place_dual_shoes_left_oppo_cam` | 600 |
| T10 | `stack_blocks_two` | `stack_blocks_two_left` | `stack_blocks_two_left_oppo_cam` | 800 |
| T11 | `stack_bowls_two` | `stack_bowls_two_left` | `stack_bowls_two_left_oppo_cam` | 900 |

`scan_object` is explicitly excluded: it has no derived task, task config, instruction, dataset, index, or evaluation entry in this project. The deliverable therefore has 22 RoboTwin task entry points and 11 task-semantic implementations. Existing source environments, configs, datasets, indexes, checkpoints, and reports must not be modified or reused as derived-task outputs.

## Mandatory Semantic Rewrite, Not a Wrapper-Only Change

The two wrappers are deliberately small, but they are the last layer of the implementation, not the implementation itself. Each of the 11 `Left*` semantic classes is a full single-arm rewrite of its source task. It must own `load_actors`, the seeded layout sampler, the serial `play_once` trajectory, `check_success`, task information placeholders, and every task-specific placement helper. It may reuse asset names, functional-point APIs, and mathematically equivalent success thresholds from the source, but it must not call the source `play_once`, select an arm from object x position, or retain a source two-arm/handover action sequence.

Before creating any actor, `load_actors` must obtain poses from its left-workspace manifest. The source pose range is not a valid default. The sampler must reject layouts outside the validated left-arm workspace, layouts whose pickup or placement pre-pose cannot be reached by the left arm, and layouts violating actor clearance or route diversity. The acceptance criterion is planning success from the sampled layout, not merely that its xy coordinates are negative.

Every source action sequence is converted into one of three explicit single-arm algorithms: (1) **serial transfer**: grasp, lift, place, release, retreat, return home, then process the next object; (2) **stable-target transfer**: make the receptacle/base static after settling, then move only the transported object; (3) **same-arm tool sequence**: place/release an object, return home, then grasp and use the tool with the same left arm. Dual grasp, concurrent `move` groups, arm handover, opposite-arm stabilizing, and right-gripper success conditions are prohibited. Where a source objective intrinsically requires two arms, the derived objective and its instruction must be explicitly narrowed, as in T07 where the basket remains on the table.

Every semantic implementation file (`<task>_left_impl.py`) must begin with a standardized English module comment, rather than carrying a separate PR attachment. The comment is maintained with the code and must state: source actors and source dual-arm actions removed; the manifest actor names, ranges, and clearances; left EEF pre-grasp/pre-place feasibility evidence and pilot seed range; serial operation order; changed success conditions; and changed instruction wording. A class cannot advance to its 100-seed pilot without this header. The static validator checks for the required `LEFT_TASK_DESIGN:` marker and all required field labels.

Required header template:

```python
"""LEFT_TASK_DESIGN:
Source task: <source-task>
Single-arm semantic change: <what was removed or narrowed>
Left workspace manifest: <manifest name and version>
Actors and clearance: <actor names and minimum distances>
Expert sequence: <serial left-arm operation order>
Success predicate: <preserved and changed conditions>
Instruction change: <left-only task wording>
Pilot evidence: seeds <start>-<end>, success rate <rate>, manifest hash <hash>
"""
```

## Source-Code Decision Matrix for the Eleven Semantic Rewrites

The following decisions are based on the current source environment code, not on task names alone. The numeric left-workspace limits are deliberately not frozen here: each row requires a feasibility pilot before its manifest is accepted. The implementation must first shift all participating source regions by the left-base offset and then expand one bound at a time while validating actual left EEF pre-grasp, pre-place, and retreat poses.

| ID | Source behavior that cannot remain | Required left-arm scene and expert rewrite | Required success/instruction decision |
| --- | --- | --- | --- |
| T01 RGB ranking | Each block chooses arm from x sign; arm switch calls `back_to_origin(opposite)`; target line is centered at x around -0.09, 0, +0.09. | Sample all three blocks in one left workspace. Move the complete three-target ordered line to a left reachable region. Pick/place red, green, blue sequentially with left arm and return home between blocks. Keep blocks dynamic. | Retain ordered-x and common-y ranking relationship, remove right-gripper-open condition, require left open and right home. Instruction says left arm ranks three colored blocks. |
| T02 size ranking | Same arm-by-x and opposite-arm reset as T01; size ordering is large/medium/small and source execution order is small, medium, large. | Use one left workspace for all blocks and a left target line. Preserve asset half-size ranges and serial source semantic order: small, medium, large. | Retain geometry ranking relation, remove right condition, and explicitly describe size ordering rather than color. |
| T03 hanging mug **(DEPRECATED 2026-08-09)** | Mug begins left, rack begins right, then the task hands mug from left to right at `middle_pos`. | Frozen. The only reliable hanging geometry rolls the rack ~58 deg about x; under an upright yaw-only rack the 19 mm bar cannot hold the 11 cm mug rim (all upright variants fail physically). Kept only to stabilise the manifest hash `28ee798eb3dfcde4`, wrappers, and tests; every instantiation warns `NotFullyImplemented` and it must not be collected. | Preserve functional-point hanging geometry and height test; replace `is_right_gripper_open()` with left open/right-home. |
| T04 bread basket | Breads can be on either side and task branches to dual grasp/dual placement. Basket is a dynamic actor at fixed center. | Sample basket and two breads in left reach with pairwise clearance. Make basket static for the derived task, because no second arm may stabilize it while the first/second bread is inserted. Place bread 0, retreat/home, then bread 1. | Require exactly two breads, each contained at basket location and above table; remove all source one-bread fallback and right-gripper condition. Instruction says left arm puts two breads into basket. |
| T05 bread skillet | Bread and skillet are deliberately put on opposite sides, then grasped concurrently. | Initialize the skillet as a static target directly in the left workspace. Sample only the bread in left reach with clearance from the skillet. The expert performs only left bread grasp, lift, place at skillet functional point, release, retreat. Do not create `position_skillet_left`, do not use `table_static=False`, and do not move the skillet. | Preserve bread-to-skillet functional-point distance and add left-open/right-home. Instruction says left arm places bread onto the skillet; it must not claim the skillet is moved. |
| T06 burger fries | Burger and fries are fixed to separate left/right source regions and grasped/lifted simultaneously. Tray is already static. | Keep tray static, relocate tray and both food objects into shared left reach, preserve tray functional point IDs, and transfer burger then fries serially with home reset in between. | Preserve both food-to-tray distance tests, remove right-gripper test, and use left-only instruction. |
| T07 can basket | Source randomly chooses active arm, then always uses opposite arm to grasp, close, lift, and pull basket; success requires basket and can height increase. | Use left arm only. Sample a static basket and can in left reach. Select the nearest left-valid basket functional point, place/release can, and retreat. Delete retry branches that lead to opposite-arm basket grasp. | This is an intentional narrowed task: can must be in upright basket, off table, with can/basket contact; basket must remain within 1 cm of initial height. Instruction must not say lift basket. |
| T08 cans plasticbox | Plastic box is dynamic; can 1/can 2 are left/right and are grasped concurrently. | Make plastic box static after initial settling or create it static. Sample both cans and box in left reach. Use deterministic point assignment can 1 -> point 1, return home, can 2 -> point 0. | Keep both-can-in-box distance threshold, remove right-gripper condition, and record deterministic point ownership in audit. |
| T09 dual shoes | Shoe box and success predicate are hard-coded at central coordinates; shoes spawn left/right and are grasped/placed in parallel. | Initialize the static shoe box inside the validated left-arm workspace, not at its source center coordinate. Sample both shoes in the same left workspace with clearance from the box and each other. Preserve shoe identity and box functional point assignment, but serially process shoe 1 then shoe 2. Do not retain source `right_shoe_middle_pose`. | Replace hard-coded world-coordinate predicate with pose-relative checks against the relocated box functional points plus orientation relative to box frame. Remove right-gripper condition. |
| T10 stack blocks two | Each block selects an arm by x. First block is placed at central target, second on first; both remain dynamic. | Sample both blocks in left reach. Move stack target to left workspace. Left places block 1, retreats/home, waits for stability, then left places block 2 at block 1 functional point. Do not make block 1 static after placement: that would change the stacking physics; reject unstable seeds instead. | Keep source relative XY/z stack threshold, replace both-gripper-open condition with left-open/right-home, and retain a post-place stability check. |
| T11 stack bowls two | Bowls choose arms by x; first bowl goes to central target, then second stacks. Both are dynamic and source sorts final state by z. | Sample both bowls and left-shift the target. Left places lower bowl, retreats/home, verifies settling, then top-down grasps/places upper bowl. Do not freeze the lower bowl after placement. | Preserve z-sorted nesting/height relation with target shifted left, remove right-gripper condition, and add a stable-lower-bowl prerequisite. |

The required semantic narrowing is T07, which removes basket lifting and must use a new instruction and dataset identity. T05 and T09 are scene-layout rewrites: skillet and shoe box must be initialized in the left workspace before the episode begins. These tasks require pilot video review in addition to numerical checks.

## Existing `place_object_scale` Implementations

`place_object_scale_left` and `place_object_scale_left_oppo_cam` are not part of this 22-entry-point delivery. They remain independent, working reference implementations during all six iterations. They provide the camera and single-arm behavioral reference, but they must not be refactored onto `LeftTaskBase` as part of this task: that would mix new-task development with regression risk for existing data, EMIX indexes, checkpoints, collection scripts, and published measurements.

The shared-base implementation may reuse proven ideas from these files, especially their left-only expert guard, separated Piper embodiment, and opposite-camera pose. It must not import either task as a superclass, because their scene geometry and scale-specific placement workaround are not general task infrastructure. Add a compatibility test that runs the existing central/opposite scale tasks unchanged on fixed seeds before and after this project.

Only after all 22 acceptance datasets pass may a separate migration change be proposed. Its explicit goal must be behavior-preserving cleanup, with fixed-seed comparisons of camera observations, success results, action dimensions, right-arm qpos, and evaluation step limits. That migration should be independently reviewable and revertible.

## Global I/O and Safety Contract

The native qpos action has this fixed ordering: `[left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]`. `left_qpos_dim=7` is the projection boundary. Expert code can command only `ArmTag("left")`. A full policy action is permitted as input, but before every simulator step indices `[7:]` are overwritten with the right home qpos captured after environment setup. The control mode is `qpos`; `ee` policy control is rejected for these tasks.

Each accepted episode writes a JSONL audit record with `task`, `camera_variant`, `seed`, `attempt`, `accepted_episode`, `sampled_layout`, `route_bins`, `expert_plan_success`, `expert_task_success`, `right_home_qpos`, `right_max_abs_deviation`, and `status`. `camera_variant` is exactly `central_cam` or `left_oppo_cam`. Valid status values are `accepted`, `scene_rejected`, `expert_failed`, `right_arm_violation`, and `invalid_observation`. Any right deviation over `1e-5` invalidates that episode.

Every `central_cam`/`left_oppo_cam` pair uses one immutable workspace manifest. The manifest defines all actor xy/yaw ranges, minimum clearances, static policy, route bins, pilot seed count, minimum 20 percent expert success rate, and minimum crossing-pair rate. It is hashed and reported. A pair may not have separate geometry tuning because camera is the only intended variation.

## New Script: `third_party/RoboTwin/envs/left_task_base.py`

This shared base implements camera setup, fixed-right-arm enforcement, deterministic sampling hooks, and audit output. All semantic task classes inherit it.

```python
ArmName = Literal["left"]
CameraVariant = Literal["central_cam", "left_oppo_cam"]

@dataclass(frozen=True)
class CameraSpec:
    name: CameraVariant
    position: tuple[float, float, float] | None
    forward: tuple[float, float, float] | None
    left: tuple[float, float, float] | None

CENTRAL_CAM = CameraSpec("central_cam", None, None, None)
LEFT_OPPO_CAM = CameraSpec(
    "left_oppo_cam", (-0.30, 0.45, 1.35), (0.0, -0.6, -0.8), (1.0, 0.0, 0.0)
)

@dataclass(frozen=True)
class XYYawRange:
    x: tuple[float, float]
    y: tuple[float, float]
    yaw_rad: tuple[float, float]

@dataclass(frozen=True)
class ActorSamplingSpec:
    name: str
    workspace: XYYawRange
    minimum_clearance_m: float
    static: bool

@dataclass(frozen=True)
class TaskManifest:
    semantic_task: str
    actor_specs: Mapping[str, ActorSamplingSpec]
    route_bin_count: int
    pilot_seed_count: int
    min_success_rate: float
    min_crossing_pair_rate: float

class LeftTaskBase(Base_Task):
    manifest: TaskManifest
    camera_variant: CameraVariant
    def setup_demo(self, **kwargs: Any) -> None: ...
    def configure_camera(self, kwargs: Mapping[str, Any]) -> dict[str, Any]: ...
    def sample_layout(self) -> dict[str, sapien.Pose]: ...
    def validate_layout(self, layout: Mapping[str, sapien.Pose]) -> bool: ...
    def move(self, actions_by_arm1: tuple, actions_by_arm2: tuple | None = None, save_freq: int | None = -1) -> Any: ...
    def capture_right_home_qpos(self) -> np.ndarray: ...
    def project_right_home(self, action: np.ndarray) -> np.ndarray: ...
    def take_action(self, action: np.ndarray, action_type: Literal["qpos", "ee"] = "qpos") -> Any: ...
    def assert_right_arm_stationary(self) -> None: ...
    def left_pick_place(self, actor: Any, target_pose: Sequence[float], *, functional_point_id: int = 0, pre_grasp_dis: float = 0.07, lift_z: float = 0.08, pre_dis: float = 0.08, dis: float = 0.02, constrain: str = "free") -> bool: ...
    def left_return_home(self) -> bool: ...
    def write_episode_audit(self, status: str, **details: Any) -> None: ...
```

`setup_demo` input is the existing RoboTwin task kwargs, including `left_embodiment_config`, `right_embodiment_config`, `seed`, `task_name`, and `domain_randomization`. It sets `arm_side="left"`, configures camera, calls `_init_task_env_`, captures the home qpos after homing, and returns `None`. It raises `ValueError` for missing camera configuration and `SceneInitRejectError` for a non-feasible layout.

`configure_camera` input is existing setup kwargs and the wrapper-selected `CameraVariant`; output is a deep-copied kwargs map and it never mutates its input. `CENTRAL_CAM` uses `None` pose values to mean preserve the asset-configured `head_camera` exactly. `LEFT_OPPO_CAM` overwrites only `position`, `forward`, and `left` of that `head_camera` with the listed values. In both cases it asserts exactly one enabled `head_camera`, asserts all wrist cameras are disabled, and rejects a missing or duplicate head camera. It must also return the selected exact variant name so collection audit, config generation, and dataset metadata cannot infer it from the task suffix.

`sample_layout` is abstract and returns a map from stable actor names to `sapien.Pose`. It uses seeded `numpy.random` only. The rejection-sampling algorithm permits 128 attempts: sample xy/yaw; derive z; reject pair clearance, prohibited-area, and orientation failures; classify pickup-to-target paths into route bins; accept when all task constraints hold. `validate_layout` is pure with regard to robot control and returns boolean legality.

`move` accepts exactly one `(ArmTag("left"), action)` pair and `None` as its second argument. It rejects a right tag, an `.opposite` tag, or parallel action group before calling `Base_Task.move`. `project_right_home` accepts one-dimensional native-width qpos, returns a copied `float32` vector with suffix `[7:]` replaced by home values, and rejects other widths. `take_action` rejects `ee`, projects the action, delegates to `Base_Task.take_action`, then checks actual right state. `assert_right_arm_stationary` computes the maximum absolute deviation and raises `RightArmMotionError` over tolerance.

`left_pick_place` input is one actor and a seven-value target pose. Algorithm: left grasp, vertical clearance, target placement, release, vertical retreat. It returns `self.plan_success` and must use guarded `move`. `left_return_home` issues left `back_to_origin` and returns plan status. `write_episode_audit` appends and flushes one audit record.

```python
class SceneInitRejectError(RuntimeError): pass
class RightArmMotionError(RuntimeError): pass
class LeftArmTaskError(ValueError): pass
```

## New Script: `third_party/RoboTwin/envs/left_task_manifests.py`

```python
MANIFESTS: dict[str, TaskManifest]
def get_manifest(semantic_task: str) -> TaskManifest: ...
def manifest_hash(manifest: TaskManifest) -> str: ...
```

This file owns the 11 shared manifests. `get_manifest` receives a source semantic name such as `place_burger_fries`, returns the immutable manifest, and raises `KeyError` for unknown names. Initial ranges are source ranges shifted to the left workspace and expanded in controlled increments. Any range/yaw change must be accompanied by a pilot result, version increase, and new hash.

## New Scripts: Semantic Classes and 22 Entry-Point Wrappers

Each row below adds one semantic class and two wrappers. A semantic class directly inherits `LeftTaskBase`. A wrapper contains no actor or expert algorithm; it only sets the class attribute consumed by `LeftTaskBase.configure_camera`. Explicit wrappers are required because RoboTwin imports `envs.<task_name>` and resolves a class with the exact task name. This avoids a second base class and a multiple-inheritance/MRO dependency.

```python
class place_burger_fries_left(PlaceBurgerFriesLeftImpl):
    camera_variant: Literal["central_cam"] = "central_cam"

class place_burger_fries_left_oppo_cam(PlaceBurgerFriesLeftImpl):
    camera_variant: Literal["left_oppo_cam"] = "left_oppo_cam"
```

| Semantic script/class | `central_cam` wrapper | `left_oppo_cam` wrapper |
| --- | --- | --- |
| `blocks_ranking_rgb_left_impl.py` / `BlocksRankingRgbLeftImpl` | `blocks_ranking_rgb_left.py` | `blocks_ranking_rgb_left_oppo_cam.py` |
| `blocks_ranking_size_left_impl.py` / `BlocksRankingSizeLeftImpl` | `blocks_ranking_size_left.py` | `blocks_ranking_size_left_oppo_cam.py` |
| `hanging_mug_left_impl.py` / `HangingMugLeftImpl` **(DEPRECATED)** | `hanging_mug_left.py` | `hanging_mug_left_oppo_cam.py` |
| `place_bread_basket_left_impl.py` / `PlaceBreadBasketLeftImpl` | `place_bread_basket_left.py` | `place_bread_basket_left_oppo_cam.py` |
| `place_bread_skillet_left_impl.py` / `PlaceBreadSkilletLeftImpl` | `place_bread_skillet_left.py` | `place_bread_skillet_left_oppo_cam.py` |
| `place_burger_fries_left_impl.py` / `PlaceBurgerFriesLeftImpl` | `place_burger_fries_left.py` | `place_burger_fries_left_oppo_cam.py` |
| `place_can_basket_left_impl.py` / `PlaceCanBasketLeftImpl` | `place_can_basket_left.py` | `place_can_basket_left_oppo_cam.py` |
| `place_cans_plasticbox_left_impl.py` / `PlaceCansPlasticboxLeftImpl` | `place_cans_plasticbox_left.py` | `place_cans_plasticbox_left_oppo_cam.py` |
| `place_dual_shoes_left_impl.py` / `PlaceDualShoesLeftImpl` | `place_dual_shoes_left.py` | `place_dual_shoes_left_oppo_cam.py` |
| `stack_blocks_two_left_impl.py` / `StackBlocksTwoLeftImpl` | `stack_blocks_two_left.py` | `stack_blocks_two_left_oppo_cam.py` |
| `stack_bowls_two_left_impl.py` / `StackBowlsTwoLeftImpl` | `stack_bowls_two_left.py` | `stack_bowls_two_left_oppo_cam.py` |

### `blocks_ranking_rgb_left_impl.py` and `blocks_ranking_size_left_impl.py`

Both classes expose `load_actors() -> None`, `place_ranked_block(block: Actor, target_pose: Sequence[float]) -> bool`, `play_once() -> dict`, and `check_success() -> bool`. RGB preserves red/green/blue ordering; size preserves large/medium/small source intervals. `load_actors` samples three blocks plus ordered targets inside left space. `play_once` serially calls `place_ranked_block`; RGB sequence is red/green/blue, size sequence is small/medium/large. The helper calls `left_pick_place(..., constrain="align")`. Success preserves the source ordered-line predicate, requires left gripper open, and asserts right immobility.

### `hanging_mug_left_impl.py` — DEPRECATED (frozen, do not collect)

**DEPRECATED 2026-08-09.** `HangingMugLeftImpl` is frozen. The only reliable hanging geometry requires a rack rolled ~58 deg about x (fp0 world z ~0.85-0.87); an upright, yaw-only rack cannot physically hang the 039_mug — the 19 mm rack bar cannot support the 11 cm mug rim, and no mug model offers a handle collision void. Every instantiation emits a `RuntimeWarning` (`NotFullyImplemented`). The task remains registered only to keep the manifest hash `28ee798eb3dfcde4` and the wrapper/test contract stable. See `docs/theory/viewpoint_experiment_summary.md`.

Legacy description (kept for reference): `HangingMugLeftImpl` exposes `load_actors() -> None`, `hang_mug_left() -> bool`, `play_once() -> dict`, and `check_success() -> bool`. It samples mug and static rack in the left workspace, rejecting inaccessible rack approach orientation. `hang_mug_left` eliminates source handover: grasp mug, lift, move through a left-side safe transit pose, align mug functional point to rack point with `pre_dis_axis="fp"`, release, retreat. Success retains source rack/mug geometry, left-open condition, and right-stationary assertion.

### `place_bread_basket_left_impl.py`

`PlaceBreadBasketLeftImpl` exposes `load_actors() -> None`, `place_bread_left(index: int) -> bool`, `play_once() -> dict`, and `check_success() -> bool`. It samples two breads and a basket with pairwise clearance in left workspace. The helper validates index 0 or 1, places that bread at basket functional point, and uses larger retreat after the first placement. `play_once` runs indexes 0 then 1. Success keeps both containment checks and shared arm assertions.

### `place_bread_skillet_left_impl.py`

`PlaceBreadSkilletLeftImpl` exposes `load_actors() -> None`, `place_bread_on_skillet_left() -> bool`, `play_once() -> dict`, and `check_success() -> bool`. `load_actors` creates a static skillet directly in the manifest's left workspace, then samples only bread in left reach with skillet/bread clearance. `place_bread_on_skillet_left` is the complete expert algorithm: left grasp bread, lift, place on skillet functional point zero, release, retreat. It must not expose or call a skillet-moving helper. Success retains bread-on-skillet geometry plus left-open/right-home checks.

### `place_burger_fries_left_impl.py`

`PlaceBurgerFriesLeftImpl` exposes `load_actors() -> None`, `place_food_left(actor: Actor, point_id: int) -> bool`, `play_once() -> dict`, and `check_success() -> bool`. It samples static tray, hamburger, fries in left space. Helper validates point ID in `{0, 1}` and calls `left_pick_place`; `play_once` uses hamburger/0 then fries/1. Success keeps two tray-distance tests plus shared checks.

### `place_can_basket_left_impl.py`

`PlaceCanBasketLeftImpl` exposes `load_actors() -> None`, `select_basket_point() -> np.ndarray`, `play_once() -> dict`, and `check_success() -> bool`. The derived objective is explicitly "place can into basket and leave basket on table" because source basket lifting requires the opposite arm. `select_basket_point` compares two basket points to current left EEF xy, returns a copied left-oriented seven-value target, and never mutates asset API output. Success requires can/basket contact, no can/table contact, source distance tolerance, basket height within 1 cm of start, left open, and right stationary. New instruction text must describe this changed objective.

### `place_cans_plasticbox_left_impl.py`

`PlaceCansPlasticboxLeftImpl` exposes `load_actors() -> None`, `place_can_left(can: Actor, point_id: int) -> bool`, `play_once() -> dict`, and `check_success() -> bool`. It samples box and two cans in left workspace. Point ownership is deterministic: object 1 uses point 1, object 2 uses point 0. `play_once` serializes the helper calls. Success retains two-can box-distance conditions plus shared checks.

### `place_dual_shoes_left_impl.py`

`PlaceDualShoesLeftImpl` exposes `load_actors() -> None`, `place_shoe_left(shoe: Actor, functional_point_id: int) -> bool`, `play_once() -> dict`, and `check_success() -> bool`. It creates the static shoe box directly inside left workspace and samples both shoes in the same workspace. Semantic shoe identity, not sampled x position, controls target point. The helper makes a left-only aligned pick-place-release. Success uses box-relative containment/orientation predicates plus shared checks.

### `stack_blocks_two_left_impl.py` and `stack_bowls_two_left_impl.py`

`StackBlocksTwoLeftImpl` exposes `stack_top_block_left() -> bool`; `StackBowlsTwoLeftImpl` exposes `stack_top_bowl_left() -> bool`; each also implements `load_actors`, `play_once`, and `check_success`. They sample base/lower and top/upper assets in left space, place the lower actor dynamically, wait for and verify settling, then pick only the top actor and make vertical alignment placement. They must not freeze the already placed lower actor because that changes stack physics; instead, unstable seeds are rejected. Both retain source contact/nesting predicates and require left open and right stationary.

## New Script: `scripts/generate_left_task_assets.py`

```python
@dataclass(frozen=True)
class VariantSpec:
    semantic_task: str
    task_name: str
    camera_variant: Literal["central_cam", "left_oppo_cam"]
    source_step_limit: int

def load_registry(path: Path) -> list[VariantSpec]: ...
def validate_registry(specs: Sequence[VariantSpec]) -> None: ...
def render_task_config(spec: VariantSpec, episodes: int) -> dict[str, Any]: ...
def render_instruction(spec: VariantSpec) -> dict[str, Any]: ...
def render_train_config(spec: VariantSpec) -> dict[str, Any]: ...
def render_eval_config(spec: VariantSpec) -> dict[str, Any]: ...
def write_or_check(path: Path, payload: Mapping[str, Any], check: bool) -> None: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

Input is `configs/left_task_registry.yaml` with exactly 22 rows. Each row's `camera_variant` value must be exactly `central_cam` or `left_oppo_cam`; outputs are `task_config/<task>.yml`, instruction JSON, train/eval stubs under matching `central_cam` or `left_oppo_cam` config directories, and 22 keys in `_eval_step_limit.yml`. The generator must preserve unrelated keys/comments. Every collection config uses `[piper, piper, 0.60]`, D435, exactly one enabled head camera, all wrist cameras disabled, RGB/endpose/qpos enabled, `save_freq: 15`, and 500 episodes. It creates separate `<task>_acceptance_10ep.yml` files changing only episode count. All instruction placeholders must match `play_once` output and state only left-arm actions.

## New Script: `scripts/validate_left_task.py`

```python
@dataclass
class ValidationResult:
    task: str
    camera_variant: str
    seeds_attempted: int
    expert_successes: int
    expert_success_rate: float
    crossing_pair_rate: float
    max_right_arm_deviation: float
    source_step_limit: int
    derived_step_limit: int
    failures: list[dict[str, Any]]

def validate_static_contract(task: str, benchmark_root: Path) -> list[str]: ...
def run_seed(task: str, config: str, seed: int, benchmark_root: Path) -> dict[str, Any]: ...
def calculate_crossing_pair_rate(records: Sequence[Mapping[str, Any]]) -> float: ...
def validate_task(task: str, config: str, seeds: Sequence[int], benchmark_root: Path, output_dir: Path) -> ValidationResult: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

Static checks cover import/class identity, config/instruction presence, an exact `central_cam` or `left_oppo_cam` value, D435/head-camera/wrist settings, exact source step limit, placeholders, output namespace, and the required `LEFT_TASK_DESIGN:` module header fields. AST checks reject direct `ArmTag("right")`, `.opposite`, and two-action `move` in semantic classes. `run_seed` always closes environment in `finally` and emits a serializable record. The crossing rate equals intersecting non-adjacent xy route-segment pairs divided by comparable pairs, excluding endpoints within 2 cm. A 100-seed pilot passes only at >=20% success, manifest crossing threshold, <=1e-5 right deviation, and exact step parity. `central_cam`/`left_oppo_cam` members must share one manifest hash and success criterion.

## New Script: `scripts/collect_left_task.py`

```python
@dataclass(frozen=True)
class CollectionRequest:
    task: str
    config: str
    expected_successes: int
    mode: Literal["acceptance", "train"]
    cuda: str | None
    output_root: Path

def build_acceptance_config(task: str, output: Path) -> Path: ...
def collect_one(request: CollectionRequest, project_root: Path) -> Path: ...
def summarize_seed_success(collection_root: Path) -> dict[str, Any]: ...
def assert_collection_contract(collection_root: Path, expected_successes: int) -> None: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

This wraps `data_collection.robotwin.collect`; the only change to that existing file is adding all 22 exact names to `ALLOWED_TASKS`. Acceptance creates a non-overwriting 10-success config; train mode uses 500-success config. Collection verification checks HDF5 count, `seed.txt`, instructions, head-camera/qpos/endpose fields, audit JSONL, and right-arm tolerance. Success statistics report attempts, successes, `successes / attempts`, maximum consecutive failures, and failure histogram.

## New Script: `scripts/convert_left_task_acceptance.py`

```python
def convert_acceptance(task: str, collection_root: Path, lerobot_root: Path, repo_id: str) -> Path: ...
def verify_lerobot_dataset(root: Path, expected_episodes: int = 10) -> dict[str, Any]: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

Every one of 22 conversions calls the existing converter with `--instruction-type seen --image-width 84 --image-height 84 --fps 17 --left-qpos-dim 7`. Repo ID is `<task>_acceptance_84`. Verification fails unless exactly ten episodes, image shape `(3, 84, 84)`, nonempty instructions, expected action/state widths, and home-fixed right qpos evidence are present.

## New Script: `scripts/run_left_task_acceptance.py`

```python
STAGES = ("generate", "static", "pilot", "collect", "convert", "report")
def stage_is_complete(task: str, stage: str, root: Path) -> bool: ...
def run_stage(task: str, stage: str, env: Mapping[str, str]) -> None: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

Inputs: `--task <one-of-22>|--all`, `--from-stage`, `--force-stage`, `--cuda`, `--twin-env` default `emix-twin`, `--train-env` default `emix`, `--dry-run`. Validation/collection run in `emix-twin`; conversion runs in `emix`. Stage completion inspects artifacts, not folder existence. Forced execution creates timestamped output and does not delete data.

## New Script: `scripts/report_left_task_results.py`

```python
def load_task_artifacts(task: str, root: Path) -> dict[str, Any]: ...
def validate_report_inputs(payload: Mapping[str, Any]) -> list[str]: ...
def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str: ...
def render_json(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

Outputs `outputs/left_task_acceptance/summary.md` and `summary.json` with 22 rows and a `central_cam`-versus-`left_oppo_cam` comparison per semantic task. Each row includes manifest hash, seed attempts, successes, success rate, max failure streak, crossing rate, max right deviation, source/derived limits, raw/LeRobot counts, and image size. Missing evidence is `FAIL`.

## Tests

Add `tests/test_left_task_base.py`: reject right/parallel moves; projection copies rather than mutates; bad widths fail; right physics motion fails; `central_cam` leaves the copied head-camera fields byte-for-byte unchanged; `left_oppo_cam` changes only `position`, `forward`, and `left` to its contract values; both have one head camera and no enabled wrist camera; same seed gives same layout; pair manifest hashes match.

Add `tests/test_left_task_assets.py`: generator check passes; 22 modules export exact class names; all limits equal sources; all configs use D435 and disable wrist; acceptance configs use 10 and train configs 500; instructions match placeholders and left-only objective.

Add `tests/test_left_task_conversion.py` using HDF5 fixtures: 84x84 arguments forwarded; wrong count/resolution fails; a changing right qpos fails; missing instructions fail. Add `@pytest.mark.robotwin` integration tests for setup/play/check/close on three fixed seeds for each of 22 task names.

## Iteration Plan and Exit Gates

1. **Iteration 0, registry/assets:** implement shared base interfaces, manifests, registry, generator, task configs, instruction files, and wrappers. Gate: 22 imports and exact static step mapping.
2. **Iteration 1, safety:** implement projection and change `evaluation/evaluate.py` to use `freeze_right_action(action: np.ndarray, home_qpos: np.ndarray, left_qpos_dim: int) -> np.ndarray`, with the same validation/copy behavior as the base action projection. Gate: malicious right expert command rejects and policy suffix is home-projected.
3. **Iteration 2, simple semantics:** implement T03/T05/T07/T10/T11, then both wrappers each. Run 30-seed diagnosis followed by 100-seed pair pilot. Gate: >=20% success, zero right violation, exact limit, shared manifest.
4. **Iteration 3, multi-object semantics:** implement T01/T02/T04/T06/T08/T09, verify deterministic target ownership and serial releases, then use the same pilot gate.
5. **Iteration 4, freeze:** pilot all eleven pairs, freeze manifests and hashes. Gate: every pair has one passing 100-seed report.
6. **Iteration 5, acceptance artifacts:** collect ten successful episodes and independently convert each of 22 variants to 84x84. Gate: 22 raw collections, 22 LeRobot roots, passing audits, statistics, and combined report.
7. **Iteration 6, optional training:** only after acceptance, collect independent 500-success data, convert 84x84, build independent EMIX index, train, evaluate. Cross-variant or source artifact reuse is prohibited.

## Agent Handoff

Assign one infrastructure agent to the shared base, manifests, generator, evaluator, and tests; semantic agents to disjoint task groups after base tests pass; one pipeline agent to validator, collector, converter, and report scripts; and one integration owner to serial pilots and collections. A variant is complete only with module, config, instruction, source-equivalent step limit, passing pilot, ten raw episodes, ten 84x84 converted episodes, right-arm audit, and seed success report. A semantic task family is complete only when both `central_cam` and `left_oppo_cam` variants meet that gate with the same semantic algorithm and manifest.
