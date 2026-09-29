# Agent Implementation Plan: EMIX Mixed Training

## 0. Execution Contract

- Implement this plan exactly. Do not redesign schema names, scheduler names, or policy behavior.
- All code, comments, logs, and technical identifiers must be ASCII-only.
- User-facing explanations outside code may be Chinese.
- Do not use global pip or user site-packages. If tests require Python packages, use the active venv/conda interpreter only.
- Before any destructive operation, create a timestamped backup of the target resource.
- This task requires code changes after this document is approved for implementation. This document itself is only the implementation specification.

## 1. Target Behavior

Add mixed training to `embodied_mixup/policies/train.py`.

Mixed training means:

- The training loop owns two data sources:
  - non-EMIX LeRobot dataset from top-level `dataset`.
  - EMIX dataset from `policy.emix_files_root`.
- Each training step samples one batch source.
- If the sampled source is `emix`, the whole batch comes from EMIX data.
- If the sampled source is `non_emix`, the whole batch comes from non-EMIX data.
- Sampling is step/batch-level, not sample-level.
- The current EMIX batch probability is named `emix_batch_prob`.
- The current EMIX batch probability is produced by `mix_training.scheduler`.
- `emix_*` policies must support both batch kinds:
  - EMIX batch: contains `emix_actions` and `emix_alpha`; use EMIX dual-branch loss.
  - non-EMIX batch: does not contain `emix_actions`; use the parent policy's normal training loss.

Do not implement sample-level mixing inside one DataLoader batch.

## 2. Required YAML Schema

All user-controlled configuration must remain inside files under `configs/policies/*.yaml`.

Use this top-level schema:

```yaml
mix_training:
  enabled: true
  seed: 0
  scheduler:
    type: constant
    emix_batch_prob: 0.10
```

Use this schema for linear annealing:

```yaml
mix_training:
  enabled: true
  seed: 0
  scheduler:
    type: linear
    start_emix_batch_prob: 0.30
    end_emix_batch_prob: 0.10
```

Policy alpha configuration must be flat under `policy`:

```yaml
policy:
  type: emix_diffusion
  emix_files_root: data/emix_data/example/emix_files
  mixup_alpha_mode: beta
  mixup_beta: 0.4
  mixup_alpha_value: null
  mixup_loss_weight: 1.0
  mixup_visual_keys: null
  mixup_state_mode: anchor
```

Allowed values:

- `mix_training.enabled`: bool. Missing or false means use existing single-source training behavior.
- `mix_training.seed`: int. If missing, use the global training seed.
- `mix_training.scheduler.type`: `constant` or `linear`.
- `constant.emix_batch_prob`: float in `[0.0, 1.0]`.
- `linear.start_emix_batch_prob`: float in `[0.0, 1.0]`.
- `linear.end_emix_batch_prob`: float in `[0.0, 1.0]`.
- `policy.mixup_alpha_mode`: `beta` or `fixed`.
- `policy.mixup_beta`: float greater than `0.0`, used only when `mixup_alpha_mode == "beta"`.
- `policy.mixup_alpha_value`: float in `(0.0, 1.0)`, required when `mixup_alpha_mode == "fixed"`.

Do not reuse `mixup_prob`.
Do not add `emix_batch_prob` under `policy`.
Do not rename `mix_training`.

## 3. Breaking Change: Delete `mixup_prob`

Delete `mixup_prob` from all EMIX policy configs and loader settings.

Required removals:

- Remove `mixup_prob` from `embodied_mixup/policies/ACT/cfg_emix_act.py`.
- Remove `mixup_prob` from `embodied_mixup/policies/DP/cfg_emix_diffusion.py`.
- Remove `mixup_prob` from `embodied_mixup/policies/SmolVLA/cfg_emix_smolvla.py`.
- Remove `mixup_prob` from `EmixLoaderSettings` in `embodied_mixup/policies/utils/emix_loader_registry_utils.py`.
- Remove all validation for `mixup_prob`.
- Remove all reads of `policy_cfg.mixup_prob`.
- Remove all reads of `loader_cfg.mixup_prob`.
- Remove all YAML examples or generated configs that include `mixup_prob`.

New rule:

- Every EMIX dataset sample must perform visual mixup.
- There is no probability gate for skipping visual mixup.
- `alpha = 1.0` must not be used as a skip-mixup path.

## 4. Alpha Sampling Rules

Implement alpha generation inside `EmixPolicyDataset`.

Required behavior:

- If `mixup_alpha_mode == "beta"`:
  - Sample `alpha` from `torch.distributions.Beta(mixup_beta, mixup_beta)`.
  - Sample independently for each dataset item.
- If `mixup_alpha_mode == "fixed"`:
  - Use `mixup_alpha_value` exactly.
- Store alpha in the returned batch item as `emix_alpha`.
- Use the same alpha for:
  - visual input interpolation.
  - EMIX dual-branch loss weighting.

Required validation:

- Unknown `mixup_alpha_mode` raises `ValueError`.
- `mixup_beta <= 0.0` raises `ValueError`.
- `fixed` mode without `mixup_alpha_value` raises `ValueError`.
- `mixup_alpha_value <= 0.0` or `mixup_alpha_value >= 1.0` raises `ValueError`.

Expected visual interpolation remains:

```python
mixed = alpha * tensor_0 + (1.0 - alpha) * tensor_1
```

Expected loss weighting remains:

```python
mixed_loss = alpha * loss_0 + (1.0 - alpha) * loss_1
```

## 5. Scheduler Module

Add a scheduler utility module.

Recommended files:

- `embodied_mixup/policies/utils/mix_scheduler.py`
- `embodied_mixup/policies/utils/mix_scheduler_utils.py`

Minimum required API:

```python
class MixSchedulerProtocol:
    def emix_batch_prob(self, step: int, total_steps: int) -> float:
        ...
```

Required built-in schedulers:

- `constant`
- `linear`

Required registry:

```python
MIX_SCHEDULER_REGISTRY = {
    "constant": ConstantMixScheduler,
    "linear": LinearMixScheduler,
}
```

Required extension hook:

```python
def register_mix_scheduler(name: str, factory: Callable[..., MixSchedulerProtocol]) -> None:
    ...
```

Do not implement YAML dynamic import paths in v1.
Do not require external modules in v1.
The registry is only a stable extension point for future external schedulers.

Constant scheduler:

- Always return `emix_batch_prob`.

Linear scheduler:

- Use the global training step and `total_steps`.
- At the first effective training step, probability is near `start_emix_batch_prob`.
- At `step == total_steps`, probability is `end_emix_batch_prob`.
- Clamp the interpolation fraction into `[0.0, 1.0]`.
- Resume must not reset the schedule.

Recommended linear formula:

```python
if total_steps <= 1:
    progress = 1.0
else:
    progress = (step - 1) / (total_steps - 1)
progress = min(1.0, max(0.0, progress))
prob = start_prob + progress * (end_prob - start_prob)
```

## 6. Training Loop Changes

Modify `embodied_mixup/policies/train.py`.

Required data construction behavior:

- If `mix_training.enabled` is false or missing:
  - Preserve current behavior.
  - One dataset.
  - One dataloader.
- If `mix_training.enabled` is true:
  - Require `policy_cfg.type` to be one of:
    - `emix_diffusion`
    - `emix_smolvla`
  - Construct non-EMIX `LeRobotDataset` from top-level `dataset`.
  - Construct EMIX dataset using `policy.emix_files_root`.
  - Construct two DataLoaders with the existing `batch_size`, `num_workers`, `collate_fn`, and `drop_last=False`.
  - Maintain two independent iterators.
  - On `StopIteration`, recreate only the exhausted iterator.

Required per-step behavior:

1. Compute `emix_batch_prob = scheduler.emix_batch_prob(step, total_steps)`.
2. Draw one random value from a dedicated Python `random.Random` initialized with `mix_training.seed`.
3. If random value is less than `emix_batch_prob`, source is `emix`; otherwise source is `non_emix`.
4. Fetch one batch from the selected iterator.
5. Move tensor values to device using the existing behavior.
6. Call `_prepare_policy_batch`.
7. Call `policy.forward(batch)`.
8. Run backward, optimizer, scheduler, logging, checkpointing as before.

Required logging:

- At existing log frequency, include:
  - `batch_source`
  - `emix_batch_prob`
  - `loss`
  - `loss_dict`

Required config snapshot:

- Include raw `mix_training` payload.
- Include resolved scheduler type.
- Include `start_step`.
- Include `total_steps`.

Do not change checkpoint directory naming.
Do not change optimizer or LR scheduler behavior.
Do not change non-mixed training behavior.

## 7. Policy Forward Changes

Modify `embodied_mixup/policies/DP/emix_diffusion.py`.

Required behavior:

- If `batch` contains `emix_actions`:
  - Keep existing EMIX loss path.
- Else:
  - Call the parent `DiffusionPolicy.forward()` normal path.

Modify `embodied_mixup/policies/SmolVLA/emix_smolvla.py`.

Required behavior:

- If `batch` contains `emix_actions`:
  - Keep existing EMIX loss path.
- Else:
  - Use the parent `SmolVLAPolicy` normal training loss path.

Do not add mixed training support to `emix_act` in v1.
Do still delete `mixup_prob` from `emix_act` config for consistency.

## 8. Loader Changes

Modify `embodied_mixup/policies/utils/emix_loader_registry_utils.py`.

Required changes:

- Add fields to `EmixLoaderSettings`:
  - `mixup_alpha_mode: str = "beta"`
  - `mixup_alpha_value: float | None = None`
- Keep:
  - `mixup_beta`
  - `mixup_visual_keys`
  - `mixup_state_mode`
  - `seed`
  - `include_task`
- Remove:
  - `mixup_prob`
- Replace `_sample_alpha()` with mode-based alpha generation.
- Do not return alpha `1.0` as a no-mixup fallback.

Modify `embodied_mixup/policies/utils/emix_loader_registry.py`.

Required changes:

- Pass `mixup_alpha_mode`.
- Pass `mixup_alpha_value`.
- Stop passing `mixup_prob`.

## 9. Config Class Changes

Modify:

- `embodied_mixup/policies/ACT/cfg_emix_act.py`
- `embodied_mixup/policies/DP/cfg_emix_diffusion.py`
- `embodied_mixup/policies/SmolVLA/cfg_emix_smolvla.py`

Required fields:

```python
mixup_beta: float = 0.4
mixup_alpha_mode: str = "beta"
mixup_alpha_value: float | None = None
mixup_loss_weight: float = 1.0
mixup_visual_keys: list[str] | None = None
mixup_state_mode: str = "anchor"
```

Required validation:

- `mixup_alpha_mode` must be `beta` or `fixed`.
- `mixup_beta > 0`.
- If `fixed`, `mixup_alpha_value` must be in `(0.0, 1.0)`.
- If `beta`, `mixup_alpha_value` may be `None`.
- `mixup_loss_weight > 0`.
- `mixup_state_mode == "anchor"`.

Do not keep `mixup_prob` for backward compatibility.

## 10. Example Configs

Add or update at least one config under `configs/policies` for mixed diffusion training.

Recommended file:

- `configs/policies/emix_mixed_diffusion_stack_cube_eef_base_nonoop.yaml`

Optional second file:

- `configs/policies/emix_mixed_smolvla_stack_cube_eef_base_nonoop.yaml`

The config must:

- Use `policy.type: emix_diffusion` or `policy.type: emix_smolvla`.
- Include `policy.emix_files_root`.
- Include `policy.mixup_alpha_mode`.
- Include `policy.mixup_beta`.
- Include `policy.mixup_alpha_value`.
- Include top-level `dataset` for non-EMIX LeRobot data.
- Include top-level `mix_training`.
- Not include `mixup_prob`.

## 11. Tests

Add tests where the repository test structure supports them.
If there is no existing test framework, add minimal tests in the closest existing pattern or document exact dry-run commands.

Required unit coverage:

- Constant scheduler returns exact probability.
- Linear scheduler returns start, middle, and end probabilities.
- Scheduler rejects invalid probability values.
- Scheduler rejects unknown type.
- Alpha config rejects invalid modes.
- Alpha config rejects invalid fixed values.
- EMIX loader no longer exposes or reads `mixup_prob`.

Required integration or smoke coverage:

- Mixed training with `emix_batch_prob: 1.0` selects EMIX source.
- Mixed training with `emix_batch_prob: 0.0` selects non-EMIX source.
- Fixed seed produces deterministic source selection.
- `emix_diffusion` accepts EMIX and non-EMIX batches.
- `emix_smolvla` accepts EMIX and non-EMIX batches.

If full training cannot run due to missing data or weights, perform:

- Python import checks.
- Config parse checks.
- Scheduler unit tests.
- Static grep check proving `mixup_prob` was removed from code and configs.

## 12. Acceptance Checklist

Implementation is complete only when all items are true:

- `rg -n "mixup_prob" embodied_mixup configs docs` returns no active code/config references except historical edit records if unavoidable.
- `mix_training.enabled: false` preserves old single-source behavior.
- `mix_training.enabled: true` creates two DataLoaders.
- Batch source is sampled per step.
- `emix_batch_prob` is logged.
- `linear` schedule is based on global step and `total_steps`.
- Resume does not restart linear schedule.
- `emix_diffusion` works with both EMIX and non-EMIX batches.
- `emix_smolvla` works with both EMIX and non-EMIX batches.
- EMIX samples always perform visual interpolation using generated alpha.
- Alpha can be generated by Beta or fixed value.
- User-editable mixed training settings live in `configs/policies/*.yaml`.
- No global pip or user-site dependency installation was used.

## 13. Non-Goals

- Do not implement sample-level EMIX/non-EMIX mixing inside one batch.
- Do not implement cosine, polynomial, or plugin-loaded scheduler in v1.
- Do not implement YAML dynamic import path for schedulers in v1.
- Do not add mixed training support for `emix_act` in v1.
- Do not change evaluation code.
- Do not change EMIX candidate generation or EmixFiles materialization.
- Do not alter checkpoint directory naming or save format beyond adding config snapshot metadata.
