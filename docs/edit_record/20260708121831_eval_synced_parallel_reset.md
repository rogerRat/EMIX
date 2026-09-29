# Eval Synced Parallel Reset

## Problem

The previous evaluation loop used `num_envs > 1` together with partial reset:

```python
env.reset(options={"env_idx": done_np})
```

This is not clean for LeRobot DiffusionPolicy evaluation. The policy is
stateful and keeps observation/action queues as full-batch tensors. A partial
environment reset changes only selected batch rows in ManiSkill, while the
policy queue has no per-env reset API.

The result is same-row cross-episode contamination:

- one sub-env can enter a new episode
- the corresponding policy queue row can still contain history or action chunks
  from the previous episode
- calling global `policy.reset()` would instead clear still-active episodes in
  other rows

This is not cross-env visibility. Batch rows remain independent during forward
passes. The issue is that episode lifetimes differ across rows while the policy
state can only be reset for the whole batch.

## Decision

Keep parallel evaluation, but remove partial reset.

The new evaluation loop uses synced batch episodes:

1. reset all envs together
2. call `policy.reset()`
3. run the whole batch for `max_episode_steps`
4. record metrics for each env, accumulating only while that env is active
5. start the next batch with another full env reset and policy reset

This preserves batch policy inference and ManiSkill vectorized rollout speed
while keeping the policy queue lifecycle aligned with the environment lifecycle.

## Files Changed

- `evaluation/evaluate.py`
  - Removed partial reset from the rollout loop.
  - Restored `EvalConfig.num_envs` default to `5`.
  - Added synced batch episode evaluation.
  - Metrics for an env stop accumulating after its first done signal in the
    current batch.

- `configs/eval/**/*.yaml`
  - Restored `num_envs: 5` for evaluation configs and template.

- `docs/sop/SOP-04-evaluation.md`
  - Documented the synced parallel reset semantics.
  - Explained why partial reset is incompatible with current LeRobot
    DiffusionPolicy queues.

## Caveat

The loop intentionally continues stepping the full vectorized environment until
`max_episode_steps`. If an env reports done early, its metrics stop
accumulating, but the env is not reset until the whole batch finishes.

This avoids policy queue contamination, but it depends on the ManiSkill
environment and wrappers allowing post-done stepping until the synced batch
horizon. If post-done stepping is invalid for a specific environment, the clean
alternatives are:

- use fixed-horizon / `ignore_terminations` semantics so success or failure does
  not end a sub-env early
- run serial evaluation with `num_envs=1`
- implement per-env policy state so partial reset only clears the done env's
  policy queue row
