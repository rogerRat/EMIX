# Task segment annotation for sin-v0-patch-oppo formal

This dataset uses task-reviewed operation order to assign the manipulated object.
Recorded gripper action values locate event boundaries; they do not infer object
identity or task success.

## Boundary convention

- A PICK is the gripper closing command. Its segment ends at the first frame
  where the gripper action signal reaches the closed endpoint and stops changing.
- A PLACE is the gripper opening command. Its segment ends at the first frame
  where the signal reaches the open endpoint and stops changing.
- The endpoint sample belongs to the PICK or PLACE that just ended. The next
  segment begins on the following frame.
- Every operation has two segments: `step_NN_pick` and `step_NN_place`.
  Both carry that operation's manipulated-object label.
- The first PICK segment includes the initial approach and the complete closing
  motion. The PLACE segment includes the transfer, placement approach, and
  complete opening motion. After one PLACE, the next operation's PICK segment
  starts at the next frame. The final return after the final PLACE is unlabeled.

The annotation tool reads the final component of each recorded `action` vector.
A boundary is accepted only when the signal reaches exactly 0 (closed) or 1
(open) and the endpoint value persists on the next frame. The tool verifies the
episode starts open and that its ordered event sequence is exactly PICK, PLACE
for each task-code operation. It fails on missing, extra, or reordered events.
It does not infer whether a grasp or placement succeeded; that requires a
separate outcome or visual review.

## Task operation order and object

The object names below are the canonical roles returned by each task's
`segment_object_actors()` method.

| Task | Ordered operation objects |
|---|---|
| `blocks_ranking_rgb_left_oppo_cam` | `block1` (red), `block2` (green), `block3` (blue) |
| `blocks_ranking_size_left_oppo_cam` | `block3` (small), `block2` (medium), `block1` (large) |
| `place_bread_basket_left_oppo_cam` | `bread_0`, `bread_1` |
| `place_bread_skillet_left_oppo_cam` | `bread` |
| `place_burger_fries_left_oppo_cam` | `hamburg`, `frenchfries` |
| `place_can_basket_left_oppo_cam` | `can` |
| `place_cans_plasticbox_left_oppo_cam` | `can1`, `can2` |
| `place_dual_shoes_left_oppo_cam` | `left_shoe`, `right_shoe` |
| `stack_blocks_two_left_oppo_cam` | `block1`, `block2` |
| `stack_bowls_two_left_oppo_cam` | `bowl1`, `bowl2` |

For each ordered object, `step_NN_pick` spans from the previous PLACE endpoint
(or episode start) through this PICK endpoint. `step_NN_place` spans from the
next frame through this PLACE endpoint. The following operation starts on the
next frame.

## Current formal-batch check

All 300 episodes in each of the 10 tasks have the expected alternating gripper
event count and order: two events per operation, ending with the gripper open.
Exact endpoint plateaus were present; the shortest observed endpoint plateau
was 21 frames. Therefore these action streams support automatic phase-boundary
and task-object labeling under this fixed task recipe. The action stream alone
does not validate physical task success.

Generated per-frame JSON inputs are stored at
`/root/autodl-tmp/sin-v0-patch-oppo/formal/segment_annotations/<task>/episodeN.json`.
Before conversion into training sidecars, canonical object names in the
existing instance-mask tables must be repaired in a separate prepared dataset
root; many current sidecars still contain the removed `patch_target` name.
