# EmixIndex v3 Specification

## Normative definition

The normative persisted data structure and runtime validator are defined in
`embodied_mixup/candidates/emix_index_spec.py`. The matching Python
dataclasses and serialization live in `embodied_mixup/candidates/emix_index.py`.
The streaming index builder validates the v3 public record and every emitted
cross-point group against the normative module. This document explains that
contract; changes to the Python specification, dataclasses, builder, reader, and
this document must be kept in sync.

All new EmixIndex artifacts use `public.schema_version = 3` and
`public.source_dataset_version = "V3.0"`. Reader support for schema v2 is
legacy read-only compatibility; v2 is not emitted by current builders and does
not satisfy segment-wise training requirements.

## Root structure

An EmixIndex JSON document has exactly two root fields:

- `public`: index identity, schema and source declarations, and build metadata.
- `content`: the ordered array of cross-point groups.

### Public record

`public` contains:

- `id`: nonempty index identifier.
- `schema_version`: integer `3`.
- `created_at`: nonempty timestamp string.
- `source_dataset_version`: `V3.0`.
- `source_datasets`: one or more source declarations. Each declares
  `repo_id`, exact `root`, source `hash` (string or null), and `media`.
- `metadata`: build summary, configuration snapshot, source fingerprint, and
  `segment_annotations` summary or null.

A source `media` record declares nonempty camera keys, per-frame alignment,
video path pattern, and storage format. The fingerprint includes `meta/`,
`data/`, `videos/`, and `annotations/`.

When segments are supplied, `metadata.segment_annotations` records the sidecar
pattern, fields `segment_id` and `segment_object`, and the number of unique
(id, object) pairs. The build configuration records
`segment_annotations_enabled: true`. With no segment arrays, the summary is
null and the configuration records false.

### Cross-point group

Each `content.cross_point_groups[]` entry contains:

- `group_id`: nonempty stable identifier.
- `member_count`: positive integer equal to the number of members.
- `members`: one or more source-frame references.
- `state`: `state` (`training` or `aborted`), `abort_reasons`, and the
  serialized derived booleans `training` and `aborted`.
- `group_metrics` and `build_trace`: objects with build diagnostics.

Each member requires `source_dataset_repo_id`, nonnegative
`source_episode_index`, and nonnegative `source_frame_index`. It may include
an optional `segment` object:

```json
{"id": "step_01_pick", "object": "block1"}
```

The segment object contains exactly two nonempty strings:

- `id`: task-specific stable phase identity. Semantically corresponding phases
  use the same ID across episodes of that task.
- `object`: canonical manipulated-object name present in that episode's
  instance-sidecar object table.

`background`, `scene_*`, and the removed historical name `patch_target`
are invalid segment objects. An unlabeled frame omits `segment` entirely.

## Instance and segment sidecars

For source episode N, the builder reads
`annotations/episode_N.instances.h5`. The sidecar contains a JSON `objects` attribute with unique names and a `labels` mask array. Manipulated task objects use canonical task-role names; background and scene objects may use reserved background/scene names, but the removed patch_target alias is forbidden. Segment-aware
sidecars additionally contain UTF-8 `segment_id` and `segment_object`
vectors, each exactly one value per converted LeRobot frame. Empty strings in
both vectors mean intentionally unlabeled. A partial pair or partially missing
episode coverage is invalid.

The segment annotation input supplied to conversion is
`--segment-annotations-root`; it contains `episodeN.json` files with one
entry per converted source frame. Each entry is null or has `id` and
`object`. The converter applies the same frame selection and transforms to
masks and segments. Task-specific phase boundaries and object identities come
from reviewed task recipes; gripper action endpoints may locate a boundary but
do not infer the manipulated object.

The formal Patch batch's v3 preparation is kept in a separate dataset root so
the v2 source and its indexes remain reproducible. Its segment object names
must resolve in the corresponding instance mask table before indexing.

## Reader behavior

A v3 EmixIndex may omit segment labels when built for non-segment workflows.
With `segment_wise.enable=true`, training requires v3, instance sidecars, and
member labels. A group is eligible only when both members have equal
`segment.id` and `segment.object`; augmentation uses that canonical object's
mask. Groups with unlabeled members are skipped, and training fails if no
eligible matching groups remain or the object mask cannot be resolved.

After editing sidecars, rebuild the index: its source fingerprint binds it to
the exact annotation files.
