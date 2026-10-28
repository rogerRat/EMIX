# EmixIndex Specification

## Versions

- Schema v2: existing group and source metadata, with no segment contract.
- Schema v3: same root shape, with optional per-member operation segment metadata.
- Readers accept v2 and v3. segment_wise.enable=true requires v3 and matching labels. New builders emit v3.
- v2 fingerprints retain their recorded included_roots; new fingerprints include meta, data, videos, and annotations.

## Member reference

Each content.cross_point_groups[].members[] record has the existing source dataset, episode, and frame fields, plus optional segment: {id: string, object: string}.

segment is omitted when the frame has no reviewed segment label. When present, it must contain exactly two nonempty strings:

- id: task-specific, stable phase ID assigned by reviewed annotation.
- object: canonical manipulated-object name in the same episode's instance annotation object table.

object may not be background, a scene_* label, or the removed historical name patch_target. A segment object must exist in the sidecar object table. Segment IDs are task-defined and must identify semantically corresponding phases across episodes when those episodes are intended to mix.

## Annotation sidecar input to the builder

For source episode N, the builder reads annotations/episode_N.instances.h5.

The HDF5 sidecar must contain:

- schema_version >= 2
- objects attribute: JSON array of canonical object names
- segment_id: one UTF-8 string per converted LeRobot frame
- segment_object: one UTF-8 string per converted LeRobot frame

Both segment arrays must have exactly the episode frame count. Empty strings in both arrays mean that the frame is not eligible for segment-wise augmentation; partial labels are invalid. Segment metadata is copied to each emitted member by source frame index. If no episode has segment datasets, the builder emits v3 without member labels and records segment_annotations: null. Partial sidecar coverage across episodes is an error.

The conversion input is supplied with --segment-annotations-root. That directory contains episodeN.json for every source episode; each file is either a list or an object with a segments list. Each list entry is null or an object with id and object strings, and the list length must match the converted frame count (RoboTwin's final observation without an action is excluded). The converter writes these labels beside instance masks in the output LeRobot root.

The sidecar is created by the reviewed annotation conversion path. Human task-specific annotations determine phase boundaries and manipulated objects; generic gripper thresholding does not produce authoritative labels. Instance masks and segment labels undergo the same frame selection and spatial transforms.

## Fingerprint and metadata

New source fingerprints include annotations/ alongside meta/, data/, and videos/. Index public metadata records segment_annotations with the sidecar storage pattern, dataset names, and unique segment count. It is null when no segment arrays are present. metadata.config.segment_annotations_enabled records whether labels were supplied during build. The fingerprint binds the index to exact annotation contents.

## Training behavior

With segment_wise.enable=true, the loader:

1. Requires schema v3 and an instance annotation root.
2. Keeps a group only when both members have equal segment.id and segment.object.
3. Uses that canonical object mask for Patch or linear Mix.
4. Fails if no eligible matching groups remain or the object cannot be resolved.

Terminal and other intentionally unlabeled frames are omitted from segment-wise training. Rebuild the EmixIndex after changing annotations.
