"""Normative Python specification for the serialized EmixIndex v3 structure."""
from __future__ import annotations

from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired

EMIX_INDEX_SCHEMA_VERSION = 3
EMIX_DATASET_VERSION = "V3.0"
LEGACY_READER_SCHEMA_VERSIONS = frozenset({2})
SUPPORTED_EMIX_INDEX_SCHEMA_VERSIONS = frozenset(
    {EMIX_INDEX_SCHEMA_VERSION, *LEGACY_READER_SCHEMA_VERSIONS}
)


class SegmentRefV3(TypedDict):
    id: str
    object: str


class MemberRefV3(TypedDict):
    source_dataset_repo_id: str
    source_episode_index: int
    source_frame_index: int
    segment: NotRequired[SegmentRefV3]


class GroupStateV3(TypedDict):
    state: str
    abort_reasons: list[str]
    training: bool
    aborted: bool


class CrossPointGroupV3(TypedDict):
    group_id: str
    member_count: int
    members: list[MemberRefV3]
    state: GroupStateV3
    group_metrics: dict[str, Any]
    build_trace: dict[str, Any]


class SourceMediaV3(TypedDict):
    camera_keys: list[str]
    frame_alignment: str
    video_path_pattern: str
    storage_format: str


class SourceDatasetV3(TypedDict):
    repo_id: str
    root: str
    hash: str | None
    media: SourceMediaV3


class EmixIndexPublicV3(TypedDict):
    id: str
    schema_version: Literal[3]
    created_at: str
    source_datasets: list[SourceDatasetV3]
    source_dataset_version: str
    metadata: dict[str, Any]


class EmixIndexContentV3(TypedDict):
    cross_point_groups: list[CrossPointGroupV3]


class EmixIndexV3(TypedDict):
    public: EmixIndexPublicV3
    content: EmixIndexContentV3


# Standard JSON Schema representation for external tooling.
EMIX_INDEX_V3_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EmixIndex v3",
    "type": "object",
    "required": ["public", "content"],
    "additionalProperties": False,
    "properties": {
        "public": {
            "type": "object",
            "required": ["id", "schema_version", "created_at", "source_datasets",
                         "source_dataset_version", "metadata"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "schema_version": {"const": 3},
                "created_at": {"type": "string", "minLength": 1},
                "source_dataset_version": {"type": "string", "minLength": 1},
                "source_datasets": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["repo_id", "root", "hash", "media"],
                        "properties": {
                            "repo_id": {"type": "string", "minLength": 1},
                            "root": {"type": "string", "minLength": 1},
                            "hash": {"type": ["string", "null"]},
                            "media": {
                                "type": "object",
                                "required": ["camera_keys", "frame_alignment",
                                             "video_path_pattern", "storage_format"],
                                "properties": {
                                    "camera_keys": {"type": "array", "minItems": 1,
                                                    "items": {"type": "string"}},
                                    "frame_alignment": {"const": "per_frame"},
                                    "video_path_pattern": {"type": "string", "minLength": 1},
                                    "storage_format": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                },
                "metadata": {
                    "type": "object",
                    "required": ["segment_annotations", "config"],
                    "properties": {
                        "segment_annotations": {
                            "oneOf": [
                                {"type": "null"},
                                {
                                    "type": "object",
                                    "required": ["storage", "fields", "unique_segments"],
                                    "properties": {
                                        "storage": {"type": "string", "minLength": 1},
                                        "fields": {
                                            "const": ["segment_id", "segment_object"]
                                        },
                                        "unique_segments": {"type": "integer", "minimum": 0},
                                    },
                                },
                            ]
                        },
                        "config": {
                            "type": "object",
                            "required": ["segment_annotations_enabled"],
                            "properties": {
                                "segment_annotations_enabled": {"type": "boolean"}
                            },
                        },
                    },
                },
            },
        },
        "content": {
            "type": "object",
            "required": ["cross_point_groups"],
            "properties": {
                "cross_point_groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["group_id", "member_count", "members", "state",
                                     "group_metrics", "build_trace"],
                        "properties": {
                            "group_id": {"type": "string", "minLength": 1},
                            "member_count": {"type": "integer", "minimum": 1},
                            "members": {
                                "type": "array", "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "required": ["source_dataset_repo_id",
                                                 "source_episode_index",
                                                 "source_frame_index"],
                                    "properties": {
                                        "source_dataset_repo_id": {"type": "string", "minLength": 1},
                                        "source_episode_index": {"type": "integer", "minimum": 0},
                                        "source_frame_index": {"type": "integer", "minimum": 0},
                                        "segment": {
                                            "type": "object",
                                            "required": ["id", "object"],
                                            "properties": {
                                                "id": {"type": "string", "minLength": 1},
                                                "object": {
                                                    "type": "string", "minLength": 1,
                                                    "not": {"enum": ["background", "patch_target"]},
                                                    "pattern": "^(?!scene_).+",
                                                },
                                            },
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                            },
                            "state": {
                                "type": "object",
                                "required": ["state", "abort_reasons", "training", "aborted"],
                                "properties": {
                                    "state": {"enum": ["training", "aborted"]},
                                    "abort_reasons": {"type": "array", "items": {"type": "string"}},
                                    "training": {"type": "boolean"},
                                    "aborted": {"type": "boolean"},
                                },
                            },
                            "group_metrics": {"type": "object"},
                            "build_trace": {"type": "object"},
                        },
                    },
                },
            },
        },
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid EmixIndex v3: {message}")


def validate_emix_index_v3_public(public: dict[str, Any]) -> None:
    """Validate the public block before a streaming v3 build writes it."""
    _require(isinstance(public, dict), "public must be an object")
    _require(public.get("schema_version") == 3, "public.schema_version must equal 3")
    _require(public.get("source_dataset_version") == EMIX_DATASET_VERSION,
             "public.source_dataset_version must equal V3.0")
    _require(isinstance(public.get("id"), str) and bool(public["id"]),
             "public.id must be a nonempty string")
    _require(isinstance(public.get("created_at"), str) and bool(public["created_at"]),
             "public.created_at must be a nonempty string")
    sources = public.get("source_datasets")
    _require(isinstance(sources, list) and bool(sources),
             "public.source_datasets must be a nonempty array")
    for source in sources:
        _require(isinstance(source, dict), "source declaration must be an object")
        _require(all(isinstance(source.get(key), str) and source[key]
                     for key in ("repo_id", "root")),
                 "source repo_id/root must be nonempty strings")
        _require(source.get("hash") is None or isinstance(source.get("hash"), str),
                 "source hash must be a string or null")
        media = source.get("media")
        _require(isinstance(media, dict), "source media must be an object")
        _require(media.get("frame_alignment") == "per_frame",
                 "source frame_alignment must be per_frame")
        _require(bool(media.get("camera_keys")), "source camera_keys must not be empty")
    metadata = public.get("metadata")
    _require(isinstance(metadata, dict), "public.metadata must be an object")
    _require("segment_annotations" in metadata,
             "public.metadata must declare segment_annotations or null")
    _require(isinstance(metadata.get("config"), dict),
             "public.metadata.config must be an object")
    _require("segment_annotations_enabled" in metadata["config"],
             "public.metadata.config must declare segment_annotations_enabled")


def validate_cross_point_group_v3(group: dict[str, Any]) -> None:
    """Validate one group, suitable for incremental streaming serialization."""
    _require(isinstance(group, dict), "cross-point group must be an object")
    members = group.get("members")
    _require(isinstance(members, list) and bool(members),
             "cross-point group members must be nonempty")
    _require(group.get("member_count") == len(members),
             "member_count must match members")
    _require(isinstance(group.get("group_id"), str) and bool(group["group_id"]),
             "group_id must be nonempty")
    state = group.get("state")
    _require(isinstance(state, dict)
             and state.get("state") in {"training", "aborted"}
             and isinstance(state.get("abort_reasons"), list)
             and isinstance(state.get("training"), bool)
             and isinstance(state.get("aborted"), bool),
             "state must follow the v3 group-state contract")
    _require(isinstance(group.get("group_metrics"), dict)
             and isinstance(group.get("build_trace"), dict),
             "group_metrics and build_trace must be objects")
    for member_index, member in enumerate(members):
        path = f"cross-point group members[{member_index}]"
        _require(isinstance(member, dict), f"{path} must be an object")
        _require(isinstance(member.get("source_dataset_repo_id"), str)
                 and bool(member["source_dataset_repo_id"]),
                 f"{path}.source_dataset_repo_id must be nonempty")
        for key in ("source_episode_index", "source_frame_index"):
            _require(type(member.get(key)) is int and member[key] >= 0,
                     f"{path}.{key} must be a nonnegative integer")
        if "segment" in member:
            segment = member["segment"]
            _require(isinstance(segment, dict) and set(segment) == {"id", "object"},
                     f"{path}.segment must contain exactly id and object")
            _require(all(isinstance(segment[key], str) and segment[key]
                         for key in ("id", "object")),
                     f"{path}.segment id/object must be nonempty strings")
            name = segment["object"]
            _require(name not in {"background", "patch_target"}
                     and not name.startswith("scene_"),
                     f"{path}.segment.object must be a canonical task object")


def validate_emix_index_v3(payload: dict[str, Any]) -> None:
    """Validate a complete v3 payload, including each optional segment label."""
    _require(isinstance(payload, dict) and set(payload) == {"public", "content"},
             "root must contain exactly public and content")
    validate_emix_index_v3_public(payload["public"])
    content = payload["content"]
    _require(isinstance(content, dict), "content must be an object")
    groups = content.get("cross_point_groups")
    _require(isinstance(groups, list), "cross_point_groups must be an array")
    for group in groups:
        validate_cross_point_group_v3(group)
