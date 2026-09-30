"""Canonical EmixIndex schema.

@input: JSON-like dictionaries that describe source datasets and cross-point groups.
@output: Validated EmixIndex dataclasses with deterministic serialization support.
@scenario: Persist the stable index artifact used for direct EmixIndex training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EMIX_INDEX_SCHEMA_VERSION = 2
EMIX_DATASET_VERSION = "V3.0"


@dataclass
class EmixSourceMediaContract:
    """Media contract for one source dataset.

    @input: camera_keys [list[str]], frame_alignment [str], video_path_pattern [str],
        storage_format [str].
    @output: validated media contract.
    @scenario: Capture frame/media alignment assumptions for source dataset replay.
    """

    camera_keys: list[str]
    frame_alignment: str
    video_path_pattern: str
    storage_format: str

    def __post_init__(self) -> None:
        self.camera_keys = [str(v) for v in self.camera_keys]
        self.frame_alignment = str(self.frame_alignment)
        self.video_path_pattern = str(self.video_path_pattern)
        self.storage_format = str(self.storage_format)
        if not self.camera_keys:
            raise ValueError("camera_keys must not be empty.")
        if self.frame_alignment != "per_frame":
            raise ValueError(f"Unsupported frame_alignment: {self.frame_alignment}")
        if not self.video_path_pattern:
            raise ValueError("video_path_pattern must not be empty.")
        if not self.storage_format:
            raise ValueError("storage_format must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmixSourceMediaContract":
        return cls(**payload)


@dataclass
class EmixSourceDatasetDeclaration:
    """Source dataset declaration embedded in EmixIndex.public.

    @input: repo_id [str], root [str], hash [str|None] field-level identity
        summary ("files=N;bytes=M"; legacy indexes hold a content hash),
        media [EmixSourceMediaContract].
    @output: normalized source dataset declaration.
    @scenario: Record the source LeRobotDataset contract used for indexing.
        The ``hash`` field name is kept for schema compatibility with
        existing indexes; per user decision 2026-09-30 no content hashing is
        used, the value is a field-level summary string.
    """

    repo_id: str
    root: str
    hash: str | None
    media: EmixSourceMediaContract

    def __post_init__(self) -> None:
        self.repo_id = str(self.repo_id)
        self.root = str(self.root)
        self.hash = None if self.hash is None else str(self.hash)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["media"] = self.media.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmixSourceDatasetDeclaration":
        media = EmixSourceMediaContract.from_dict(payload["media"])
        return cls(
            repo_id=payload["repo_id"],
            root=payload["root"],
            hash=payload.get("hash"),
            media=media,
        )


@dataclass
class EmixGroupState:
    """Group state stored in EmixIndex.

    @input: state [str], abort_reasons [list[str]].
    @output: validated group state.
    @scenario: Track whether a group is trainable or should be skipped.
    """

    state: str = "training"
    abort_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = str(self.state)
        self.abort_reasons = [str(v) for v in self.abort_reasons]
        if self.state not in {"training", "aborted"}:
            raise ValueError(f"Unsupported group state: {self.state}")

    @property
    def training(self) -> bool:
        return self.state == "training" and not self.abort_reasons

    @property
    def aborted(self) -> bool:
        return self.state == "aborted"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training"] = self.training
        payload["aborted"] = self.aborted
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmixGroupState":
        return cls(
            state=payload.get("state", "training"),
            abort_reasons=list(payload.get("abort_reasons", [])),
        )


@dataclass
class EmixGroupMemberRef:
    """Reference to one source frame member.

    @input: source_dataset_repo_id [str], source_episode_index [int],
        source_frame_index [int].
    @output: normalized member reference.
    @scenario: Point from a cross-point group back to the source dataset.
    """

    source_dataset_repo_id: str
    source_episode_index: int
    source_frame_index: int

    def __post_init__(self) -> None:
        self.source_dataset_repo_id = str(self.source_dataset_repo_id)
        self.source_episode_index = int(self.source_episode_index)
        self.source_frame_index = int(self.source_frame_index)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmixGroupMemberRef":
        return cls(**payload)


@dataclass
class EmixCrossPointGroup:
    """One cross-point group in the canonical index.

    @input: group_id [str], members [list[EmixGroupMemberRef]], state [EmixGroupState].
    @output: validated group object.
    @scenario: Keep N-member cross-point references for direct training loading.
    """

    group_id: str
    member_count: int
    members: list[EmixGroupMemberRef]
    state: EmixGroupState
    group_metrics: dict[str, Any] = field(default_factory=dict)
    build_trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.group_id = str(self.group_id)
        self.member_count = int(self.member_count)
        self.group_metrics = dict(self.group_metrics)
        self.build_trace = dict(self.build_trace)
        if self.member_count != len(self.members):
            raise ValueError("member_count must match len(members).")
        if self.member_count <= 0:
            raise ValueError("member_count must be positive.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["members"] = [member.to_dict() for member in self.members]
        payload["state"] = self.state.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmixCrossPointGroup":
        return cls(
            group_id=payload["group_id"],
            member_count=payload["member_count"],
            members=[EmixGroupMemberRef.from_dict(v) for v in payload.get("members", [])],
            state=EmixGroupState.from_dict(payload["state"]),
            group_metrics=dict(payload.get("group_metrics", {})),
            build_trace=dict(payload.get("build_trace", {})),
        )


@dataclass
class EmixIndexPublic:
    """Public EmixIndex metadata.

    @input: id [str], schema_version [int], created_at [str], source_datasets [list],
        source_dataset_version [str], metadata [dict].
    @output: canonical public section.
    @scenario: Store source declarations and build parameters.
    """

    id: str
    schema_version: int
    created_at: str
    source_datasets: list[EmixSourceDatasetDeclaration]
    source_dataset_version: str = EMIX_DATASET_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.schema_version = int(self.schema_version)
        self.created_at = str(self.created_at)
        self.source_dataset_version = str(self.source_dataset_version)
        self.metadata = dict(self.metadata)
        if self.schema_version != EMIX_INDEX_SCHEMA_VERSION:
            raise ValueError(f"Unsupported EmixIndex schema_version: {self.schema_version}")
        if not self.source_datasets:
            raise ValueError("source_datasets must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_datasets"] = [dataset.to_dict() for dataset in self.source_datasets]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmixIndexPublic":
        return cls(
            id=payload["id"],
            schema_version=payload["schema_version"],
            created_at=payload["created_at"],
            source_datasets=[
                EmixSourceDatasetDeclaration.from_dict(v)
                for v in payload.get("source_datasets", [])
            ],
            source_dataset_version=payload.get("source_dataset_version", EMIX_DATASET_VERSION),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class EmixIndexContent:
    """Canonical EmixIndex content section."""

    cross_point_groups: list[EmixCrossPointGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cross_point_groups": [group.to_dict() for group in self.cross_point_groups],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmixIndexContent":
        return cls(
            cross_point_groups=[
                EmixCrossPointGroup.from_dict(v) for v in payload.get("cross_point_groups", [])
            ]
        )


@dataclass
class EmixIndex:
    """Canonical EmixIndex root object."""

    public: EmixIndexPublic
    content: EmixIndexContent

    def to_dict(self) -> dict[str, Any]:
        return {
            "public": self.public.to_dict(),
            "content": self.content.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmixIndex":
        return cls(
            public=EmixIndexPublic.from_dict(payload["public"]),
            content=EmixIndexContent.from_dict(payload["content"]),
        )
