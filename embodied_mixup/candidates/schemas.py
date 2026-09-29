"""Typed transient data structures for candidate generation."""

from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class CandidatePublic:
    """Public candidate fields."""

    candidate_id: str
    sample_id: str
    dataset_repo_id: str
    source_dataset_id: str
    source_dataset_root: str | None = None
    created_at: str | None = None
    schema_version: int = 2
    build_stage: str = "build"
    build_config: dict[str, Any] = field(default_factory=dict)
    cluster_id: int = -1
    selected_for_training: bool = False

    def __post_init__(self) -> None:
        self.candidate_id = str(self.candidate_id)
        self.sample_id = str(self.sample_id)
        self.dataset_repo_id = str(self.dataset_repo_id)
        self.source_dataset_id = str(self.source_dataset_id)
        if self.schema_version != 2:
            raise ValueError(f"Unsupported CandidatePublic schema_version: {self.schema_version}")


@dataclass
class CandidateParticipant:
    """One participant in a candidate relation."""

    role: str
    episode_index: int
    frame_index: int
    branch_frame_in_window: int
    pose: list[float]
    local_window: dict[str, Any]

    def __post_init__(self) -> None:
        self.role = str(self.role)
        self.episode_index = int(self.episode_index)
        self.frame_index = int(self.frame_index)
        self.branch_frame_in_window = int(self.branch_frame_in_window)
        self.pose = [float(v) for v in self.pose]


@dataclass
class CandidateRelation:
    """Relation fields for one candidate."""

    source_pair_index: int
    anchor_participant: CandidateParticipant
    neighbor_participant: CandidateParticipant
    pose_distance_xyz: float
    pose_distance_rot: float
    action_divergence: float
    future_angle_xyz: float = 0.0
    future_disp_a: float = 0.0
    future_disp_b: float = 0.0
    gripper_action_filter_flag: bool = False
    action_filter_valid_by_future_angle: bool = False
    action_filter_valid_by_current_action: bool = False
    action_filter_valid: bool = False
    cluster_id: int = -1
    cluster_probability: float | None = None
    midpoint_pose: list[float] | None = None
    aligned_index_pairs: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_pair_index = int(self.source_pair_index)
        self.pose_distance_xyz = float(self.pose_distance_xyz)
        self.pose_distance_rot = float(self.pose_distance_rot)
        self.action_divergence = float(self.action_divergence)
        self.future_angle_xyz = float(self.future_angle_xyz)
        self.future_disp_a = float(self.future_disp_a)
        self.future_disp_b = float(self.future_disp_b)
        self.gripper_action_filter_flag = bool(self.gripper_action_filter_flag)
        self.action_filter_valid_by_future_angle = bool(self.action_filter_valid_by_future_angle)
        self.action_filter_valid_by_current_action = bool(
            self.action_filter_valid_by_current_action
        )
        self.action_filter_valid = bool(self.action_filter_valid)
        self.cluster_id = int(self.cluster_id)
        if self.midpoint_pose is not None:
            self.midpoint_pose = [float(v) for v in self.midpoint_pose]


@dataclass
class CandidateMediaAsset:
    """Explicit media bookkeeping for one candidate."""

    family: str
    source_uri: str | None = None
    source_dataset_root: str | None = None
    source_episode_index: int | None = None
    source_frame_indices: list[int] = field(default_factory=list)
    source_camera_key: str | None = None
    export_uri: str | None = None
    export_format: str | None = None
    export_checksum: str | None = None
    is_present: bool = False
    is_required: bool = False
    is_self_contained: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.family = str(self.family)
        self.source_frame_indices = [int(v) for v in self.source_frame_indices]
        if self.source_episode_index is not None:
            self.source_episode_index = int(self.source_episode_index)


@dataclass
class CandidateMedia:
    """Media family container."""

    assets: list[CandidateMediaAsset] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateRecord:
    """Unified candidate record with peer public, relation, and media families."""

    public: CandidatePublic
    relation: CandidateRelation
    media: CandidateMedia = field(default_factory=CandidateMedia)

    @property
    def candidate_id(self) -> str:
        return self.public.candidate_id

    @property
    def sample_id(self) -> str:
        return self.public.sample_id

    @property
    def cluster_id(self) -> int:
        return self.public.cluster_id

    @property
    def anchor_participant(self) -> CandidateParticipant:
        return self.relation.anchor_participant

    @property
    def neighbor_participant(self) -> CandidateParticipant:
        return self.relation.neighbor_participant

    @property
    def anchor_idx(self) -> int:
        return self.relation.anchor_participant.frame_index

    @property
    def neighbor_idx(self) -> int:
        return self.relation.neighbor_participant.frame_index

    @property
    def anchor_episode(self) -> int:
        return self.relation.anchor_participant.episode_index

    @property
    def neighbor_episode(self) -> int:
        return self.relation.neighbor_participant.episode_index

    @property
    def pose_distance_xyz(self) -> float:
        return self.relation.pose_distance_xyz

    @property
    def pose_distance_rpy(self) -> float:
        return self.relation.pose_distance_rot

    @property
    def action_divergence(self) -> float:
        return self.relation.action_divergence

    @property
    def anchor_pose(self) -> list[float]:
        return self.relation.anchor_participant.pose

    @property
    def neighbor_pose(self) -> list[float]:
        return self.relation.neighbor_participant.pose

    @property
    def source_dataset_id(self) -> str:
        return self.public.source_dataset_id

    @property
    def source_frame_a(self) -> int:
        return self.anchor_idx

    @property
    def source_frame_b(self) -> int:
        return self.neighbor_idx

    @property
    def branch_frame_a_in_window(self) -> int:
        return self.relation.anchor_participant.branch_frame_in_window

    @property
    def branch_frame_b_in_window(self) -> int:
        return self.relation.neighbor_participant.branch_frame_in_window

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "anchor_idx": self.anchor_idx,
                "neighbor_idx": self.neighbor_idx,
                "anchor_episode": self.anchor_episode,
                "neighbor_episode": self.neighbor_episode,
                "anchor_pose": list(self.anchor_pose),
                "neighbor_pose": list(self.neighbor_pose),
                "pose_distance_rpy": self.pose_distance_rpy,
                "source_frame_a": self.source_frame_a,
                "source_frame_b": self.source_frame_b,
                "branch_frame_a_in_window": self.branch_frame_a_in_window,
                "branch_frame_b_in_window": self.branch_frame_b_in_window,
                "source_dataset_id": self.source_dataset_id,
            }
        )
        return data

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        return data[key]

    def get(self, key: str, default: Any = None) -> Any:
        data = self.to_dict()
        return data.get(key, default)
