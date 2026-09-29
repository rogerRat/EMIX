"""Configuration for candidate generation."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_OUTPUTS_ROOT = (PROJECT_ROOT / "outputs").resolve()


def resolve_project_output_path(path: str | Path, *, field: str) -> Path:
    """Resolve an artifact path and require it to live under ``outputs/``.

    Per-batch experiment outputs live under ``outputs/run_<label>/``; the
    guard only demands the outputs/ root, not a specific batch folder.
    """

    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_OUTPUTS_ROOT)
    except ValueError as error:
        raise ValueError(f"{field} must be under the project outputs/ directory: {path}") from error
    return resolved


@dataclass
class EmbodiedMixupConfig:
    """Configuration for Embodied Mixup candidate generation.

    All distance-based parameters are in SI units unless otherwise noted.
    window_len and raw_candidate_feature_keys have been removed; they were
    only used for the removed materialized export path.
    """

    xyz_epsilon: float = 0.02
    # Orientation similarity threshold in radians as the norm across the three
    # Euler angles.  0.26 rad (~15 deg) replaces the original 0.01 rad
    # (~0.57 deg): after the placement-envelope widening the wrist orientation
    # varies per episode, and the 0.01 gate rejected 83-98% of the position
    # matches and left only home/approach waypoints eligible for pairing.
    rpy_epsilon: float = 0.26

    future_horizon: int = 8
    future_angle_thresh: float = 0.5235987755982988
    min_future_disp: float = 0.002
    gripper_action_filter_enabled: bool = True

    cross_episode_only: bool = True
    gripper_consistency: bool = True

    # Trim each episode's tail out of pairing.  The final home-return frames
    # sit at the same pose as the episode start, so untrimmed tails pair
    # "start task" with "go home" and teach a spurious do-nothing mode.  The
    # trim is specified in seconds and converted to frames at the dataset fps
    # (0.1 s at 17 fps = the last 2 frames).
    exclude_tail_seconds: float = 0.1
    dataset_fps: float = 17.0

    training_candidates_total_budget: int | None = None
    training_candidates_cap_per_cluster: int | None = None
    training_candidates_include_noise: bool = False
    training_candidates_seed: int = 0

    hdbscan_min_cluster_size_ratio: float = 0.01
    hdbscan_min_samples: int = 3
    hdbscan_cluster_selection_epsilon: float = 0.05
    hdbscan_alpha: float = 1.0
    hdbscan_cluster_selection_method: str = "eom"

    # Hard cap on candidate pairs entering clustering and the index.  Dense
    # 500-episode collections produce tens of millions of valid pairs at the
    # default gates; the marginal EMIX value of pairs beyond a few hundred
    # thousand is nil while clustering/serialization cost explodes.  The cap
    # scales with collection size: max_candidate_pairs_per_episode pairs per
    # episode (1000/ep keeps a 500-episode batch at 500k pairs — validated to
    # build in minutes on CPU — and is a no-op for the 200-episode batches
    # whose natural pair count is ~60k).  An explicit max_candidate_pairs
    # overrides the ratio; the effective cap is the smaller of the two when
    # both are set.  Both None disables capping (legacy behaviour).
    max_candidate_pairs: int | None = None
    max_candidate_pairs_per_episode: float | None = 1000.0

    # Drop candidate pairs whose midpoint xyz sits within this radius of the
    # arm's shared start (home) point.  All cen_arm episodes launch from one
    # fixed EEF pose, so pairs clustered there supervise "go home / wait" on
    # mixed images instead of task-relevant grasp/place behaviour.  None
    # disables the filter (legacy behaviour); 0.02 m matches the pairing
    # xyz_epsilon and covers the home cluster measured on the 500ep batches.
    exclude_start_radius: float | None = None

    # Drop candidate pairs whose two frames are too far apart in time.  The
    # span is measured as the absolute difference of the two frames'
    # within-episode time positions, capped at max_pair_time_ratio x the mean
    # episode length of the collection.  Pairs spanning most of a trajectory
    # mix e.g. an episode's start with another's finish at coincidental
    # spatial overlaps (0904 finding 2: phase-mismatched pseudo-multimodal
    # pairs).  None disables the filter; 0.6 = 60% of the mean length.
    max_pair_time_ratio: float | None = None

    action_delta_threshold: float = 0.05
    p_mixup: float = 0.5
    lambda_mix: float = 0.4

    index_cache_path: str | None = None

    # Click-to-inspect image pre-computation budget.
    # Total cap on pairs to pre-compute camera thumbnails for.
    click_image_total_budget: int = 100
    # Minimum number of image pairs pre-computed per cluster.  The total
    # budget is expanded when necessary to satisfy this floor.
    click_image_min_per_cluster: int = 10

    @property
    def xyz_thresh(self) -> float:
        return self.xyz_epsilon

    @property
    def rot_thresh(self) -> float:
        return self.rpy_epsilon

    @property
    def exclude_tail_frames(self) -> int:
        """Episode-tail trim in frames, converted from seconds at ``dataset_fps``."""
        return int(round(self.exclude_tail_seconds * self.dataset_fps))

    def __post_init__(self) -> None:
        if self.xyz_epsilon <= 0:
            raise ValueError(f"xyz_epsilon must be positive, got {self.xyz_epsilon}")
        if self.rpy_epsilon <= 0:
            raise ValueError(f"rpy_epsilon must be positive, got {self.rpy_epsilon}")
        if self.exclude_tail_seconds < 0:
            raise ValueError(
                f"exclude_tail_seconds must be non-negative, got {self.exclude_tail_seconds}"
            )
        if self.dataset_fps <= 0:
            raise ValueError(f"dataset_fps must be positive, got {self.dataset_fps}")
        if self.future_horizon <= 0:
            raise ValueError(f"future_horizon must be positive, got {self.future_horizon}")
        if self.future_angle_thresh < 0:
            raise ValueError(
                f"future_angle_thresh must be non-negative, got {self.future_angle_thresh}"
            )
        if self.min_future_disp < 0:
            raise ValueError(f"min_future_disp must be non-negative, got {self.min_future_disp}")
        if self.action_delta_threshold < 0:
            raise ValueError(
                "action_delta_threshold must be non-negative, got "
                f"{self.action_delta_threshold}"
            )
        if self.max_candidate_pairs is not None and self.max_candidate_pairs <= 0:
            raise ValueError(
                "max_candidate_pairs must be positive when provided, got "
                f"{self.max_candidate_pairs}"
            )
        if (
            self.max_candidate_pairs_per_episode is not None
            and self.max_candidate_pairs_per_episode <= 0
        ):
            raise ValueError(
                "max_candidate_pairs_per_episode must be positive when provided, got "
                f"{self.max_candidate_pairs_per_episode}"
            )
        if self.exclude_start_radius is not None and self.exclude_start_radius <= 0:
            raise ValueError(
                "exclude_start_radius must be positive when provided, got "
                f"{self.exclude_start_radius}"
            )
        if self.max_pair_time_ratio is not None and self.max_pair_time_ratio <= 0:
            raise ValueError(
                "max_pair_time_ratio must be positive when provided, got "
                f"{self.max_pair_time_ratio}"
            )
        if self.hdbscan_min_cluster_size_ratio <= 0 or self.hdbscan_min_cluster_size_ratio > 1.0:
            raise ValueError(
                "hdbscan_min_cluster_size_ratio must be in (0, 1], got "
                f"{self.hdbscan_min_cluster_size_ratio}"
            )
        if self.hdbscan_min_samples < 1:
            raise ValueError(
                f"hdbscan_min_samples must be at least 1, got {self.hdbscan_min_samples}"
            )
        if self.hdbscan_cluster_selection_epsilon < 0:
            raise ValueError(
                "hdbscan_cluster_selection_epsilon must be non-negative, got "
                f"{self.hdbscan_cluster_selection_epsilon}"
            )
        if self.hdbscan_alpha <= 0:
            raise ValueError(f"hdbscan_alpha must be positive, got {self.hdbscan_alpha}")
        if self.hdbscan_cluster_selection_method not in ("eom", "leaf"):
            raise ValueError(
                "hdbscan_cluster_selection_method must be 'eom' or 'leaf', got "
                f"{self.hdbscan_cluster_selection_method}"
            )
        if not 0.0 <= self.p_mixup <= 1.0:
            raise ValueError(f"p_mixup must be in [0, 1], got {self.p_mixup}")
        if self.lambda_mix <= 0:
            raise ValueError(f"lambda_mix must be positive, got {self.lambda_mix}")
        if self.training_candidates_total_budget is not None and self.training_candidates_total_budget <= 0:
            raise ValueError(
                "training_candidates_total_budget must be positive when provided, got "
                f"{self.training_candidates_total_budget}"
            )
        if (
            self.training_candidates_cap_per_cluster is not None
            and self.training_candidates_cap_per_cluster <= 0
        ):
            raise ValueError(
                "training_candidates_cap_per_cluster must be positive when provided, got "
                f"{self.training_candidates_cap_per_cluster}"
            )
        if self.click_image_total_budget < 1:
            raise ValueError(
                f"click_image_total_budget must be at least 1, got {self.click_image_total_budget}"
            )
        if self.click_image_min_per_cluster < 0:
            raise ValueError(
                "click_image_min_per_cluster must be non-negative, got "
                f"{self.click_image_min_per_cluster}"
            )


DEFAULT_EMBODIED_MIXUP_CONFIG = EmbodiedMixupConfig()
