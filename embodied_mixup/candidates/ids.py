"""Stable identifier helpers for Emix artifacts."""

import hashlib
import json


def make_emix_sample_id(
    dataset_repo_id: str,
    anchor_idx: int,
    neighbor_idx: int,
    future_horizon: int,
    window_len: int,
) -> str:
    """Build a deterministic Emix sample id from source frames and config."""
    raw = json.dumps(
        {
            "dataset_repo_id": dataset_repo_id,
            "anchor_idx": int(anchor_idx),
            "neighbor_idx": int(neighbor_idx),
            "future_horizon": int(future_horizon),
            "window_len": int(window_len),
        },
        sort_keys=True,
    )
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"emix_{digest}"


def make_emix_group_id(
    source_repo_id: str,
    members,
    salt: int = 0,
) -> str:
    """Build a deterministic Emix group id from source members."""
    raw = json.dumps(
        {
            "source_repo_id": str(source_repo_id),
            "members": [
                {
                    "source_dataset_repo_id": str(member.source_dataset_repo_id),
                    "source_episode_index": int(member.source_episode_index),
                    "source_frame_index": int(member.source_frame_index),
                }
                for member in members
            ],
            "salt": int(salt),
        },
        sort_keys=True,
    )
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"emix_group_{digest}"


def make_emix_index_id(
    source_repo_id: str,
    dataset_length: int,
    source_dataset_summary: str,
    config_snapshot: dict,
) -> str:
    """Build a deterministic EmixIndex identifier.

    @input: source_repo_id [str], dataset_length [int],
        source_dataset_summary [str] field-level identity of the source
        dataset (file/byte counts per user decision 2026-09-30),
        config_snapshot [dict] candidate generation config without window_len.
    @output: str id that changes when source data or config changes.
    @scenario: Bind the index id to the source dataset identity and config.
    """
    raw = json.dumps(
        {
            "source_repo_id": str(source_repo_id),
            "dataset_length": int(dataset_length),
            "source_dataset_summary": str(source_dataset_summary),
            "config_snapshot": config_snapshot,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"emix_index_{digest}"
