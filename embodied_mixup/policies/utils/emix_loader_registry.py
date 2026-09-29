"""Registry for policy-wise EmixIndex loaders.

@input: policy type strings, source dataset root/repo_id, EmixIndex paths.
@output: policy-specific dataset instances backed by source LeRobotDataset.
@scenario: Keep loader selection explicit and centralized.
"""

from __future__ import annotations

from dataclasses import dataclass

from .batch import PolicyTemporalSpec
from .emix_loader_registry_utils import EmixLoaderSettings, EmixPolicyDataset


@dataclass(frozen=True)
class EmixLoaderRegistration:
    """Policy-to-loader registration."""

    policy_type: str
    include_task: bool


EMIX_LOADER_REGISTRY: dict[str, EmixLoaderRegistration] = {
    "emix_act": EmixLoaderRegistration(policy_type="emix_act", include_task=False),
    "emix_diffusion": EmixLoaderRegistration(policy_type="emix_diffusion", include_task=False),
    "emix_smolvla": EmixLoaderRegistration(policy_type="emix_smolvla", include_task=True),
}


def make_emix_loader(
    policy_type: str,
    source_root: str,
    source_repo_id: str,
    emix_index_path: str,
    policy_temporal_spec: PolicyTemporalSpec,
    loader_cfg,
) -> EmixPolicyDataset:
    """Instantiate the policy-specific EMIX dataset loader.

    @input: policy_type [str], source_root [str], source_repo_id [str],
        emix_index_path [str], policy_temporal_spec [PolicyTemporalSpec],
        loader_cfg [Any] object with optional mixup fields.
    @output: EmixPolicyDataset backed by source LeRobotDataset.
    @scenario: Centralised registration lookup; all callers go through here.
    """

    registration = EMIX_LOADER_REGISTRY.get(policy_type)
    if registration is None:
        raise ValueError(f"Unsupported Emix policy type: {policy_type}")

    settings = EmixLoaderSettings(
        policy_type=policy_type,
        mixup_alpha_mode=str(getattr(loader_cfg, "mixup_alpha_mode", "beta")),
        mixup_alpha_value=getattr(loader_cfg, "mixup_alpha_value", None),
        mixup_beta=float(getattr(loader_cfg, "mixup_beta", 1.6)),
        mixup_visual_keys=getattr(loader_cfg, "mixup_visual_keys", None),
        mixup_state_mode=str(getattr(loader_cfg, "mixup_state_mode", "anchor")),
        seed=int(getattr(loader_cfg, "seed", 0)),
        include_task=registration.include_task,
    )
    return EmixPolicyDataset(
        source_root=source_root,
        source_repo_id=source_repo_id,
        emix_index_path=emix_index_path,
        policy_temporal_spec=policy_temporal_spec,
        loader_cfg=settings,
    )
