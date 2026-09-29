"""Runtime registry helpers for Embodied Mixup policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmixPolicyRegistration:
    """Describe one Emix policy registration pair."""

    policy_type: str
    config_import: str
    config_name: str
    model_import: str
    model_name: str
    processor_import: str
    processor_name: str


EMIX_POLICY_REGISTRATIONS = {
    "emix_act": EmixPolicyRegistration(
        policy_type="emix_act",
        config_import="embodied_mixup.policies.ACT.cfg_emix_act",
        config_name="EmixACTConfig",
        model_import="embodied_mixup.policies.ACT.emix_act",
        model_name="EmixACTPolicy",
        processor_import="lerobot.policies.act.processor_act",
        processor_name="make_act_pre_post_processors",
    ),
    "emix_diffusion": EmixPolicyRegistration(
        policy_type="emix_diffusion",
        config_import="embodied_mixup.policies.DP.cfg_emix_diffusion",
        config_name="EmixDiffusionConfig",
        model_import="embodied_mixup.policies.DP.emix_diffusion",
        model_name="EmixDiffusionPolicy",
        processor_import="lerobot.policies.diffusion.processor_diffusion",
        processor_name="make_diffusion_pre_post_processors",
    ),
    "emix_smolvla": EmixPolicyRegistration(
        policy_type="emix_smolvla",
        config_import="embodied_mixup.policies.SmolVLA.cfg_emix_smolvla",
        config_name="EmixSmolVLAConfig",
        model_import="embodied_mixup.policies.SmolVLA.emix_smolvla",
        model_name="EmixSmolVLAPolicy",
        processor_import="lerobot.policies.smolvla.processor_smolvla",
        processor_name="make_smolvla_pre_post_processors",
    ),
}


def is_emix_policy_type(policy_type: str) -> bool:
    """Return True when the policy type is an Emix runtime policy."""

    return policy_type in EMIX_POLICY_REGISTRATIONS
