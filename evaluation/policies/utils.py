from .base import PolicyConfig, BasePolicy
from .lerobot import LeRobotPolicyConfig, LeRobotPolicy


def make_policy_from_config(cfg: PolicyConfig) -> BasePolicy:
    if isinstance(cfg, LeRobotPolicyConfig):
        return LeRobotPolicy(cfg)
    # elif isinstance(cfg, YourNewPolicyConfig):
    #     return YourNewPolicy(cfg)
    else:
        raise ValueError(f"Unknown policy config type: {type(cfg)}")