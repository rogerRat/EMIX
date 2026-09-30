"""EMIX YAML schema validation and policy config hydration.

@input: emix YAML payload dict and a mutable policy config object.
@output: None; raises ValueError on missing or illegal schema keys.
@scenario: Enforce that YAML-level emix config uses the v2 schema
    (index_path replaces files_root) and hydrate all EMIX mixup settings on
    the policy config used by both the loader and the policy loss.
"""

from __future__ import annotations

from typing import Any


def apply_emix_schema(policy_cfg: Any, emix_payload: dict[str, Any]) -> None:
    """Validate EMIX YAML and hydrate index, mixing, and loss settings.

    @input: policy_cfg [Any] mutable policy config dataclass,
        emix_payload [dict] the top-level emix: mapping from YAML.
    @output: None; raises ValueError on schema errors.
    @scenario: Reject legacy files_root key; require index_path; set
        policy_cfg.emix_index_path and the policy's mixup_* fields from the
        nested emix.mixup mapping.
    """

    if not emix_payload:
        return

    if "files_root" in emix_payload:
        raise ValueError(
            "emix.files_root is no longer supported. "
            "Use emix.index_path and dataset.root."
        )

    index_path = emix_payload.get("index_path")
    if index_path is None:
        raise ValueError(
            "emix.index_path is required for EMIX training. "
            "Set emix.index_path to the path of your emix_index.json file."
        )

    if not isinstance(index_path, str):
        raise ValueError(
            f"emix.index_path must be a string path, got {type(index_path).__name__}."
        )

    if not hasattr(policy_cfg, "emix_index_path"):
        raise ValueError(
            f"Policy config type '{getattr(policy_cfg, 'type', '?')}' does not have "
            f"an emix_index_path field. Use an emix_* policy type."
        )

    policy_cfg.emix_index_path = index_path

    mixup_payload = emix_payload.get("mixup") or {}
    if not isinstance(mixup_payload, dict):
        raise ValueError("emix.mixup must be a mapping when present")

    alpha_mode = str(mixup_payload.get("alpha_mode", policy_cfg.mixup_alpha_mode))
    if alpha_mode not in ("beta", "fixed"):
        raise ValueError(
            "emix.mixup.alpha_mode must be 'beta' or 'fixed', "
            f"got {alpha_mode!r}"
        )

    beta = float(mixup_payload.get("beta", policy_cfg.mixup_beta))
    if beta <= 0.0:
        raise ValueError(f"emix.mixup.beta must be positive, got {beta}")

    alpha_value_payload = mixup_payload.get(
        "alpha_value", policy_cfg.mixup_alpha_value
    )
    alpha_value = None if alpha_value_payload is None else float(alpha_value_payload)
    if alpha_mode == "fixed":
        if alpha_value is None:
            raise ValueError(
                "emix.mixup.alpha_value is required when alpha_mode='fixed'"
            )
        if not 0.0 < alpha_value < 1.0:
            raise ValueError(
                "emix.mixup.alpha_value must be in (0.0, 1.0), "
                f"got {alpha_value}"
            )

    loss_weight = float(
        mixup_payload.get("loss_weight", policy_cfg.mixup_loss_weight)
    )
    if loss_weight <= 0.0:
        raise ValueError(f"emix.mixup.loss_weight must be positive, got {loss_weight}")

    visual_keys = mixup_payload.get(
        "visual_keys", policy_cfg.mixup_visual_keys
    )
    if visual_keys is not None:
        if not isinstance(visual_keys, (list, tuple)):
            raise ValueError("emix.mixup.visual_keys must be a list or null")
        visual_keys = [str(key) for key in visual_keys]

    state_mode = str(
        mixup_payload.get("state_mode", policy_cfg.mixup_state_mode)
    )
    if state_mode != "anchor":
        raise ValueError(
            "emix.mixup.state_mode must be 'anchor', "
            f"got {state_mode!r}"
        )
    method = str(mixup_payload.get("method", policy_cfg.mixup_method))
    if method not in ("overlay", "collage"):
        raise ValueError(f"emix.mixup.method must be overlay or collage, got {method!r}")
    if method == "collage" and getattr(policy_cfg, "type", None) != "emix_diffusion":
        raise ValueError("Emix-collage currently requires emix_diffusion")

    # These fields are consumed later by both _build_emix_dataset() and the
    # EMIX policy forward() loss implementation.
    policy_cfg.mixup_alpha_mode = alpha_mode
    policy_cfg.mixup_beta = beta
    policy_cfg.mixup_alpha_value = alpha_value
    policy_cfg.mixup_loss_weight = loss_weight
    policy_cfg.mixup_visual_keys = visual_keys
    policy_cfg.mixup_state_mode = state_mode
    policy_cfg.mixup_method = method
