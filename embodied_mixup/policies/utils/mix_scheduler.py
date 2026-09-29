"""Factory for EMIX mixed-training schedulers.

@input: scheduler config dict with keys 'type' and scheduler-specific params.
@output: MixSchedulerProtocol instance.
@scenario: Instantiate the correct scheduler from emix.training YAML config.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from .mix_scheduler_utils import (
    MIX_SCHEDULER_REGISTRY,
    ConstantMixScheduler,
    FunctionMixScheduler,
    LinearMixScheduler,
    MixSchedulerProtocol,
    StepMixScheduler,
    register_mix_scheduler,
)

__all__ = [
    "MixSchedulerProtocol",
    "ConstantMixScheduler",
    "StepMixScheduler",
    "FunctionMixScheduler",
    "LinearMixScheduler",
    "MIX_SCHEDULER_REGISTRY",
    "register_mix_scheduler",
    "make_mix_scheduler",
]


def make_mix_scheduler(scheduler_cfg: dict[str, Any]) -> MixSchedulerProtocol:
    """Instantiate a scheduler from a config dict.

    @input: scheduler_cfg dict with 'type' key plus scheduler params.
    @output: MixSchedulerProtocol instance.
    @scenario: Called once at training startup from emix.training YAML config.
    """
    scheduler_type = scheduler_cfg.get("type")
    if scheduler_type is None:
        raise ValueError("emix.training.scheduler.type is required")

    if scheduler_type == "function":
        function_name = scheduler_cfg.get("function")
        if not isinstance(function_name, str) or not function_name:
            raise ValueError(
                "emix.training.scheduler.function is required when type='function'"
            )
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name) is None:
            raise ValueError(
                "emix.training.scheduler.function must be a Python module name, "
                f"got '{function_name}'"
            )
        parameters = scheduler_cfg.get("params", {})
        if not isinstance(parameters, dict):
            raise ValueError("emix.training.scheduler.params must be a mapping")
        module_name = f"{__package__}.sampling_functions.{function_name}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"Sampling function module not found: sampling_functions/{function_name}.py"
            ) from exc
        sampling_function = getattr(module, "sampling_rate", None)
        if not callable(sampling_function):
            raise ValueError(
                f"Sampling function module '{function_name}' must define callable sampling_rate"
            )
        return FunctionMixScheduler(sampling_function, parameters)

    factory = MIX_SCHEDULER_REGISTRY.get(scheduler_type)
    if factory is None:
        raise ValueError(
            f"Unknown mix scheduler type: '{scheduler_type}'. "
            f"Available: {sorted(MIX_SCHEDULER_REGISTRY.keys())}"
        )

    if scheduler_type == "constant":
        return factory(emix_batch_prob=float(scheduler_cfg["batch_prob"]))
    if scheduler_type == "step":
        steps = scheduler_cfg.get("steps")
        if steps is None:
            raise ValueError("emix.training.scheduler.steps is required when type='step'")
        return factory(steps=steps)
    if scheduler_type == "linear":
        return factory(
            start_emix_batch_prob=float(scheduler_cfg["start_batch_prob"]),
            end_emix_batch_prob=float(scheduler_cfg["end_batch_prob"]),
            start_percentage=float(scheduler_cfg.get("start_percentage", 0.0)),
            end_percentage=float(scheduler_cfg.get("end_percentage", 1.0)),
        )
    if scheduler_type == "logarithmic":
        return factory(
            start_emix_batch_prob=float(scheduler_cfg["start_batch_prob"]),
            end_emix_batch_prob=float(scheduler_cfg["end_batch_prob"]),
            start_percentage=float(scheduler_cfg.get("start_percentage", 0.0)),
            end_percentage=float(scheduler_cfg.get("end_percentage", 1.0)),
            log_base=float(scheduler_cfg.get("log_base", 2.718281828459045)),
        )
    kwargs = {k: v for k, v in scheduler_cfg.items() if k != "type"}
    return factory(**kwargs)
