"""Utilities and built-in implementations for mix training schedulers.

@input: step int, total_steps int, scheduler config dict.
@output: emix_batch_prob float in [0.0, 1.0].
@scenario: Provide pluggable per-step EMIX batch probability for mixed training.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MixSchedulerProtocol:
    """Interface every mix scheduler must satisfy."""

    def emix_batch_prob(self, step: int, total_steps: int) -> float:
        raise NotImplementedError


class ConstantMixScheduler(MixSchedulerProtocol):
    """Always return the same probability.

    @input: emix_batch_prob float in [0.0, 1.0].
    @output: float, identical on every call.
    @scenario: Stable EMIX fraction throughout training.
    """

    def __init__(self, emix_batch_prob: float) -> None:
        if not 0.0 <= emix_batch_prob <= 1.0:
            raise ValueError(
                f"emix_batch_prob must be in [0.0, 1.0], got {emix_batch_prob}"
            )
        self._prob = float(emix_batch_prob)

    def emix_batch_prob(self, step: int, total_steps: int) -> float:
        return self._prob


class StepMixScheduler(MixSchedulerProtocol):
    """Return a constant probability for each configured training interval.

    Each step is a mapping with ``start_percentage``, ``end_percentage`` and
    ``ratio`` keys. Intervals use half-open bounds ``[start, end)``; an
    interval ending at ``1.0`` also includes the final training step. Percent
    ranges not covered by a configured interval return ``0.0``.

    @input: steps non-empty sequence of mappings with percentages in [0.0, 1.0]
        and ratios in [0.0, 1.0]; step int; total_steps int.
    @output: float in [0.0, 1.0] for the matching interval.
    @scenario: Use multiple constant EMIX fractions over training, including
        a constant schedule as one interval spanning [0.0, 1.0].
    """

    def __init__(self, steps: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
            raise ValueError("steps must be a non-empty sequence of mappings")
        if not steps:
            raise ValueError("steps must contain at least one interval")

        normalized_steps: list[tuple[float, float, float]] = []
        previous_end = 0.0
        for index, step_config in enumerate(steps):
            if not isinstance(step_config, Mapping):
                raise ValueError(f"steps[{index}] must be a mapping")
            missing_keys = {
                "start_percentage",
                "end_percentage",
                "ratio",
            } - step_config.keys()
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"steps[{index}] is missing required key(s): {missing}")

            start_percentage = self._finite_float(
                step_config["start_percentage"],
                f"steps[{index}].start_percentage",
            )
            end_percentage = self._finite_float(
                step_config["end_percentage"],
                f"steps[{index}].end_percentage",
            )
            ratio = self._finite_float(step_config["ratio"], f"steps[{index}].ratio")

            if not 0.0 <= start_percentage < 1.0:
                raise ValueError(
                    f"steps[{index}].start_percentage must be in [0.0, 1.0), "
                    f"got {start_percentage}"
                )
            if not 0.0 < end_percentage <= 1.0:
                raise ValueError(
                    f"steps[{index}].end_percentage must be in (0.0, 1.0], "
                    f"got {end_percentage}"
                )
            if start_percentage >= end_percentage:
                raise ValueError(
                    f"steps[{index}].start_percentage ({start_percentage}) must be "
                    f"less than end_percentage ({end_percentage})"
                )
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    f"steps[{index}].ratio must be in [0.0, 1.0], got {ratio}"
                )
            if start_percentage < previous_end:
                raise ValueError(
                    f"steps must be ordered and non-overlapping: steps[{index}] "
                    f"starts at {start_percentage}, before the previous end "
                    f"{previous_end}"
                )

            normalized_steps.append((start_percentage, end_percentage, ratio))
            previous_end = end_percentage

        self._steps = tuple(normalized_steps)

    @staticmethod
    def _finite_float(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        return numeric_value

    def emix_batch_prob(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            global_progress = 0.0
        else:
            global_progress = (step - 1) / (total_steps - 1)
            global_progress = min(1.0, max(0.0, global_progress))

        for start_percentage, end_percentage, ratio in self._steps:
            if start_percentage <= global_progress < end_percentage:
                return ratio
            if global_progress == 1.0 and end_percentage == 1.0:
                return ratio
        return 0.0


class LinearMixScheduler(MixSchedulerProtocol):
    """Linearly anneal emix_batch_prob from start to end over a configurable
    sub-interval of total_steps.

    @input: start_emix_batch_prob, end_emix_batch_prob floats in [0.0, 1.0];
        start_percentage [0.0, 1.0) fraction of total_steps where linear
        change begins (clamped at start value before this point);
        end_percentage (0.0, 1.0] fraction of total_steps where linear
        change ends (clamped at end value after this point).
        step int; total_steps int.
    @output: float, interpolated probability clamped to [0.0, 1.0].
    @scenario: Gradually reduce EMIX fraction as training progresses, with
        optional flat plateaus before/after the linear ramp.
    """

    def __init__(
        self,
        start_emix_batch_prob: float,
        end_emix_batch_prob: float,
        start_percentage: float = 0.0,
        end_percentage: float = 1.0,
    ) -> None:
        if not 0.0 <= start_emix_batch_prob <= 1.0:
            raise ValueError(
                f"start_emix_batch_prob must be in [0.0, 1.0], got {start_emix_batch_prob}"
            )
        if not 0.0 <= end_emix_batch_prob <= 1.0:
            raise ValueError(
                f"end_emix_batch_prob must be in [0.0, 1.0], got {end_emix_batch_prob}"
            )
        if not 0.0 <= start_percentage < 1.0:
            raise ValueError(
                f"start_percentage must be in [0.0, 1.0), got {start_percentage}"
            )
        if not 0.0 < end_percentage <= 1.0:
            raise ValueError(
                f"end_percentage must be in (0.0, 1.0], got {end_percentage}"
            )
        if start_percentage >= end_percentage:
            raise ValueError(
                f"start_percentage ({start_percentage}) must be less than "
                f"end_percentage ({end_percentage})"
            )
        self._start = float(start_emix_batch_prob)
        self._end = float(end_emix_batch_prob)
        self._start_pct = float(start_percentage)
        self._end_pct = float(end_percentage)

    def emix_batch_prob(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return self._end

        # Map step to a normalized progress in [0.0, 1.0].
        global_progress = (step - 1) / (total_steps - 1)
        global_progress = min(1.0, max(0.0, global_progress))

        # Before the linear ramp: hold at start value.
        if global_progress <= self._start_pct:
            return self._start

        # After the linear ramp: hold at end value.
        if global_progress >= self._end_pct:
            return self._end

        # Inside the ramp: linear interpolation.
        ramp_progress = (global_progress - self._start_pct) / (self._end_pct - self._start_pct)
        return self._start + ramp_progress * (self._end - self._start)


class LogarithmicMixScheduler(MixSchedulerProtocol):
    """Logarithmically anneal emix_batch_prob from start to end over a
    configurable sub-interval of total_steps.

    The interpolation follows a log-shaped curve that changes rapidly at the
    beginning of the ramp and gradually levels off — producing a natural
    ``-log`` decay when ``start > end`` and a ``+log`` growth when
    ``start < end``.

    @input: start_emix_batch_prob, end_emix_batch_prob floats in [0.0, 1.0];
        start_percentage [0.0, 1.0) fraction of total_steps where log change
        begins (clamped at start value before this point);
        end_percentage (0.0, 1.0] fraction of total_steps where log change
        ends (clamped at end value after this point);
        log_base float > 1.0 (default math.e), the logarithm base controlling
        curvature — larger values produce more extreme early change.
    @output: float, log-interpolated probability clamped to [0.0, 1.0].
    @scenario: Rapidly reduce EMIX fraction early in the ramp, with a long
        tail approaching the end value.
    """

    def __init__(
        self,
        start_emix_batch_prob: float,
        end_emix_batch_prob: float,
        start_percentage: float = 0.0,
        end_percentage: float = 1.0,
        log_base: float = math.e,
    ) -> None:
        if not 0.0 <= start_emix_batch_prob <= 1.0:
            raise ValueError(
                f"start_emix_batch_prob must be in [0.0, 1.0], got {start_emix_batch_prob}"
            )
        if not 0.0 <= end_emix_batch_prob <= 1.0:
            raise ValueError(
                f"end_emix_batch_prob must be in [0.0, 1.0], got {end_emix_batch_prob}"
            )
        if not 0.0 <= start_percentage < 1.0:
            raise ValueError(
                f"start_percentage must be in [0.0, 1.0), got {start_percentage}"
            )
        if not 0.0 < end_percentage <= 1.0:
            raise ValueError(
                f"end_percentage must be in (0.0, 1.0], got {end_percentage}"
            )
        if start_percentage >= end_percentage:
            raise ValueError(
                f"start_percentage ({start_percentage}) must be less than "
                f"end_percentage ({end_percentage})"
            )
        if not isinstance(log_base, (int, float)) or log_base <= 1.0:
            raise ValueError(
                f"log_base must be > 1.0, got {log_base}"
            )
        if not math.isfinite(log_base):
            raise ValueError(
                f"log_base must be finite, got {log_base}"
            )
        self._start = float(start_emix_batch_prob)
        self._end = float(end_emix_batch_prob)
        self._start_pct = float(start_percentage)
        self._end_pct = float(end_percentage)
        self._log_base = float(log_base)
        # Pre-compute: log_base(b) = 1.0 when natural log, so denominator works generically.
        self._log_denom = math.log(self._log_base)

    def emix_batch_prob(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return self._end

        # Map step to a normalized progress in [0.0, 1.0].
        global_progress = (step - 1) / (total_steps - 1)
        global_progress = min(1.0, max(0.0, global_progress))

        # Before the log ramp: hold at start value.
        if global_progress <= self._start_pct:
            return self._start

        # After the log ramp: hold at end value.
        if global_progress >= self._end_pct:
            return self._end

        # Inside the ramp: logarithmic interpolation.
        # ramp_progress ∈ [0.0, 1.0] maps to log(1 + t*(b-1)) / log(b).
        ramp_progress = (global_progress - self._start_pct) / (self._end_pct - self._start_pct)
        log_factor = math.log(1.0 + ramp_progress * (self._log_base - 1.0)) / self._log_denom
        return self._start + log_factor * (self._end - self._start)


class FunctionMixScheduler(MixSchedulerProtocol):
    """Compute the EMIX batch probability with a configured function.

    @input: sampling_function callable accepting ``step`` and keyword
        parameters; parameters dict containing function-specific values.
    @output: float in [0.0, 1.0] returned by the configured function.
    @scenario: Use a step-dependent EMIX sampling policy loaded from a module.
    """

    def __init__(self, sampling_function: Callable[..., Any], parameters: dict[str, Any]) -> None:
        if not callable(sampling_function):
            raise TypeError("sampling_function must be callable")
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a mapping")
        self._sampling_function = sampling_function
        self._parameters = dict(parameters)

    def emix_batch_prob(self, step: int, total_steps: int) -> float:
        """Evaluate and validate the configured sampling function.

        @input: step [int] current 1-based optimizer step; total_steps [int]
            total optimizer steps, accepted for scheduler interface compatibility.
        @output: float in [0.0, 1.0] EMIX batch sampling probability.
        @scenario: Select the EMIX or non-EMIX dataloader for one training step.
        """

        del total_steps
        probability = self._sampling_function(step=step, **self._parameters)
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError(
                "Sampling function must return a numeric probability, "
                f"got {type(probability).__name__}"
            )
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Sampling function must return a finite probability in [0.0, 1.0], "
                f"got {probability}"
            )
        return probability


MIX_SCHEDULER_REGISTRY: dict[str, type] = {
    "constant": ConstantMixScheduler,
    "step": StepMixScheduler,
    "linear": LinearMixScheduler,
    "logarithmic": LogarithmicMixScheduler,
}


def register_mix_scheduler(name: str, factory: Callable[..., MixSchedulerProtocol]) -> None:
    """Register an external scheduler factory under a name.

    @input: name str, factory callable returning MixSchedulerProtocol.
    @output: None; mutates MIX_SCHEDULER_REGISTRY.
    @scenario: Allow downstream code to plug in custom schedulers.
    """
    if name in MIX_SCHEDULER_REGISTRY:
        logger.warning("[mix_scheduler] Overwriting existing scheduler: %s", name)
    MIX_SCHEDULER_REGISTRY[name] = factory
