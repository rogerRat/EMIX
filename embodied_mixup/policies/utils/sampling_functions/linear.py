"""Linear EMIX sampling-rate function.

@input: step and interval/probability parameters from the training scheduler.
@output: EMIX sampling probability in [0.0, 1.0].
@scenario: Linearly reduce or increase EMIX sampling over a bounded step range.
"""

from __future__ import annotations


def sampling_rate(
    step: int,
    start_step: int,
    end_step: int,
    start_pct: float,
    end_pct: float,
) -> float:
    """Return a clamped linear sampling rate for one training step.

    @input: step [int] current optimizer step; start_step/end_step [int] inclusive
        interval boundaries with start_step <= end_step; start_pct/end_pct [float]
        endpoint probabilities in [0.0, 1.0].
    @output: float in [0.0, 1.0], equal to an endpoint outside the interval and
        linearly interpolated inside it.
    @scenario: Configure a schedule such as 0.50 at step 0 to 0.00 at step 5000,
        then keep 0.00 for all later steps.
    """

    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"step must be an integer, got {type(step).__name__}")
    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise TypeError(f"start_step must be an integer, got {type(start_step).__name__}")
    if isinstance(end_step, bool) or not isinstance(end_step, int):
        raise TypeError(f"end_step must be an integer, got {type(end_step).__name__}")
    if start_step > end_step:
        raise ValueError(
            f"start_step must be less than or equal to end_step, got {start_step} > {end_step}"
        )
    if not 0.0 <= float(start_pct) <= 1.0:
        raise ValueError(f"start_pct must be in [0.0, 1.0], got {start_pct}")
    if not 0.0 <= float(end_pct) <= 1.0:
        raise ValueError(f"end_pct must be in [0.0, 1.0], got {end_pct}")

    start_pct = float(start_pct)
    end_pct = float(end_pct)
    if start_step == end_step:
        return end_pct
    if step <= start_step:
        return start_pct
    if step >= end_step:
        return end_pct

    progress = (step - start_step) / float(end_step - start_step)
    return start_pct + progress * (end_pct - start_pct)
