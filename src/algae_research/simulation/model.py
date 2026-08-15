"""Explicit, parameterized version of the original normalized growth model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FACTOR_NAMES = ("light", "nutrient", "temperature", "ultrasound", "trace_elements")
ACTION_NAMES = ("light", "nutrient", "ultrasound", "trace_elements")


@dataclass(frozen=True, slots=True)
class GrowthModelParameters:
    """Coefficients for the prototype's normalized, non-biological model."""

    light_weight: float = 0.4
    nutrient_weight: float = 0.3
    ultrasound_weight: float = 0.2
    trace_element_weight: float = 0.1
    growth_scale: float = 0.2
    decay: float = 0.05
    temperature_drift: float = 0.01

    def __post_init__(self) -> None:
        weights = (
            self.light_weight,
            self.nutrient_weight,
            self.ultrasound_weight,
            self.trace_element_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Growth weights must be non-negative.")
        if self.growth_scale < 0 or self.decay < 0 or self.temperature_drift < 0:
            raise ValueError("Growth scale, decay, and drift must be non-negative.")


@dataclass(frozen=True, slots=True)
class GrowthStep:
    state: NDArray[np.float32]
    algae_amount: float
    growth: float
    decay: float
    reward: float


def apply_growth_step(
    state: NDArray[np.floating],
    action: NDArray[np.floating],
    algae_amount: float,
    *,
    rng: np.random.Generator,
    parameters: GrowthModelParameters | None = None,
) -> GrowthStep:
    """Apply one deterministic-except-for-temperature simulation step."""

    params = parameters or GrowthModelParameters()
    state_array = np.asarray(state, dtype=np.float32).copy()
    action_array = np.asarray(action, dtype=np.float32)
    if state_array.shape != (5,):
        raise ValueError(f"State must have shape (5,), received {state_array.shape}.")
    if action_array.shape != (4,):
        raise ValueError(f"Action must have shape (4,), received {action_array.shape}.")
    if not np.isfinite(state_array).all() or not np.isfinite(action_array).all():
        raise ValueError("State and action values must be finite.")

    bounded_action = np.clip(action_array, 0, 1)
    state_array[[0, 1, 3, 4]] = bounded_action
    growth = float(
        (
            state_array[0] * params.light_weight
            + state_array[1] * params.nutrient_weight
            + state_array[3] * params.ultrasound_weight
            + state_array[4] * params.trace_element_weight
        )
        * params.growth_scale
    )
    updated_amount = max(float(algae_amount) + growth - params.decay, 0.0)
    state_array[2] = np.clip(
        state_array[2] + rng.uniform(-params.temperature_drift, params.temperature_drift),
        0,
        1,
    )
    return GrowthStep(
        state=state_array,
        algae_amount=updated_amount,
        growth=growth,
        decay=params.decay,
        reward=growth - params.decay,
    )

