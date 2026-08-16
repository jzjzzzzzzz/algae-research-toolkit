import numpy as np
import pytest

from algae_research.simulation.model import GrowthModelParameters, apply_growth_step


def test_default_model_matches_original_equation_without_drift():
    params = GrowthModelParameters(temperature_drift=0)
    result = apply_growth_step(
        np.full(5, 0.5, dtype=np.float32),
        np.array([1.0, 0.5, 0.25, 0.0], dtype=np.float32),
        0.1,
        rng=np.random.default_rng(1),
        parameters=params,
    )
    expected_growth = (1.0 * 0.4 + 0.5 * 0.3 + 0.25 * 0.2) * 0.2
    assert result.growth == pytest.approx(expected_growth)
    assert result.reward == pytest.approx(expected_growth - 0.05)
    assert result.algae_amount == pytest.approx(0.1 + expected_growth - 0.05)
    assert result.state[2] == pytest.approx(0.5)


def test_action_is_bounded_without_mutating_input():
    state = np.full(5, 0.5, dtype=np.float32)
    action = np.array([-1, 2, 0.5, 0.25], dtype=np.float32)
    result = apply_growth_step(state, action, 0.1, rng=np.random.default_rng(1))
    assert result.state[[0, 1, 3, 4]].tolist() == [0.0, 1.0, 0.5, 0.25]
    assert state.tolist() == [0.5] * 5


@pytest.mark.parametrize("shape", [(4,), (6,)])
def test_invalid_state_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="State"):
        apply_growth_step(
            np.zeros(shape, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            0.1,
            rng=np.random.default_rng(1),
        )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), float("-inf"), True])
def test_invalid_parameters_are_rejected(value):
    with pytest.raises(ValueError, match="light_weight"):
        GrowthModelParameters(light_weight=value)


def test_finite_integer_parameters_remain_supported():
    parameters = GrowthModelParameters(light_weight=1, temperature_drift=0)
    assert parameters.light_weight == 1
