import builtins

import numpy as np
import pytest

from algae_research.rl.training import _evaluate_policy, train_ppo
from algae_research.simulation.environment import AlgaeGrowthEnv


def test_training_dependency_error_is_actionable(monkeypatch):
    original_import = builtins.__import__

    def reject_stable_baselines(name, *args, **kwargs):
        if name == "stable_baselines3":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_stable_baselines)
    with pytest.raises(RuntimeError, match="Stable-Baselines3"):
        train_ppo(total_timesteps=1)


def test_evaluation_factors_match_controls_applied_by_environment():
    class OutOfRangePolicy:
        def predict(self, _observation, *, deterministic):
            assert deterministic is True
            return np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32), None

    records, factors = _evaluate_policy(
        OutOfRangePolicy(),
        AlgaeGrowthEnv(max_steps=1),
        seed=7,
    )

    assert len(records) == 1
    record = records[0]
    assert (record.light, record.nutrient, record.ultrasound, record.trace_elements) == (
        1.0,
        0.0,
        0.5,
        1.0,
    )
    assert factors == {
        "Light": [record.light],
        "Nutrient": [record.nutrient],
        "Ultrasound": [record.ultrasound],
        "Trace Elements": [record.trace_elements],
    }
