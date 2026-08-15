import builtins

import pytest

from algae_research.rl.training import train_ppo


def test_training_dependency_error_is_actionable(monkeypatch):
    original_import = builtins.__import__

    def reject_stable_baselines(name, *args, **kwargs):
        if name == "stable_baselines3":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_stable_baselines)
    with pytest.raises(RuntimeError, match="Stable-Baselines3"):
        train_ppo(total_timesteps=1)
