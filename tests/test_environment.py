import numpy as np
import pytest

from algae_research.simulation.environment import AlgaeGrowthEnv


def test_seeded_reset_and_step_are_reproducible():
    first = AlgaeGrowthEnv(max_steps=2)
    second = AlgaeGrowthEnv(max_steps=2)
    first_obs, first_info = first.reset(seed=7)
    second_obs, second_info = second.reset(seed=7)
    assert np.array_equal(first_obs, second_obs)
    assert first_info == second_info == {"algae_amount": 0.1}

    action = np.full(4, 0.5, dtype=np.float32)
    first_result = first.step(action)
    second_result = second.step(action)
    assert np.array_equal(first_result[0], second_result[0])
    assert first_result[1:] == second_result[1:]


def test_episode_truncates_at_configured_limit():
    env = AlgaeGrowthEnv(max_steps=2)
    env.reset(seed=1)
    assert env.step(env.action_space.sample())[3] is False
    assert env.step(env.action_space.sample())[3] is True


def test_step_requires_reset():
    env = AlgaeGrowthEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4, dtype=np.float32))


def test_spaces_remain_compatible_with_archived_ppo_model():
    env = AlgaeGrowthEnv()
    assert env.observation_space.shape == (5,)
    assert env.action_space.shape == (4,)
