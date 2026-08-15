"""Gymnasium-compatible environment preserving the source model's spaces."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .model import GrowthModelParameters, apply_growth_step


class AlgaeGrowthEnv(gym.Env):
    """Normalized research simulation; not a validated biological model."""

    metadata: ClassVar[dict[str, object]] = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        *,
        max_steps: int = 200,
        initial_algae_amount: float = 0.1,
        parameters: GrowthModelParameters | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if initial_algae_amount < 0:
            raise ValueError("initial_algae_amount cannot be negative.")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'.")

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.max_steps = max_steps
        self.initial_algae_amount = float(initial_algae_amount)
        self.parameters = parameters or GrowthModelParameters()
        self.render_mode = render_mode
        self.state: np.ndarray | None = None
        self.algae_amount = self.initial_algae_amount
        self.step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        self.state = self.np_random.uniform(0.3, 0.7, size=5).astype(np.float32)
        self.algae_amount = self.initial_algae_amount
        self.step_count = 0
        return self.state.copy(), {"algae_amount": self.algae_amount}

    def step(self, action: np.ndarray):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        result = apply_growth_step(
            self.state,
            action,
            self.algae_amount,
            rng=self.np_random,
            parameters=self.parameters,
        )
        self.state = result.state
        self.algae_amount = result.algae_amount
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {
            "algae_amount": self.algae_amount,
            "growth": result.growth,
            "decay": result.decay,
        }
        return self.state.copy(), result.reward, terminated, truncated, info

    def render(self) -> None:
        if self.state is None:
            print("AlgaeGrowthEnv has not been reset.")
            return
        print(f"Step {self.step_count}: Algae={self.algae_amount:.4f}, State={self.state}")


# Compatibility alias used by both source repositories.
AlgaeEnv = AlgaeGrowthEnv
