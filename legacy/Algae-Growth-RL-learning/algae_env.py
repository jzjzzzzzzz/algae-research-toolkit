# env/algae_env.py
##################################################
###### Copyright (c) 2026 jzjzzzzzzz
###### All rights reserved.
##################################################

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AlgaeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self):
        super(AlgaeEnv, self).__init__()

        # Observation space: Light, Nutrient, Temperature, Ultrasound, Trace Elements
        self.obs_low = np.array([0.0]*5, dtype=np.float32)
        self.obs_high = np.array([1.0]*5, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        # Action space: Light, Nutrient, Ultrasound, Trace Elements (adjustable)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.state = None
        self.algae_amount = None
        self.step_count = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize state with random values
        self.state = np.random.uniform(low=0.3, high=0.7, size=(5,))
        self.algae_amount = 0.1
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1

        # Apply the action to the four controllable factors
        self.state[0] = np.clip(action[0], 0, 1)  # Light
        self.state[1] = np.clip(action[1], 0, 1)  # Nutrient
        self.state[3] = np.clip(action[2], 0, 1)  # Ultrasound
        self.state[4] = np.clip(action[3], 0, 1)  # Trace Elements

        # Simple growth model: considering all controllable factors
        growth = (self.state[0]*0.4 + self.state[1]*0.3 + self.state[3]*0.2 + self.state[4]*0.1) * 0.2
        decay = 0.05
        self.algae_amount = max(self.algae_amount + growth - decay, 0)

        reward = growth - decay

        # Random disturbance for Temperature (state[2])
        self.state[2] = np.clip(self.state[2] + np.random.uniform(-0.01, 0.01), 0, 1)

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {"algae_amount": self.algae_amount}

        return self.state.copy(), reward, terminated, truncated, info

    def render(self, mode="human"):
        # Print the current step, algae amount, and state
        print(f"Step {self.step_count}: Algae={self.algae_amount:.4f}, State={self.state}")
