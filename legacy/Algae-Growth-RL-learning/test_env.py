##################################################
###### Copyright (c) 2026 jzjzzzzzzz
###### All rights reserved.
##################################################

import numpy as np
from env.algae_env import AlgaeEnv
from utils.plot_utils import plot_growth

def test_algae_env(steps=50, save_fig=True):
    """
    Test AlgaeEnv by running random actions and plotting results.
    """
    # Initialize the environment
    env = AlgaeEnv()
    obs, _ = env.reset()

    # Initialize records
    algae_history = []
    factor_histories = {
        "light": [],
        "nutrient": [],
        "temp": [],
        "ultrasound": [],
        "trace_element": []
    }

    print("Starting environment test with random actions...")
    for step in range(steps):
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, _ = env.step(action)

        # Save data
        algae_history.append(obs[-1])
        factor_histories["light"].append(obs[0])
        factor_histories["nutrient"].append(obs[1])
        factor_histories["temp"].append(obs[2])
        factor_histories["ultrasound"].append(obs[3])
        factor_histories["trace_element"].append(obs[4])

        # Print the current state
        print(f"Step {step+1}: Algae={obs[-1]:.2f}, Reward={reward:.2f}")

        if done:
            obs, _ = env.reset()

    # Plot the results
    plot_growth(
        algae_history,
        factor_histories=factor_histories,
        title="Test: Algae Growth with Random Actions",
        save_path="test_algae_growth.png" if save_fig else None,
        show=True
    )
    print("Test completed.")

if __name__ == "__main__":
    test_algae_env()
