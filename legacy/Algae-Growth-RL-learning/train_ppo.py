# agents/train_ppo.py

##################################################
###### Copyright (c) 2026 jzjzzzzzzz
###### All rights reserved.
##################################################

import gymnasium as gym
from stable_baselines3 import PPO
from env.algae_env import AlgaeEnv
from utils.plot_utils import plot_growth


def quick_train(total_timesteps=10000):
    # Create the environment for algae growth
    env = AlgaeEnv()

    # Initialize the PPO model with MLP policy
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model for the specified number of timesteps
    model.learn(total_timesteps=total_timesteps)

    # Test and collect data after training
    obs, _ = env.reset()

    # Store the history of algae growth and environmental factors
    algae_history = []
    factor_histories = {
        "Light": [],
        "Nutrient": [],
        "Ultrasound": [],
        "Trace Elements": []
    }

    # Simulate the environment for the maximum number of steps
    for _ in range(env.max_steps):
        # Predict the action based on the current observation
        action, _ = model.predict(obs)

        # Step the environment using the predicted action
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if the episode has ended
        done = terminated or truncated

        # Collect data at each step
        algae_history.append(info["algae_amount"])
        factor_histories["Light"].append(action[0])
        factor_histories["Nutrient"].append(action[1])
        factor_histories["Ultrasound"].append(action[2])
        factor_histories["Trace Elements"].append(action[3])

        if done:
            break

    # Render the final environment state
    env.render()

    # Plot the growth results
    plot_growth(
        algae_history=algae_history,
        factor_histories=factor_histories,
        title="PPO Control of Algae Growth (4 Factors)",
        save_path="algae_result.png"
    )


if __name__ == "__main__":
    # Start the training process with the specified number of timesteps
    quick_train(total_timesteps=10000)
