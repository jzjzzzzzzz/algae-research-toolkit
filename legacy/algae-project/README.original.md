# Algae Growth RL Prediction

Python research sandbox for simulating algae growth and training a reinforcement-learning controller. The custom Gymnasium environment models light, nutrient level, temperature drift, ultrasound exposure, and trace elements; a PPO agent can learn action settings that maximize simulated growth.

## Project Structure

- `env/algae_env.py`: Gymnasium environment used by training and tests.
- `agents/train_ppo.py`: PPO training script using Stable-Baselines3.
- `utils/plot_utils.py`: plotting helper for growth and factor histories.
- `tests/test_env.py`: manual environment smoke test.
- `quick_algae_report.csv`, `algae_result.png`: generated experiment outputs kept as reference artifacts.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## Run

Train and plot a quick PPO run:

```bash
.venv/bin/python agents/train_ppo.py
```

Run the environment smoke test:

```bash
.venv/bin/python tests/test_env.py
```

## Notes

- Training requires `torch`, `gymnasium`, and `stable-baselines3`.
- The growth model is a simplified simulation, not a validated biological model.
- Keep large local virtual environments and regenerated scratch outputs out of git.
