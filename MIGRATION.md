# Migration guide

## Repository mapping

| Archived repository | Retained scope | Maintained location |
|---|---|---|
| [`algae-project`](https://github.com/jzjzzzzzzz/algae-project) | structured Gymnasium environment, PPO runner, plotting, artifacts | `simulation/`, `rl/`, `analysis/`, `artifacts/legacy/` |
| [`Algae-Growth-RL-learning`](https://github.com/jzjzzzzzzz/Algae-Growth-RL-learning) | original flat-layout environment and PPO prototype | same maintained modules plus `legacy/` snapshot |
| [`base-algae`](https://github.com/jzjzzzzzzz/base-algae) | Edge tone automation, logs, grouped lux plot, presentation/references | `experiments/`, `analysis/turbidity.py`, `data/legacy/`, `docs/` |

No source repository was deleted. Each archive retains its complete Git
history, stars, issues, and attachments. Maintained functionality was rewritten
behind stable package interfaces; original Python files remain under `legacy/`.

## Command migration

| Previous command | Replacement |
|---|---|
| `python tests/test_env.py` | `algae-research simulate --steps 50 --seed 0` |
| `python agents/train_ppo.py` | `algae-research train-ppo --timesteps 10000` |
| `python train_ppo.py` | `algae-research train-ppo --timesteps 10000` |
| `python algae_sound_experiment_edge.py` | `algae-research sound-experiment ...` |
| `python graph.py` | `algae-research analyze-turbidity` |

## Important behavior corrections

- The simulation now uses Gymnasium's seeded RNG rather than global
  `numpy.random`, making reset/step sequences reproducible.
- Algae amount is read from `info["algae_amount"]`; the final observation value
  is trace-element level and is no longer mislabeled as algae growth.
- State/action shapes and finite values are validated.
- Growth coefficients are a typed configuration instead of hidden literals.
- Browser session failures are retained as data rather than printed and lost.
- Windows volume control is explicit rather than enabled by default.
- Plotting supports headless output for CI and reproducible runs.

## Historical PPO model

The original PPO ZIP is stored once in `artifacts/legacy/shared/` because both
RL repositories contained identical SHA-256 content. The new environment keeps
the original observation/action shapes for inspection, but dependency-version
and serialization compatibility should be checked before loading the model.

