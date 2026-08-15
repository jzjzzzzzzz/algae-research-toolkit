# Algae Research Toolkit

[![CI](https://github.com/jzjzzzzzzz/algae-research-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/jzjzzzzzzz/algae-research-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

A consolidated, reproducible workspace for normalized algae-growth simulation,
reinforcement-learning experiments, browser-assisted sound exposure, and
turbidity/lux analysis.

This repository supersedes three prototypes while preserving their source,
artifacts, and complete archived Git histories:

- [`algae-project`](https://github.com/jzjzzzzzzz/algae-project)
- [`Algae-Growth-RL-learning`](https://github.com/jzjzzzzzzz/Algae-Growth-RL-learning)
- [`base-algae`](https://github.com/jzjzzzzzzz/base-algae)

## Research boundary

The growth environment is a **normalized computational prototype**, not a
validated biological or cultivation model. Its coefficients encode an explicit
reward landscape for software experiments; they are not fitted dose-response
estimates. Historical plots and PPO outputs are retained as project artifacts,
not as evidence of causal biological effects.

The repository separates four evidence levels:

| Area | What it contains | Interpretation |
|---|---|---|
| Simulation | parameterized normalized state transition | software model |
| Reinforcement learning | PPO optimization against that transition | policy behavior inside the model |
| Sound exposure | browser automation and auditable session logs | experiment operations |
| Turbidity/lux analysis | retained grouped measurements and standard errors | descriptive observations |

## Features

- Seeded Gymnasium environment with the original 5-value observation and
  4-value action spaces.
- Explicit model parameters, input validation, reproducible RNG use, and
  correct separation of algae amount from environmental state.
- Random-policy simulation with structured CSV reports.
- Optional Stable-Baselines3 PPO training with model, evaluation CSV, and plot
  outputs.
- Configurable Microsoft Edge tone-exposure sessions with one audit-log row per
  frequency.
- Grouped turbidity/lux means and standard errors plus headless plotting.
- Original scripts, presentation, logs, model, reports, and figures retained in
  clearly labelled legacy directories.

## Installation

Core simulation:

```bash
git clone https://github.com/jzjzzzzzzz/algae-research-toolkit.git
cd algae-research-toolkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install only the workflow extras you need:

```bash
python -m pip install -e '.[analysis]'
python -m pip install -e '.[rl,analysis]'
python -m pip install -e '.[experiment]'
```

On Windows, activate with `.venv\Scripts\activate`.

## Quick start

### Reproducible simulation

```bash
algae-research simulate \
  --steps 50 \
  --seed 7 \
  --output outputs/simulation.csv
```

### Turbidity/lux analysis

```bash
algae-research analyze-turbidity --output-directory outputs/turbidity
```

This writes `summary.csv` and `turbidity.png`. The y-axis is inverted in the
historical visualization because the project treated lower lux transmission as
greater turbidity; that interpretation should be checked against instrument
calibration before scientific reuse.

### PPO training

```bash
algae-research train-ppo \
  --timesteps 10000 \
  --seed 7 \
  --output-directory outputs/ppo
```

The trained policy optimizes the repository's normalized reward function. It
does not identify real-world cultivation settings without external calibration
and validation data.

### Sound-exposure session

```bash
algae-research sound-experiment \
  --frequency 200 \
  --frequency 10000 \
  --frequency 18000 \
  --duration 10 \
  --log outputs/sound_experiment_log.csv
```

This workflow requires Microsoft Edge, Selenium, and an accessible external
tone-generator page. Windows master/application volume changes are disabled by
default; use `--set-windows-volume` only when the experimental protocol
requires the retained 100% volume behavior.

## Python API

```python
import numpy as np

from algae_research.simulation.model import apply_growth_step

result = apply_growth_step(
    state=np.full(5, 0.5),
    action=np.array([0.8, 0.7, 0.2, 0.1]),
    algae_amount=0.1,
    rng=np.random.default_rng(7),
)
print(result.algae_amount)
```

## Repository map

```text
src/algae_research/
├── simulation/          normalized model, Gymnasium environment, runners
├── rl/                  optional PPO training
├── experiments/         sound-exposure automation and logging
└── analysis/            growth and turbidity/lux analysis

tests/                   unit and workflow-boundary tests
data/legacy/             original session log, unchanged
artifacts/legacy/        original model, reports, and generated figures
legacy/                  original Python source snapshots
docs/                    methodology, references, and presentation
```

## Reproducibility

- Record the command, seed, package versions, operating system, and output
  hashes for every simulation or training run.
- Treat browser sessions as incomplete unless their log status is `success`.
- Keep raw measurements immutable; write cleaned/derived tables separately.
- Do not compare simulation output with measured lux data without documenting a
  calibration model and units.
- See [docs/methodology.md](docs/methodology.md) and
  [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Development

```bash
python -m pip install -e '.[dev]'
ruff check src tests
MPLBACKEND=Agg pytest
```

## Project status

Active consolidation repository. New work belongs here; the three source
repositories are read-only archives with migration banners linking back to this
project.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Source
provenance and the original MIT/Apache-2.0 licenses are recorded in
[MIGRATION.md](MIGRATION.md).

