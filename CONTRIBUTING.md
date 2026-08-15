# Contributing

## Local checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check src tests
MPLBACKEND=Agg pytest
```

## Research changes

- State whether a change affects code behavior, model assumptions, measured
  data, or interpretation.
- Add units and provenance for new measured variables.
- Keep raw data immutable and derived data reproducible.
- Add a seeded test for stochastic model changes.
- Do not present simulation optimization as experimental validation.

