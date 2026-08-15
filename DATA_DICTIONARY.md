# Data dictionary

## Simulation CSV

| Column | Meaning |
|---|---|
| `step` | one-based environment step |
| `algae_amount` | accumulated amount in the normalized model |
| `reward` | current modeled growth minus decay |
| `light` | normalized light state |
| `nutrient` | normalized nutrient state |
| `temperature` | normalized drifting temperature proxy |
| `ultrasound` | normalized ultrasound state |
| `trace_elements` | normalized trace-element state |

## Sound-experiment log

| Column | Meaning |
|---|---|
| `date` | local calendar date at session start |
| `frequency_hz` | requested web tone frequency |
| `duration_seconds` | requested playback duration |
| `start_time`, `end_time` | timezone-aware local timestamps |
| `system`, `browser` | automation environment |
| `website` | exact requested tone URL |
| `status` | `success` or captured failure text |

The historical log in `data/legacy/` is unmodified and contains a schema
transition. It should not be parsed as a clean table without an explicit repair
step.

## Turbidity/lux summary

For each group and day, `Mean` is the arithmetic mean of three retained sample
columns and `SE` is sample standard deviation divided by `sqrt(3)`.

