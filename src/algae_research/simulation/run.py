"""Repeatable simulation runners and tabular output."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .environment import AlgaeGrowthEnv


@dataclass(frozen=True, slots=True)
class SimulationRecord:
    step: int
    algae_amount: float
    reward: float
    light: float
    nutrient: float
    temperature: float
    ultrasound: float
    trace_elements: float


def run_random_policy(*, steps: int = 50, seed: int = 0) -> list[SimulationRecord]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    env = AlgaeGrowthEnv(max_steps=steps)
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    records: list[SimulationRecord] = []
    for step_number in range(1, steps + 1):
        action = env.action_space.sample()
        observation, reward, _, truncated, info = env.step(action)
        records.append(
            SimulationRecord(
                step=step_number,
                algae_amount=float(info["algae_amount"]),
                reward=float(reward),
                light=float(observation[0]),
                nutrient=float(observation[1]),
                temperature=float(observation[2]),
                ultrasound=float(observation[3]),
                trace_elements=float(observation[4]),
            )
        )
        if truncated:
            break
    return records


def write_simulation_csv(records: list[SimulationRecord], destination: str | Path) -> Path:
    if not records:
        raise ValueError("At least one simulation record is required.")
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return output

