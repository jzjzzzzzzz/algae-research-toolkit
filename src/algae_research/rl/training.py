"""PPO training wrapper with reproducible outputs and lazy heavy dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from algae_research.analysis.growth import plot_growth
from algae_research.simulation.environment import AlgaeGrowthEnv
from algae_research.simulation.run import SimulationRecord, write_simulation_csv


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_path: Path
    report_path: Path
    figure_path: Path
    evaluation_steps: int


def _evaluate_policy(
    model: Any,
    env: AlgaeGrowthEnv,
    *,
    seed: int,
) -> tuple[list[SimulationRecord], dict[str, list[float]]]:
    observation, _ = env.reset(seed=seed)
    records: list[SimulationRecord] = []
    factors: dict[str, list[float]] = {
        "Light": [],
        "Nutrient": [],
        "Ultrasound": [],
        "Trace Elements": [],
    }
    for step_number in range(1, env.max_steps + 1):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        record = SimulationRecord(
            step=step_number,
            algae_amount=float(info["algae_amount"]),
            reward=float(reward),
            light=float(observation[0]),
            nutrient=float(observation[1]),
            temperature=float(observation[2]),
            ultrasound=float(observation[3]),
            trace_elements=float(observation[4]),
        )
        records.append(record)
        factors["Light"].append(record.light)
        factors["Nutrient"].append(record.nutrient)
        factors["Ultrasound"].append(record.ultrasound)
        factors["Trace Elements"].append(record.trace_elements)
        if terminated or truncated:
            break
    return records, factors


def train_ppo(
    *,
    total_timesteps: int = 10_000,
    seed: int = 0,
    output_directory: str | Path = "outputs/ppo",
) -> TrainingResult:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "PPO training requires Stable-Baselines3 and PyTorch. Install with "
            "pip install 'algae-research-toolkit[rl,analysis]'."
        ) from exc

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = AlgaeGrowthEnv()
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)

    records, factors = _evaluate_policy(model, env, seed=seed)

    model_base = output / "ppo_algae_model"
    model.save(str(model_base))
    report_path = write_simulation_csv(records, output / "evaluation.csv")
    figure_path = output / "evaluation.png"
    plot_growth(
        [record.algae_amount for record in records],
        factors,
        title="PPO Control of the Normalized Algae Simulation",
        destination=figure_path,
    )
    return TrainingResult(
        model_path=model_base.with_suffix(".zip"),
        report_path=report_path,
        figure_path=figure_path,
        evaluation_steps=len(records),
    )
