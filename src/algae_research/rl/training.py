"""PPO training wrapper with reproducible outputs and lazy heavy dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from algae_research.analysis.growth import plot_growth
from algae_research.simulation.environment import AlgaeGrowthEnv
from algae_research.simulation.run import SimulationRecord, write_simulation_csv


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_path: Path
    report_path: Path
    figure_path: Path
    evaluation_steps: int


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

    observation, _ = env.reset(seed=seed)
    records: list[SimulationRecord] = []
    factors = {"Light": [], "Nutrient": [], "Ultrasound": [], "Trace Elements": []}
    for step_number in range(1, env.max_steps + 1):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
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
        factors["Light"].append(float(action[0]))
        factors["Nutrient"].append(float(action[1]))
        factors["Ultrasound"].append(float(action[2]))
        factors["Trace Elements"].append(float(action[3]))
        if terminated or truncated:
            break

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

