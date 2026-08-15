import csv
import json
from pathlib import Path

from algae_research.cli import main
from algae_research.rl.training import TrainingResult
from algae_research.simulation.run import run_random_policy, write_simulation_csv


def test_seeded_random_policy_is_reproducible():
    assert run_random_policy(steps=4, seed=9) == run_random_policy(steps=4, seed=9)


def test_csv_contains_algae_amount_not_trace_element_proxy(tmp_path: Path):
    records = run_random_policy(steps=3, seed=1)
    output = write_simulation_csv(records, tmp_path / "nested" / "simulation.csv")
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert float(rows[-1]["algae_amount"]) == records[-1].algae_amount
    assert "trace_elements" in rows[-1]


def test_cli_writes_requested_output(tmp_path: Path, capsys):
    output = tmp_path / "run.csv"
    assert main(["simulate", "--steps", "2", "--seed", "4", "--output", str(output)]) == 0
    assert output.exists()
    assert '"steps": 2' in capsys.readouterr().out


def test_training_cli_preserves_numeric_summary_fields(monkeypatch, tmp_path: Path, capsys):
    result = TrainingResult(
        model_path=tmp_path / "model.zip",
        report_path=tmp_path / "report.csv",
        figure_path=tmp_path / "figure.png",
        evaluation_steps=200,
    )
    monkeypatch.setattr("algae_research.cli.train_ppo", lambda **_kwargs: result)
    assert main(["train-ppo", "--timesteps", "1", "--output-directory", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evaluation_steps"] == 200
    assert payload["model_path"].endswith("model.zip")
