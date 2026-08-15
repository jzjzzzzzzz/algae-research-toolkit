from pathlib import Path

import pytest

from algae_research.analysis.growth import plot_growth
from algae_research.analysis.turbidity import summarize_turbidity, write_turbidity_summary


def test_summary_retains_days_and_calculates_standard_error():
    summary = summarize_turbidity()
    assert summary["Day"].tolist() == [0, 3, 8, 9, 10, 11, 12]
    assert summary.loc[0, "Silent Mean"] == pytest.approx((8063 + 8184 + 8090) / 3)
    assert summary.loc[0, "Silent SE"] > 0


def test_summary_and_plot_can_be_written_headlessly(tmp_path: Path):
    csv_path = write_turbidity_summary(tmp_path / "summary.csv")
    image_path = tmp_path / "growth.png"
    plot_growth([0.1, 0.2], {"light": [0.5, 0.6]}, destination=image_path)
    assert csv_path.exists()
    assert image_path.exists()


def test_growth_plot_rejects_mismatched_factor_length():
    with pytest.raises(ValueError, match="length"):
        plot_growth([0.1, 0.2], {"light": [0.5]})
