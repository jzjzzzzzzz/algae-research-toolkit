from math import isnan
from pathlib import Path

import pytest

from algae_research.analysis.growth import plot_growth
from algae_research.analysis.turbidity import (
    build_turbidity_frame,
    summarize_turbidity,
    write_turbidity_summary,
)


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


def test_summary_uses_observed_replicate_count_for_standard_error():
    frame = build_turbidity_frame()
    frame.loc[0, "3"] = None

    summary = summarize_turbidity(frame)

    expected = frame.loc[0, ["1", "2"]].std() / (2**0.5)
    assert summary.loc[0, "Silent SE"] == pytest.approx(expected)


def test_summary_leaves_standard_error_undefined_with_one_observation():
    frame = build_turbidity_frame()
    frame.loc[0, ["2", "3"]] = None

    summary = summarize_turbidity(frame)

    assert isnan(summary.loc[0, "Silent SE"])


def test_summary_rejects_missing_or_non_numeric_required_data():
    missing = build_turbidity_frame().drop(columns="3")
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_turbidity(missing)

    non_numeric = build_turbidity_frame()
    non_numeric["1"] = non_numeric["1"].astype(object)
    non_numeric.loc[0, "1"] = "not-a-number"
    with pytest.raises(ValueError, match="must be numeric"):
        summarize_turbidity(non_numeric)
