"""Grouped turbidity/lux analysis retained from the sound experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MEASUREMENTS = {
    "Day": [0, 3, 8, 9, 10, 11, 12],
    "1": [8063, 5890, 6098, 5781, 5678, 5930, 5102],
    "2": [8184, 5850, 5962, 5882, 5593, 5642, 5423],
    "3": [8090, 5830, 5935, 5657, 5520, 5576, 5520],
    "X": [8142, 5567, 5725, 5541, 5523, 5255, 5080],
    "Y": [8063, 5115, 5651, 5505, 5720, 5620, 5116],
    "Z": [7969, 5314, 5425, 5243, 5077, 5329, 4996],
    "A": [8050, 5635, 5536, 5535, 5203, 5412, 5141],
    "B": [7950, 5492, 5620, 5790, 5667, 5845, 5026],
    "C": [8286, 5990, 5673, 5780, 5589, 5681, 5173],
    "alpha": [8023, 5619, 5419, 5525, 5701, 5163, 4894],
    "beta": [8171, 5934, 5725, 5451, 5472, 5538, 5382],
    "gamma": [7935, 5881, 5525, 5482, 5368, 5403, 5286],
}

GROUPS = {
    "Silent": ("1", "2", "3"),
    "200 Hz": ("X", "Y", "Z"),
    "10,000 Hz": ("A", "B", "C"),
    "20,000 Hz": ("alpha", "beta", "gamma"),
}


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Turbidity analysis requires pandas. Install with "
            "pip install 'algae-research-toolkit[analysis]'."
        ) from exc
    return pd


def build_turbidity_frame():
    """Return the original measurements as a new DataFrame."""

    return _pandas().DataFrame(MEASUREMENTS)


def summarize_turbidity(frame=None):
    """Calculate group means and standard errors for each measured day."""

    data = build_turbidity_frame() if frame is None else frame.copy()
    required_columns = [
        "Day",
        *(column for group_columns in GROUPS.values() for column in group_columns),
    ]
    missing_columns = sorted(set(required_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Turbidity frame is missing required columns: {missing_columns}")

    try:
        data[required_columns] = data[required_columns].apply(
            _pandas().to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Turbidity days and measurements must be numeric.") from exc
    if data["Day"].isna().any() or np.isinf(data[required_columns].to_numpy(dtype=float)).any():
        raise ValueError("Turbidity days and measurements must be finite when present.")

    summary = data[["Day"]].copy()
    for group_name, columns in GROUPS.items():
        observations = data[list(columns)]
        observed_counts = observations.count(axis=1)
        summary[f"{group_name} Mean"] = observations.mean(axis=1)
        summary[f"{group_name} SE"] = observations.std(axis=1).div(observed_counts.pow(0.5))
    return summary


def write_turbidity_summary(destination: str | Path) -> Path:
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summarize_turbidity().to_csv(output, index=False)
    return output


def plot_turbidity(destination: str | Path | None = None, *, show: bool = False):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install with "
            "pip install 'algae-research-toolkit[analysis]'."
        ) from exc

    summary = summarize_turbidity()
    colors = {"Silent": "teal", "200 Hz": "red", "10,000 Hz": "orange", "20,000 Hz": "black"}
    offsets = {"Silent": -0.18, "200 Hz": -0.06, "10,000 Hz": 0.06, "20,000 Hz": 0.18}
    figure, axis = plt.subplots(figsize=(12, 7))
    for group_name in GROUPS:
        axis.errorbar(
            summary["Day"] + offsets[group_name],
            summary[f"{group_name} Mean"],
            yerr=summary[f"{group_name} SE"],
            marker="o",
            capsize=3,
            linewidth=2,
            color=colors[group_name],
            label=group_name,
        )
    axis.set(title="Various Sound Frequencies on Algae Turbidity", xlabel="Day", ylabel="Lux")
    axis.invert_yaxis()
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    if destination is not None:
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure
