"""Plot simulation or policy growth histories."""

from __future__ import annotations

from pathlib import Path


def plot_growth(
    algae_history,
    factor_histories=None,
    *,
    title: str = "Algae Growth and Environmental Factors",
    destination: str | Path | None = None,
    show: bool = False,
):
    if len(algae_history) == 0:
        raise ValueError("algae_history cannot be empty.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install with "
            "pip install 'algae-research-toolkit[analysis]'."
        ) from exc

    steps = range(1, len(algae_history) + 1)
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(steps, algae_history, label="Algae", color="green", linewidth=2)
    for name, values in (factor_histories or {}).items():
        if len(values) != len(algae_history):
            raise ValueError(f"Factor {name!r} length does not match algae history.")
        axis.plot(steps, values, label=name, linestyle="--")
    axis.set(xlabel="Step", ylabel="Normalized value / algae amount", title=title)
    axis.legend()
    axis.grid(True, alpha=0.25)
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

