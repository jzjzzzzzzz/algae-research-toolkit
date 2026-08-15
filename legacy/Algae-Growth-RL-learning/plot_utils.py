##################################################
###### Copyright (c) 2026 jzjzzzzzzz
###### All rights reserved.
##################################################

import matplotlib.pyplot as plt

import itertools

def plot_growth(
        algae_history,
        factor_histories=None,
        factor_names=None,
        title="Algae Growth and Environmental Factors",
        save_path=None,
        show=True
):
    """
    Plot algae growth over time, optionally with environmental factors.

    Parameters:
    - algae_history: list of algae amounts per step
    - factor_histories: dict of factor_name -> list of values per step
    - factor_names: optional list of factor names to plot
    - title: plot title
    - save_path: if provided, save figure to this path
    - show: whether to display the plot
    """

    if not algae_history:
        print("No algae history data to plot!")
        return

    steps = range(1, len(algae_history) + 1)

    plt.figure(figsize=(12, 6))

    # Plot algae growth
    plt.plot(steps, algae_history, label="Algae", color="green", linewidth=2)

    # Plot environmental factors if provided
    if factor_histories:
        if factor_names is None:
            factor_names = list(factor_histories.keys())
        # Automatically cycle through colors
        colors = itertools.cycle(["orange", "blue", "red", "purple", "brown", "cyan", "magenta", "black"])
        for name in factor_names:
            if name in factor_histories:
                plt.plot(steps, factor_histories[name], label=name, linestyle="--", color=next(colors))
            else:
                print(f"Warning: Factor '{name}' not found in factor_histories!")

    plt.xlabel("Step")
    plt.ylabel("Value / Algae Amount")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

    # Display the plot
    if show:
        plt.show()
    else:
        plt.close()
