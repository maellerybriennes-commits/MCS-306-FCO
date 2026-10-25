"""Utilities for plotting and exporting mini-research results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_population_trajectory(time_points, population_history, output_path: Path) -> None:
    """Generate and save a line chart for the simulated trajectory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(time_points, population_history, color="tab:blue", linewidth=2)
    axis.set_title("Starter Simulation Output")
    axis.set_xlabel("Time step")
    axis.set_ylabel("Population")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_results_table(time_points, population_history, output_path: Path) -> None:
    """Export simulation values to a CSV table for downstream analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_table = pd.DataFrame(
        {
            "time_step": time_points,
            "population": population_history,
        }
    )
    results_table.to_csv(output_path, index=False)
