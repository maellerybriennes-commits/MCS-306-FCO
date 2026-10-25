"""Entry point for the collaborative AI mini-research starter project."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model import simulate_logistic_growth
from visualization import plot_population_trajectory, save_results_table


def run_starter_simulation() -> None:
    """Run a baseline numerical experiment and export plot + table outputs."""
    initial_population = 10.0
    intrinsic_growth_rate = 0.18
    carrying_capacity = 250.0
    simulation_steps = 40

    # Placeholder: tune these model parameters based on your dataset and hypothesis.
    population_history = simulate_logistic_growth(
        initial_population=initial_population,
        growth_rate=intrinsic_growth_rate,
        carrying_capacity=carrying_capacity,
        number_of_steps=simulation_steps,
    )
    time_points = np.arange(simulation_steps + 1)

    project_root = Path(__file__).resolve().parents[1]
    results_directory = project_root / "results"

    # Placeholder: swap these outputs for figures/tables aligned with your research question.
    plot_population_trajectory(time_points, population_history, results_directory / "population_trajectory.png")
    save_results_table(time_points, population_history, results_directory / "population_results.csv")

    print("Simulation complete. Outputs written to the results/ directory.")


if __name__ == "__main__":
    run_starter_simulation()
