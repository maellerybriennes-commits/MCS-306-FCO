"""Numerical model utilities for the mini-research starter project."""

from __future__ import annotations

import numpy as np


def logistic_growth_step(current_population: float, growth_rate: float, carrying_capacity: float) -> float:
    """Compute one Euler-style update for the logistic growth model."""
    # Placeholder: replace or extend this equation if your project uses a different model.
    rate_of_change = growth_rate * current_population * (1 - current_population / carrying_capacity)
    return current_population + rate_of_change


def simulate_logistic_growth(
    initial_population: float,
    growth_rate: float,
    carrying_capacity: float,
    number_of_steps: int,
) -> np.ndarray:
    """Simulate a population trajectory using a simple iterative update."""
    population_history = np.zeros(number_of_steps + 1, dtype=float)
    population_history[0] = initial_population

    for step_index in range(number_of_steps):
        # Placeholder: if needed, replace with a more advanced integrator or stochastic update.
        population_history[step_index + 1] = logistic_growth_step(
            current_population=population_history[step_index],
            growth_rate=growth_rate,
            carrying_capacity=carrying_capacity,
        )

    return population_history
