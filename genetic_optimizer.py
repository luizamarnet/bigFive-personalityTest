"""Genetic algorithm for finding extreme factor scores."""

import random
import numpy as np
from deap import base, creator, tools, algorithms
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def _create_evaluator(
    fa_model,
    mode: str = "sum",
    factor_idx: int | None = None,
    minimize: bool = False,
):
    """Create an evaluation function for the genetic algorithm."""

    def evaluate(individual):
        factors = fa_model.transform([individual])[0]

        if mode == "sum":
            result = sum(factors)
        elif mode == "factor" and factor_idx is not None:
            result = factors[factor_idx]
        else:
            raise ValueError("Invalid mode or factor index not provided.")

        return (-result,) if minimize else (result,)

    return evaluate


def find_optimal_individual(
    fa_model,
    mode: str = "sum",
    factor_idx: int | None = None,
    minimize: bool = False,
    n_generations: int = 50,
    pop_size: int = 100,
) -> tuple[list, np.ndarray, float]:
    """
    Use a genetic algorithm to find the 50-answer vector that maximizes/minimizes a factor score.

    Parameters
    ----------
    fa_model : FactorAnalyzer
        Fitted factor analysis model.
    mode : str, optional
        "sum" or "factor" (default "sum").
    factor_idx : int | None, optional
        Index of factor to optimize when mode="factor".
    minimize : bool, optional
        If True, minimize the objective (default False).
    n_generations : int, optional
        Number of generations (default 50).
    pop_size : int, optional
        Population size (default 100).

    Returns
    -------
    tuple[list, np.ndarray, float]
        Best individual (list of 50 ints), factor scores, and objective value.
    """
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=50)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    evaluator = _create_evaluator(fa_model, mode, factor_idx, minimize)
    toolbox.register("evaluate", evaluator)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=n_generations,
        halloffame=hof,
        verbose=False,
    )

    best_individual = hof[0]
    factors = fa_model.transform([best_individual])[0]
    objective = sum(factors) if mode == "sum" else factors[factor_idx]

    return best_individual, factors, objective
