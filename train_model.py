"""Training pipeline for the Big Five personality model."""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_FILE_PATH,
    MODEL_PATH,
    N_GENERATIONS,
    POP_SIZE,
    R_HOME,
    TEMPO_CURTO,
    FATOR_IQR,
    USAR_LIMITE_SUPERIOR,
)
from data_loader import load_data
from data_cleaner import clean_by_response_time
from correlation import polychoric_correlation
from factor_analysis import perform_factor_analysis
from genetic_optimizer import find_optimal_individual
from clustering import cluster_and_visualize


def main() -> None:
    """Run the full training pipeline."""
    os.environ["R_HOME"] = R_HOME

    df, df_items = load_data(DATA_FILE_PATH)
    df, df_items = clean_by_response_time(
        df,
        df_items,
        tempo_curto=TEMPO_CURTO,
        fator=FATOR_IQR,
        usar_limite_superior=USAR_LIMITE_SUPERIOR,
    )

    print("Number of responses after cleaning: ", len(df))
    print("Number of responses after cleaning: ", len(df_items))

    df = df.sort_index()
    df_items = df_items.loc[df.index]
    print("Check that both datasets have same indices in same order: ", (df_items.index == df.index).all())

    pcor_matrix = polychoric_correlation(df_items)
    print("Shape of correlation matrix: ", np.shape(pcor_matrix))

    fa_model, factor_names = perform_factor_analysis(pcor_matrix, df_items.columns)

    mean_ = df_items.mean(axis=0)
    std_ = df_items.std(axis=0)

    # Normalization using genetic algorithm
    factor_max = np.zeros(5)
    factor_min = np.zeros(5)

    for factor_idx in range(5):
        # Maximize
        _, _, objective_max = find_optimal_individual(
            fa_model,
            mode="factor",
            factor_idx=factor_idx,
            minimize=False,
            n_generations=N_GENERATIONS,
            pop_size=POP_SIZE,
        )
        factor_max[factor_idx] = objective_max

        # Minimize
        _, _, objective_min = find_optimal_individual(
            fa_model,
            mode="factor",
            factor_idx=factor_idx,
            minimize=True,
            n_generations=N_GENERATIONS,
            pop_size=POP_SIZE,
        )
        factor_min[factor_idx] = objective_min

    print("Factor max values:", factor_max)
    print("Factor min values:", factor_min)

    # Transform and normalize
    df_items_transform = fa_model.transform(df_items)
    df_items_transform = (df_items_transform - factor_min) / (factor_max - factor_min)

    # Save model
    model_to_save = {
        "model": fa_model,
        "nome_fatores": factor_names,
        "fatores_minimos": factor_min,
        "fatores_maximos": factor_max,
    }
    joblib.dump(model_to_save, MODEL_PATH)

    # Clustering
    trait_names = {
        "EXT": "Extraversion",
        "EST": "Neuroticism",
        "AGR": "Agreeableness",
        "CSN": "Conscientiousness",
        "OPN": "Openness",
    }
    factor_display_names = [
        trait_names.get(prefix, prefix) for prefix in factor_names.values()
    ]

    clusters = cluster_and_visualize(df_items_transform, factor_display_names)

    print("Clustering completed.")


if __name__ == "__main__":
    main()
