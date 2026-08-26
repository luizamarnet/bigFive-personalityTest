"""Training pipeline for the Big Five personality model."""

import os
import logging
import joblib
import numpy as np
import pandas as pd

from src.config import (
    DATA_FILE_PATH,
    MODEL_PATH,
    R_HOME,
    TEMPO_CURTO,
    FATOR_IQR,
    USAR_LIMITE_SUPERIOR,
    TEST_NUMBER_CLUSTERS,
)
from src.data.data_loader import load_data
from src.data.data_cleaner import clean_by_response_time
from src.data.correlation import polychoric_correlation
from src.models.FactorAnalyzer import perform_factor_analysis
from src.models.clustering import cluster_and_visualize
from src.models.number_clusters_choice import number_clusters_choice

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data and clean it."""
    df, df_items = load_data(DATA_FILE_PATH)
    df, df_items = clean_by_response_time(
        df,
        df_items,
        tempo_curto=TEMPO_CURTO,
        fator=FATOR_IQR,
        usar_limite_superior=USAR_LIMITE_SUPERIOR,
    )
    logger.info(f"Number of responses after cleaning: {len(df_items)}")
    return df, df_items


def compute_correlation(df_items: pd.DataFrame) -> pd.DataFrame:
    """Compute polychoric correlation matrix."""
    pcor_matrix = polychoric_correlation(df_items)
    logger.info(f"Shape of correlation matrix: {np.shape(pcor_matrix)}")
    return pcor_matrix


def fit_factor_analysis(pcor_matrix: pd.DataFrame, column_names: list[str]):
    """Fit factor analysis model."""
    fa_model, factor_names = perform_factor_analysis(pcor_matrix, column_names)
    return fa_model, factor_names


def compute_normalization_bounds(fa_model, df_items: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute min and max factor scores for normalization."""
    B = np.linalg.pinv(df_items) @ fa_model.transform(df_items)

    aux_df = np.zeros((50, 5))
    aux_df[B < 0] = 1
    aux_df[B >= 0] = 5
    aux_df_trans = fa_model.transform(aux_df.T)

    factor_max = np.zeros(5)
    for i in range(5):
        factor_max[i] = aux_df_trans[i, i]

    aux_df = np.zeros((50, 5))
    aux_df[B >= 0] = 1
    aux_df[B < 0] = 5
    aux_df_trans = fa_model.transform(aux_df.T)

    factor_min = np.zeros(5)
    for i in range(5):
        factor_min[i] = aux_df_trans[i, i]

    return factor_min, factor_max


def save_model(fa_model, factor_names, factor_min, factor_max) -> None:
    """Save the trained model to disk."""
    model_to_save = {
        "model": fa_model,
        "factor_names": factor_names,
        "factor_min": factor_min,
        "factor_max": factor_max,
    }
    joblib.dump(model_to_save, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")


def run_clustering(df_items_transform: np.ndarray, factor_names: dict) -> None:
    """Perform clustering and visualization."""
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

    cluster_and_visualize(df_items_transform, factor_display_names)
    logger.info("Clustering completed.")


def main() -> None:
    """Run the full training pipeline."""
    os.environ["R_HOME"] = R_HOME

    df, df_items = load_and_clean_data()
    pcor_matrix = compute_correlation(df_items)
    fa_model, factor_names = fit_factor_analysis(pcor_matrix, df_items.columns)
    fa_model.mean_ = 0#mean_.values
    fa_model.std_ = 1#std_.values

    factor_min, factor_max = compute_normalization_bounds(fa_model, df_items)

    df_items_transform = fa_model.transform(df_items)
    df_items_transform = (df_items_transform - factor_min) / (factor_max - factor_min)

    save_model(fa_model, factor_names, factor_min, factor_max)

    if TEST_NUMBER_CLUSTERS:
        n_clusters=number_clusters_choice(df_items_transform)
        run_clustering(df_items_transform, factor_names,k=n_clusters)
    else: 
        run_clustering(df_items_transform, factor_names)


if __name__ == "__main__":
    main()
