"""Training pipeline for the Big Five personality model."""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DATA_FILE_PATH,
    FILTER_LONG_RESPONSE_TIMES,
    IQR_FACTOR,
    K_RANGE,
    MIN_TIME,
    MODEL_PATH,
    N_FACTORS,
    TEST_NUMBER_CLUSTERS,
    OPTIMAL_K,
)
from src.data.correlation import polychoric_correlation
from src.data.data_cleaner import clean_by_response_time
from src.data.data_loader import load_data
from src.models.clustering import cluster_and_visualize
from src.models.FactorAnalyzer import perform_factor_analysis
from src.models.number_clusters_choice import number_clusters_choice

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean_data(
    data_file_path: Path,
    min_time: int,
    iqr_factor: float,
    filter_long_response_times: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data and clean it."""
    df, df_items = load_data(data_file_path)
    df, df_items = clean_by_response_time(
        df,
        df_items,
        min_response_time=min_time,
        iqr_factor=iqr_factor,
        filter_long_response_times=filter_long_response_times,
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
    fa_model, factor_names = perform_factor_analysis(
        pcor_matrix, column_names, n_factors=N_FACTORS
    )
    return fa_model, factor_names


def compute_normalization_bounds(
    fa_model, df_items: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Compute min and max factor scores for normalization."""
    B = np.linalg.pinv(df_items) @ fa_model.transform(df_items)

    aux_df = np.zeros((50, fa_model.n_factors))
    aux_df[B < 0] = 1
    aux_df[B >= 0] = 5
    aux_df_trans = fa_model.transform(aux_df.T)

    factor_max = np.zeros(fa_model.n_factors)
    for i in range(fa_model.n_factors):
        factor_max[i] = aux_df_trans[i, i]

    aux_df = np.zeros((50, fa_model.n_factors))
    aux_df[B >= 0] = 1
    aux_df[B < 0] = 5
    aux_df_trans = fa_model.transform(aux_df.T)

    factor_min = np.zeros(fa_model.n_factors)
    for i in range(fa_model.n_factors):
        factor_min[i] = aux_df_trans[i, i]

    return factor_min, factor_max


def save_model(fa_model, factor_names, factor_min, factor_max, model_path: Path) -> None:
    """Save the trained model to disk."""
    model_to_save = {
        "model": fa_model,
        "factor_names": factor_names,
        "factor_min": factor_min,
        "factor_max": factor_max,
    }
    joblib.dump(model_to_save, model_path)
    logger.info(f"Model saved to {model_path}")


def run_clustering(df_items_transform: np.ndarray, factor_names: dict, k: int) -> None:
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

    cluster_and_visualize(df_items_transform, factor_display_names, k)
    logger.info("Clustering completed.")


def parse_args() -> argparse.Namespace:
    """Parse training settings, defaulting to values from ``src.config``."""
    parser = argparse.ArgumentParser(description="Train the Big Five personality model.")
    parser.add_argument(
        "--data-file-path",
        type=Path,
        default=DATA_FILE_PATH,
        help=f"Path to the training CSV (default: {DATA_FILE_PATH})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Path where the trained model is saved (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--min-time",
        type=int,
        default=MIN_TIME,
        help=f"Minimum response time in seconds (default: {MIN_TIME})",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=IQR_FACTOR,
        help=f"IQR factor for long response times (default: {IQR_FACTOR})",
    )
    parser.add_argument(
        "--filter-long-response-times",
        action=argparse.BooleanOptionalAction,
        default=FILTER_LONG_RESPONSE_TIMES,
        help="Whether to discard unusually long response times.",
    )
    parser.add_argument(
        "--optimal-k",
        type=int,
        default=OPTIMAL_K,
        help=f"Number of clusters when not testing K (default: {OPTIMAL_K})",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=N_FACTORS,
        help=f"Number of factors to fit (default: {N_FACTORS})",
    )
    parser.add_argument(
        "--k-range",
        type=int,
        default=K_RANGE,
        help=f"Maximum K to test when choosing clusters (default: {K_RANGE})",
    )
    parser.add_argument(
        "--test-number-clusters",
        action=argparse.BooleanOptionalAction,
        default=TEST_NUMBER_CLUSTERS,
        help="Whether to determine the number of clusters before clustering.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()

    _df, df_items = load_and_clean_data(
        data_file_path=args.data_file_path,
        min_time=args.min_time,
        iqr_factor=args.iqr_factor,
        filter_long_response_times=args.filter_long_response_times,
    )
    pcor_matrix = compute_correlation(df_items)
    fa_model, factor_names = perform_factor_analysis(
        pcor_matrix, df_items.columns, n_factors=args.n_factors
    )
    fa_model.mean_ = 0  # mean_.values
    fa_model.std_ = 1  # std_.values

    factor_min, factor_max = compute_normalization_bounds(fa_model, df_items)

    df_items_transform = fa_model.transform(df_items)
    df_items_transform = (df_items_transform - factor_min) / (factor_max - factor_min)

    save_model(fa_model, factor_names, factor_min, factor_max, args.model_path)

    if args.test_number_clusters:
        n_clusters = number_clusters_choice(df_items_transform, max_k=args.k_range)
        run_clustering(df_items_transform, factor_names, k=n_clusters)
    else:
        run_clustering(df_items_transform, factor_names, args.optimal_k)


if __name__ == "__main__":
    main()
