"""K-means clustering and radar visualization."""

import joblib
import numpy as np
from sklearn.cluster import KMeans
from src.config import K_OTIMO, KMEANS_MODEL_PATH
from src.visualization.visualization import plot_radar_interactive


def cluster_and_visualize(
    data: np.ndarray, factor_names: list[str], k: int = K_OTIMO
) -> np.ndarray:
    """
    Perform K-means clustering and generate an interactive radar chart.

    Parameters
    ----------
    data : np.ndarray
        Normalized factor scores (n_samples, n_factors).
    factor_names : list[str]
        Names of the factors.
    k : int, optional
        Number of clusters (default from config).

    Returns
    -------
    np.ndarray
        Cluster labels for each sample.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_

    joblib.dump(kmeans, KMEANS_MODEL_PATH)

    plot_radar_interactive(centroids, factor_names)

    return clusters
