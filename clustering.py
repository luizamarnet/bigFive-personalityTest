"""K-means clustering and radar visualization."""

import joblib
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from config import K_OTIMO, KMEANS_MODEL_PATH


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
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_

    joblib.dump(kmeans, KMEANS_MODEL_PATH)

    n_factors = centroids.shape[1]
    radar_labels = factor_names + [factor_names[0]]

    fig = go.Figure()

    for i, centroid in enumerate(centroids):
        values = centroid.tolist()
        values += values[:1]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=radar_labels,
                mode="lines+markers",
                name=f"Cluster {i + 1}",
                fill="toself",
                opacity=0.35,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{theta}: %{r:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Perfis dos Clusters",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )
        ),
        legend=dict(title="Clusters", orientation="v"),
        template="plotly_white",
        width=900,
        height=700,
    )

    fig.show()
    fig.write_html("perfis_clusters.html", include_plotlyjs=True)
    print("Radar interativo salvo em: perfis_clusters.html")

    return clusters
