"""Centralized plotting functions for the Big Five personality analysis."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

def plot_scree(eigenvalues: np.ndarray) -> None:
    """Plot scree plot."""
    plt.scatter(range(1, len(eigenvalues) + 1), eigenvalues)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues)
    plt.title("Scree Plot")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.show()


def plot_heatmap(loadings: np.ndarray, column_names: list[str], factor_names: list[str]) -> None:
    """Plot heatmap of factor loadings."""
    import seaborn as sns
    import pandas as pd

    df_loadings = pd.DataFrame(
        loadings,
        index=column_names,
        columns=factor_names,
    )

    plt.figure(figsize=(8, 4))
    sns.heatmap(df_loadings, annot=True, cmap="coolwarm", center=0, cbar_kws={"label": "Factor Loading"})
    plt.title("Heatmap of Factor Loadings")
    plt.ylabel("Variables")
    plt.xlabel("Factors")
    plt.tight_layout()
    plt.savefig("./results/factors_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    logger.info("Factors heatmap saved to: ./results/factors_heatmap.png")


def plot_boxplot(
    valid_times: list[np.ndarray],
    labels: list[str],
    min_time: int,
    max_time: float | None,
    iqr_factor: float,
) -> None:
    """Plot boxplot of response times."""
    flierprops = dict(marker="o", markerfacecolor="red", markersize=3, linestyle="none")

    plt.figure(figsize=(14, 6))
    plt.boxplot(valid_times, tick_labels=labels, showfliers=True, flierprops=flierprops, whis=iqr_factor)
    plt.title("Boxplot of valid response times per Big Five item")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Items")
    '''if max_time is not None:
        if max_time < 1800:
            plt.ylim(-1, max_time)
        else:
            plt.ylim(-1, 1800)
    else:
        plt.ylim(-1, 1800)'''
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results/boxplot_tempos_bigfive.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_radar_interactive(centroids: np.ndarray, factor_names: list[str]) -> None:
    """Plot interactive radar chart using Plotly."""
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
        title="Cluster Profiles",
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
    fig.write_html("./results/perfis_clusters.html", include_plotlyjs=True)
    logger.info("Interactive radar saved to: ./results/perfis_clusters.html")


def plot_radar_matplotlib(results: dict, lang: str = "en") -> None:
    """Plot radar chart using Matplotlib."""
    categories = list(results.keys())
    values = list(results.values())
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="purple", linewidth=2)
    ax.fill(angles, values, color="purple", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.set_ylim(0, 1)

    title = "Perfil Baseado nos traços de Personalidade" if lang == "pt" else "Profile Based on Personality Traits"
    ax.set_title(title, size=15, color="black", pad=20)
    plt.show()
