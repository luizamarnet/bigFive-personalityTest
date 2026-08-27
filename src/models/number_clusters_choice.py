import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src.config import K_RANGE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)

def number_clusters_choice (
    data: np.ndarray) -> int:

    logger.info("Testing the number of clusters...")

    max_k = K_RANGE
    k_range = range(2, max_k + 1)
    inertias=[]
    silhouette_scores=[]
    davies_bouldin_scores=[]
    for k in k_range:

        logger.info(f"Performing k-means with {k} clusters...")

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto"
        )

        clusters_k = kmeans.fit_predict(data)

        # Inércia funciona também para k=1
        inertias.append(kmeans.inertia_)

        # Silhouette e Davies-Bouldin precisam de pelo menos 2 clusters
        if k >= 2:
            silhouette_scores.append(
                silhouette_score(data, clusters_k)
            )

            davies_bouldin_scores.append(
                davies_bouldin_score(data, clusters_k)
            )

    # ============================================================
    # 3. Plot das métricas para escolher K
    # ============================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    # ----------------------------
    # Inércia
    # ----------------------------

    axes[0].plot(
        list(k_range),
        inertias,
        marker="o"
    )

    axes[0].set_title("Método do Cotovelo")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inércia")
    axes[0].set_xticks(list(k_range))
    axes[0].grid(True)

    # ----------------------------
    # Silhouette
    # ----------------------------

    silhouette_k_range = list(range(2, max_k + 1))

    axes[1].plot(
        silhouette_k_range,
        silhouette_scores,
        marker="o"
    )

    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Silhouette")
    axes[1].set_xticks(list(k_range))
    axes[1].grid(True)

    # Melhor silhouette
    best_silhouette_k = (
        silhouette_k_range[
            np.argmax(silhouette_scores)
        ]
    )

    axes[1].axvline(
        best_silhouette_k,
        linestyle="--",
        alpha=0.7
    )

    axes[1].text(
        best_silhouette_k,
        max(silhouette_scores),
        f" Melhor k = {best_silhouette_k}"
    )

    # ----------------------------
    # Davies-Bouldin
    # ----------------------------

    axes[2].plot(
        silhouette_k_range,
        davies_bouldin_scores,
        marker="o"
    )

    axes[2].set_title("Davies-Bouldin Index")
    axes[2].set_xlabel("Número de clusters (k)")
    axes[2].set_ylabel("Davies-Bouldin")
    axes[2].set_xticks(list(k_range))
    axes[2].grid(True)

    # Melhor DB = menor valor
    best_db_k = (
        silhouette_k_range[
            np.argmin(davies_bouldin_scores)
        ]
    )

    axes[2].axvline(
        best_db_k,
        linestyle="--",
        alpha=0.7
    )

    axes[2].text(
        best_db_k,
        min(davies_bouldin_scores),
        f" Melhor k = {best_db_k}"
    )

    plt.tight_layout()
    plt.savefig("./results/number_clusters_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    logger.info("Number of clusters analysis saved to: ./results/number_clusters_analysis.png")
    
    
    print("\nBased on these plots, how many clusters would you want to choose?")
    number_clusters_chosen_by_user = int(input("> "))

    return number_clusters_chosen_by_user