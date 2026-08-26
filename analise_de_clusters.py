import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import joblib
import plotly.graph_objects as go


def clusterizacao(df_itens_transform,nomes):

    # ============================================================
    # 1. Dados
    # ============================================================

    # Nome das colunas = nomes dos fatores
    labels = nomes

    # Converte para numpy para o sklearn
    X = df_itens_transform

    # Número máximo de clusters
    max_k = 15#min(20, len(X) - 1)
    k_range = range(1, max_k + 1)

    # ============================================================
    # 2. Avaliação de diferentes valores de K
    # ============================================================

    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []

    '''for k in k_range:

        print("k: ", k)

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto"
        )

        clusters_k = kmeans.fit_predict(X)

        # Inércia funciona também para k=1
        inertias.append(kmeans.inertia_)

        # Silhouette e Davies-Bouldin precisam de pelo menos 2 clusters
        if k >= 2:
            silhouette_scores.append(
                silhouette_score(X, clusters_k)
            )

            davies_bouldin_scores.append(
                davies_bouldin_score(X, clusters_k)
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
    plt.show()
    
    print(f"Melhor Silhouette: k = {best_silhouette_k}")
    print(f"Melhor Davies-Bouldin: k = {best_db_k}")'''

    # ============================================================
    # 4. Escolher K
    # ============================================================

    # Você pode alterar isso depois de olhar os gráficos
    k_otimo = 5

    print(f"\nK escolhido: {k_otimo}")
    

    # ============================================================
    # 5. Treinar modelo final
    # ============================================================

    kmeans = KMeans(
        n_clusters=k_otimo,
        random_state=42,
        n_init="auto"
    )

    clusters = kmeans.fit_predict(X)

    centroids = kmeans.cluster_centers_

    # ============================================================
    # 6. Salvar modelo
    # ============================================================

    joblib.dump(
        kmeans,
        "modelo_kmeans.pkl"
    )

    # ============================================================
    # 7. Radar plot interativo
    # ============================================================

    # 8. Preparar radar
    # ============================================================

    # Número de fatores
    n_fatores = centroids.shape[1]

    # Fechar o radar repetindo o primeiro fator
    radar_labels = labels + [labels[0]]

    # ============================================================
    # 9. Criar radar Plotly
    # ============================================================

    fig = go.Figure()

    for i, centroide in enumerate(centroids):

        valores = centroide.tolist()

        # Fecha o polígono
        valores += valores[:1]

        fig.add_trace(
            go.Scatterpolar(
                r=valores,
                theta=radar_labels,

                mode="lines+markers",

                name=f"Cluster {i + 1}",

                fill="toself",

                opacity=0.35,

                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{theta}: %{r:.2f}"
                    "<extra></extra>"
                )
            )
        )

    # ============================================================
    # 10. Configurar radar
    # ============================================================

    fig.update_layout(

        title="Perfis dos Clusters",

        polar=dict(

            radialaxis=dict(

                # Mostrar eixo
                visible=True,

                # IMPORTANTE:
                # eixo sempre vai de 0 até 1
                range=[0, 1],

                # Marcações
                tickvals=[
                    0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0
                ]
            )
        ),

        legend=dict(
            title="Clusters",
            orientation="v"
        ),

        template="plotly_white",

        width=900,
        height=700
    )

    # ============================================================
    # 11. Mostrar radar interativo
    # ============================================================

    fig.show()

    # ============================================================
    # 12. Salvar radar interativo
    # ============================================================

    fig.write_html(
        "perfis_clusters.html",
        include_plotlyjs=True
    )

    print(
        "Radar interativo salvo em: perfis_clusters.html"
    )

    # ============================================================
    # 13. Retornar clusters
    # ============================================================

    return clusters
