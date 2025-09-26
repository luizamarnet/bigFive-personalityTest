import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def clusterizacao(df_itens_transform):

    '''print(type(df_itens_transform))  # should output: <class 'numpy.ndarray'>
    # Optionally, check the new mean and std of each feature (column)
    print("Mean per feature:", df_itens_transform.mean(axis=0))
    print("Std per feature:", df_itens_transform.std(axis=0))

    # Normalize
    scaler = StandardScaler()
    df_itens_transform = scaler.fit_transform(df_itens_transform)

    print("Mean per feature:", df_itens_transform.mean(axis=0))
    print("Std per feature:", df_itens_transform.std(axis=0))'''
    
    # Testar diferentes números de clusters (k) para o método do cotovelo
    inertias = []
    k_range = range(1, 50)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_itens_transform)
        inertias.append(kmeans.inertia_)  # Soma das distâncias quadradas intra-cluster

    # Plot do cotovelo
    plt.figure(figsize=(6, 4))
    plt.plot(k_range, inertias, marker='o')
    plt.title('Método do Cotovelo para escolher k')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia (Soma das distâncias quadradas)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    k_otimo=5
    kmeans = KMeans(n_clusters=k_otimo, random_state=42)
    clusters = kmeans.fit(df_itens_transform)
    centroids = kmeans.cluster_centers_
    clusters = kmeans.predict(df_itens_transform)
    # Salvando o modelo
    joblib.dump(kmeans, 'modelo_kmeans.pkl')

    # Número de fatores (eixos)
    n_fatores = centroids.shape[1]
    angles = np.linspace(0, 2 * np.pi, n_fatores, endpoint=False).tolist()
    angles += angles[:1]  # Fechar o círculo

    # Nomes dos fatores
    labels = [f'Fator {i+1}' for i in range(n_fatores)]

    # Criar gráfico
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)

    # Adicionar cada cluster
    for i, centroide in enumerate(centroids):
        valores = centroide.tolist()
        valores += valores[:1]  # Fechar o círculo
        ax.plot(angles, valores, label=f'Cluster {i+1}')
        ax.fill(angles, valores, alpha=0.2)

    # Ajustes do gráfico
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Perfis dos Clusters (Centroides) - Gráfico Aranha')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

    return clusters