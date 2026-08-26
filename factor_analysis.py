"""Factor analysis utilities."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
from config import N_FACTORS


def perform_factor_analysis(
    correlation_matrix: pd.DataFrame, column_names: list[str]
) -> tuple[FactorAnalyzer, dict]:
    """
    Perform factor analysis on a correlation matrix.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Polychoric correlation matrix.
    column_names : list[str]
        Names of the 50 personality items.

    Returns
    -------
    tuple[FactorAnalyzer, dict]
        Fitted FactorAnalyzer model and a dict mapping factor index to dominant trait prefix.
    """
    fa = FactorAnalyzer(n_factors=50, rotation="varimax", is_corr_matrix=True)
    fa.fit(correlation_matrix)
    ev, v = fa.get_eigenvalues()
    print(ev.sum())
    print(v.sum())

    plt.scatter(range(1, correlation_matrix.shape[1] + 1), ev)
    plt.plot(range(1, correlation_matrix.shape[1] + 1), ev)
    plt.title("Scree Plot")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.show()

    fa = FactorAnalyzer(n_factors=N_FACTORS, rotation="varimax", is_corr_matrix=True)
    fa.fit(correlation_matrix)

    print("fa.get_factor_variance(): ", fa.get_factor_variance())
    print("total var: ", np.sum(fa.get_factor_variance()[0]))

    loadings = pd.DataFrame(
        fa.loadings_,
        index=column_names,
        columns=[f"Fator {i+1}" for i in range(fa.loadings_.shape[1])],
    )

    plt.figure(figsize=(8, 4))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0, cbar_kws={"label": "Carga Fatorial"})
    plt.title("Heatmap das Cargas Fatoriais")
    plt.ylabel("Variáveis")
    plt.xlabel("Fatores")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.heatmap(np.abs(loadings), annot=True, cmap="coolwarm", center=0, cbar_kws={"label": "Carga Fatorial"})
    plt.title("Heatmap das Cargas Fatoriais")
    plt.ylabel("Variáveis")
    plt.xlabel("Fatores")
    plt.tight_layout()
    plt.show()

    df_fatores = np.abs(loadings)
    fator_dominante = df_fatores.idxmax(axis=1)
    fator_variaveis = {}
    for fator in df_fatores.columns:
        variaveis = df_fatores.index[fator_dominante == fator]
        fator_variaveis[fator] = list(variaveis)

    fator_nomes = {}
    print("fator_variaveis: ", fator_variaveis.items())
    for fator, variaveis in fator_variaveis.items():
        prefixos = [v[:3] for v in variaveis]
        if prefixos:
            mais_comum = pd.Series(prefixos).value_counts().idxmax()
            fator_nomes[fator] = mais_comum
        else:
            fator_nomes[fator] = "SEM_VARIAVEIS"

    print("Nomes atribuídos aos fatores:")
    print(fator_nomes)

    return fa, fator_nomes
