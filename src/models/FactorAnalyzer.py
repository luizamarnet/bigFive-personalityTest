"""Factor analysis utilities."""

import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import FactorAnalyzer
from src.config import N_FACTORS
from src.visualization.visualization import plot_scree, plot_heatmap
import logging

logger = logging.getLogger(__name__)



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
    logger.info("Starting factor analysis...")
    fa = FactorAnalyzer(n_factors=50, rotation="varimax", is_corr_matrix=True)
    fa.fit(correlation_matrix)
    ev, v = fa.get_eigenvalues()
    print(ev.sum())
    print(v.sum())

    plot_scree(ev)

    fa = FactorAnalyzer(n_factors=N_FACTORS, rotation="varimax", is_corr_matrix=True)
    fa.fit(correlation_matrix)

    #print("fa.get_factor_variance(): ", fa.get_factor_variance())
    #print("total var: ", np.sum(fa.get_factor_variance()[0]))

    loadings = fa.loadings_
    factor_names = [f"Factor {i+1}" for i in range(loadings.shape[1])]
    plot_heatmap(loadings, column_names, factor_names)

    df_fatores = np.abs(loadings)
    fator_dominante = df_fatores.argmax(axis=1)
    fator_variaveis = {}
    for fator_idx in range(loadings.shape[1]):
        variaveis = [column_names[i] for i in range(len(column_names)) if fator_dominante[i] == fator_idx]
        fator_variaveis[fator_idx] = variaveis

    fator_nomes = {}
    #print("fator_variaveis: ", fator_variaveis.items())
    for fator_idx, variaveis in fator_variaveis.items():
        prefixos = [v[:3] for v in variaveis]
        if prefixos:
            mais_comum = pd.Series(prefixos).value_counts().idxmax()
            fator_nomes[fator_idx] = mais_comum
        else:
            fator_nomes[fator_idx] = "NO_VARIABLES"

    #print("Assigned factor names:")
    #print(fator_nomes)

    return fa, fator_nomes
