"""Factor analysis utilities."""

import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import FactorAnalyzer
from src.config import N_FACTORS
from src.visualization.visualization import plot_scree, plot_heatmap
import logging

logger = logging.getLogger(__name__)



def perform_factor_analysis(
    correlation_matrix: np.ndarray, column_names: list[str], n_factors: int = N_FACTORS
) -> tuple[FactorAnalyzer, dict[int,str]]:
    """
    Perform factor analysis on a correlation matrix.

    Parameters
    ----------
    correlation_matrix : np.ndarray (n_variables, n_variable) / n_variable = 50 (the 50 personality items)
        Polychoric correlation matrix.
    column_names : list[str]
        Names of the 50 personality items.
    n_factors: int
        Number of factors to choose from the factorial analysis 

    Returns
    -------
    fa : FactorAnalyzer
        Fitted FactorAnalyzer model  
    factor_names: dict
        Dictionary mapping factor index to dominant trait question prefix.
    """
    logger.info("Starting factor analysis...")
    fa = FactorAnalyzer(n_factors=len(column_names), rotation="varimax", is_corr_matrix=True)
    fa.fit(correlation_matrix)
    ev, v = fa.get_eigenvalues()
    #print(ev.sum())
    #print(v.sum())

    plot_scree(ev)

    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", is_corr_matrix=True)
    fa.fit(correlation_matrix)

    #print("fa.get_factor_variance(): ", fa.get_factor_variance())
    #print("total var: ", np.sum(fa.get_factor_variance()[0]))

    loadings = fa.loadings_

    df_fatores = np.abs(loadings)
    fator_dominante = df_fatores.argmax(axis=1)
    fator_variaveis = {}
    for fator_idx in range(loadings.shape[1]):
        variaveis = [column_names[i] for i in range(len(column_names)) if fator_dominante[i] == fator_idx]
        fator_variaveis[fator_idx] = variaveis

    factor_names = {}
    #print("fator_variaveis: ", fator_variaveis.items())
    for fator_idx, variaveis in fator_variaveis.items():
        prefixos = [v[:3] for v in variaveis]
        if prefixos:
            mais_comum = pd.Series(prefixos).value_counts().idxmax()
            factor_names[fator_idx] = mais_comum
        else:
            factor_names[fator_idx] = "NO_VARIABLES"

    #print("factor_names: ", factor_names)
    plot_heatmap(loadings, column_names, factor_names.values())


    return fa, factor_names
