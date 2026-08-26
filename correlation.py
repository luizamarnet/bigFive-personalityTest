"""Polychoric correlation using R."""

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd


def polychoric_correlation(df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Compute polychoric correlation matrix using R's psych package.

    Parameters
    ----------
    df_items : pd.DataFrame
        DataFrame with ordinal item responses.

    Returns
    -------
    pd.DataFrame
        Polychoric correlation matrix.
    """
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_items = ro.conversion.py2rpy(df_items)

    ro.globalenv["df"] = r_df_items
    n = df_items.shape[0]
    ro.globalenv["N"] = ro.IntVector([n])

    ro.r(
        """
        library(psych)
        pcor <- polychoric(df)$rho
        cat("\\n=== KMO ===\\n")
        kmo_result <- KMO(pcor)
        print(kmo_result$MSA)
        print(kmo_result$MSAi)
        cat("\\n=== Bartlett's Test ===\\n")
        bartlett_result <- cortest.bartlett(pcor, n = N)
        print(bartlett_result$chisq)
        print(bartlett_result$df)
        print(bartlett_result$p.value)
        """
    )

    with localconverter(ro.default_converter + pandas2ri.converter):
        pcor_matrix = ro.conversion.rpy2py(ro.r("pcor"))

    return pcor_matrix
