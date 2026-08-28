"""Polychoric correlation using R."""

import logging
import rpy2.robjects as ro
import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def polychoric_correlation(df_items: pd.DataFrame) -> NDArray[np.float64]:
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
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df_items = ro.conversion.get_conversion().py2rpy(df_items)

        ro.globalenv["df"] = r_df_items
        n = df_items.shape[0]
        ro.globalenv["N"] = ro.IntVector([n])

        ro.r(
            """
            library(psych)
            pcor <- polychoric(df)$rho
            kmo_result <- KMO(pcor)
            bartlett_result <- cortest.bartlett(pcor, n = N)
            """
        )
        

        with localconverter(ro.default_converter + pandas2ri.converter):
            pcor_matrix = ro.conversion.get_conversion().rpy2py(ro.r("pcor"))

            kmo_result = ro.conversion.get_conversion().rpy2py(ro.r("kmo_result$MSA"))
            bartlett_result = ro.conversion.get_conversion().rpy2py(ro.r("bartlett_result$p.value"))

        if kmo_result<0.6:
                raise ValueError(
                    "This dataset is not adequate for Factor Analysis. The overall KMO value for the dataset is bellow 0.6"
                )
        if bartlett_result[0]>= 0.05:
            raise ValueError(
                "P-value for Bartlett's test greater than 0.05. Meaning the variables are uncorrelated and unsuitable for factor analysis."
            )
        


        return pcor_matrix
    except Exception as e:
        logger.error(f"Error computing polychoric correlation: {e}")
        raise



#library(psych)
#pcor <- polychoric(df)$rho
#cat("\\n=== KMO ===\\n")
#kmo_result <- KMO(pcor)
#print(kmo_result$MSA)
#print(kmo_result$MSAi)
#cat("\\n=== Bartlett's Test ===\\n")
#bartlett_result <- cortest.bartlett(pcor, n = N)
#print(bartlett_result$chisq)
#print(bartlett_result$df)
#print(bartlett_result$p.value)
