"""Tests for factor_analysis module."""

import numpy as np
import pandas as pd
from factor_analysis import perform_factor_analysis


def test_perform_factor_analysis_returns_model_and_names():
    # Create a simple correlation matrix
    n = 10
    corr = np.eye(n)
    corr[0, 1] = corr[1, 0] = 0.8
    corr[2, 3] = corr[3, 2] = 0.7
    corr[4, 5] = corr[5, 4] = 0.6
    corr[6, 7] = corr[7, 6] = 0.5
    corr[8, 9] = corr[9, 8] = 0.4

    corr_df = pd.DataFrame(corr)
    column_names = [f"VAR{i}" for i in range(n)]

    fa_model, factor_names = perform_factor_analysis(corr_df, column_names)

    assert fa_model is not None
    assert isinstance(factor_names, dict)
    assert len(factor_names) == 5  # N_FACTORS
