"""Tests for correlation module."""

import numpy as np
import pandas as pd
from src.data.correlation import polychoric_correlation

def test_perform_correlation_matrix_format():

    dataset = pd.DataFrame({
            "EXT1": [1, 2, 3, 1, 4, 2, 1, 3, 5],
            "EXT2": [1, 2, 3, 1, 4, 2, 1, 3, 5],
            "EXT3": [1, 2, 3, 1, 4, 2, 1, 3, 5],
        })
    
    correlation_matrix = polychoric_correlation(dataset)

    assert isinstance(correlation_matrix, np.ndarray)
    assert correlation_matrix.dtype == np.float64
    assert correlation_matrix.shape == (len(dataset.columns),len(dataset.columns))

