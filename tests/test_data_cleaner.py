"""Tests for data_cleaner module."""

import pandas as pd
from data_cleaner import clean_by_response_time


def test_clean_by_response_time_removes_short_times():
    df = pd.DataFrame({
        "EXT1_E": [500, 2000, 3000],
        "EXT2_E": [1000, 2500, 3500],
    })
    df_items = pd.DataFrame({
        "EXT1": [1, 2, 3],
        "EXT2": [2, 3, 4],
    })

    cleaned_df, cleaned_items = clean_by_response_time(
        df, df_items, tempo_curto=2, fator=1.5, usar_limite_superior=False
    )

    # Only rows with time >= 2 seconds remain
    assert len(cleaned_df) == 2
    assert len(cleaned_items) == 2
