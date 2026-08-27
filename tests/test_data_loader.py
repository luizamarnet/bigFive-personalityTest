"""Tests for data_loader module."""

import pandas as pd
from pathlib import Path
from src.data.data_loader import load_data


def test_load_data_returns_two_dataframes():
    tmp_path=Path("./tests/")
    # Create a minimal CSV file
    data = {
        "IPC": [1, 2, 3],
        "EXT1": [1, 2, 3],
        "EXT1_E": [1000, 2000, 3000],
        "EXT2": [2, 3, 4],
        "EXT2_E": [1500, 2500, 3500],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, sep="\t", index=False)

    full_df, items_df = load_data(file_path)

    assert isinstance(full_df, pd.DataFrame)
    assert isinstance(items_df, pd.DataFrame)
    assert items_df.shape[1] == 2  # EXT1 and EXT2
