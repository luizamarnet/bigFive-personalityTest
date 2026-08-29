"""Data loading utilities."""

from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data from CSV and return the full DataFrame and the 50 personality items.

    Parameters
    ----------
    data_path : Path
        Path to the raw data file.

    Returns
    -------
    df : pd.DataFrame
        Full dataframe read from the file containing only non empty valid answers
    df_items : pd.DataFrame
        similar to df, but contains only the answers for the 50 personality questions.
    """
    try:
        df = pd.read_csv(data_path, sep="\t", index_col=False)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise

    # Remove responses from same IP
    ipc_counts = df["IPC"].value_counts()
    for ipc_value, count in ipc_counts.items():
        if ipc_value > 1 and count % ipc_value != 0:
            logger.warning(f"IPC = {ipc_value} appears {count} times (not a multiple)")

    logger.info(f"Number of responses: {len(df)}")
    aux = len(df)
    df = df[df["IPC"] < 2]
    logger.info(f"Responses removed due to same IP: {aux - len(df)}")

    # Remove missing values
    aux = len(df)
    df = df.dropna()
    df = df[~df.isin(["NONE"]).any(axis=1)]
    logger.info(f"Responses removed due to NaN: {aux - len(df)}")

    # Select the 50 personality items
    columns = [
        col for col in df.columns
        if (
            col.startswith("EXT")
            or col.startswith("EST")
            or col.startswith("AGR")
            or col.startswith("CSN")
            or col.startswith("OPN")
        )
        and not col.endswith("_E")
    ]
    df = df[df[columns].isin([1, 2, 3, 4, 5]).all(axis=1)]
    if df.empty:
        raise ValueError(
            "The dataset has no rows with valid data. "
            "The data must contain integers with values 1, 2, 3, 4, or 5."
        )
        
    df_items = df[columns]

    logger.info(f"Number of variables: {len(df.columns)}")
    logger.info(f"Number of selected variables: {len(df_items.columns)}")
    logger.info(f"Number of responses after cleaning: {len(df_items)}")

    return df, df_items
