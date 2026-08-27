"""Data cleaning based on response time."""

import pandas as pd
from src.config import MIN_TIME, IQR_FACTOR, FILTER_LONG_RESPONSE_TIMES
from src.visualization.visualization import plot_boxplot


def clean_by_response_time(
    df: pd.DataFrame,
    df_items: pd.DataFrame,
    min_response_time: int = MIN_TIME,
    max_response_time: float | None = None,
    filter_long_response_times: bool = FILTER_LONG_RESPONSE_TIMES,
    iqr_factor: float = IQR_FACTOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows where response time is too short or too long.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with time columns.
    df_items : pd.DataFrame
        DataFrame with only the 50 personality items.
    min_response_time : int, optional
        Minimum response time in seconds (default from config).
    max_response_time : float | None, optional
        Maximum response time in seconds (default None).
    filter_long_response_times : bool, optional
        Whether to also filter by upper IQR limit (default from config).
    iqr_factor : float, optional
        IQR multiplier for outlier detection (default from config).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Cleaned (df, df_items).
    """
    time_columns = [
        col for col in df.columns
        if (
            col.startswith("EXT")
            or col.startswith("EST")
            or col.startswith("AGR")
            or col.startswith("CSN")
            or col.startswith("OPN")
        )
        and col.endswith("_E")
    ]
    df_times = df[time_columns]

    valid_times = []
    labels = []
    all_indices = df_times.index

    for var in df_times.columns:
        times = df_times[var] / 1000  # ms -> s
        times = times[times >= min_response_time]

        if max_response_time is not None:
            times = times[times <= max_response_time]

        if len(times) == 0:
            print(f"[!] No valid data in {var} after time filtering ")
            continue

        filtered_times = times[times >= min_response_time]
        discarded_indices = all_indices.difference(filtered_times.index)
        df = df.drop(index=discarded_indices, errors="ignore")
        df_items = df_items.drop(index=discarded_indices, errors="ignore")

        

        Q1 = times.quantile(0.25)
        Q3 = times.quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - iqr_factor * IQR
        upper_limit = Q3 + iqr_factor * IQR

        if filter_long_response_times:
            long_response_indices = times[
                times > upper_limit
            ].index

            df = df.drop(index=long_response_indices, errors="ignore")
            df_items = df_items.drop(
                index=long_response_indices,
                errors="ignore",
            )

            times = times[times <= upper_limit]

        valid_times.append(times)
        labels.append(var)

    plot_boxplot(valid_times, labels, min_response_time, max_response_time, iqr_factor)

    return df, df_items
