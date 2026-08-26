"""Data cleaning based on response time."""

import pandas as pd
from config import TEMPO_CURTO, FATOR_IQR, USAR_LIMITE_SUPERIOR
from visualization import plot_boxplot


def clean_by_response_time(
    df: pd.DataFrame,
    df_items: pd.DataFrame,
    tempo_curto: int = TEMPO_CURTO,
    tempo_longo: float | None = None,
    usar_limite_superior: bool = USAR_LIMITE_SUPERIOR,
    fator: float = FATOR_IQR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows where response time is too short or too long.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with time columns.
    df_items : pd.DataFrame
        DataFrame with only the 50 personality items.
    tempo_curto : int, optional
        Minimum response time in seconds (default from config).
    tempo_longo : float | None, optional
        Maximum response time in seconds (default None).
    usar_limite_superior : bool, optional
        Whether to also filter by upper IQR limit (default from config).
    fator : float, optional
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
        times = times[times >= tempo_curto]

        if tempo_longo is not None:
            times = times[times <= tempo_longo]

        if len(times) == 0:
            print(f"[!] No valid data in {var} after filter > {tempo_curto}s")
            continue

        filtered_times = times[times >= tempo_curto]
        discarded_indices = all_indices.difference(filtered_times.index)
        df = df.drop(index=discarded_indices, errors="ignore")
        df_items = df_items.drop(index=discarded_indices, errors="ignore")

        valid_times.append(times)
        labels.append(var)

        Q1 = times.quantile(0.25)
        Q3 = times.quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - fator * IQR
        upper_limit = Q3 + fator * IQR

        outliers = times[times < lower_limit]
        if usar_limite_superior:
            outliers = times[(times < lower_limit) | (times > upper_limit)]
        else:
            outliers = times[times < lower_limit]

    plot_boxplot(valid_times, labels, tempo_curto, tempo_longo, fator)

    return df, df_items
