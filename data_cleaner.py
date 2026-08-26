"""Data cleaning based on response time."""

import matplotlib.pyplot as plt
import pandas as pd
from config import TEMPO_CURTO, FATOR_IQR, USAR_LIMITE_SUPERIOR


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
    colunas = [
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
    df_tempos = df[colunas]

    tempos_validos = []
    labels = []
    todos_indices = df_tempos.index

    for var in df_tempos.columns:
        tempos = df_tempos[var] / 1000  # ms -> s
        tempos = tempos[tempos >= tempo_curto]

        if tempo_longo is not None:
            tempos = tempos[tempos <= tempo_longo]

        if len(tempos) == 0:
            print(f"[!] No valid data in {var} after filter > {tempo_curto}s")
            continue

        tempos_filtrados = tempos[tempos >= tempo_curto]
        indices_descartados = todos_indices.difference(tempos_filtrados.index)
        df = df.drop(index=indices_descartados, errors="ignore")
        df_items = df_items.drop(index=indices_descartados, errors="ignore")

        tempos_validos.append(tempos)
        labels.append(var)

        Q1 = tempos.quantile(0.25)
        Q3 = tempos.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - fator * IQR
        limite_superior = Q3 + fator * IQR

        outliers = tempos[tempos < limite_inferior]
        if usar_limite_superior:
            outliers = tempos[(tempos < limite_inferior) | (tempos > limite_superior)]
        else:
            outliers = tempos[tempos < limite_inferior]

    flierprops = dict(marker="o", markerfacecolor="red", markersize=3, linestyle="none")

    plt.figure(figsize=(14, 6))
    plt.boxplot(tempos_validos, labels=labels, showfliers=True, flierprops=flierprops, whis=fator)
    plt.title(f"Boxplot of response times {tempo_curto} per Big Five item")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Items")
    if tempo_longo is not None:
        if tempo_longo < 1800:
            plt.ylim(-1, tempo_longo)
        else:
            plt.ylim(-1, 1800)
    else:
        plt.ylim(-1, 1800)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("boxplot_tempos_bigfive.png", dpi=300, bbox_inches="tight")
    plt.show()

    return df, df_items
