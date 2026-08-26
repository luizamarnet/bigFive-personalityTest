"""Training pipeline for the Big Five personality model."""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_FILE_PATH,
    MODEL_PATH,
    R_HOME,
    TEMPO_CURTO,
    FATOR_IQR,
    USAR_LIMITE_SUPERIOR,
)
from data_loader import load_data
from data_cleaner import clean_by_response_time
from correlation import polychoric_correlation
from factor_analysis import perform_factor_analysis
from clustering import cluster_and_visualize


def main() -> None:
    """Run the full training pipeline."""
    os.environ["R_HOME"] = R_HOME

    df, df_items = load_data(DATA_FILE_PATH)
    df, df_items = clean_by_response_time(
        df,
        df_items,
        tempo_curto=TEMPO_CURTO,
        fator=FATOR_IQR,
        usar_limite_superior=USAR_LIMITE_SUPERIOR,
    )

    print("Number of responses after cleaning: ", len(df_items))


    pcor_matrix = polychoric_correlation(df_items)
    print("Shape of correlation matrix: ", np.shape(pcor_matrix))

    fa_model, factor_names = perform_factor_analysis(pcor_matrix, df_items.columns)

    fa_model.mean_ = 0#mean_.values
    fa_model.std_ = 1#std_.values


    B = np.linalg.pinv(df_items) @ fa_model.transform(df_items)

    aux_df = np.zeros((50,5))
    aux_df[B<0] = 1
    aux_df[B>=0] = 5
    aux_df_trans = fa_model.transform(aux_df.T)

    factor_max  = np.zeros(5)
    factor_max [0] = aux_df_trans[0,0]
    factor_max [1] = aux_df_trans[1,1]
    factor_max [2] = aux_df_trans[2,2]
    factor_max [3] = aux_df_trans[3,3]
    factor_max [4] = aux_df_trans[4,4]

    aux_df = np.zeros((50,5))
    aux_df[B>=0] = 1
    aux_df[B<0] = 5
    aux_df_trans = fa_model.transform(aux_df.T)

    factor_min = np.zeros(5)
    factor_min[0] = aux_df_trans[0,0]
    factor_min[1] = aux_df_trans[1,1]
    factor_min[2] = aux_df_trans[2,2]
    factor_min[3] = aux_df_trans[3,3]
    factor_min[4] = aux_df_trans[4,4]

    print("Factor max values:", factor_max)
    print("Factor min values:", factor_min)

    

    # Transform and normalize
    df_items_transform = fa_model.transform(df_items)
    df_items_transform = (df_items_transform - factor_min) / (factor_max - factor_min)

    # Save model
    model_to_save = {
        "model": fa_model,
        "nome_fatores": factor_names,
        "fatores_minimos": factor_min,
        "fatores_maximos": factor_max,
    }
    joblib.dump(model_to_save, MODEL_PATH)

    # Clustering
    trait_names = {
        "EXT": "Extraversion",
        "EST": "Neuroticism",
        "AGR": "Agreeableness",
        "CSN": "Conscientiousness",
        "OPN": "Openness",
    }
    factor_display_names = [
        trait_names.get(prefix, prefix) for prefix in factor_names.values()
    ]

    cluster_and_visualize(df_items_transform, factor_display_names)

    print("Clustering completed.")


if __name__ == "__main__":
    main()
