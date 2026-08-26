"""Configuration constants for the Big Five personality analysis project."""

from pathlib import Path

# Paths
DATA_FILE_PATH = Path("dataset-IPIP-FFM-data-8Nov2018") / "data-final.csv"
MODEL_PATH = Path("./models/modelo_factoranalyzer.pkl")
KMEANS_MODEL_PATH = Path("./models/modelo_kmeans.pkl")

# Data cleaning parameters
TEMPO_CURTO = 2  # seconds
FATOR_IQR = 1.5
USAR_LIMITE_SUPERIOR = True

# Factor analysis
N_FACTORS = 5

# Clustering
K_OTIMO = 5

# Genetic algorithm
N_GENERATIONS = 500
POP_SIZE = 200

# R environment
R_HOME = r"C:\Program Files\R\R-4.5.1"
