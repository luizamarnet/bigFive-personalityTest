"""Configuration constants for the Big Five personality analysis project."""

from pathlib import Path

# Paths
DATA_FILE_PATH = Path("dataset-IPIP-FFM-data-8Nov2018") / "data-final.csv"
MODEL_PATH = Path("./models/modelo_factoranalyzer.pkl")
KMEANS_MODEL_PATH = Path("./models/modelo_kmeans.pkl")

# Data cleaning parameters
MIN_TIME = 2  # [seconds] - Inputs with questions answered in less than MIN_TIME seconds will be discarded
IQR_FACTOR = 1.5 # Factor used to identify unusually long response times
FILTER_LONG_RESPONSE_TIMES = False # Whether to discard inputs with unusually long response times

# Factor analysis
N_FACTORS = 5

# Clustering
OPTIMAL_K = 5
K_RANGE = 20 # max number of clusters to test
TEST_NUMBER_CLUSTERS = True # Whether to test the number of clusters to choose before clusterization


