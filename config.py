"""
ReviewScope - Configuration centrale
=====================================
Projet final 8INF867 - UQAC
"""
import os
from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Fichiers de donnees
RAW_REVIEWS = DATA_RAW / "Reviews.csv"

# Parametres du dataset
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000

# Decoupage temporel
TRAIN_END_YEAR = 2010
TEST_START_YEAR = 2011

# Module 1 : Coherence texte/note
SENTIMENT_CLASSES = {
    "negatif": [1, 2],
    "neutre": [3],
    "positif": [4, 5],
}

# Module 2 : Prediction d'utilite
HELPFULNESS_THRESHOLD = 0.75
MIN_VOTES = 3

# Module 3 : Aspects
ASPECT_KEYWORDS = {
    "gout": ["taste", "flavor", "flavour", "delicious", "bland", "sweet",
             "bitter", "sour", "salty", "spicy", "yummy", "tasty"],
    "prix": ["price", "expensive", "cheap", "cost", "value", "worth",
             "affordable", "overpriced", "bargain", "deal"],
    "emballage": ["packaging", "package", "wrapped", "box", "container",
                  "seal", "sealed", "broken", "damaged", "crushed"],
    "livraison": ["delivery", "shipping", "arrived", "shipped", "late",
                  "delayed", "fast", "slow", "transit"],
    "fraicheur": ["fresh", "stale", "expired", "shelf life", "mold",
                  "rotten", "spoiled"],
    "sante": ["healthy", "organic", "natural", "gluten", "sugar",
              "calories", "nutrition", "diet", "vegan", "allergies"],
    "qualite": ["quality", "premium", "inferior", "great", "poor",
                "excellent", "terrible", "good", "bad"],
    "quantite": ["quantity", "amount", "portion", "size", "small",
                 "large", "enough", "less", "more"],
}

# Module 4 : Clustering
N_CLUSTERS_RANGE = range(3, 15)
DBSCAN_EPS_RANGE = [0.3, 0.5, 0.7, 1.0]
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256
