"""Configuration globale du pipeline Zenith."""
from pathlib import Path

# ---- Chemins ----
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"

for d in [PROCESSED_DIR, RESULTS_DIR, FIG_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_TRANSACTIONS = RAW_DIR / "zenith_dataset_brut.csv"
RAW_CATALOGUE = RAW_DIR / "catalogue_produits_250.csv"

# ---- Partitionnement temporel (cf. mémoire §3.3.3) ----
TRAIN_END = "2024-12-31"
VAL_END = "2025-03-31"
# Test = 2025-04-01 → 2025-07-31

# ---- Classification ABC (cf. mémoire §3.4.1) ----
ABC_A_THRESHOLD = 0.70
ABC_B_THRESHOLD = 0.90

# ---- Analyse XYZ ----
XYZ_X_CV = 0.5
XYZ_Y_CV = 1.0

# ---- Isolation Forest ----
IFOREST_CONTAMINATION = 0.10
IFOREST_N_ESTIMATORS = 100
IFOREST_MAX_SAMPLES = 256
IFOREST_RANDOM_STATE = 42

# ---- Optimisation linéaire (cf. mémoire §3.7) ----
COUT_COMMANDE_FIXE = 50.0           # USD par commande passée
TAUX_STOCKAGE_JOURNALIER = 0.001    # 0.1% du coût d'achat / jour
DELAI_LIVRAISON_DUBAI = 35          # jours
DELAI_LIVRAISON_CHINE = 55          # jours
NIVEAU_SERVICE_A = 0.95             # ≥ 95% disponibilité pour classe A
BUDGET_MENSUEL_DEFAUT = 500_000.0   # USD — calibré sur ~356k$ d'achats mensuels observés
HORIZON_PLANIFICATION_MOIS = 6   # couvre au moins un cycle d'importation complet

# ---- Forecasting ----
LGBM_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": -1,
    "n_estimators": 500,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "random_state": 42,
}
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
LSTM_SEQ_LENGTH = 30

RANDOM_STATE = 42
