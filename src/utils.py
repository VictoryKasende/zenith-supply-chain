"""Fonctions utilitaires partagées par tous les modules du pipeline.

Inclut :
- Constantes de chemins (ROOT, DATA_DIR, OUTPUTS_DIR, ...).
- Configuration du logging.
- Lecture du dataset brut avec parsing robuste des dates mixtes.
- Helpers de partitionnement temporel et de sauvegarde reproductible.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- #
# Constantes globales
# --------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
OUTPUTS_DIR = ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
TAB_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
POWERBI_DIR = OUTPUTS_DIR / "powerbi"

for _d in [PROCESSED_DIR, FEATURES_DIR, FIG_DIR, TAB_DIR, MODELS_DIR, POWERBI_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RAW_TRANSACTIONS = RAW_DIR / "zenith_dataset_brut.csv"
RAW_CATALOGUE = RAW_DIR / "catalogue_produits_250.csv"

RANDOM_STATE = 42

# Partitionnement temporel (mémoire §3.3.3)
TRAIN_END = "2024-12-31"
VAL_END = "2025-03-31"


# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #
def setup_logger(name: str = "zenith", level: int = logging.INFO) -> logging.Logger:
    """Configure un logger console + format homogène pour tout le pipeline."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------- #
# Parsing dates
# --------------------------------------------------------------------- #
def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Convertit une colonne date au format datetime64 en gérant les formats mixtes.

    Accepte `YYYY-MM-DD` et `DD/MM/YYYY` (deux formats observés dans le dataset).
    Les valeurs non-parsables deviennent ``NaT`` (gérées par le module de nettoyage).
    """
    iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    fallback = iso.isna() & series.notna()
    if fallback.any():
        alt = pd.to_datetime(series[fallback], format="%d/%m/%Y", errors="coerce")
        iso.loc[fallback] = alt
    return iso


# --------------------------------------------------------------------- #
# I/O dataset brut
# --------------------------------------------------------------------- #
def load_raw_transactions(path: Path | str | None = None) -> pd.DataFrame:
    """Charge le dataset brut Zenith avec typage explicite + parsing des dates."""
    path = Path(path) if path else RAW_TRANSACTIONS
    string_cols = [
        "transaction_id", "magasin", "ville", "produit_id", "produit_nom",
        "famille", "marque", "origine_fournisseur", "client_id", "client_nom",
        "type_client", "mode_paiement",
    ]
    df = pd.read_csv(path, dtype={c: "string" for c in string_cols}, low_memory=False)
    df["date"] = parse_mixed_dates(df["date"])
    return df


def load_catalogue(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW_CATALOGUE
    return pd.read_csv(path)


# --------------------------------------------------------------------- #
# Partitionnement temporel
# --------------------------------------------------------------------- #
def temporal_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologique train / validation / test (cf. mémoire §3.3.3)."""
    train = df[df[date_col] <= pd.Timestamp(train_end)].copy()
    val = df[(df[date_col] > pd.Timestamp(train_end)) & (df[date_col] <= pd.Timestamp(val_end))].copy()
    test = df[df[date_col] > pd.Timestamp(val_end)].copy()
    return train, val, test


# --------------------------------------------------------------------- #
# Helpers numériques
# --------------------------------------------------------------------- #
def linear_slope(values: Iterable[float]) -> float:
    """Pente OLS d'un mini-historique. Renvoie 0 si moins de 2 points valides."""
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return 0.0
    x = np.arange(len(v), dtype=float)
    return float(np.polyfit(x, v, 1)[0])


def coefficient_of_variation(values: Iterable[float]) -> float:
    """CV = écart-type / moyenne. NaN si moyenne nulle/négative."""
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or v.mean() <= 0:
        return float("nan")
    return float(v.std(ddof=1) / v.mean())


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    """Division protégée contre la division par zéro."""
    if den == 0 or pd.isna(den):
        return default
    return num / den
