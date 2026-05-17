"""Détection d'obsolescence via Isolation Forest (cf. mémoire §3.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import (
    IFOREST_CONTAMINATION,
    IFOREST_MAX_SAMPLES,
    IFOREST_N_ESTIMATORS,
    IFOREST_RANDOM_STATE,
)

OBSOLESCENCE_FEATURES = (
    "jours_depuis_derniere_vente",
    "tendance_3_mois",
    "tendance_6_mois",
    "ratio_ventes_3m_vs_12m",
    "nombre_mois_consecutifs_sans_vente",
    "valeur_stock_dormant",
)


def detect_obsolescence(
    features: pd.DataFrame,
    feature_cols: tuple[str, ...] = OBSOLESCENCE_FEATURES,
    contamination: float = IFOREST_CONTAMINATION,
) -> pd.DataFrame:
    """Score d'obsolescence par Isolation Forest sur les features de vente.

    Convention: plus le score est *bas* (≈ -1), plus le produit est suspect.
    On renvoie aussi un drapeau `a_risque_obsolescence`.
    """
    df = features.copy()
    X = df[list(feature_cols)].fillna(0).to_numpy(dtype=float)

    iforest = IsolationForest(
        n_estimators=IFOREST_N_ESTIMATORS,
        max_samples=IFOREST_MAX_SAMPLES,
        contamination=contamination,
        bootstrap=False,
        random_state=IFOREST_RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X)
    df["score_obsolescence"] = iforest.decision_function(X)
    df["a_risque_obsolescence"] = (iforest.predict(X) == -1).astype(int)

    # Règles métier additionnelles (filet de sécurité)
    rule_mask = (
        (df["jours_depuis_derniere_vente"] >= 180)
        | (df["nombre_mois_consecutifs_sans_vente"] >= 6)
        | ((df["ratio_ventes_3m_vs_12m"] < 0.1) & (df["age_produit_jours"] > 365))
    )
    df.loc[rule_mask, "a_risque_obsolescence"] = 1
    return df
