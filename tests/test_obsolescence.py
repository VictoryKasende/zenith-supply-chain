"""Tests unitaires du module ``src.obsolescence``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.obsolescence import (
    DEFAULT_FEATURES,
    build_obsolescence_features,
    detect_obsolescence,
    sensitivity_analysis,
)


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Mini-historique sur 36 mois avec : 1 produit actif, 1 dormant, 1 récent."""
    rng = np.random.default_rng(0)
    months = pd.period_range("2022-08", "2025-07", freq="M").to_timestamp()
    rows = []
    # Produit actif (P_A) : ventes régulières
    for d in months:
        for _ in range(rng.integers(2, 6)):
            rows.append({"transaction_id": f"T{len(rows):05d}", "date": d + pd.Timedelta(days=int(rng.integers(1, 27))),
                         "produit_id": "P_A", "quantite_vendue": float(rng.integers(1, 5)),
                         "prix_vente_unitaire": 100.0, "cout_achat_unitaire": 60.0,
                         "stock_apres_vente": 20.0, "montant_total": 200.0})
    # Produit dormant (P_D) : ventes uniquement sur les 6 premiers mois
    for d in months[:6]:
        rows.append({"transaction_id": f"T{len(rows):05d}", "date": d,
                     "produit_id": "P_D", "quantite_vendue": 1.0,
                     "prix_vente_unitaire": 50.0, "cout_achat_unitaire": 30.0,
                     "stock_apres_vente": 100.0, "montant_total": 50.0})
    # Produit récent (P_N) : ventes uniquement sur les 3 derniers mois
    for d in months[-3:]:
        rows.append({"transaction_id": f"T{len(rows):05d}", "date": d,
                     "produit_id": "P_N", "quantite_vendue": 2.0,
                     "prix_vente_unitaire": 200.0, "cout_achat_unitaire": 120.0,
                     "stock_apres_vente": 5.0, "montant_total": 400.0})
    return pd.DataFrame(rows)


def test_build_features_returns_one_row_per_product(transactions):
    feats = build_obsolescence_features(transactions)
    assert set(feats["produit_id"]) == {"P_A", "P_D", "P_N"}
    assert set(DEFAULT_FEATURES).issubset(feats.columns)


def test_dormant_product_has_high_jours_sans_vente(transactions):
    feats = build_obsolescence_features(transactions)
    p_d = feats[feats["produit_id"] == "P_D"].iloc[0]
    p_a = feats[feats["produit_id"] == "P_A"].iloc[0]
    assert p_d["jours_depuis_derniere_vente"] > p_a["jours_depuis_derniere_vente"]
    assert p_d["nombre_mois_consecutifs_sans_vente"] > 6


def test_detect_obsolescence_flags_dormant_product(transactions):
    feats = build_obsolescence_features(transactions)
    age = transactions.groupby("produit_id")["date"].min().pipe(
        lambda s: (transactions["date"].max() - s).dt.days
    ).rename("age_produit_jours")
    feats = feats.merge(age.reset_index(), on="produit_id", how="left")
    result = detect_obsolescence(feats, contamination=0.34)
    flagged = result.df[result.df["a_risque_obsolescence"] == 1]["produit_id"].tolist()
    assert "P_D" in flagged


def test_business_rules_catch_evident_cases():
    df = pd.DataFrame({
        "produit_id": ["P1", "P2"],
        "jours_depuis_derniere_vente": [400, 5],
        "tendance_3_mois": [-2.0, 1.0],
        "tendance_6_mois": [-1.0, 0.5],
        "ratio_ventes_3m_vs_12m": [0.0, 0.8],
        "nombre_mois_consecutifs_sans_vente": [12, 0],
        "valeur_stock_dormant": [500.0, 200.0],
        "variation_relative_prix": [0.0, 0.05],
        "age_produit_jours": [800, 800],
    })
    result = detect_obsolescence(df, contamination=0.5)
    assert int(result.df[result.df["produit_id"] == "P1"]["a_risque_obsolescence"].iloc[0]) == 1


def test_sensitivity_analysis_monotonic_relation(transactions):
    feats = build_obsolescence_features(transactions)
    age = transactions.groupby("produit_id")["date"].min().pipe(
        lambda s: (transactions["date"].max() - s).dt.days
    ).rename("age_produit_jours")
    feats = feats.merge(age.reset_index(), on="produit_id", how="left")
    sens = sensitivity_analysis(feats, contaminations=(0.10, 0.20, 0.30))
    # Plus la contamination est grande, plus on flagge (ou autant)
    assert sens["n_flagged_iforest"].is_monotonic_increasing
