"""Tests unitaires du module ``src.preprocessing``.

Couvre les règles critiques :
- Parsing de dates mixtes (ISO + DD/MM/YYYY).
- Détection et correction des prix aberrants (×10 vs imputation médiane).
- Normalisation des libellés famille.
- Isolation des retours.
- Variables temporelles cycliques.
- Cohérence montant_total après recalcul.
- Pipeline complet sur un mini-dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    add_financial_features,
    add_temporal_features,
    clean_dataset,
    fix_aberrant_prices,
    flag_returns,
    normalize_famille,
    reconcile_amount,
)
from src.utils import parse_mixed_dates


# --------------------------------------------------------------------- #
# Fixture : mini-dataset
# --------------------------------------------------------------------- #
@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": ["T001", "T002", "T003", "T004", "T005", "T001"],
        "date": ["2024-01-15", "15/02/2024", "2024-03-10", "2024-04-01", "2024-05-20", "2024-01-15"],
        "magasin": ["Mobutu 2"] * 6,
        "ville": ["Lubumbashi"] * 6,
        "produit_id": ["P001", "P001", "P001", "P002", "P002", "P001"],
        "produit_nom": ["Cartouche A"] * 3 + ["Souris B"] * 2 + ["Cartouche A"],
        "famille": ["Cartouches", "Cartouche", "Cartouche", "Accessoire", "Accessoires", "Cartouches"],
        "marque": ["HP"] * 3 + ["Logitech"] * 2 + ["HP"],
        "origine_fournisseur": ["Dubaï"] * 6,
        "prix_vente_unitaire": [100.0, 100.0, 1000.0, 20.0, 20.0, 100.0],  # T003 = ×10 erreur
        "cout_achat_unitaire": [60.0, 60.0, 60.0, 12.0, 12.0, 60.0],
        "quantite_vendue": [2.0, 3.0, 1.0, -1.0, 4.0, 2.0],  # T004 = retour
        "montant_total": [200.0, 300.0, 1000.0, -20.0, 80.0, 200.0],
        "client_id": ["B001", "B001", "C001", "C002", "C002", "B001"],
        "client_nom": ["KICC", "KICC", "Anonyme", "Anonyme", "Anonyme", "KICC"],
        "type_client": ["Entreprise"] * 2 + ["Personne courante"] * 3 + ["Entreprise"],
        "mode_paiement": ["Crédit"] * 2 + ["Comptant"] * 3 + ["Crédit"],
        "stock_apres_vente": [10.0, 7.0, 6.0, 50.0, 46.0, 10.0],
    })


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #
def test_parse_mixed_dates_handles_iso_and_european():
    s = pd.Series(["2024-01-15", "15/02/2024", "invalid"])
    parsed = parse_mixed_dates(s)
    assert parsed.iloc[0] == pd.Timestamp("2024-01-15")
    assert parsed.iloc[1] == pd.Timestamp("2024-02-15")
    assert pd.isna(parsed.iloc[2])


def test_normalize_famille_corrects_typos(sample_df):
    df = sample_df.assign(famille=sample_df["famille"].astype("string"))
    out, n_fix = normalize_famille(df)
    assert n_fix == 3
    assert set(out["famille"].unique()) == {"Cartouche", "Accessoire"}


def test_fix_aberrant_prices_divides_by_10_when_extreme(sample_df):
    df = sample_df.copy()
    df["famille"] = df["famille"].astype("string").replace({"Cartouches": "Cartouche", "Accessoires": "Accessoire"})
    out, stats = fix_aberrant_prices(df)
    # T003 a un prix 1000 vs médiane 100 → divisé par 10
    assert out.loc[2, "prix_vente_unitaire"] == 100.0
    assert stats["prix_aberrants_x10_corriges"] >= 1


def test_fix_aberrant_prices_recomputes_montant(sample_df):
    df = sample_df.copy()
    df["famille"] = df["famille"].astype("string").replace({"Cartouches": "Cartouche", "Accessoires": "Accessoire"})
    out, _ = fix_aberrant_prices(df)
    # Après correction du prix de T003, montant_total = 100 × 1 = 100
    assert out.loc[2, "montant_total"] == 100.0


def test_flag_returns_isolates_negative_quantities(sample_df):
    out, n_returns = flag_returns(sample_df)
    assert n_returns == 1
    assert (out["quantite_vendue"] > 0).all()
    # Le retour est exclu de l'output principal
    assert "T004" not in out["transaction_id"].values


def test_add_temporal_features_cyclical(sample_df):
    df = sample_df.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    out = add_temporal_features(df)
    # Variables cycliques entre [-1, 1]
    assert out["mois_sin"].between(-1, 1).all()
    assert out["mois_cos"].between(-1, 1).all()
    # Rentrée scolaire = août-septembre
    august = pd.DataFrame({"date": [pd.Timestamp("2024-08-15")]})
    assert add_temporal_features(august)["est_rentree_scolaire"].iloc[0] == 1
    january = pd.DataFrame({"date": [pd.Timestamp("2024-01-15")]})
    assert add_temporal_features(january)["est_rentree_scolaire"].iloc[0] == 0


def test_financial_features_marge_unitaire(sample_df):
    df = sample_df.copy()
    out = add_financial_features(df)
    assert (out["marge_unitaire"] == df["prix_vente_unitaire"] - df["cout_achat_unitaire"]).all()
    assert (out["taux_marge_pct"].dropna() > 0).any()


def test_reconcile_amount_fixes_inconsistency():
    df = pd.DataFrame({
        "prix_vente_unitaire": [10.0],
        "quantite_vendue": [3.0],
        "montant_total": [25.0],  # incorrect, devrait être 30.0
    })
    out, n_fix = reconcile_amount(df)
    assert n_fix == 1
    assert out["montant_total"].iloc[0] == 30.0


def test_clean_dataset_pipeline_removes_duplicates_and_returns(sample_df):
    df = sample_df.copy()
    df["date"] = parse_mixed_dates(df["date"])
    out = clean_dataset(df)
    # 6 lignes brutes : 1 doublon exact + 1 retour → max 4 conservées
    assert len(out) <= 4
    assert "est_retour" in out.columns or out["quantite_vendue"].gt(0).all()
