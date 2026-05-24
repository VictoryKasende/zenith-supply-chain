"""Tests unitaires du module ``src.evaluation``."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import (
    build_evaluation_report,
    evaluate_h1_differentiation,
    evaluate_h2_complementarite,
    evaluate_h3_obsolescence,
    evaluate_h4_optimisation,
    evaluate_h5_impact_financier,
    evaluate_h6_faisabilite_pme,
)


@pytest.fixture
def metrics_df() -> pd.DataFrame:
    return pd.DataFrame({
        "produit_id": [f"P{i}" for i in range(6)],
        "modele": ["lstm", "lstm", "lightgbm", "lightgbm", "sarima", "sarima"],
        "mae": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "classe_abc": ["A", "A", "B", "B", "C", "C"],
    })


@pytest.fixture
def comparison_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"indicateur": "cout_total_simule_usd", "politique_empirique": 100.0,
         "politique_optimisee": 70.0, "delta": -30.0, "delta_pct": -30.0},
        {"indicateur": "ca_realise_usd", "politique_empirique": 1000.0,
         "politique_optimisee": 1010.0, "delta": 10.0, "delta_pct": 1.0},
        {"indicateur": "marge_perdue_usd", "politique_empirique": 50.0,
         "politique_optimisee": 40.0, "delta": -10.0, "delta_pct": -20.0},
        {"indicateur": "stock_moyen_immo_usd", "politique_empirique": 200.0,
         "politique_optimisee": 80.0, "delta": -120.0, "delta_pct": -60.0},
    ])


@pytest.fixture
def obs_features() -> pd.DataFrame:
    return pd.DataFrame({
        "produit_id": ["P_obs1", "P_obs2", "P_actif"],
        "nombre_mois_consecutifs_sans_vente": [12, 8, 0],
    })


def test_h1_validee_si_pipeline_meilleur(metrics_df):
    v = evaluate_h1_differentiation(metrics_df)
    assert v.valeur < 1.0
    assert v.verdict == "Validée"


def test_h2_complementarite_avec_modeles_attendus(metrics_df):
    # min_sample=2 car la fixture n'a que 2 produits par classe
    v = evaluate_h2_complementarite(metrics_df, min_sample=2)
    assert v.verdict == "Validée"
    assert v.details["observé"] == {"A": "lstm", "B": "lightgbm", "C": "sarima"}


def test_h3_obsolescence_rappel(obs_features):
    flagged = pd.DataFrame({"produit_id": ["P_obs1", "P_obs2", "P_actif"]})
    v = evaluate_h3_obsolescence(flagged, obs_features)
    assert v.valeur == 1.0
    assert v.verdict == "Validée"


def test_h3_obsolescence_rappel_faible(obs_features):
    flagged = pd.DataFrame({"produit_id": ["P_actif"]})  # tous manqués
    v = evaluate_h3_obsolescence(flagged, obs_features)
    assert v.valeur < 0.5
    assert v.verdict == "Non validée"


def test_h4_optimisation_meilleure(comparison_df):
    v = evaluate_h4_optimisation(comparison_df)
    assert v.verdict == "Validée"
    assert v.details["gain_usd"] == 30.0


def test_h5_benefice_positif(comparison_df):
    v = evaluate_h5_impact_financier(comparison_df)
    assert v.valeur > 0
    assert v.verdict == "Validée"


def test_h6_faisabilite_pme_validee():
    v = evaluate_h6_faisabilite_pme(total_runtime_sec=200, n_python_deps=15, requires_gpu=False)
    assert v.verdict == "Validée"


def test_h6_non_validee_si_gpu_requis():
    v = evaluate_h6_faisabilite_pme(total_runtime_sec=200, n_python_deps=15, requires_gpu=True)
    assert v.verdict != "Validée"


def test_build_evaluation_report_has_expected_columns(metrics_df, comparison_df, obs_features):
    flagged = pd.DataFrame({"produit_id": ["P_obs1", "P_obs2"]})
    verdicts = [
        evaluate_h1_differentiation(metrics_df),
        evaluate_h2_complementarite(metrics_df, min_sample=2),
        evaluate_h3_obsolescence(flagged, obs_features),
        evaluate_h4_optimisation(comparison_df),
        evaluate_h5_impact_financier(comparison_df),
        evaluate_h6_faisabilite_pme(200, 15, False),
    ]
    report = build_evaluation_report(verdicts)
    assert {"code", "libelle", "verdict", "valeur_mesuree"}.issubset(report.columns)
    assert len(report) == 6
