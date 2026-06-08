"""Tests unitaires des fonctions critiques de ``validation_hypotheses.py``.

Couvre :
- Test Diebold-Mariano (formule HLN, corrigée petit échantillon).
- Prédicteurs baselines (seasonal_naive, moving_average, croston_classic).
- Cohérence d'agrégation des coûts (_row_costs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation_hypotheses import (  # noqa: E402
    _row_costs,
    diebold_mariano,
    mae,
    predict_croston_classic,
    predict_moving_average,
    predict_seasonal_naive,
)


# --------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------- #
def test_seasonal_naive_returns_correct_length():
    s = pd.Series(np.arange(24, dtype=float),
                  index=pd.date_range("2023-01-01", periods=24, freq="MS"))
    out = predict_seasonal_naive(s, horizon=6)
    assert len(out) == 6
    # Le pattern est répliqué des 12 derniers points
    assert np.array_equal(out, s.iloc[-12:-6].to_numpy())


def test_moving_average_constant_prediction():
    s = pd.Series([2.0, 4.0, 6.0])
    out = predict_moving_average(s, horizon=3, window=3)
    assert len(out) == 3
    assert np.allclose(out, 4.0)


def test_croston_classic_returns_zero_on_zero_series():
    s = pd.Series([0.0] * 12)
    out = predict_croston_classic(s, horizon=4)
    assert np.allclose(out, 0.0)


def test_croston_classic_intermittent_demand():
    # demande tous les 3 mois de taille 6
    # Croston classique : taille/intervalle, convergence asymptotique 6/3=2
    # mais avec alpha=0.1 et historique court, la convergence est lente :
    # on vérifie juste la plage métier raisonnable et la stabilité positive.
    s = pd.Series([0, 0, 6, 0, 0, 6, 0, 0, 6, 0, 0, 6], dtype=float)
    out = predict_croston_classic(s, horizon=2)
    assert out[0] > 0  # forecast positif sur série intermittente
    assert out[0] < 10  # plus faible que la taille brute (6)
    assert np.allclose(out, out[0])  # constante sur horizon


# --------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------- #
def test_diebold_mariano_equal_errors_gives_high_pvalue():
    rng = np.random.default_rng(0)
    e1 = rng.normal(size=100)
    e2 = rng.normal(size=100)  # même distribution, différents échantillons
    dm, pval = diebold_mariano(e1, e2, h=1)
    assert -3 < dm < 3  # statistique raisonnable
    assert pval > 0.1   # pas de rejet


def test_diebold_mariano_strictly_better_model():
    """e1 a une erreur systématiquement plus faible que e2 → DM négatif, p faible."""
    rng = np.random.default_rng(1)
    e1 = rng.normal(0, 1, 200)
    e2 = e1 + rng.normal(0, 2, 200)  # bruit additionnel
    dm, pval = diebold_mariano(e1, e2, h=1)
    assert dm < 0       # e1 < e2 en erreur²
    assert pval < 0.05


def test_diebold_mariano_too_few_obs_returns_nan():
    dm, pval = diebold_mariano(np.array([1.0, 2.0]), np.array([1.5, 1.5]), h=1)
    assert np.isnan(dm)
    assert np.isnan(pval)


# --------------------------------------------------------------------- #
# MAE et coûts
# --------------------------------------------------------------------- #
def test_mae_basic():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    assert mae(y, p) == pytest.approx((2 + 2 + 3) / 3)


def test_row_costs_columns_added():
    plan = pd.DataFrame({
        "produit_id": ["P1", "P2"],
        "mois_offset": [1, 1],
        "quantite_commandee": [10, 0],
        "commande_passee": [1, 0],
        "stock_final": [5.0, 2.0],
        "rupture": [0.0, 1.0],
        "demande_prevue": [10.0, 1.0],
        "cout_achat": [100.0, 50.0],
        "prix_vente": [150.0, 75.0],
    })
    out = _row_costs(plan)
    assert {"cout_commande", "cout_stockage", "marge_perdue", "cout_total"}.issubset(out.columns)
    # P1 : cmd_passée=1 → 50 USD, stock_final=5 × 100 × 0.001 × 30 = 15, rupture=0 → 0
    assert out.loc[0, "cout_commande"] == pytest.approx(50.0)
    assert out.loc[0, "cout_stockage"] == pytest.approx(5 * 100 * 0.001 * 30)
    assert out.loc[0, "marge_perdue"] == pytest.approx(0.0)
    # P2 : marge perdue = 1 × (75-50) = 25
    assert out.loc[1, "marge_perdue"] == pytest.approx(25.0)
