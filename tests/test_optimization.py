"""Tests unitaires du module ``src.optimization``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization import (
    compare_policies,
    distribute_to_stores,
    financial_kpis,
    fournisseur_label,
    lead_time_months,
    optimize_orders,
    simulate_baseline_policy,
)


@pytest.fixture
def mini_products() -> pd.DataFrame:
    return pd.DataFrame({
        "produit_id": ["P1", "P2", "P3"],
        "classe_abc": ["A", "B", "C"],
        "classe_xyz": ["X", "Y", "Z"],
        "a_risque_obsolescence": [0, 0, 1],
        "cout_achat_unitaire": [100.0, 50.0, 10.0],
        "prix_vente_unitaire": [150.0, 75.0, 15.0],
        "stock_courant": [20.0, 50.0, 5.0],
        "origine_fournisseur": ["Dubaï", "Chine", "Dubaï"],
    })


@pytest.fixture
def mini_forecasts() -> pd.DataFrame:
    return pd.DataFrame({
        "produit_id": (["P1"] * 3 + ["P2"] * 3 + ["P3"] * 3),
        "date": pd.date_range("2025-08-01", periods=3, freq="MS").tolist() * 3,
        "prevision": [10, 12, 8, 30, 25, 20, 0, 0, 0],
        "modele_utilise": ["lstm"] * 3 + ["lightgbm"] * 3 + ["obsolete"] * 3,
    })


def test_lead_time_dubai_one_month():
    assert lead_time_months("Dubaï") == 1


def test_lead_time_chine_two_months():
    assert lead_time_months("Chine") == 2


def test_fournisseur_label_dispatch():
    assert fournisseur_label("Chine") == "Chine"
    assert fournisseur_label("Dubaï") == "Dubaï"


def test_optimize_orders_excludes_obsoletes(mini_products, mini_forecasts):
    plan, kpis = optimize_orders(mini_products, mini_forecasts, horizon=3, budget_mensuel=100_000)
    assert kpis["statut"] == "Optimal"
    # P3 est obsolète → aucune ligne dans le plan
    assert "P3" not in plan["produit_id"].values


def test_optimize_orders_produces_horizon_rows(mini_products, mini_forecasts):
    plan, _ = optimize_orders(mini_products, mini_forecasts, horizon=3)
    # P1, P2 actifs × 3 mois = 6 lignes
    assert len(plan) == 6


def test_simulate_baseline_policy_returns_dataframe(mini_products, mini_forecasts):
    plan = simulate_baseline_policy(mini_products, mini_forecasts, horizon=3)
    assert not plan.empty
    assert set(plan.columns) >= {"produit_id", "mois_offset", "quantite_commandee", "rupture"}


def test_financial_kpis_has_expected_keys(mini_products, mini_forecasts):
    plan = simulate_baseline_policy(mini_products, mini_forecasts, horizon=3)
    kpi = financial_kpis(plan)
    for k in ["nb_commandes", "ruptures_unites", "ca_realise_usd", "taux_service_pct"]:
        assert k in kpi


def test_compare_policies_returns_table(mini_products, mini_forecasts):
    plan_lp, _ = optimize_orders(mini_products, mini_forecasts, horizon=3, budget_mensuel=50_000)
    plan_emp = simulate_baseline_policy(mini_products, mini_forecasts, horizon=3)
    cmp = compare_policies(plan_lp, plan_emp)
    assert {"indicateur", "politique_empirique", "politique_optimisee", "delta_pct"}.issubset(cmp.columns)


def test_distribute_to_stores_preserves_quantities(mini_products, mini_forecasts):
    plan_lp, _ = optimize_orders(mini_products, mini_forecasts, horizon=3, budget_mensuel=50_000)
    tx = pd.DataFrame({
        "produit_id": ["P1"] * 4 + ["P2"] * 6,
        "magasin": ["Mobutu 2", "Lomami", "Mobutu 2", "Maniema"] + ["Kolwezi Ville"] * 6,
        "quantite_vendue": [3, 1, 2, 1] + [1] * 6,
    })
    distributed = distribute_to_stores(plan_lp, tx)
    # Total quantité après distribution == total quantité du plan central
    assert (distributed["quantite_commandee"].sum()
            == plan_lp[plan_lp["quantite_commandee"] > 0]["quantite_commandee"].sum())
