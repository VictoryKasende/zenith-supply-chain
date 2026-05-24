"""Tests unitaires du module ``src.classification``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification import (
    abc_xyz_matrix,
    classify_abc,
    classify_xyz,
    kmeans_pipeline,
)


@pytest.fixture
def fake_products() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # 30 produits avec CA décroissant + CV varié
    n = 30
    ca = np.sort(rng.uniform(100, 100_000, n))[::-1]
    ventes = ca / rng.uniform(5, 50, n)
    cv = rng.uniform(0.1, 2.5, n)
    return pd.DataFrame({
        "produit_id": [f"P{i:04d}" for i in range(n)],
        "ca_total_36mois": ca,
        "ventes_totales_36mois": ventes,
        "coefficient_variation": cv,
        "nombre_mois_avec_ventes": rng.integers(1, 36, n),
        "tendance_3_mois": rng.normal(0, 1, n),
        "jours_depuis_derniere_vente": rng.integers(0, 400, n),
        "prix_vente_unitaire_moyen": rng.uniform(5, 1000, n),
    })


def test_classify_abc_respects_pareto_thresholds(fake_products):
    out = classify_abc(fake_products)
    total = out["ca_total_36mois"].sum()
    a_cum = out[out["classe_abc"] == "A"]["ca_total_36mois"].sum() / total
    ab_cum = out[out["classe_abc"].isin(["A", "B"])]["ca_total_36mois"].sum() / total
    # La classe A contient le CA jusqu'à environ 70 %
    assert 0.55 <= a_cum <= 0.80
    assert 0.80 <= ab_cum <= 0.95


def test_classify_abc_all_classes_present(fake_products):
    out = classify_abc(fake_products)
    assert set(out["classe_abc"].unique()) == {"A", "B", "C"}


def test_classify_xyz_uses_quantile_fallback_if_no_x(fake_products):
    # On force toutes les CV > 0.5 → classe X devrait être 0 → fallback quantile
    df = fake_products.copy()
    df["coefficient_variation"] = np.random.default_rng(0).uniform(0.6, 2.0, len(df))
    df = classify_abc(df)
    out = classify_xyz(df)
    assert set(out["classe_xyz"].unique()) >= {"X", "Y", "Z"}


def test_classify_xyz_creates_abc_xyz(fake_products):
    df = classify_abc(fake_products)
    out = classify_xyz(df)
    assert "classe_abc_xyz" in out.columns
    assert out["classe_abc_xyz"].str.len().eq(2).all()


def test_abc_xyz_matrix_dimensions(fake_products):
    df = classify_xyz(classify_abc(fake_products))
    mat = abc_xyz_matrix(df)
    assert mat.shape[0] == 3  # 3 classes ABC


def test_kmeans_pipeline_returns_valid_labels(fake_products):
    df = classify_xyz(classify_abc(fake_products))
    out, diag = kmeans_pipeline(df, k_candidates=(2, 3, 4))
    assert "cluster_kmeans" in out.columns
    assert "libelle_cluster" in out.columns
    assert diag.best_k in (2, 3, 4)
    assert -1 <= diag.best_silhouette <= 1
