"""Tests unitaires de ``src.powerbi_export``."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.powerbi_export import (
    build_alertes_powerbi,
    build_dim_clients,
    build_dim_magasins,
    build_dim_produits,
    build_dim_temps,
    build_fact_ventes,
)


@pytest.fixture
def mini_tx() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "date": pd.to_datetime(["2024-01-15", "2024-02-01", "2024-03-10"]),
        "magasin": ["Mobutu 2", "Lomami", "Mobutu 2"],
        "ville": ["Lubumbashi"] * 3,
        "produit_id": ["P001", "P002", "P001"],
        "produit_nom": ["Cartouche", "Souris", "Cartouche"],
        "famille": ["Cartouche", "Accessoire", "Cartouche"],
        "client_id": ["B001", "C001", "B001"],
        "client_nom": ["KICC", "Anonyme", "KICC"],
        "type_client": ["Entreprise", "Personne courante", "Entreprise"],
        "mode_paiement": ["Crédit", "Comptant", "Crédit"],
        "prix_vente_unitaire": [100.0, 20.0, 100.0],
        "cout_achat_unitaire": [60.0, 12.0, 60.0],
        "quantite_vendue": [2.0, 5.0, 1.0],
        "montant_total": [200.0, 100.0, 100.0],
        "stock_apres_vente": [10.0, 50.0, 9.0],
    })


def test_dim_temps_complete_range():
    df = build_dim_temps("2024-01-01", "2024-01-10")
    assert len(df) == 10
    assert df["date"].is_monotonic_increasing
    assert set(["annee", "mois", "trimestre", "jour_nom"]).issubset(df.columns)


def test_dim_temps_event_flags():
    df = build_dim_temps("2024-08-01", "2024-09-30")
    assert df["est_rentree_scolaire"].sum() == len(df)


def test_dim_clients_aggregates_ca(mini_tx):
    df = build_dim_clients(mini_tx)
    kicc = df[df["client_id"] == "B001"].iloc[0]
    assert kicc["ca_total_usd"] == 300.0
    assert kicc["nb_transactions"] == 2
    assert kicc["fidele"] in (0, 1)


def test_dim_magasins_part_ca_sums_100(mini_tx):
    df = build_dim_magasins(mini_tx)
    assert round(df["part_ca_pct"].sum()) == 100


def test_fact_ventes_adds_marge_columns(mini_tx):
    df = build_fact_ventes(mini_tx)
    assert "marge_unitaire" in df.columns
    assert "benefice_transaction" in df.columns
    assert df["marge_unitaire"].iloc[0] == 40.0
    assert df["benefice_transaction"].iloc[0] == 80.0


def test_dim_produits_flag_obsolescence(mini_tx):
    cat = pd.DataFrame({
        "produit_id": ["P001", "P002"],
        "produit_nom": ["Cartouche", "Souris"],
        "famille": ["Cartouche", "Accessoire"],
        "marque": ["HP", "Logitech"],
        "origine_fournisseur": ["Dubaï", "Chine"],
        "cout_achat_unitaire": [60.0, 12.0],
        "prix_vente_unitaire": [100.0, 20.0],
        "marge_theorique": [0.4, 0.4],
    })
    classes = pd.DataFrame({
        "produit_id": ["P001", "P002"],
        "classe_abc": ["A", "C"],
        "classe_xyz": ["X", "Y"],
        "classe_abc_xyz": ["AX", "CY"],
        "cluster_kmeans": [0, 1],
        "libelle_cluster": ["Bestseller", "Rotation modérée"],
        "ca_total_36mois": [10000.0, 500.0],
        "ventes_totales_36mois": [100.0, 50.0],
        "coefficient_variation": [0.3, 0.7],
    })
    obs_feats = pd.DataFrame({
        "produit_id": ["P001", "P002"],
        "jours_depuis_derniere_vente": [5, 250],
        "nombre_mois_consecutifs_sans_vente": [0, 8],
        "valeur_stock_dormant": [100.0, 600.0],
    })
    df = build_dim_produits(cat, classes, obs_feats, obsoletes_ids={"P002"})
    p2 = df[df["produit_id"] == "P002"].iloc[0]
    assert p2["a_risque_obsolescence"] == 1
    assert p2["statut"] == "À risque"


def test_alertes_powerbi_assigns_severity():
    obs = pd.DataFrame({"produit_id": ["P1", "P2", "P3"], "valeur_stock_dormant": [50, 300, 1000]})
    cat = pd.DataFrame({"produit_id": ["P1", "P2", "P3"], "produit_nom": ["A", "B", "C"]})
    out = build_alertes_powerbi(obs, cat)
    severities = dict(zip(out["produit_id"], out["severite"]))
    assert severities == {"P1": "Faible", "P2": "Modérée", "P3": "Élevée"}
