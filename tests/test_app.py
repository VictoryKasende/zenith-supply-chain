"""Tests d'intégration légers pour l'application Streamlit.

On ne lance pas un vrai serveur : on vérifie que les loaders cachés renvoient
des DataFrames non vides et compatibles avec les pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _has_outputs() -> bool:
    return (ROOT / "outputs/tables/classification_produits.csv").exists()


pytestmark = pytest.mark.skipif(not _has_outputs(),
                                 reason="Pipeline outputs absent — run scripts/run_*.py first")


def test_app_module_imports():
    # Import-only test : vérifie qu'on peut importer le module sans crash.
    import importlib
    mod = importlib.import_module("app.zenith_tool")
    assert hasattr(mod, "page_dashboard")
    assert hasattr(mod, "page_classification")
    assert hasattr(mod, "page_obsolescence")
    assert hasattr(mod, "page_previsions")
    assert hasattr(mod, "page_commandes")
    assert hasattr(mod, "page_simulation")


def test_loaders_return_dataframes():
    from app.zenith_tool import (
        load_classification, load_commandes, load_obsolescence,
        load_previsions, load_transactions, build_products_input,
    )
    for fn in [load_classification, load_commandes, load_obsolescence,
               load_previsions, load_transactions, build_products_input]:
        df = fn()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


def test_products_input_has_required_columns():
    from app.zenith_tool import build_products_input
    df = build_products_input()
    required = {"produit_id", "classe_abc", "a_risque_obsolescence",
                "cout_achat_unitaire", "prix_vente_unitaire", "stock_courant",
                "origine_fournisseur"}
    assert required.issubset(set(df.columns))
