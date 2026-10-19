"""Critère d'acceptation P0-1 : aucune fuite temporelle dans le rolling-origin.

Modifier les données brutes postérieures à t0 (quantités, prix, stocks,
suppression et ajout de lignes) ne doit changer ni les prévisions, ni les
classes ABC/XYZ/K-Means, ni les drapeaux d'obsolescence, ni le stock central,
ni les dénominateurs MASE à cette origine. Un contrôle positif vérifie que le
test est sensible : une modification ≤ t0 change bien les sorties.
"""
import experiments  # noqa: F401  (fixe les threads avant numpy)

import numpy as np
import pandas as pd
import pytest

from experiments.data import build_state, load_cat, load_raw
from experiments.forecasters import ALL_METHODS
from experiments.rolling_origin import run_origin
from src.forecasting import assign_model

T0 = pd.Timestamp("2024-06-30")


def _products_covering_every_model() -> list[str]:
    """Deux produits par modèle affecté à t0 (LSTM/MLP, LightGBM, SARIMA, Croston, obsolète)."""
    state = build_state(load_raw(), load_cat(), T0)
    picked: dict[str, list[str]] = {}
    for pid in state.products:
        abc, xyz = state.class_of(pid)
        model = assign_model(state.monthly[pid], abc, xyz, pid in state.obsolete)
        picked.setdefault(model, [])
        if len(picked[model]) < 2:
            picked[model].append(pid)
    assert {"lstm", "lightgbm", "sarima", "croston_sba", "obsolete"} <= set(picked)
    return sorted(p for ps in picked.values() for p in ps)


def _perturb(raw: pd.DataFrame, future: bool) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    out = raw.copy()
    mask = (out["date"] > T0) if future else (out["date"] > T0 - pd.Timedelta(days=60)) & (out["date"] <= T0)
    n = int(mask.sum())
    out.loc[mask, "quantite_vendue"] = out.loc[mask, "quantite_vendue"] * 3 + rng.integers(1, 50, n)
    out.loc[mask, "prix_vente_unitaire"] = out.loc[mask, "prix_vente_unitaire"] * 0.5
    out.loc[mask, "stock_apres_vente"] = 999.0
    if future:
        drop = out.index[mask][rng.random(n) < 0.2]
        out = out.drop(drop)
        extra = out[out["date"] > T0].sample(500, random_state=42).copy()
        extra["transaction_id"] = [f"TFAKE{i:05d}" for i in range(len(extra))]
        extra["produit_id"] = extra["produit_id"].iloc[0]
        out = pd.concat([out, extra], ignore_index=True)
    return out


@pytest.fixture(scope="module")
def runs():
    raw, cat = load_raw(), load_cat()
    products = _products_covering_every_model()
    base = run_origin(T0, 3, ALL_METHODS, raw=raw, catalogue=cat, products=products)
    fut = run_origin(T0, 3, ALL_METHODS, raw=_perturb(raw, future=True), catalogue=cat, products=products)
    past = run_origin(T0, 3, ALL_METHODS, raw=_perturb(raw, future=False), catalogue=cat, products=products)
    return base, fut, past


def _sorted(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values(keys).reset_index(drop=True)


def test_future_changes_do_not_change_forecasts(runs):
    base, fut, _ = runs
    keys = ["product_id", "method", "horizon"]
    assert len(base["forecasts"]) > 0
    pd.testing.assert_frame_equal(_sorted(base["forecasts"], keys), _sorted(fut["forecasts"], keys),
                                  check_exact=True)


def test_future_changes_do_not_change_classes_flags_stock(runs):
    base, fut, _ = runs
    pd.testing.assert_frame_equal(_sorted(base["states"], ["produit_id"]),
                                  _sorted(fut["states"], ["produit_id"]), check_exact=True)
    pd.testing.assert_frame_equal(_sorted(base["scales"], ["product_id"]),
                                  _sorted(fut["scales"], ["product_id"]), check_exact=True)


def test_past_changes_are_detected(runs):
    """Contrôle positif : le test ci-dessus n'est pas trivialement vrai."""
    base, _, past = runs
    keys = ["product_id", "method", "horizon"]
    a = _sorted(base["forecasts"], keys)["y_pred"].to_numpy()
    b = _sorted(past["forecasts"], keys)["y_pred"].to_numpy()
    assert not np.array_equal(a, b)
    assert not base["states"]["central_stock"].equals(past["states"]["central_stock"])
