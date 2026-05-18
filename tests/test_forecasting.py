"""Tests unitaires du module ``src.forecasting``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting import (
    choose_model,
    compute_metrics,
    croston_sba,
    forecast_product,
    lightgbm_forecast,
    monthly_series_by_product,
    naive_seasonal,
)


@pytest.fixture
def long_history() -> pd.Series:
    idx = pd.date_range("2022-08-01", periods=36, freq="MS")
    np.random.seed(0)
    vals = 50 + 20 * np.sin(np.arange(36) * 2 * np.pi / 12) + np.random.normal(0, 5, 36)
    return pd.Series(np.clip(vals, 0, None), index=idx, name="P_TEST")


def test_choose_model_obsolete_overrides_everything():
    assert choose_model("A", "X", False, True, False) == "obsolete"


def test_choose_model_cold_start_uses_analogy():
    assert choose_model("B", "Y", False, False, True) == "analogy"


def test_choose_model_intermittent_uses_croston():
    assert choose_model("A", "Z", True, False, False) == "croston_sba"


def test_choose_model_class_routing():
    assert choose_model("A", "X", False, False, False) == "lstm"
    assert choose_model("B", "Y", False, False, False) == "lightgbm"
    assert choose_model("C", "X", False, False, False) == "sarima"


def test_naive_seasonal_length(long_history):
    out = naive_seasonal(long_history, 6)
    assert len(out) == 6
    assert (out >= 0).all()


def test_croston_sba_returns_constant(long_history):
    # Croston SBA renvoie une constante (moyenne pondérée)
    h = long_history.copy()
    h.iloc[::4] = 0  # intermittence
    out = croston_sba(h, 6)
    assert len(out) == 6
    assert np.allclose(out, out[0])


def test_lightgbm_forecast_returns_horizon(long_history):
    out, model = lightgbm_forecast(long_history, 4)
    assert len(out) == 4
    assert (out >= 0).all()


def test_forecast_product_obsolete_zero(long_history):
    r = forecast_product(long_history, 3, "obsolete", product_id="P_TEST")
    assert (r.forecast == 0).all()


def test_compute_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y, y.copy())
    assert m["mae"] == 0.0 and m["rmse"] == 0.0


def test_monthly_series_returns_zero_filled_pivot():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-03-20", "2024-01-22"]),
        "produit_id": ["P1", "P1", "P2"],
        "quantite_vendue": [2.0, 3.0, 5.0],
    })
    m = monthly_series_by_product(df)
    # Doit couvrir janvier, février, mars
    assert len(m) == 3
    assert m.loc["2024-02-01", "P1"] == 0
    assert m.loc["2024-01-01", "P2"] == 5
