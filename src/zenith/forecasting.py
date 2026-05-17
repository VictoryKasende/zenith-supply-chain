"""Prévisions adaptées par classe (cf. mémoire §3.6).

- SARIMA pour les produits saisonniers stables (baseline + classe CX/CY).
- LightGBM pour les produits courants (classes B, et la majorité).
- LSTM léger (Keras si dispo, sinon fallback baseline) pour la classe A.
- Croston / Naïve pour les produits intermittents (classe Z).
- Prévision par analogie (moyenne famille) pour le cold-start.

Toutes les prévisions sont produites en pas mensuel sur l'horizon de planification.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Outils de séries mensuelles
# ------------------------------------------------------------------ #
def monthly_series_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Renvoie un DataFrame indexé (mois) × colonnes (produits) en quantités."""
    tmp = df.assign(mois=df["date"].values.astype("datetime64[M]"))
    pivot = (
        tmp.groupby(["mois", "produit_id"])["quantite_vendue"]
        .sum()
        .unstack("produit_id")
        .fillna(0)
        .sort_index()
    )
    # Index continu
    if len(pivot) > 0:
        full = pd.date_range(pivot.index.min(), pivot.index.max(), freq="MS")
        pivot = pivot.reindex(full, fill_value=0)
    return pivot


# ------------------------------------------------------------------ #
# Métriques
# ------------------------------------------------------------------ #
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ------------------------------------------------------------------ #
# Modèles
# ------------------------------------------------------------------ #
@dataclass
class ForecastResult:
    produit_id: str
    model: str
    horizon: int
    forecast: np.ndarray
    test_mae: float | None = None
    test_rmse: float | None = None
    test_mape: float | None = None


# -- Naïve saisonnier --
def naive_seasonal(history: pd.Series, horizon: int) -> np.ndarray:
    if len(history) >= 12:
        last_year = history.iloc[-12:].to_numpy()
        reps = int(np.ceil(horizon / 12))
        return np.tile(last_year, reps)[:horizon]
    val = float(history.tail(3).mean()) if len(history) else 0.0
    return np.full(horizon, val)


# -- Croston --
def croston(history: pd.Series, horizon: int, alpha: float = 0.1) -> np.ndarray:
    y = history.to_numpy(dtype=float)
    if (y > 0).sum() == 0:
        return np.zeros(horizon)
    # Initialisation
    first = int(np.argmax(y > 0))
    a, p = y[first], 1.0
    interval = 1
    for t in range(first + 1, len(y)):
        if y[t] > 0:
            a = alpha * y[t] + (1 - alpha) * a
            p = alpha * interval + (1 - alpha) * p
            interval = 1
        else:
            interval += 1
    forecast_value = a / p if p > 0 else 0.0
    return np.full(horizon, max(0.0, forecast_value))


# -- SARIMA --
def sarima_forecast(
    history: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if len(history) < 18 or history.sum() < 5:
        return naive_seasonal(history, horizon)
    try:
        model = SARIMAX(
            history.astype(float),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=50)
        pred = fit.forecast(steps=horizon).to_numpy()
        # Garde-fou : si SARIMA diverge, on retombe sur le naïve saisonnier
        upper = max(history.max() * 3.0, 1.0)
        if not np.isfinite(pred).all() or pred.max() > upper * 5:
            return naive_seasonal(history, horizon)
        return np.clip(pred, 0, upper)
    except Exception as e:
        logger.debug("SARIMA fail: %s", e)
        return naive_seasonal(history, horizon)


# -- LightGBM --
def _make_supervised_monthly(series: pd.Series, lags: tuple[int, ...] = (1, 2, 3, 6, 12)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    df["mois"] = df.index.month
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)
    df["trimestre"] = ((df.index.month - 1) // 3) + 1
    for L in lags:
        df[f"lag_{L}"] = series.shift(L)
    df["roll3"] = series.shift(1).rolling(3).mean()
    df["roll6"] = series.shift(1).rolling(6).mean()
    df["roll12"] = series.shift(1).rolling(12).mean()
    return df


def lightgbm_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    import lightgbm as lgb

    from .config import LGBM_PARAMS

    if len(history) < 18:
        return naive_seasonal(history, horizon)
    train = _make_supervised_monthly(history)
    train = train.dropna()
    if len(train) < 6:
        return naive_seasonal(history, horizon)
    X = train.drop(columns="y")
    y = train["y"]
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)

    preds: list[float] = []
    work = history.copy()
    for h in range(horizon):
        new_idx = work.index[-1] + pd.offsets.MonthBegin(1)
        extended = pd.concat([work, pd.Series([np.nan], index=[new_idx])])
        feats = _make_supervised_monthly(extended).iloc[[-1]].drop(columns="y")
        if feats.isna().any(axis=1).iloc[0]:
            feats = feats.fillna(0)
        yhat = float(np.clip(model.predict(feats)[0], 0, None))
        preds.append(yhat)
        work = pd.concat([work, pd.Series([yhat], index=[new_idx])])
    return np.array(preds)


# -- LSTM (avec fallback) --
def lstm_forecast(history: pd.Series, horizon: int, seq_len: int = 12) -> np.ndarray:
    try:
        from sklearn.neural_network import MLPRegressor
    except Exception:
        return lightgbm_forecast(history, horizon)

    if len(history) < seq_len + 6:
        return naive_seasonal(history, horizon)
    y = history.to_numpy(dtype=float)
    max_y = max(y.max(), 1.0)
    yn = y / max_y

    X, Y = [], []
    for i in range(len(yn) - seq_len):
        X.append(yn[i : i + seq_len])
        Y.append(yn[i + seq_len])
    X = np.array(X)
    Y = np.array(Y)

    try:
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=False,
        )
        model.fit(X, Y)
    except Exception:
        return lightgbm_forecast(history, horizon)

    window = yn[-seq_len:].copy()
    preds = []
    for _ in range(horizon):
        pred = float(model.predict(window.reshape(1, -1))[0])
        pred = max(0.0, pred)
        preds.append(pred * max_y)
        window = np.concatenate([window[1:], [pred]])
    return np.array(preds)


# ------------------------------------------------------------------ #
# Sélection automatique du modèle selon la classe ABC × XYZ
# ------------------------------------------------------------------ #
def choose_model_for_class(classe_abc: str, classe_xyz: str, intermittent: bool) -> str:
    """Stratégie d'affectation (cf. mémoire Tab. 3.3)."""
    if intermittent:
        return "croston"
    if classe_abc == "A":
        return "lstm"
    if classe_abc == "B":
        return "lightgbm"
    # Classe C
    if classe_xyz == "X":
        return "sarima"
    if classe_xyz == "Z":
        return "croston"
    return "sarima"


MODEL_DISPATCH = {
    "sarima": sarima_forecast,
    "lightgbm": lightgbm_forecast,
    "lstm": lstm_forecast,
    "croston": croston,
    "naive": naive_seasonal,
}


def _clip_to_history(yhat: np.ndarray, history: pd.Series) -> np.ndarray:
    """Plafonne les prévisions au quantile 95 × 1.5 de l'historique non nul.

    Garde-fou contre les sur-prévisions SARIMA/LSTM qui rendent l'optimisation
    irréaliste. La borne min reste 0.
    """
    if len(history) == 0:
        return np.clip(yhat, 0, None)
    nonzero = history[history > 0]
    if nonzero.empty:
        return np.zeros_like(yhat)
    upper = max(float(nonzero.quantile(0.95)) * 1.5, float(nonzero.max()))
    return np.clip(yhat, 0, upper)


def forecast_one_product(
    history: pd.Series,
    horizon: int,
    model_name: str,
    test_history: pd.Series | None = None,
) -> ForecastResult:
    """Entraîne sur `history` et renvoie un ForecastResult.

    Si `test_history` est fourni, calcule MAE/RMSE/MAPE sur les premiers points.
    """
    func = MODEL_DISPATCH[model_name]
    yhat = _clip_to_history(func(history, horizon), history)

    metrics = {"mae": None, "rmse": None, "mape": None}
    if test_history is not None and len(test_history) > 0:
        n = min(len(test_history), len(yhat))
        if n > 0:
            yt = test_history.iloc[:n].to_numpy(dtype=float)
            yp = yhat[:n]
            metrics = {"mae": mae(yt, yp), "rmse": rmse(yt, yp), "mape": mape(yt, yp)}

    return ForecastResult(
        produit_id=history.name if hasattr(history, "name") else "",
        model=model_name,
        horizon=horizon,
        forecast=yhat,
        test_mae=metrics["mae"],
        test_rmse=metrics["rmse"],
        test_mape=metrics["mape"],
    )
