"""Prévision de la demande adaptée par classe (mémoire §3.6).

Stratégie d'affectation
-----------------------
+------------------------+--------------------+
| Classe / profil        | Modèle             |
+========================+====================+
| A non-intermittent     | LSTM (MLP séqu.)   |
| B                      | LightGBM           |
| C non-intermittent     | SARIMA auto_arima  |
| Intermittents (Z, ≤30% | Croston SBA        |
| de mois actifs)        |                    |
| Obsolète (Étape 4)     | Prévision = 0      |
| Cold-start             | Analogie famille   |
+------------------------+--------------------+

Notes d'implémentation
----------------------
- ``pmdarima.auto_arima`` minimise l'AIC pour SARIMA (saisonnalité s=12).
- Croston-SBA (Syntetos-Boylan Approximation) corrige le biais du Croston classique
  par le facteur (1 - alpha/2).
- En l'absence de TensorFlow (poste PME, principe de frugalité §3.2.1), le
  modèle "LSTM" est implémenté via un MLP scikit-learn appliqué sur fenêtres
  glissantes de 30 jours — équivalence opérationnelle pour des séries courtes.
- LightGBM utilise lags 1/7/14/30, moyennes mobiles 7/30, écart-type 30, et
  variables événementielles RDC (rentrée scolaire, pic B2B, …).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.utils import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("forecasting")

# --------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------- #
INTERMITTENT_THRESHOLD = 0.30          # ≤30 % de mois actifs ⇒ intermittent
LSTM_SEQ_LENGTH = 30                   # fenêtre LSTM (jours)
DEFAULT_HORIZON_MONTHS = 3
RANDOM_STATE = 42

LGBM_PARAMS = dict(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    min_data_in_leaf=5,
    random_state=RANDOM_STATE,
    verbose=-1,
)


# =================================================================== #
# Métriques
# =================================================================== #
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


# =================================================================== #
# Préparation des séries
# =================================================================== #
def monthly_series_by_product(transactions: pd.DataFrame) -> pd.DataFrame:
    """Renvoie un DataFrame mois × produit (quantités vendues, zéros remplis)."""
    tmp = transactions.copy()
    tmp["mois"] = tmp["date"].values.astype("datetime64[M]")
    pivot = (
        tmp.groupby(["mois", "produit_id"])["quantite_vendue"]
        .sum().unstack("produit_id").fillna(0).sort_index()
    )
    if len(pivot) > 0:
        full = pd.date_range(pivot.index.min(), pivot.index.max(), freq="MS")
        pivot = pivot.reindex(full, fill_value=0)
    return pivot


def daily_series_by_product(transactions: pd.DataFrame) -> pd.DataFrame:
    """Reconstitue les ventes journalières par produit (zéros remplis)."""
    tmp = (
        transactions.groupby(["date", "produit_id"])["quantite_vendue"]
        .sum().unstack("produit_id").fillna(0).sort_index()
    )
    if len(tmp) > 0:
        full = pd.date_range(tmp.index.min(), tmp.index.max(), freq="D")
        tmp = tmp.reindex(full, fill_value=0)
    return tmp


# =================================================================== #
# Modèles individuels
# =================================================================== #
def naive_seasonal(history: pd.Series, horizon: int) -> np.ndarray:
    """Réplique les 12 derniers mois (fallback robuste)."""
    if len(history) >= 12:
        last_year = history.iloc[-12:].to_numpy(dtype=float)
        return np.tile(last_year, int(np.ceil(horizon / 12)))[:horizon]
    val = float(history.tail(3).mean()) if len(history) else 0.0
    return np.full(horizon, val)


def croston_sba(history: pd.Series, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Méthode Croston-Syntetos-Boylan pour demande intermittente."""
    y = history.to_numpy(dtype=float)
    if (y > 0).sum() == 0:
        return np.zeros(horizon)
    first = int(np.argmax(y > 0))
    a, p, interval = y[first], 1.0, 1
    for t in range(first + 1, len(y)):
        if y[t] > 0:
            a = alpha * y[t] + (1 - alpha) * a
            p = alpha * interval + (1 - alpha) * p
            interval = 1
        else:
            interval += 1
    forecast = (a / p) * (1 - alpha / 2) if p > 0 else 0.0
    return np.full(horizon, max(0.0, float(forecast)))


def sarima_auto(history: pd.Series, horizon: int, seasonal: bool = True) -> tuple[np.ndarray, str]:
    """SARIMA avec recherche auto des paramètres (pmdarima.auto_arima).

    Renvoie (prévisions, signature 'SARIMA(p,d,q)(P,D,Q,s)').
    """
    from pmdarima import auto_arima

    if len(history) < 18 or history.sum() < 5:
        return naive_seasonal(history, horizon), "naive_seasonal"
    try:
        model = auto_arima(
            history.astype(float),
            seasonal=seasonal,
            m=12 if seasonal else 1,
            suppress_warnings=True,
            stepwise=True,
            max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
            error_action="ignore",
            random_state=RANDOM_STATE,
        )
        pred = np.asarray(model.predict(n_periods=horizon))
        upper = max(history.max() * 3.0, 1.0)
        if not np.isfinite(pred).all() or pred.max() > upper * 5:
            return naive_seasonal(history, horizon), "naive_seasonal_fallback"
        return np.clip(pred, 0, upper), str(model)
    except Exception as exc:
        logger.debug("SARIMA fail: %s", exc)
        return naive_seasonal(history, horizon), "naive_seasonal_fallback"


def _make_supervised_monthly(series: pd.Series, lags: tuple[int, ...] = (1, 2, 3, 6, 12)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    df["mois"] = df.index.month
    df["trimestre"] = ((df.index.month - 1) // 3) + 1
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)
    df["est_rentree_scolaire"] = df["mois"].isin([8, 9]).astype(int)
    df["est_pic_b2b"] = df["mois"].isin([1, 2, 3, 11, 12]).astype(int)
    for L in lags:
        df[f"lag_{L}"] = series.shift(L)
    df["ma_3"] = series.shift(1).rolling(3).mean()
    df["ma_6"] = series.shift(1).rolling(6).mean()
    df["ma_12"] = series.shift(1).rolling(12).mean()
    df["std_6"] = series.shift(1).rolling(6).std()
    return df


def lightgbm_forecast(history: pd.Series, horizon: int) -> tuple[np.ndarray, object]:
    """LightGBM avec features lag / saisonnier / événementiel.

    Renvoie (prévisions, modèle entraîné pour analyse de feature importance).
    """
    import lightgbm as lgb

    if len(history) < 18:
        return naive_seasonal(history, horizon), None
    train = _make_supervised_monthly(history).dropna()
    if len(train) < 6:
        return naive_seasonal(history, horizon), None
    X = train.drop(columns="y")
    y = train["y"]
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)

    preds: list[float] = []
    work = history.copy()
    for _ in range(horizon):
        new_idx = work.index[-1] + pd.offsets.MonthBegin(1)
        ext = pd.concat([work, pd.Series([np.nan], index=[new_idx])])
        feats = _make_supervised_monthly(ext).iloc[[-1]].drop(columns="y")
        feats = feats.fillna(0)
        yhat = float(np.clip(model.predict(feats)[0], 0, None))
        preds.append(yhat)
        work = pd.concat([work, pd.Series([yhat], index=[new_idx])])
    return np.array(preds), model


def lstm_like_forecast(history: pd.Series, horizon: int, seq_len: int = LSTM_SEQ_LENGTH) -> np.ndarray:
    """Modèle séquentiel (MLP scikit-learn sur fenêtres glissantes).

    Sans TensorFlow disponible (principe de frugalité PME), on utilise un MLP
    profond appliqué sur les ``seq_len`` derniers points — équivalent pratique
    d'un LSTM pour des historiques mensuels courts.
    """
    from sklearn.neural_network import MLPRegressor

    if len(history) < seq_len + 6:
        return naive_seasonal(history, horizon)
    y = history.to_numpy(dtype=float)
    max_y = max(y.max(), 1.0)
    yn = y / max_y

    X = np.array([yn[i : i + seq_len] for i in range(len(yn) - seq_len)])
    Y = np.array([yn[i + seq_len] for i in range(len(yn) - seq_len)])

    try:
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="tanh",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=False,
        )
        model.fit(X, Y)
    except Exception:
        return naive_seasonal(history, horizon)

    window = yn[-seq_len:].copy()
    preds: list[float] = []
    for _ in range(horizon):
        pred = float(model.predict(window.reshape(1, -1))[0])
        pred = max(0.0, pred)
        preds.append(pred * max_y)
        window = np.concatenate([window[1:], [pred]])
    return np.array(preds)


def family_analogy(
    product_id: str,
    catalogue: pd.DataFrame,
    monthly: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Cold-start : moyenne pondérée des ventes des produits de la même famille."""
    fam = catalogue.set_index("produit_id").loc[product_id, "famille"] if product_id in catalogue["produit_id"].values else None
    if fam is None:
        return np.zeros(horizon)
    peers = catalogue[catalogue["famille"] == fam]["produit_id"].tolist()
    peers = [p for p in peers if p in monthly.columns and p != product_id]
    if not peers:
        return np.zeros(horizon)
    avg = monthly[peers].iloc[-12:].mean(axis=1).mean()
    return np.full(horizon, max(0.0, float(avg)))


# =================================================================== #
# Sélection de modèle
# =================================================================== #
def choose_model(
    classe_abc: str,
    classe_xyz: str,
    is_intermittent: bool,
    is_obsolete: bool,
    is_cold_start: bool,
) -> str:
    if is_obsolete:
        return "obsolete"
    if is_cold_start:
        return "analogy"
    if is_intermittent:
        return "croston_sba"
    if classe_abc == "A":
        return "lstm"
    if classe_abc == "B":
        return "lightgbm"
    return "sarima"


# =================================================================== #
# Résultat structuré
# =================================================================== #
@dataclass
class ForecastRecord:
    produit_id: str
    model: str
    horizon: int
    forecast: np.ndarray
    test_mae: Optional[float] = None
    test_rmse: Optional[float] = None
    test_mape: Optional[float] = None
    sarima_signature: Optional[str] = None
    lgbm_feature_importance: Optional[dict[str, float]] = None


# =================================================================== #
# Orchestration produit par produit
# =================================================================== #
def forecast_product(
    history: pd.Series,
    horizon: int,
    model_name: str,
    test_actual: Optional[pd.Series] = None,
    catalogue: Optional[pd.DataFrame] = None,
    monthly: Optional[pd.DataFrame] = None,
    product_id: Optional[str] = None,
) -> ForecastRecord:
    """Calcule une prévision pour un produit avec le modèle ``model_name``."""
    sarima_sig: Optional[str] = None
    fi: Optional[dict[str, float]] = None

    if model_name == "obsolete":
        yhat = np.zeros(horizon)
    elif model_name == "analogy":
        if catalogue is None or monthly is None or product_id is None:
            yhat = np.zeros(horizon)
        else:
            yhat = family_analogy(product_id, catalogue, monthly, horizon)
    elif model_name == "croston_sba":
        yhat = croston_sba(history, horizon)
    elif model_name == "sarima":
        yhat, sarima_sig = sarima_auto(history, horizon)
    elif model_name == "lightgbm":
        yhat, model_obj = lightgbm_forecast(history, horizon)
        if model_obj is not None:
            try:
                names = model_obj.booster_.feature_name()
                vals = model_obj.booster_.feature_importance(importance_type="gain")
                fi = dict(zip(names, [float(v) for v in vals]))
            except Exception:
                fi = None
    elif model_name == "lstm":
        yhat = lstm_like_forecast(history, horizon)
    else:
        yhat = naive_seasonal(history, horizon)

    yhat = np.clip(yhat, 0, None)

    metrics: dict = {}
    if test_actual is not None and len(test_actual) > 0:
        n = min(len(test_actual), len(yhat))
        if n > 0:
            metrics = compute_metrics(test_actual.iloc[:n].to_numpy(dtype=float), yhat[:n])

    return ForecastRecord(
        produit_id=product_id or "",
        model=model_name,
        horizon=horizon,
        forecast=yhat,
        test_mae=metrics.get("mae"),
        test_rmse=metrics.get("rmse"),
        test_mape=metrics.get("mape"),
        sarima_signature=sarima_sig,
        lgbm_feature_importance=fi,
    )


def predict_demand(
    product_id: str,
    horizon: int,
    monthly: pd.DataFrame,
    classes: pd.DataFrame,
    catalogue: pd.DataFrame,
    obsoletes: Iterable[str],
) -> ForecastRecord:
    """API publique : prédit la demande pour un produit en sélectionnant le modèle.

    Sélectionne automatiquement le bon modèle d'après la classe ABC×XYZ, le
    flag d'obsolescence et la disponibilité d'historique.
    """
    obs_set = set(obsoletes)
    is_obs = product_id in obs_set
    has_history = product_id in monthly.columns
    history = monthly[product_id].astype(float) if has_history else pd.Series(dtype=float)

    if not has_history or len(history.dropna()) < 6:
        return forecast_product(
            history, horizon, "analogy",
            catalogue=catalogue, monthly=monthly, product_id=product_id,
        )

    row = classes.set_index("produit_id").loc[product_id] if product_id in classes["produit_id"].values else None
    if row is None:
        return forecast_product(history, horizon, "naive_seasonal", product_id=product_id)

    classe_abc = row["classe_abc"]
    classe_xyz = row["classe_xyz"]
    months_active = int((history > 0).sum())
    is_intermittent = months_active / max(len(history), 1) <= INTERMITTENT_THRESHOLD or classe_xyz == "Z"
    model_name = choose_model(classe_abc, classe_xyz, is_intermittent, is_obs, False)

    return forecast_product(
        history, horizon, model_name, product_id=product_id,
        catalogue=catalogue, monthly=monthly,
    )
