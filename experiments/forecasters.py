"""Méthodes de prévision évaluées en rolling-origin.

Le pipeline et les baselines appellent les fonctions existantes de ``src``
(aucune ré-implémentation) afin que les chiffres restent comparables à ceux
de l'article. Seul ``lightgbm_global`` (modèle poolé) est nouveau.

Correspondance avec les noms de l'article
-----------------------------------------
- ``sarima_uniform``   = « SARIMA global » de l'article : SARIMA(1,1,1)(1,1,1,12)
  fixe, ajusté **série par série** (``predict_sarima_quick``).
- ``lightgbm_uniform`` = « LightGBM global » de l'article : un LightGBM
  **par produit** (``predict_lightgbm_quick``).
- ``lightgbm_global``  = vrai modèle global : un seul LightGBM entraîné sur
  toutes les séries ≤ t0, normalisées par leur échelle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.config import SEED
from experiments.data import OriginState
from src.forecasting import (
    LSTM_SEQ_LENGTH,
    _make_supervised_monthly,
    assign_model,
    croston_sba,
    forecast_product,
    lstm_like_forecast,
    naive_seasonal,
)
from src.validation_hypotheses import (
    predict_croston_classic,
    predict_lightgbm_quick,
    predict_moving_average,
    predict_sarima_quick,
)

PIPELINE_METHODS = ("pipeline", "pipeline_mlp12", "pipeline_no_obso")
SIMPLE_BASELINES = ("seasonal_naive", "moving_average_3", "croston", "croston_sba")
MODEL_BASELINES = ("sarima_uniform", "lightgbm_uniform", "lightgbm_global")
ALL_METHODS = PIPELINE_METHODS + SIMPLE_BASELINES + MODEL_BASELINES

MLP_VARIANT_SEQ_LEN = 12


# --------------------------------------------------------------------- #
# Pipeline (tel que codé) et variantes
# --------------------------------------------------------------------- #
def _effective(model: str, record, history: pd.Series, horizon: int, seq_len: int = LSTM_SEQ_LENGTH) -> str:
    """Modèle réellement utilisé après les replis internes de ``src.forecasting``."""
    if model == "sarima":
        sig = record.sarima_signature or ""
        return "naive_seasonal_fallback" if sig.startswith("naive") else "sarima"
    if model == "lightgbm":
        return "lightgbm" if record.lgbm_feature_importance is not None else "naive_seasonal_fallback"
    if model == "lstm":
        if len(history) < seq_len + 6:
            return "naive_seasonal_fallback"
        same = np.allclose(record.forecast, np.clip(naive_seasonal(history, horizon), 0, None))
        return "naive_seasonal_fallback" if same else "mlp"
    return model


def pipeline_forecast(state: OriginState, catalogue: pd.DataFrame, pid: str, horizon: int,
                      ignore_obsolescence: bool = False) -> tuple[np.ndarray, str, str]:
    """Prévision du pipeline différencié pour un produit (Étape 5 à l'identique)."""
    history = state.monthly[pid].astype(float)
    abc, xyz = state.class_of(pid)
    is_obs = (pid in state.obsolete) and not ignore_obsolescence
    model = assign_model(history, abc, xyz, is_obs, has_history=True)
    rec = forecast_product(history, horizon, model, catalogue=catalogue,
                           monthly=state.monthly, product_id=pid)
    return rec.forecast.astype(float), model, _effective(model, rec, history, horizon)


def mlp12_forecast(history: pd.Series, horizon: int) -> tuple[np.ndarray, str]:
    """Variante : même MLP que l'Étape 5 mais fenêtre de 12 mois (au lieu de 30)."""
    pred = np.clip(lstm_like_forecast(history, horizon, seq_len=MLP_VARIANT_SEQ_LEN), 0, None)
    if len(history) < MLP_VARIANT_SEQ_LEN + 6:
        return pred, "naive_seasonal_fallback"
    same = np.allclose(pred, np.clip(naive_seasonal(history, horizon), 0, None))
    return pred, ("naive_seasonal_fallback" if same else "mlp12")


# --------------------------------------------------------------------- #
# Baselines série par série
# --------------------------------------------------------------------- #
def baseline_forecast(method: str, history: pd.Series, horizon: int) -> np.ndarray:
    if method == "seasonal_naive":
        out = naive_seasonal(history, horizon)
    elif method == "moving_average_3":
        out = predict_moving_average(history, horizon, window=3)
    elif method == "croston":
        out = predict_croston_classic(history, horizon)
    elif method == "croston_sba":
        out = croston_sba(history, horizon)
    elif method == "sarima_uniform":
        out = predict_sarima_quick(history, horizon)
    elif method == "lightgbm_uniform":
        out = predict_lightgbm_quick(history, horizon)
    else:
        raise ValueError(method)
    return np.clip(np.asarray(out, dtype=float), 0, None)


# --------------------------------------------------------------------- #
# LightGBM global (poolé)
# --------------------------------------------------------------------- #
GLOBAL_LGBM_PARAMS = dict(
    num_leaves=31, learning_rate=0.05, n_estimators=500, min_data_in_leaf=20,
    random_state=SEED, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1,
)


def _series_scale(history: pd.Series) -> float:
    """Échelle de normalisation : moyenne des 12 derniers mois, sinon de tout l'historique."""
    s = float(history.tail(12).mean())
    if s <= 0:
        s = float(history[history > 0].mean()) if (history > 0).any() else 1.0
    return max(s, 1e-6)


def lightgbm_global_forecasts(state: OriginState, horizon: int,
                              products: list[str] | None = None) -> dict[str, np.ndarray]:
    """Un LightGBM unique entraîné sur toutes les séries connues à t0.

    Chaque série est divisée par son échelle (moyenne des 12 derniers mois ≤ t0) ;
    mêmes features que le LightGBM de l'Étape 5 ; lignes antérieures au premier
    mois de vente exclues ; prévision récursive h = 1..H puis remise à l'échelle.
    L'apprentissage utilise tous les produits connus à t0, même si
    ``products`` restreint les produits prédits.
    """
    import lightgbm as lgb

    known = state.products
    X_parts, y_parts = [], []
    scales: dict[str, float] = {}
    for pid in known:
        hist = state.monthly[pid].astype(float)
        scales[pid] = _series_scale(hist)
        sup = _make_supervised_monthly(hist / scales[pid])
        first = hist.ne(0).idxmax()
        sup = sup.loc[sup.index >= first].dropna()
        if len(sup):
            X_parts.append(sup.drop(columns="y"))
            y_parts.append(sup["y"])
    X = pd.concat(X_parts)
    y = pd.concat(y_parts)
    model = lgb.LGBMRegressor(**GLOBAL_LGBM_PARAMS)
    model.fit(X, y)

    out: dict[str, np.ndarray] = {}
    for pid in (products if products is not None else known):
        work = state.monthly[pid].astype(float) / scales[pid]
        preds = []
        for _ in range(horizon):
            new_idx = work.index[-1] + pd.offsets.MonthBegin(1)
            ext = pd.concat([work, pd.Series([np.nan], index=[new_idx])])
            feats = _make_supervised_monthly(ext).iloc[[-1]].drop(columns="y").fillna(0)
            yhat = float(np.clip(model.predict(feats)[0], 0, None))
            preds.append(yhat)
            work = pd.concat([work, pd.Series([yhat], index=[new_idx])])
        out[pid] = np.asarray(preds) * scales[pid]
    return out
