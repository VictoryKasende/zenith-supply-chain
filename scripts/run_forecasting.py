"""Exécute l'Étape 5 — Prévisions de la demande.

Lit :
- data/processed/zenith_clean.csv
- outputs/tables/classification_produits.csv
- outputs/tables/produits_obsoletes.csv
- data/raw/catalogue_produits_250.csv

Produit :
- outputs/tables/previsions_complet.csv (produit × mois × prevision × modèle)
- outputs/tables/comparaison_modeles.csv (MAE/RMSE/MAPE par produit)
- outputs/tables/forecast_metrics_by_class.csv (synthèse par classe ABC × modèle)
- outputs/tables/lgbm_feature_importance.csv (gain moyen par feature)
- outputs/figures/fc_01..04.png (4 figures)
- outputs/models/sarima_signatures.csv (paramètres SARIMA retenus)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting import (
    INTERMITTENT_THRESHOLD,
    choose_model,
    forecast_product,
    monthly_series_by_product,
)
from src.utils import (
    FIG_DIR, MODELS_DIR, PROCESSED_DIR, RAW_CATALOGUE, TAB_DIR,
    TRAIN_END, VAL_END, setup_logger, temporal_split,
)

logger = setup_logger("pipeline.forecasting")
sns.set_theme(style="whitegrid", context="talk")

HORIZON_TEST = 4   # mois d'évaluation (avril → juillet 2025)
HORIZON_DEPLOY = 6  # mois de prévision opérationnelle


def fig_metrics_by_class(metrics_df: pd.DataFrame) -> None:
    sub = metrics_df.dropna(subset=["mae"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=sub, x="classe_abc", y="mae", hue="modele",
                order=["A", "B", "C"], ax=ax)
    ax.set_title("MAE par classe ABC × modèle (test 4 mois)")
    ax.set_ylabel("MAE (unités/mois)"); ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fc_01_mae_by_class.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_real_vs_pred(records: list, monthly_full: pd.DataFrame, classes: pd.DataFrame, n_per_class: int = 2) -> None:
    """4 produits emblématiques (A LSTM, B LightGBM, C SARIMA, Z Croston)."""
    by_model: dict[str, list] = {"lstm": [], "lightgbm": [], "sarima": [], "croston_sba": []}
    for r in records:
        if r.model in by_model and r.test_mae is not None:
            by_model[r.model].append(r)
    for k in by_model:
        by_model[k].sort(key=lambda r: r.test_mae)
        by_model[k] = by_model[k][:n_per_class]

    flat = [r for k in ["lstm", "lightgbm", "sarima", "croston_sba"] for r in by_model[k]]
    if not flat:
        return
    n = len(flat)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), sharex=False)
    axes = np.atleast_2d(axes).flatten()
    for i, r in enumerate(flat):
        if r.produit_id not in monthly_full.columns:
            continue
        ts = monthly_full[r.produit_id].astype(float)
        last_history = ts.iloc[:-r.horizon] if len(ts) > r.horizon else ts
        future_idx = ts.index[-r.horizon:] if len(ts) >= r.horizon else pd.date_range(ts.index[-1], periods=r.horizon, freq="MS")
        ax = axes[i]
        ax.plot(ts.index, ts.values, color="#1f4e79", linewidth=1.5, label="réel")
        ax.plot(future_idx, r.forecast, color="#ff6b6b", linewidth=2, label=f"prévu ({r.model})")
        ax.set_title(f"{r.produit_id} — MAE {r.test_mae:.2f}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Courbes réel vs prévu — meilleurs produits par classe/modèle", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fc_02_real_vs_pred.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_model_distribution(metrics_df: pd.DataFrame) -> None:
    counts = metrics_df["modele"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="viridis", ax=ax, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=12)
    ax.set_title("Nombre de produits par modèle de prévision retenu")
    ax.set_xlabel("Produits")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fc_03_model_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_lgbm_importance(records: list) -> None:
    bag: dict[str, list[float]] = {}
    for r in records:
        if r.model == "lightgbm" and r.lgbm_feature_importance:
            for k, v in r.lgbm_feature_importance.items():
                bag.setdefault(k, []).append(v)
    if not bag:
        return
    df = pd.DataFrame({
        "feature": list(bag.keys()),
        "importance_moy": [float(np.mean(v)) for v in bag.values()],
    }).sort_values("importance_moy", ascending=False)
    df.to_csv(TAB_DIR / "lgbm_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df, x="importance_moy", y="feature", hue="feature",
                palette="flare", ax=ax, legend=False)
    ax.set_title("Importance moyenne des features LightGBM (gain)")
    ax.set_xlabel("Gain moyen")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fc_04_lgbm_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logger.info("Lancement des prévisions")
    transactions = pd.read_csv(PROCESSED_DIR / "zenith_clean.csv", parse_dates=["date"])
    classes = pd.read_csv(TAB_DIR / "classification_produits.csv")
    obsoletes = pd.read_csv(TAB_DIR / "produits_obsoletes.csv")["produit_id"].tolist()
    catalogue = pd.read_csv(RAW_CATALOGUE)

    # Split temporel : on entraîne sur train+val, on évalue sur test
    train_df = transactions[transactions["date"] <= pd.Timestamp(VAL_END)].copy()
    test_df = transactions[transactions["date"] > pd.Timestamp(VAL_END)].copy()

    monthly_train = monthly_series_by_product(train_df)
    monthly_test = monthly_series_by_product(test_df)
    monthly_full = monthly_series_by_product(transactions)
    logger.info(
        "Séries mensuelles — train %d × %d, test %d × %d, full %d × %d",
        *monthly_train.shape, *monthly_test.shape, *monthly_full.shape,
    )

    records: list = []
    rows_forecast: list[dict] = []
    sarima_sigs: list[dict] = []
    obs_set = set(obsoletes)

    t0 = time.time()
    for _, row in tqdm(classes.iterrows(), total=len(classes), desc="Prévisions", ncols=80):
        pid = row["produit_id"]
        classe_abc = row["classe_abc"]
        classe_xyz = row["classe_xyz"]
        is_obs = pid in obs_set
        has_history = pid in monthly_train.columns
        history = monthly_train[pid].astype(float) if has_history else pd.Series(dtype=float)

        months_active = int((history > 0).sum())
        is_intermittent = (
            months_active / max(len(history), 1) <= INTERMITTENT_THRESHOLD
            or classe_xyz == "Z"
        )
        is_cold = (not has_history) or len(history.dropna()) < 6
        model_name = choose_model(classe_abc, classe_xyz, is_intermittent, is_obs, is_cold)

        test_actual = monthly_test[pid].astype(float) if pid in monthly_test.columns else None
        r = forecast_product(
            history, horizon=HORIZON_TEST, model_name=model_name,
            test_actual=test_actual, catalogue=catalogue, monthly=monthly_train,
            product_id=pid,
        )
        records.append(r)

        if r.sarima_signature:
            sarima_sigs.append({"produit_id": pid, "signature": r.sarima_signature})

        # Re-prévoit sur historique complet pour les mois opérationnels
        if not is_obs and not is_cold:
            r_deploy = forecast_product(
                monthly_full[pid].astype(float), horizon=HORIZON_DEPLOY,
                model_name=model_name, product_id=pid,
            )
            deploy_forecast = r_deploy.forecast
        else:
            deploy_forecast = np.zeros(HORIZON_DEPLOY) if is_obs else r.forecast[:HORIZON_DEPLOY]
            if len(deploy_forecast) < HORIZON_DEPLOY:
                deploy_forecast = np.pad(deploy_forecast,
                                         (0, HORIZON_DEPLOY - len(deploy_forecast)),
                                         constant_values=0.0)

        last_date = transactions["date"].max()
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                                     periods=HORIZON_DEPLOY, freq="MS")
        for k, d in enumerate(future_dates):
            yhat = float(deploy_forecast[k])
            # Intervalle de confiance grossier : ±20 % pour Croston/SARIMA, ±15 % LightGBM/LSTM
            band = 0.20 if model_name in ("croston_sba", "sarima") else 0.15
            rows_forecast.append({
                "produit_id": pid,
                "date": d.date(),
                "prevision": yhat,
                "modele_utilise": model_name,
                "intervalle_confiance_bas": max(0.0, yhat * (1 - band)),
                "intervalle_confiance_haut": yhat * (1 + band),
            })

    logger.info("Prévisions terminées en %.1f s", time.time() - t0)

    # ---- Évaluation comparée ----
    metrics_df = pd.DataFrame([{
        "produit_id": r.produit_id,
        "modele": r.model,
        "mae": r.test_mae,
        "rmse": r.test_rmse,
        "mape": r.test_mape,
    } for r in records]).merge(
        classes[["produit_id", "classe_abc", "classe_xyz", "classe_abc_xyz"]],
        on="produit_id", how="left",
    )
    metrics_df.to_csv(TAB_DIR / "comparaison_modeles.csv", index=False)

    # Synthèse par classe ABC × modèle
    agg = (
        metrics_df.dropna(subset=["mae"]).groupby(["classe_abc", "modele"])
        .agg(n_produits=("produit_id", "nunique"),
             mae_moy=("mae", "mean"),
             rmse_moy=("rmse", "mean"),
             mape_moy=("mape", "mean"))
        .round(3).reset_index()
    )
    agg.to_csv(TAB_DIR / "forecast_metrics_by_class.csv", index=False)

    # ---- Export prévisions ----
    pd.DataFrame(rows_forecast).to_csv(TAB_DIR / "previsions_complet.csv", index=False)

    # ---- Signatures SARIMA ----
    if sarima_sigs:
        pd.DataFrame(sarima_sigs).to_csv(MODELS_DIR / "sarima_signatures.csv", index=False)

    # ---- Figures ----
    fig_metrics_by_class(metrics_df)
    fig_real_vs_pred(records, monthly_full, classes)
    fig_model_distribution(metrics_df)
    fig_lgbm_importance(records)

    logger.info("Sorties : previsions_complet.csv, comparaison_modeles.csv, forecast_metrics_by_class.csv")
    logger.info("Synthèse par classe ABC × modèle :\n%s", agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
