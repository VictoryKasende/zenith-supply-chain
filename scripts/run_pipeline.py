"""Pipeline complet Zenith Supply Chain.

Exécution :
    python scripts/run_pipeline.py

Produit :
- data/processed/transactions_clean.parquet
- data/processed/products_features.parquet
- data/results/classification.csv
- data/results/obsolescence.csv
- data/results/forecasts.csv
- data/results/forecast_metrics.csv
- data/results/optimization_plan.csv
- data/results/baseline_plan.csv
- data/results/financial_comparison.csv
- reports/figures/*.png
- reports/tables/summary.md
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zenith import config as cfg
from zenith.classification import classify_abc, classify_xyz, kmeans_clustering
from zenith.evaluation import (
    aggregate_by_class,
    compare_policies,
    evaluate_forecasts,
    simulate_baseline_policy,
)
from zenith.feature_engineering import (
    add_financial_features,
    add_temporal_features,
    build_product_features,
)
from zenith.forecasting import (
    choose_model_for_class,
    forecast_one_product,
    monthly_series_by_product,
)
from zenith.obsolescence import detect_obsolescence
from zenith.optimization import optimize_orders
from zenith.preprocessing import clean_dataset, temporal_split
from zenith.viz import (
    fig_abc_distribution,
    fig_financial_comparison,
    fig_kmeans,
    fig_metrics_by_class,
    fig_obsolescence,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("zenith.pipeline")


def step_load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 70)
    logger.info("ÉTAPE 1 — Chargement et nettoyage des données")
    logger.info("=" * 70)
    raw = pd.read_csv(cfg.RAW_TRANSACTIONS)
    catalogue = pd.read_csv(cfg.RAW_CATALOGUE)
    logger.info("Transactions brutes: %d  |  Catalogue: %d produits", len(raw), len(catalogue))

    clean = clean_dataset(raw, catalogue=catalogue)
    clean = add_temporal_features(clean)
    clean = add_financial_features(clean)

    out = cfg.PROCESSED_DIR / "transactions_clean.parquet"
    clean.to_parquet(out, index=False)
    logger.info("→ Sauvegardé : %s (%d lignes)", out.relative_to(cfg.ROOT), len(clean))
    return clean, catalogue


def step_build_features(clean: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("ÉTAPE 2 — Feature engineering (statistiques produit)")
    logger.info("=" * 70)
    features = build_product_features(clean)
    out = cfg.PROCESSED_DIR / "products_features.parquet"
    features.to_parquet(out, index=False)
    logger.info("→ Sauvegardé : %s (%d produits)", out.relative_to(cfg.ROOT), len(features))
    return features


def step_classify(features: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("ÉTAPE 3 — Classification ABC × XYZ + K-Means")
    logger.info("=" * 70)
    abc = classify_abc(features)
    abc_xyz = classify_xyz(abc)
    clustered, diag = kmeans_clustering(abc_xyz)

    counts_abc = clustered["classe_abc"].value_counts().to_dict()
    counts_xyz = clustered["classe_xyz"].value_counts().to_dict()
    counts_abcxyz = clustered["classe_abc_xyz"].value_counts().to_dict()
    logger.info("Répartition ABC : %s", counts_abc)
    logger.info("Répartition XYZ : %s", counts_xyz)
    logger.info("Répartition ABC×XYZ : %s", counts_abcxyz)
    logger.info("K-Means : k* = %s (silhouette = %.3f)", diag["best_k"], diag["best_silhouette"])

    out = cfg.RESULTS_DIR / "classification.csv"
    clustered.to_csv(out, index=False)
    diag["cluster_profile"].to_csv(cfg.RESULTS_DIR / "cluster_profile.csv")
    logger.info("→ Sauvegardé : %s", out.relative_to(cfg.ROOT))
    return clustered


def step_obsolescence(clustered: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("ÉTAPE 4 — Détection d'obsolescence (Isolation Forest)")
    logger.info("=" * 70)
    obs = detect_obsolescence(clustered)
    n_risk = int(obs["a_risque_obsolescence"].sum())
    logger.info("Produits à risque d'obsolescence : %d / %d (%.1f%%)", n_risk, len(obs), 100 * n_risk / len(obs))
    out = cfg.RESULTS_DIR / "obsolescence.csv"
    obs.to_csv(out, index=False)
    logger.info("→ Sauvegardé : %s", out.relative_to(cfg.ROOT))
    return obs


def step_forecast(clean: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    logger.info("=" * 70)
    logger.info("ÉTAPE 5 — Prévision adaptée par classe (SARIMA / LightGBM / LSTM / Croston)")
    logger.info("=" * 70)

    train_df, val_df, test_df = temporal_split(clean, cfg.TRAIN_END, cfg.VAL_END)
    logger.info("Split temporel — train: %d | val: %d | test: %d", len(train_df), len(val_df), len(test_df))

    series_train = monthly_series_by_product(train_df)
    series_full = monthly_series_by_product(clean)
    series_test = monthly_series_by_product(pd.concat([val_df, test_df]))

    horizon = cfg.HORIZON_PLANIFICATION_MOIS
    test_horizon = max(horizon, min(len(series_test), 6))

    results: list = []
    fc_rows = []

    iterable = products.set_index("produit_id").iterrows()
    for pid, prod in tqdm(list(iterable), desc="Prévisions", ncols=80):
        if prod.get("a_risque_obsolescence", 0) == 1:
            # Pas de prévision pour les produits jugés obsolètes
            yhat = np.zeros(horizon)
            results.append(
                type(
                    "ObsoleteForecast",
                    (),
                    {
                        "produit_id": pid,
                        "model": "obsolete",
                        "horizon": horizon,
                        "forecast": yhat,
                        "test_mae": None,
                        "test_rmse": None,
                        "test_mape": None,
                    },
                )()
            )
            fc_rows.append({"produit_id": pid, **{f"m{t + 1}": 0 for t in range(horizon)}})
            continue

        # Sélection du modèle
        intermittent = prod.get("nombre_mois_avec_ventes", 99) <= max(1, 0.4 * 36)
        model_name = choose_model_for_class(
            prod["classe_abc"], prod["classe_xyz"], intermittent=intermittent
        )

        # Historique d'entraînement
        if pid not in series_train.columns:
            history = pd.Series(dtype=float)
        else:
            history = series_train[pid].astype(float)
            history.name = pid

        # Évaluation sur la fenêtre val+test si dispo
        if pid in series_test.columns and len(history) > 12:
            r = forecast_one_product(
                history, horizon=test_horizon, model_name=model_name,
                test_history=series_test[pid].astype(float),
            )
        else:
            r = forecast_one_product(history, horizon=horizon, model_name=model_name)

        # Prévision finale = ré-entraîner sur historique complet pour les mois à venir
        if pid in series_full.columns and len(series_full[pid]) > 6:
            full_hist = series_full[pid].astype(float)
            full_hist.name = pid
            try:
                final = forecast_one_product(full_hist, horizon=horizon, model_name=model_name)
                forecast_array = final.forecast
            except Exception:
                forecast_array = r.forecast[:horizon]
        else:
            forecast_array = r.forecast[:horizon]

        results.append(r)
        fc_rows.append({
            "produit_id": pid,
            **{f"m{t + 1}": float(forecast_array[t]) if t < len(forecast_array) else 0.0
               for t in range(horizon)},
        })

    metrics = evaluate_forecasts(results, by_class=products)
    metrics_out = cfg.RESULTS_DIR / "forecast_metrics.csv"
    metrics.to_csv(metrics_out, index=False)

    forecast_df = pd.DataFrame(fc_rows).set_index("produit_id")
    forecast_out = cfg.RESULTS_DIR / "forecasts.csv"
    forecast_df.to_csv(forecast_out)

    agg = aggregate_by_class(metrics, group_col="classe_abc")
    agg.to_csv(cfg.RESULTS_DIR / "forecast_metrics_by_class.csv", index=False)
    logger.info("→ Sauvegardé : %s, %s", forecast_out.relative_to(cfg.ROOT), metrics_out.relative_to(cfg.ROOT))
    if not agg.empty:
        logger.info("Synthèse MAE par classe ABC :\n%s", agg.to_string(index=False))
    return forecast_df, metrics, results


def step_optimize(products: pd.DataFrame, forecast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 70)
    logger.info("ÉTAPE 6 — Optimisation linéaire des commandes (PuLP / CBC)")
    logger.info("=" * 70)

    horizon = cfg.HORIZON_PLANIFICATION_MOIS
    plan_optim = optimize_orders(products, forecast_df, horizon=horizon)
    plan_baseline = simulate_baseline_policy(products, forecast_df, horizon=horizon)

    plan_optim.to_csv(cfg.RESULTS_DIR / "optimization_plan.csv", index=False)
    plan_baseline.to_csv(cfg.RESULTS_DIR / "baseline_plan.csv", index=False)

    compare = compare_policies(plan_optim, plan_baseline, products)
    compare.to_csv(cfg.RESULTS_DIR / "financial_comparison.csv", index=False)
    logger.info("→ Plans sauvegardés (%d commandes optimisées, %d empiriques)",
                len(plan_optim), len(plan_baseline))
    logger.info("Comparaison politique optimisée vs empirique :\n%s",
                compare.to_string(index=False))
    return plan_optim, plan_baseline, compare


def step_visualize(
    products: pd.DataFrame,
    metrics: pd.DataFrame,
    compare: pd.DataFrame,
) -> None:
    logger.info("=" * 70)
    logger.info("ÉTAPE 7 — Visualisations & rapport")
    logger.info("=" * 70)
    fig_abc_distribution(products, cfg.FIG_DIR / "abc_pareto.png")
    fig_obsolescence(products, cfg.FIG_DIR / "obsolescence.png")
    fig_kmeans(products, cfg.FIG_DIR / "kmeans_clusters.png")
    fig_metrics_by_class(metrics, cfg.FIG_DIR / "metrics_by_class.png")
    fig_financial_comparison(compare, cfg.FIG_DIR / "financial_comparison.png")
    logger.info("Figures écrites dans %s", cfg.FIG_DIR.relative_to(cfg.ROOT))


def step_write_summary(
    clean: pd.DataFrame,
    products: pd.DataFrame,
    metrics: pd.DataFrame,
    compare: pd.DataFrame,
) -> None:
    out = cfg.REPORTS_DIR / "summary.md"
    n_obs_risk = int(products["a_risque_obsolescence"].sum())
    abc = products["classe_abc"].value_counts().reindex(["A", "B", "C"]).fillna(0).astype(int)
    abcxyz = products["classe_abc_xyz"].value_counts().to_dict()
    period = (clean["date"].min().date(), clean["date"].max().date())

    agg = aggregate_by_class(metrics)
    cmp_txt = compare.to_markdown(index=False) if not compare.empty else "(vide)"
    agg_txt = agg.to_markdown(index=False) if not agg.empty else "(vide)"

    lines = [
        "# Synthèse exécution pipeline Zenith Supply Chain",
        "",
        f"- **Période couverte** : {period[0]} → {period[1]}",
        f"- **Transactions nettoyées** : {len(clean):,}".replace(",", " "),
        f"- **Produits analysés** : {len(products)}",
        "",
        "## Classification ABC",
        f"- Classe A : {abc['A']} produits",
        f"- Classe B : {abc['B']} produits",
        f"- Classe C : {abc['C']} produits",
        "",
        "## Répartition ABC × XYZ",
    ]
    for k, v in sorted(abcxyz.items()):
        lines.append(f"- {k} : {v} produits")
    lines += [
        "",
        f"## Détection d'obsolescence",
        f"- Produits à risque : **{n_obs_risk}** / {len(products)} ({100 * n_obs_risk / len(products):.1f}%)",
        "",
        "## Performance des modèles de prévision (MAE moyen, par classe ABC × modèle)",
        "",
        agg_txt,
        "",
        "## Comparaison politique empirique vs politique optimisée",
        "",
        cmp_txt,
        "",
        "## Fichiers générés",
        "- `data/processed/transactions_clean.parquet`",
        "- `data/processed/products_features.parquet`",
        "- `data/results/classification.csv`",
        "- `data/results/obsolescence.csv`",
        "- `data/results/forecasts.csv`",
        "- `data/results/forecast_metrics.csv`",
        "- `data/results/forecast_metrics_by_class.csv`",
        "- `data/results/optimization_plan.csv`",
        "- `data/results/baseline_plan.csv`",
        "- `data/results/financial_comparison.csv`",
        "- `reports/figures/*.png`",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("→ Synthèse écrite : %s", out.relative_to(cfg.ROOT))


def main() -> int:
    clean, _catalogue = step_load_and_clean()
    features = step_build_features(clean)
    clustered = step_classify(features)
    products = step_obsolescence(clustered)
    forecast_df, metrics, _ = step_forecast(clean, products)
    plan_optim, plan_baseline, compare = step_optimize(products, forecast_df)
    step_visualize(products, metrics, compare)
    step_write_summary(clean, products, metrics, compare)

    logger.info("=" * 70)
    logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
