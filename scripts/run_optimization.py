"""Exécute l'Étape 6 — Optimisation linéaire des commandes.

Lit :
- data/processed/zenith_clean.csv (pour répartition magasin)
- data/features/product_features.csv (stock_courant, coût d'achat médian)
- outputs/tables/classification_produits.csv (classe ABC)
- outputs/tables/produits_obsoletes.csv (filtre)
- outputs/tables/previsions_complet.csv (demande prévue Étape 5)
- data/raw/catalogue_produits_250.csv

Produit :
- outputs/tables/commandes_recommandees.csv (produit × magasin × mois)
- outputs/tables/commandes_centrales.csv (plan central avant distribution)
- outputs/tables/baseline_policy_plan.csv
- outputs/tables/comparaison_avant_apres.csv
- outputs/figures/opt_01..04.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization import (
    compare_policies,
    distribute_to_stores,
    financial_kpis,
    optimize_orders,
    simulate_baseline_policy,
)
from src.utils import (
    FEATURES_DIR, FIG_DIR, PROCESSED_DIR, RAW_CATALOGUE, TAB_DIR, setup_logger,
)

logger = setup_logger("pipeline.optimization")
sns.set_theme(style="whitegrid", context="talk")


def build_products_input() -> pd.DataFrame:
    """Assemble les colonnes nécessaires à l'optimiseur."""
    feats = pd.read_csv(FEATURES_DIR / "product_features.csv")
    classes = pd.read_csv(TAB_DIR / "classification_produits.csv")[
        ["produit_id", "classe_abc", "classe_xyz"]
    ]
    obsoletes = pd.read_csv(TAB_DIR / "produits_obsoletes.csv")["produit_id"].tolist()
    cat = pd.read_csv(RAW_CATALOGUE)[
        ["produit_id", "origine_fournisseur", "cout_achat_unitaire", "prix_vente_unitaire"]
    ]
    products = feats.merge(classes, on="produit_id", how="left").merge(cat, on="produit_id", how="left")
    products["a_risque_obsolescence"] = products["produit_id"].isin(obsoletes).astype(int)
    # Compatibilité noms attendus par optimize_orders
    products["prix_vente_unitaire"] = products["prix_vente_unitaire"].fillna(products["prix_vente_unitaire_moyen"])
    if "stock_courant" not in products.columns:
        products["stock_courant"] = 0.0
    return products


def fig_orders_by_class(plan: pd.DataFrame) -> None:
    agg = plan.groupby("classe_abc").agg(
        ruptures=("rupture", "sum"),
        quantite=("quantite_commandee", "sum"),
        montant=("montant_total", "sum"),
    ).reindex(["A", "B", "C"]).fillna(0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title, color in zip(
        axes, ["quantite", "montant", "ruptures"],
        ["Quantité commandée", "Montant (USD)", "Ruptures (unités)"],
        ["#1f4e79", "#ff9f1c", "#ef476f"],
    ):
        ax.bar(agg.index, agg[col], color=color, edgecolor="black")
        ax.set_title(title); ax.set_xlabel("Classe ABC")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "opt_01_par_classe_abc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_comparison(compare: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.4
    x = np.arange(len(compare))
    ax.bar(x - width / 2, compare["politique_empirique"], width, label="Empirique", color="#ff9f1c")
    ax.bar(x + width / 2, compare["politique_optimisee"], width, label="Optimisée", color="#1f4e79")
    ax.set_xticks(x); ax.set_xticklabels(compare["indicateur"], rotation=35, ha="right", fontsize=10)
    ax.set_yscale("symlog", linthresh=10); ax.set_title("Politique empirique vs optimisée")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "opt_02_comparison_kpis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_budget_breakdown(plan: pd.DataFrame) -> None:
    agg = plan.groupby(["mois_offset", "fournisseur"])["montant_total"].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    agg.plot(kind="bar", stacked=True, ax=ax, color=["#1f4e79", "#ff9f1c"])
    ax.set_title("Budget mensuel par fournisseur — politique optimisée")
    ax.set_xlabel("Mois (1=mois prochain)"); ax.set_ylabel("Montant commande (USD)")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "opt_03_budget_par_fournisseur.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_service_per_class(plan_lp: pd.DataFrame, plan_emp: pd.DataFrame) -> None:
    def _service(plan):
        return (
            plan.groupby("classe_abc")
            .apply(lambda g: 100 * (1 - g["rupture"].sum() / max(g["demande_prevue"].sum(), 1)))
            .reindex(["A", "B", "C"])
        )

    s_lp = _service(plan_lp); s_emp = _service(plan_emp)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3); width = 0.4
    ax.bar(x - width / 2, s_emp, width, label="Empirique", color="#ff9f1c")
    ax.bar(x + width / 2, s_lp, width, label="Optimisée", color="#1f4e79")
    ax.axhline(95, ls="--", color="green", label="Objectif A ≥ 95%")
    ax.set_xticks(x); ax.set_xticklabels(["A", "B", "C"])
    ax.set_ylabel("Taux de service (%)"); ax.set_title("Taux de service par classe ABC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "opt_04_service_par_classe.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logger.info("Étape 6 — Optimisation linéaire des commandes")
    products = build_products_input()
    forecasts = pd.read_csv(TAB_DIR / "previsions_complet.csv", parse_dates=["date"])
    transactions = pd.read_csv(PROCESSED_DIR / "zenith_clean.csv", parse_dates=["date"])

    n_obs = int(products["a_risque_obsolescence"].sum())
    logger.info("Produits actifs : %d / %d (obsolètes : %d)", len(products) - n_obs, len(products), n_obs)

    # 1) Plan central optimisé
    plan_lp, kpis_lp = optimize_orders(products, forecasts)
    plan_lp.to_csv(TAB_DIR / "commandes_centrales.csv", index=False)
    logger.info("Solveur LP : %s — %d commandes, %d unités, %.0f USD",
                kpis_lp["statut"], kpis_lp["nb_commandes_passees"],
                kpis_lp["quantite_totale"], kpis_lp["valeur_commandes_usd"])

    # 2) Distribution multi-magasins
    plan_store = distribute_to_stores(plan_lp, transactions)
    cols = [
        "produit_id", "magasin", "mois_offset", "date_decision", "fournisseur",
        "classe_abc", "quantite_commandee", "cout_achat", "montant_total",
        "demande_prevue", "stock_final", "rupture", "lead_time_mois",
    ]
    plan_store[cols].to_csv(TAB_DIR / "commandes_recommandees.csv", index=False)
    logger.info("Plan magasin écrit : %d lignes (%d magasins)",
                len(plan_store), plan_store["magasin"].nunique())

    # 3) Baseline empirique
    plan_emp = simulate_baseline_policy(products, forecasts)
    plan_emp.to_csv(TAB_DIR / "baseline_policy_plan.csv", index=False)

    # 4) Comparaison
    compare = compare_policies(plan_lp, plan_emp)
    compare.to_csv(TAB_DIR / "comparaison_avant_apres.csv", index=False)
    logger.info("Comparaison politique optimisée vs empirique :\n%s",
                compare.to_string(index=False))

    # 5) KPI synthétiques
    kpi_lp = financial_kpis(plan_lp)
    kpi_emp = financial_kpis(plan_emp)
    logger.info(
        "KPI LP — taux service %.1f %%, marge perdue %.0f USD, stock immo moyen %.0f USD",
        kpi_lp["taux_service_pct"], kpi_lp["marge_perdue_usd"], kpi_lp["stock_moyen_immo_usd"],
    )
    logger.info(
        "KPI EMP — taux service %.1f %%, marge perdue %.0f USD, stock immo moyen %.0f USD",
        kpi_emp["taux_service_pct"], kpi_emp["marge_perdue_usd"], kpi_emp["stock_moyen_immo_usd"],
    )

    # 6) Figures
    fig_orders_by_class(plan_lp)
    fig_comparison(compare)
    fig_budget_breakdown(plan_lp)
    fig_service_per_class(plan_lp, plan_emp)

    logger.info("Sorties écrites dans outputs/tables/ et outputs/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
