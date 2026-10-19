"""Reproduction des chiffres de politique publiés, par le chemin de code de l'article.

Recalcule, avec ``optimize_orders`` / ``simulate_baseline_policy`` /
``financial_kpis`` sur les entrées versionnées (``outputs/tables``), les
chiffres de la Table III et de H4 de l'article, et les met en regard du même
calcul sur un périmètre apparié (mêmes 177 produits actifs).

Sortie : ``paper_results/paper_method_reproduction.json``.
"""
from __future__ import annotations

import json

import numpy as np

from experiments.config import PAPER_DIR, ensure_dirs
from experiments.milp_vs_lp import paper_inputs
from src.optimization import financial_kpis, optimize_orders, simulate_baseline_policy


def data_audit() -> dict:
    """Constats d'audit cités en V1–V3 (stock par magasin, ventes à stock nul, repli du MLP)."""
    import pandas as pd

    from experiments.data import full_clean
    from src.utils import TAB_DIR

    c = full_clean()
    g = c.sort_values(["produit_id", "magasin", "date", "transaction_id"]).copy()
    g["prev"] = g.groupby(["produit_id", "magasin"])["stock_apres_vente"].shift()
    ok_store = (g["prev"] - g["quantite_vendue"] - g["stock_apres_vente"]).abs() < 1e-6
    g2 = c.sort_values(["produit_id", "date", "transaction_id"]).copy()
    g2["prev"] = g2.groupby("produit_id")["stock_apres_vente"].shift()
    ok_prod = (g2["prev"] - g2["quantite_vendue"] - g2["stock_apres_vente"]).abs() < 1e-6
    zero = c[c["stock_apres_vente"] == 0]
    cm = pd.read_csv(TAB_DIR / "comparaison_modeles.csv")
    vb = pd.read_csv(TAB_DIR / "validation_baselines.csv")
    m = cm.merge(vb[["produit_id", "seasonal_naive"]], on="produit_id")
    lstm = m[m["modele"] == "lstm"]
    return {
        "share_consecutive_rows_stock_consistent_product_store": float(ok_store[g["prev"].notna()].mean()),
        "share_consecutive_rows_stock_consistent_product_only": float(ok_prod[g2["prev"].notna()].mean()),
        "share_rows_stock_increase_product_store": float((g["stock_apres_vente"] > g["prev"])[g["prev"].notna()].mean()),
        "sales_rows_leaving_store_at_zero_stock": int(len(zero)),
        "products_with_such_rows": int(zero["produit_id"].nunique()),
        "article_lstm_products": int(len(lstm)),
        "article_lstm_products_mae_identical_to_seasonal_naive": int(((lstm["mae"] - lstm["seasonal_naive"]).abs() < 1e-9).sum()),
        "data_period": [str(c["date"].min().date()), str(c["date"].max().date())],
        "n_products": int(c["produit_id"].nunique()), "n_stores": int(c["magasin"].nunique()),
    }


def run() -> dict:
    ensure_dirs()
    products, forecasts = paper_inputs()
    plan_lp, _ = optimize_orders(products, forecasts)
    base_all = simulate_baseline_policy(products, forecasts)
    base_paired = base_all[base_all["produit_id"].isin(plan_lp["produit_id"].unique())]
    k_lp, k_all, k_pair = financial_kpis(plan_lp), financial_kpis(base_all), financial_kpis(base_paired)

    def _pct(a, b):
        return 100 * (a - b) / b

    out = {
        "description": "plan août → octobre 2025 jugé contre la demande PRÉVUE (méthode de l'article)",
        "n_products_lp": int(plan_lp["produit_id"].nunique()),
        "n_products_incumbent_as_published": int(base_all["produit_id"].nunique()),
        "lp": {"total_cost": k_lp["cout_total_simule_usd"], "unit_fill_rate_vs_forecast_pct": k_lp["taux_service_pct"]},
        "incumbent_as_published_250": {"total_cost": k_all["cout_total_simule_usd"],
                                       "unit_fill_rate_vs_forecast_pct": k_all["taux_service_pct"]},
        "incumbent_paired_177": {"total_cost": k_pair["cout_total_simule_usd"],
                                 "unit_fill_rate_vs_forecast_pct": k_pair["taux_service_pct"]},
        "savings_pct_as_published": _pct(k_lp["cout_total_simule_usd"], k_all["cout_total_simule_usd"]),
        "savings_pct_paired": _pct(k_lp["cout_total_simule_usd"], k_pair["cout_total_simule_usd"]),
        "paired_difference_usd_3_months": k_pair["cout_total_simule_usd"] - k_lp["cout_total_simule_usd"],
        "paired_difference_annualised_x4": 4 * (k_pair["cout_total_simule_usd"] - k_lp["cout_total_simule_usd"]),
        "n_product_month_pairs": int(len(base_paired)),
        "published": {"lp": 55029, "incumbent": 76740, "pct": -28.3, "service": [69.19, 69.94],
                      "h4_pairs": 531, "h4_diff": 19381, "h4_annual": 77525},
        "note": "les valeurs publiées 55 029 / 76 740 / 69,19 / 69,94 viennent de "
                "outputs/tables/optimization_summary.md (exécution antérieure, texte saisi à la main)",
        "data_audit": data_audit(),
    }
    out = json.loads(json.dumps(out, default=lambda o: float(o) if isinstance(o, np.floating) else int(o)))
    (PAPER_DIR / "paper_method_reproduction.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
    return out


def main() -> int:
    print(json.dumps(run(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
