"""Évaluation double : précision prédictive + impact financier (cf. mémoire §3.8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecasting import mae, mape, rmse


# ------------------------------------------------------------------ #
# Précision prédictive
# ------------------------------------------------------------------ #
def evaluate_forecasts(
    results: list,
    by_class: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construit un tableau MAE/RMSE/MAPE par produit, avec agrégation par classe."""
    rows = []
    for r in results:
        rows.append({
            "produit_id": r.produit_id,
            "modele": r.model,
            "mae": r.test_mae,
            "rmse": r.test_rmse,
            "mape": r.test_mape,
        })
    df = pd.DataFrame(rows)
    if by_class is not None and not df.empty:
        df = df.merge(
            by_class[["produit_id", "classe_abc", "classe_xyz", "classe_abc_xyz"]],
            on="produit_id",
            how="left",
        )
    return df


def aggregate_by_class(df: pd.DataFrame, group_col: str = "classe_abc") -> pd.DataFrame:
    """Synthèse MAE/RMSE/MAPE par classe ABC (ou ABC×XYZ)."""
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby([group_col, "modele"])
        .agg(
            n_produits=("produit_id", "nunique"),
            mae_moy=("mae", "mean"),
            rmse_moy=("rmse", "mean"),
            mape_moy=("mape", "mean"),
        )
        .round(3)
        .reset_index()
    )
    return out


# ------------------------------------------------------------------ #
# Impact financier — comparaison politique optimisée vs politique empirique
# ------------------------------------------------------------------ #
def simulate_baseline_policy(
    products: pd.DataFrame,
    forecast_df: pd.DataFrame,
    horizon: int = 3,
) -> pd.DataFrame:
    """Politique empirique (point de commande réactif) — pratique actuelle.

    Règle métier reproduite :
      - On commande quand le stock courant tombe sous la demande prévue du mois suivant.
      - La quantité commandée couvre 2 mois de demande prévue.
      - Le délai d'importation réel (Dubaï/Chine) est subi mais non anticipé.
      - Aucune segmentation ABC ni détection d'obsolescence.
    """
    from .optimization import lead_time_months

    rows = []
    for _, p in products.iterrows():
        pid = p["produit_id"]
        if pid not in forecast_df.index:
            continue
        forecast = forecast_df.loc[pid].to_numpy(dtype=float)[:horizon]
        stock = float(p.get("stock_courant", 0) or 0)
        cost = float(p.get("cout_achat_unitaire", 0) or 0)
        L = lead_time_months(p.get("origine_fournisseur"))
        avg_demand = forecast.mean() if len(forecast) else 0.0

        pipeline: dict[int, float] = {}  # mois d'arrivée → quantité
        for t in range(1, horizon + 1):
            d = float(forecast[t - 1])

            # Quantités reçues à ce mois
            received = pipeline.pop(t, 0.0)
            stock += received

            # Sert la demande
            served = min(stock, d)
            rupture = max(0.0, d - served)
            stock -= served

            # Politique réactive : déclencher commande si stock bas
            order_qty = 0
            if stock < avg_demand:
                order_qty = int(np.ceil(2 * avg_demand))
                pipeline[t + L] = pipeline.get(t + L, 0.0) + order_qty

            rows.append({
                "produit_id": pid,
                "mois_offset": t,
                "quantite_commandee_emp": order_qty,
                "commande_passee_emp": int(order_qty > 0),
                "stock_final_emp": stock,
                "rupture_emp": rupture,
                "demande_prevue": d,
                "cout_achat": cost,
            })
    return pd.DataFrame(rows)


def financial_kpis(plan: pd.DataFrame, products: pd.DataFrame, suffix: str = "") -> dict:
    """KPIs financiers globaux pour un plan de commandes.

    Le coût total simulé combine :
      - coût fixe par commande (50 USD)
      - coût d'immobilisation du stock (0.1%/jour × coût d'achat × 30j)
      - coût de rupture (marge perdue)
    """
    from .config import COUT_COMMANDE_FIXE, TAUX_STOCKAGE_JOURNALIER

    s = suffix
    qcol = f"quantite_commandee{s}"
    pcol = f"commande_passee{s}"
    scol = f"stock_final{s}"
    rcol = f"rupture{s}"

    df = plan.merge(
        products[["produit_id", "prix_vente_unitaire", "cout_achat_unitaire"]],
        on="produit_id",
        how="left",
        suffixes=("", "_prod"),
    )
    df["valeur_commande"] = df[qcol] * df["cout_achat_unitaire"]
    df["valeur_stock_immo"] = df[scol] * df["cout_achat_unitaire"]
    df["cout_stockage"] = df["valeur_stock_immo"] * TAUX_STOCKAGE_JOURNALIER * 30
    df["marge_perdue"] = df[rcol] * (df["prix_vente_unitaire"] - df["cout_achat_unitaire"])
    df["cout_commande"] = df[pcol] * COUT_COMMANDE_FIXE
    df["cout_total"] = df["cout_commande"] + df["cout_stockage"] + df["marge_perdue"]

    # Ventes effectivement servies (demande - rupture)
    df["ventes_servies"] = (df["demande_prevue"] - df[rcol]).clip(lower=0)
    df["ca_realise"] = df["ventes_servies"] * df["prix_vente_unitaire"]
    return {
        "nb_commandes": int(df[pcol].sum()),
        "qte_commandee_totale": float(df[qcol].sum()),
        "valeur_commande_totale": float(df["valeur_commande"].sum()),
        "stock_moyen_immo_usd": float(df["valeur_stock_immo"].mean()),
        "nb_ruptures_unites": float(df[rcol].sum()),
        "marge_perdue_usd": float(df["marge_perdue"].sum()),
        "cout_commande_total_usd": float(df["cout_commande"].sum()),
        "cout_stockage_total_usd": float(df["cout_stockage"].sum()),
        "cout_total_simule_usd": float(df["cout_total"].sum()),
        "ca_realise_usd": float(df["ca_realise"].sum()),
        "taux_service_pct": float(
            100 * (1 - df[rcol].sum() / max(df["demande_prevue"].sum(), 1))
        ),
    }


def compare_policies(
    plan_optim: pd.DataFrame,
    plan_baseline: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Construit un tableau comparatif entre politique optimisée et empirique."""
    k_opt = financial_kpis(plan_optim, products, suffix="")
    k_emp = financial_kpis(plan_baseline, products, suffix="_emp")

    rows = []
    for k in k_opt:
        opt = k_opt[k]
        emp = k_emp[k]
        delta = opt - emp
        pct = (delta / emp * 100) if emp not in (0, 0.0) else np.nan
        rows.append({
            "indicateur": k,
            "politique_empirique": round(emp, 2),
            "politique_optimisee": round(opt, 2),
            "delta": round(delta, 2),
            "delta_pct": round(pct, 2) if not np.isnan(pct) else np.nan,
        })
    return pd.DataFrame(rows)
