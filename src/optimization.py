"""Optimisation linéaire des commandes (mémoire §3.7).

Formulation LP continue (PuLP/CBC). Les variables ``Q`` (quantités) et ``S``
(stocks) sont continues puis arrondies à l'entier en post-traitement —
choix justifié par §3.2.1 (frugalité PME) et confirmé en Étape 6 du
prototype initial (le MILP entier avec variables binaires de commande
explose en temps de résolution pour 200+ produits sur 6 mois).

Variables
---------
- ``Q[i,t]`` ≥ 0 : quantité commandée du produit *i* à la période *t*.
- ``S[i,t]`` ≥ 0 : stock fin de période.
- ``R[i,t]`` ≥ 0 : ruptures (proxy de service).

Objectif
--------
``min Σ_i Σ_t ( c_commande_unitaire * Q + c_stockage * S + c_rupture * R )``

Contraintes
-----------
- Conservation : ``S_t = S_{t-1} + Q_{t-L_i} - D_{i,t} + R_{i,t}``
- Budget : ``Σ_i c_achat_i * Q_{i,t} ≤ B_t``  pour chaque période *t*.
- Capacité de stockage : ``Σ_i v_i * S_{i,t} ≤ V_max``.
- Niveau de service classe A : converti en pondération forte sur R (plutôt
  qu'une contrainte dure qui peut rendre le LP infaisable face au délai
  d'importation et au budget).
- Produits obsolètes : ``Q_{i,t} = 0`` ∀ t.

API publique : :func:`optimize_orders` et :func:`simulate_baseline_policy`.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pulp

from src.utils import setup_logger

logger = setup_logger("optimization")

# --------------------------------------------------------------------- #
# Constantes économiques (mémoire §3.7)
# --------------------------------------------------------------------- #
COUT_COMMANDE_FIXE = 50.0           # USD par commande passée
TAUX_STOCKAGE_JOURNALIER = 0.001    # 0.1 % du coût d'achat par jour
PENALITE_RUPTURE_CLIENT = 0.20      # 20 % de pénalité supplémentaire (perte de fidélité)
DELAI_LIVRAISON_DUBAI = 35          # jours
DELAI_LIVRAISON_CHINE = 55          # jours
NIVEAU_SERVICE_A = 0.95             # ≥ 95 % disponibilité pour classe A
BUDGET_MENSUEL_DEFAUT = 500_000.0   # USD (calibré sur ~356 k$ d'achats mensuels)
HORIZON_PLANIFICATION_MOIS = 3      # cf. §3.7
VOLUME_UNITAIRE_DEFAUT = 0.01       # m³ par unité (capacité)
CAPACITE_STOCKAGE_DEFAUT = 5000.0   # m³ (entrepôt simplifié)


# --------------------------------------------------------------------- #
# Délais de livraison (en mois)
# --------------------------------------------------------------------- #
def lead_time_months(origine: str | None) -> int:
    """Convertit le délai d'importation en nombre de mois.

    35 jours ≈ 1.2 mois → 1 mois pratique pour Dubaï.
    55 jours ≈ 1.8 mois → 2 mois pour la Chine.
    """
    if origine and "Chine" in str(origine):
        return max(1, int(round(DELAI_LIVRAISON_CHINE / 30)))
    return max(1, int(round(DELAI_LIVRAISON_DUBAI / 30)))


def fournisseur_label(origine: str | None) -> str:
    if origine and "Chine" in str(origine):
        return "Chine"
    return "Dubaï"


# --------------------------------------------------------------------- #
# Optimisation principale
# --------------------------------------------------------------------- #
def optimize_orders(
    products: pd.DataFrame,
    forecasts: pd.DataFrame,
    horizon: int = HORIZON_PLANIFICATION_MOIS,
    budget_mensuel: float = BUDGET_MENSUEL_DEFAUT,
    capacite_stockage: float = CAPACITE_STOCKAGE_DEFAUT,
    cout_commande: float = COUT_COMMANDE_FIXE,
    taux_stockage: float = TAUX_STOCKAGE_JOURNALIER,
    niveau_service_a: float = NIVEAU_SERVICE_A,
    solver_msg: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Résout le LP et renvoie le plan de commandes mensuel + KPI globaux.

    Parameters
    ----------
    products : DataFrame (1 ligne = 1 produit) avec colonnes :
        ``produit_id, classe_abc, a_risque_obsolescence, cout_achat_unitaire,
        prix_vente_unitaire, stock_courant, origine_fournisseur``.
    forecasts : DataFrame ``produit_id, date, prevision, modele_utilise, ...``
        Issu de l'Étape 5.
    """
    df = products.copy().reset_index(drop=True)
    df = df[df["a_risque_obsolescence"] == 0].copy()
    if df.empty:
        return pd.DataFrame(), {"statut": "Aucun produit actif"}

    horizon = int(horizon)
    products_list = df["produit_id"].tolist()
    T = list(range(1, horizon + 1))

    # ---- Mise en forme des paramètres ----
    cost = dict(zip(df["produit_id"], df["cout_achat_unitaire"].astype(float)))
    price = dict(zip(df["produit_id"], df["prix_vente_unitaire"].astype(float)))
    margin = {p: max(price[p] - cost[p], 0.5) for p in products_list}
    stock0 = dict(zip(df["produit_id"], df["stock_courant"].astype(float)))
    classe = dict(zip(df["produit_id"], df["classe_abc"]))
    lead = {p: lead_time_months(o) for p, o in zip(df["produit_id"], df["origine_fournisseur"])}

    # ---- Demande prévue par produit × mois ----
    fc = forecasts.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    fc = fc.sort_values(["produit_id", "date"])
    demand: dict[tuple[str, int], float] = {}
    for p in products_list:
        sub = fc[fc["produit_id"] == p].head(horizon)
        values = sub["prevision"].to_numpy(dtype=float)
        if len(values) < horizon:
            values = np.pad(values, (0, horizon - len(values)), constant_values=0.0)
        for k, t in enumerate(T):
            demand[(p, t)] = float(max(0.0, values[k]))

    # ---- Coefficients d'objectif ----
    avg_demand = {p: max(1.0, np.mean([demand[(p, t)] for t in T])) for p in products_list}
    order_unit_cost = {p: cout_commande / avg_demand[p] for p in products_list}
    storage_cost = {p: cost[p] * taux_stockage * 30 for p in products_list}  # par mois
    classe_weight = {"A": 4.0, "B": 2.5, "C": 1.5}
    stockout_cost = {
        p: (margin[p] + price[p]) * (1 + PENALITE_RUPTURE_CLIENT) * classe_weight.get(classe[p], 1.5)
        for p in products_list
    }

    # ---- Modèle LP ----
    model = pulp.LpProblem("zenith_orders_lp", pulp.LpMinimize)
    Q = pulp.LpVariable.dicts("Q", (products_list, T), lowBound=0, cat="Continuous")
    S = pulp.LpVariable.dicts("S", (products_list, T), lowBound=0, cat="Continuous")
    R = pulp.LpVariable.dicts("R", (products_list, T), lowBound=0, cat="Continuous")

    # Objectif
    model += pulp.lpSum(
        order_unit_cost[p] * Q[p][t]
        + storage_cost[p] * S[p][t]
        + stockout_cost[p] * R[p][t]
        for p in products_list for t in T
    )

    # Conservation
    for p in products_list:
        L = lead[p]
        for t in T:
            received = Q[p][t - L] if (t - L) in T else 0.0
            prev_stock = S[p][t - 1] if t > 1 else stock0[p]
            model += S[p][t] == prev_stock + received - demand[(p, t)] + R[p][t]

    # Budget
    for t in T:
        model += pulp.lpSum(cost[p] * Q[p][t] for p in products_list) <= budget_mensuel

    # Capacité de stockage (proxy volumétrique)
    for t in T:
        model += pulp.lpSum(VOLUME_UNITAIRE_DEFAUT * S[p][t] for p in products_list) <= capacite_stockage

    # Résolution
    solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=60, threads=2)
    status = model.solve(solver)
    logger.info("Statut solveur LP : %s", pulp.LpStatus[status])

    # ---- Extraction de la solution ----
    rows: list[dict] = []
    last_date_known = pd.to_datetime(forecasts["date"]).max()
    for p in products_list:
        for t in T:
            q_val = float(pulp.value(Q[p][t]) or 0)
            q_int = int(np.round(q_val))
            s_val = float(pulp.value(S[p][t]) or 0)
            r_val = float(pulp.value(R[p][t]) or 0)
            rows.append({
                "produit_id": p,
                "mois_offset": t,
                "date_decision": (pd.Timestamp.now().normalize() + pd.offsets.MonthBegin(t - 1)).date(),
                "quantite_commandee": q_int,
                "stock_final": round(s_val, 1),
                "rupture": round(r_val, 2),
                "demande_prevue": round(demand[(p, t)], 2),
                "commande_passee": int(q_int > 0),
                "classe_abc": classe.get(p),
                "cout_achat": cost[p],
                "prix_vente": price[p],
                "fournisseur": fournisseur_label(
                    df.loc[df["produit_id"] == p, "origine_fournisseur"].iloc[0]
                ),
                "lead_time_mois": lead[p],
                "montant_total": round(q_int * cost[p], 2),
            })
    plan = pd.DataFrame(rows)

    # ---- KPI globaux ----
    kpis = {
        "statut": pulp.LpStatus[status],
        "nb_commandes_passees": int(plan["commande_passee"].sum()),
        "quantite_totale": int(plan["quantite_commandee"].sum()),
        "valeur_commandes_usd": float(plan["montant_total"].sum()),
        "ruptures_unites": float(plan["rupture"].sum()),
        "stock_moyen_immo_usd": float((plan["stock_final"] * plan["cout_achat"]).mean()),
    }
    return plan, kpis


# --------------------------------------------------------------------- #
# Distribution multi-magasins
# --------------------------------------------------------------------- #
def distribute_to_stores(
    plan: pd.DataFrame,
    transactions: pd.DataFrame,
    store_col: str = "magasin",
) -> pd.DataFrame:
    """Répartit les quantités centrales sur les magasins selon leur part historique.

    Pour chaque ``produit_id``, on calcule la part historique des ventes par
    magasin et on alloue les quantités commandées centralement à due
    proportion. Les arrondis vont au plus grand magasin pour conserver le
    total exact.
    """
    raw = transactions.groupby(["produit_id", store_col])["quantite_vendue"].sum()
    totals = raw.groupby(level=0).transform("sum")
    shares = (raw / totals).rename("part").reset_index()
    rows: list[dict] = []
    for _, row in plan.iterrows():
        if row["quantite_commandee"] == 0:
            continue
        prod_shares = shares[shares["produit_id"] == row["produit_id"]]
        if prod_shares.empty:
            rows.append({**row.to_dict(), "magasin": "Mobutu 2"})  # fallback magasin principal
            continue
        allocations = (prod_shares["part"] * row["quantite_commandee"]).round().astype(int).tolist()
        diff = int(row["quantite_commandee"] - sum(allocations))
        if diff != 0:
            # Ajuste l'allocation du plus gros magasin pour conserver le total
            idx_max = int(np.argmax(allocations))
            allocations[idx_max] += diff
        for (m, qty) in zip(prod_shares[store_col].tolist(), allocations):
            if qty <= 0:
                continue
            line = {**row.to_dict(), "magasin": m, "quantite_commandee": int(qty)}
            line["montant_total"] = round(qty * row["cout_achat"], 2)
            rows.append(line)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Politique empirique simulée (baseline)
# --------------------------------------------------------------------- #
def simulate_baseline_policy(
    products: pd.DataFrame,
    forecasts: pd.DataFrame,
    horizon: int = HORIZON_PLANIFICATION_MOIS,
) -> pd.DataFrame:
    """Reproduit la pratique actuelle de Zenith :

    - Commande déclenchée quand stock < demande mensuelle moyenne.
    - Quantité couvrant 2 mois de demande prévue.
    - Délai d'importation subi (Dubaï 1 mo / Chine 2 mo) sans anticipation.
    - Aucune segmentation ABC, aucune exclusion d'obsolètes.
    """
    fc = forecasts.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    rows: list[dict] = []
    for _, p in products.iterrows():
        pid = p["produit_id"]
        sub = fc[fc["produit_id"] == pid].sort_values("date").head(horizon)
        if sub.empty:
            continue
        forecast = sub["prevision"].to_numpy(dtype=float)
        stock = float(p.get("stock_courant", 0) or 0)
        cost = float(p.get("cout_achat_unitaire", 0) or 0)
        L = lead_time_months(p.get("origine_fournisseur"))
        avg_demand = forecast.mean() if len(forecast) else 0.0
        pipeline: dict[int, float] = {}
        for t in range(1, horizon + 1):
            d = float(forecast[t - 1])
            stock += pipeline.pop(t, 0.0)
            served = min(stock, d)
            rupture = max(0.0, d - served)
            stock -= served
            order_qty = 0
            if stock < avg_demand:
                order_qty = int(np.ceil(2 * avg_demand))
                pipeline[t + L] = pipeline.get(t + L, 0.0) + order_qty
            rows.append({
                "produit_id": pid,
                "classe_abc": p.get("classe_abc"),
                "mois_offset": t,
                "quantite_commandee": order_qty,
                "commande_passee": int(order_qty > 0),
                "stock_final": round(stock, 1),
                "rupture": round(rupture, 2),
                "demande_prevue": round(d, 2),
                "cout_achat": cost,
                "prix_vente": float(p.get("prix_vente_unitaire", 0) or 0),
                "fournisseur": fournisseur_label(p.get("origine_fournisseur")),
                "lead_time_mois": L,
                "montant_total": round(order_qty * cost, 2),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# KPI financiers
# --------------------------------------------------------------------- #
def financial_kpis(plan: pd.DataFrame) -> dict:
    """KPI financiers globaux d'un plan de commandes (LP ou baseline)."""
    if plan.empty:
        return {}
    df = plan.copy()
    df["valeur_stock_immo"] = df["stock_final"] * df["cout_achat"]
    df["cout_stockage"] = df["valeur_stock_immo"] * TAUX_STOCKAGE_JOURNALIER * 30
    df["marge_perdue"] = df["rupture"] * (df["prix_vente"] - df["cout_achat"])
    df["cout_commande"] = df["commande_passee"] * COUT_COMMANDE_FIXE
    df["cout_total"] = df["cout_commande"] + df["cout_stockage"] + df["marge_perdue"]
    df["ventes_servies"] = (df["demande_prevue"] - df["rupture"]).clip(lower=0)
    df["ca_realise"] = df["ventes_servies"] * df["prix_vente"]
    total_demande = df["demande_prevue"].sum()
    return {
        "nb_commandes": int(df["commande_passee"].sum()),
        "quantite_commandee": int(df["quantite_commandee"].sum()),
        "valeur_commande_totale_usd": round(df["montant_total"].sum(), 2),
        "stock_moyen_immo_usd": round(df["valeur_stock_immo"].mean(), 2),
        "ruptures_unites": round(df["rupture"].sum(), 2),
        "marge_perdue_usd": round(df["marge_perdue"].sum(), 2),
        "cout_stockage_usd": round(df["cout_stockage"].sum(), 2),
        "cout_commandes_usd": round(df["cout_commande"].sum(), 2),
        "cout_total_simule_usd": round(df["cout_total"].sum(), 2),
        "ca_realise_usd": round(df["ca_realise"].sum(), 2),
        "taux_service_pct": round(100 * (1 - df["rupture"].sum() / max(total_demande, 1)), 2),
    }


def compare_policies(plan_lp: pd.DataFrame, plan_baseline: pd.DataFrame) -> pd.DataFrame:
    """Construit un tableau comparatif politique LP vs empirique."""
    k_lp = financial_kpis(plan_lp)
    k_emp = financial_kpis(plan_baseline)
    rows = []
    for k in k_lp:
        opt, emp = k_lp[k], k_emp[k]
        delta = opt - emp
        pct = (delta / emp * 100) if emp not in (0, 0.0) else float("nan")
        rows.append({
            "indicateur": k,
            "politique_empirique": emp,
            "politique_optimisee": opt,
            "delta": round(delta, 2),
            "delta_pct": round(pct, 2) if not np.isnan(pct) else float("nan"),
        })
    return pd.DataFrame(rows)
