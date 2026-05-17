"""Optimisation linéaire des commandes (cf. mémoire §3.7).

Formulation LP continue résolue avec PuLP/CBC, choix justifié par :
  - le principe de frugalité (§3.2.1) — solveur en quelques secondes ;
  - l'absence de coût marginal de commande dominant face aux coûts de
    stockage et de rupture dans le contexte Zenith ;
  - les quantités optimales sont arrondies à l'entier en post-traitement.

Variables :
  Q[i,t] : quantité commandée du produit i pour la période t (continue ≥ 0)
  S[i,t] : stock final du produit i en fin de période t      (continue ≥ 0)
  R[i,t] : ruptures du produit i à la période t              (continue ≥ 0)

Objectif :
  minimiser Σ ( c_commande_unitaire * Q + c_stockage * S + c_rupture * R )

Contraintes :
  S[i,t] = S[i,t-1] + Q[i,t-L_i] - D[i,t] + R[i,t]    (conservation du stock)
  Σ_i (c_i * Q[i,t]) <= Budget[t]                      (budget mensuel)
  Q[i,t] = 0 si produit obsolète                       (filtrage)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pulp

from .config import (
    BUDGET_MENSUEL_DEFAUT,
    COUT_COMMANDE_FIXE,
    DELAI_LIVRAISON_CHINE,
    DELAI_LIVRAISON_DUBAI,
    HORIZON_PLANIFICATION_MOIS,
    NIVEAU_SERVICE_A,
    TAUX_STOCKAGE_JOURNALIER,
)

logger = logging.getLogger(__name__)


def lead_time_months(origine: str | None) -> int:
    """Convertit le délai d'importation en nombre de mois.

    35 jours ≈ 1,2 mois → 1 mois en pratique pour Dubaï.
    55 jours ≈ 1,8 mois → 2 mois pour la Chine.
    """
    if origine and "Chine" in str(origine):
        return max(1, int(np.round(DELAI_LIVRAISON_CHINE / 30)))
    return max(1, int(np.round(DELAI_LIVRAISON_DUBAI / 30)))


def optimize_orders(
    products: pd.DataFrame,
    forecast_df: pd.DataFrame,
    horizon: int = HORIZON_PLANIFICATION_MOIS,
    budget_mensuel: float = BUDGET_MENSUEL_DEFAUT,
    cout_commande: float = COUT_COMMANDE_FIXE,
    cout_stockage_taux: float = TAUX_STOCKAGE_JOURNALIER,
    niveau_service_a: float = NIVEAU_SERVICE_A,
    solver_msg: bool = False,
) -> pd.DataFrame:
    """Résout le MILP et renvoie le plan de commandes mensuel.

    Paramètres
    ----------
    products : DataFrame
        Une ligne par produit, doit contenir :
        produit_id, classe_abc, a_risque_obsolescence, cout_achat_unitaire,
        prix_vente_unitaire, stock_courant, origine_fournisseur.
    forecast_df : DataFrame
        Index = produits (produit_id en colonne), colonnes = mois futurs (1..H),
        valeurs = demande prévue.
    """
    df = products.copy().reset_index(drop=True)
    horizon = int(horizon)

    # Garde uniquement les produits non obsolètes pour la décision de commande
    active = df[df["a_risque_obsolescence"] == 0].copy()
    if active.empty:
        logger.warning("Aucun produit actif à optimiser.")
        return pd.DataFrame()

    # Indexation
    products_list = active["produit_id"].tolist()
    T = list(range(1, horizon + 1))

    # Délais (en mois)
    lead = {pid: lead_time_months(o) for pid, o in zip(active["produit_id"], active["origine_fournisseur"])}
    cost = dict(zip(active["produit_id"], active["cout_achat_unitaire"].astype(float)))
    price = dict(zip(active["produit_id"], active["prix_vente_unitaire"].astype(float)))
    margin = {p: max(price[p] - cost[p], 0.5) for p in products_list}
    stock0 = dict(zip(active["produit_id"], active["stock_courant"].astype(float)))
    classe = dict(zip(active["produit_id"], active["classe_abc"]))

    # Coût de stockage mensuel par produit
    storage_cost = {p: cost[p] * cout_stockage_taux * 30 for p in products_list}

    # Coût de rupture : on combine la marge perdue immédiate et la probabilité
    # de perte définitive du client (cf. mémoire §3.7.2). Pour la classe A,
    # nous appliquons une pondération forte afin de faire émerger un niveau
    # de service élevé sans introduire de contrainte dure qui rendrait le
    # MILP infaisable face aux délais d'importation et au budget.
    classe_weight = {"A": 4.0, "B": 2.5, "C": 1.5}
    stockout_cost = {
        p: (margin[p] + price[p]) * classe_weight.get(classe[p], 1.5)
        for p in products_list
    }

    # Demande prévue (par produit, par mois)
    demand = {}
    for p in products_list:
        if p in forecast_df.index:
            row = forecast_df.loc[p].to_numpy(dtype=float)
        else:
            row = np.zeros(horizon)
        if len(row) < horizon:
            row = np.concatenate([row, np.zeros(horizon - len(row))])
        for k, t in enumerate(T):
            demand[(p, t)] = float(max(0.0, row[k]))

    # Coût d'amortissement par unité commandée (fee fixe / quantité-cible)
    # afin de garder un coût de passation de commande dans l'objectif tout en
    # restant linéaire.
    avg_demand = {p: max(1.0, np.mean([demand[(p, t)] for t in T])) for p in products_list}
    order_unit_cost = {p: cout_commande / max(avg_demand[p], 1.0) for p in products_list}

    # --- Modèle ---
    model = pulp.LpProblem("zenith_orders", pulp.LpMinimize)

    Q = pulp.LpVariable.dicts("Q", (products_list, T), lowBound=0, cat="Continuous")
    S = pulp.LpVariable.dicts("S", (products_list, T), lowBound=0, cat="Continuous")
    R = pulp.LpVariable.dicts("R", (products_list, T), lowBound=0, cat="Continuous")

    # Objectif : Σ (commande/unité + stockage + rupture)
    model += pulp.lpSum(
        order_unit_cost[p] * Q[p][t]
        + storage_cost[p] * S[p][t]
        + stockout_cost[p] * R[p][t]
        for p in products_list
        for t in T
    )

    for p in products_list:
        L = lead[p]
        for t in T:
            received = Q[p][t - L] if (t - L) in T else 0.0
            prev_stock = S[p][t - 1] if t > 1 else stock0[p]
            # Conservation : S_t = S_{t-1} + Q_{t-L} - D + R
            model += S[p][t] == prev_stock + received - demand[(p, t)] + R[p][t]

    # Contrainte budgétaire par période
    for t in T:
        model += pulp.lpSum(cost[p] * Q[p][t] for p in products_list) <= budget_mensuel

    # Résolution — LP continue, CBC simplex (rapide, gap nul exact).
    solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=30, threads=2)
    status = model.solve(solver)
    logger.info("Statut solveur: %s", pulp.LpStatus[status])

    rows = []
    for p in products_list:
        for t in T:
            q_val = float(pulp.value(Q[p][t]) or 0)
            q_int = int(np.round(q_val))
            rows.append({
                "produit_id": p,
                "mois_offset": t,
                "quantite_commandee": q_int,
                "stock_final": float(pulp.value(S[p][t]) or 0),
                "rupture": float(pulp.value(R[p][t]) or 0),
                "demande_prevue": demand[(p, t)],
                "commande_passee": int(q_int > 0),
                "classe_abc": classe.get(p),
                "cout_achat": cost[p],
                "lead_time_mois": lead[p],
            })
    plan = pd.DataFrame(rows)
    return plan
