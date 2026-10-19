"""Modèles d'optimisation des commandes : LP du code (réplique exacte) et MILP.

``solve_lp`` reproduit coefficient par coefficient :func:`src.optimization.optimize_orders`
(même construction PuLP, mêmes noms de variables, même solveur) et ajoute
seulement les arrivées déjà en transit ρ[i,t] (nulles par défaut, auquel cas
le modèle est strictement celui du code ; cf. ``tests/test_exp_lp_model.py``).

``solve_milp`` est le modèle de référence décrit dans l'article :
q ∈ Z≥0, x ∈ {0,1}, I ≥ 0, s ≥ 0, coût fixe K·x, liaison q ≤ M·x,
MOQ q ≥ m·x, budget, capacité, obsolètes exclus, ρ en transit.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from src.optimization import (
    BUDGET_MENSUEL_DEFAUT,
    CAPACITE_STOCKAGE_DEFAUT,
    COUT_COMMANDE_FIXE,
    PENALITE_RUPTURE_CLIENT,
    TAUX_STOCKAGE_JOURNALIER,
    VOLUME_UNITAIRE_DEFAUT,
    lead_time_months,
)

CLASS_WEIGHTS = {"A": 4.0, "B": 2.5, "C": 1.5}


@dataclass(frozen=True)
class Params:
    horizon: int = 3
    budget: float = BUDGET_MENSUEL_DEFAUT
    capacity: float = CAPACITE_STOCKAGE_DEFAUT
    volume: float = VOLUME_UNITAIRE_DEFAUT
    order_cost: float = COUT_COMMANDE_FIXE
    holding_rate: float = TAUX_STOCKAGE_JOURNALIER
    penalty: float = PENALITE_RUPTURE_CLIENT
    class_weights: tuple = tuple(CLASS_WEIGHTS.items())
    lead_extra: int = 0

    def weights(self) -> dict[str, float]:
        return dict(self.class_weights)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


@dataclass
class Instance:
    """Données d'un problème de planification (produits actifs uniquement)."""
    products: list[str]
    cost: dict[str, float]
    price: dict[str, float]
    classe: dict[str, str]
    lead: dict[str, int]
    demand: dict[str, np.ndarray]           # prévision d̂[i, 1..H] (≥ 0)
    stock0: dict[str, float]
    in_transit: dict[str, np.ndarray] = field(default_factory=dict)  # ρ[i, 1..H]

    def rho(self, p: str, t: int) -> float:
        arr = self.in_transit.get(p)
        return float(arr[t - 1]) if arr is not None and t - 1 < len(arr) else 0.0


def make_instance(products: pd.DataFrame, forecasts: dict[str, np.ndarray], stock0: dict[str, float],
                  params: Params, in_transit: dict[str, np.ndarray] | None = None) -> Instance:
    """``products`` : produit_id, classe_abc, cout_achat_unitaire, prix_vente_unitaire, origine_fournisseur."""
    pids = products["produit_id"].tolist()
    H = params.horizon
    demand = {}
    for p in pids:
        v = np.asarray(forecasts.get(p, np.zeros(H)), dtype=float)[:H]
        if len(v) < H:
            v = np.pad(v, (0, H - len(v)))
        demand[p] = np.maximum(v, 0.0)
    return Instance(
        products=pids,
        cost=dict(zip(pids, products["cout_achat_unitaire"].astype(float))),
        price=dict(zip(pids, products["prix_vente_unitaire"].astype(float))),
        classe=dict(zip(pids, products["classe_abc"])),
        lead={p: lead_time_months(o) + params.lead_extra
              for p, o in zip(pids, products["origine_fournisseur"])},
        demand=demand,
        stock0={p: float(stock0.get(p, 0.0)) for p in pids},
        in_transit=in_transit or {},
    )


# --------------------------------------------------------------------- #
# Coefficients (identiques à optimize_orders)
# --------------------------------------------------------------------- #
def code_coefficients(inst: Instance, params: Params) -> dict[str, dict[str, float]]:
    T = range(1, params.horizon + 1)
    w = params.weights()
    margin = {p: max(inst.price[p] - inst.cost[p], 0.5) for p in inst.products}
    avg_demand = {p: max(1.0, np.mean([inst.demand[p][t - 1] for t in T])) for p in inst.products}
    return {
        "order_unit_cost": {p: params.order_cost / avg_demand[p] for p in inst.products},
        "storage_cost": {p: inst.cost[p] * params.holding_rate * 30 for p in inst.products},
        "stockout_cost": {p: (margin[p] + inst.price[p]) * (1 + params.penalty) * w.get(inst.classe[p], 1.5)
                          for p in inst.products},
    }


def paper_coefficients(inst: Instance, params: Params) -> dict[str, dict[str, float]]:
    """Objectif tel que formulé dans l'article : w_i · (K·x + h_i·I + p_i·s), p_i = (1+pénalité)·prix."""
    w = params.weights()
    return {
        "weight": {p: w.get(inst.classe[p], 1.5) for p in inst.products},
        "storage_cost": {p: inst.cost[p] * params.holding_rate * 30 for p in inst.products},
        "stockout_cost": {p: (1 + params.penalty) * inst.price[p] for p in inst.products},
    }


# --------------------------------------------------------------------- #
# LP du code
# --------------------------------------------------------------------- #
def solve_lp(inst: Instance, params: Params, msg: bool = False) -> dict:
    """Relaxation continue telle que codée dans ``optimize_orders`` (+ ρ optionnel)."""
    T = list(range(1, params.horizon + 1))
    co = code_coefficients(inst, params)
    P = inst.products
    model = pulp.LpProblem("zenith_orders_lp", pulp.LpMinimize)
    Q = pulp.LpVariable.dicts("Q", (P, T), lowBound=0, cat="Continuous")
    S = pulp.LpVariable.dicts("S", (P, T), lowBound=0, cat="Continuous")
    R = pulp.LpVariable.dicts("R", (P, T), lowBound=0, cat="Continuous")
    model += pulp.lpSum(co["order_unit_cost"][p] * Q[p][t] + co["storage_cost"][p] * S[p][t]
                        + co["stockout_cost"][p] * R[p][t] for p in P for t in T)
    for p in P:
        L = inst.lead[p]
        for t in T:
            received = Q[p][t - L] if (t - L) in T else 0.0
            rho = inst.rho(p, t)
            if rho:
                received = received + rho
            prev = S[p][t - 1] if t > 1 else inst.stock0[p]
            model += S[p][t] == prev + received - inst.demand[p][t - 1] + R[p][t]
    for t in T:
        model += pulp.lpSum(inst.cost[p] * Q[p][t] for p in P) <= params.budget
    for t in T:
        model += pulp.lpSum(params.volume * S[p][t] for p in P) <= params.capacity
    tic = time.perf_counter()
    status = model.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=60, threads=2))
    return {
        "status": pulp.LpStatus[status],
        "objective": float(pulp.value(model.objective) or 0.0),
        "seconds": time.perf_counter() - tic,
        "Q": {p: np.array([float(pulp.value(Q[p][t]) or 0.0) for t in T]) for p in P},
        "S": {p: np.array([float(pulp.value(S[p][t]) or 0.0) for t in T]) for p in P},
        "R": {p: np.array([float(pulp.value(R[p][t]) or 0.0) for t in T]) for p in P},
        "n_vars": len(model.variables()),
        "n_constraints": len(model.constraints),
    }


def round_like_code(q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Arrondi de ``optimize_orders`` : ``int(np.round(q))`` (au plus proche, pas au supérieur)."""
    return {p: np.array([int(np.round(v)) for v in arr]) for p, arr in q.items()}


def round_up(q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {p: np.ceil(np.round(arr, 9)).astype(int) for p, arr in q.items()}


# --------------------------------------------------------------------- #
# MILP
# --------------------------------------------------------------------- #
def big_m(inst: Instance, moq: dict[str, float]) -> dict[str, float]:
    """Borne valide : commander plus que la demande totale de l'horizon est dominé."""
    return {p: max(float(moq.get(p, 0.0)), float(np.ceil(inst.demand[p].sum())), 1.0) for p in inst.products}


_CBC_PATTERNS = {
    "result": r"Result - (.+)",
    "objective": r"Objective value:\s+([-\d.eE+]+)",
    "bound": r"Lower bound:\s+([-\d.eE+]+)",
    "gap": r"Gap:\s+([-\d.eE+]+)",
    "nodes": r"Enumerated nodes:\s+(\d+)",
    "iterations": r"Total iterations:\s+(\d+)",
    "cpu_seconds": r"Time \(CPU seconds\):\s+([\d.]+)",
    "wall_seconds": r"Time \(Wallclock seconds\):\s+([\d.]+)",
}


def parse_cbc_log(path: Path) -> dict:
    text = Path(path).read_text(errors="replace") if Path(path).exists() else ""
    out: dict = {}
    for key, pat in _CBC_PATTERNS.items():
        m = re.findall(pat, text)
        if m:
            v = m[-1].strip()
            out[key] = v if key == "result" else float(v)
    ver = re.search(r"Version:\s*([\w.]+)", text)
    if ver:
        out["cbc_version"] = ver.group(1)
    return out


def solve_milp(inst: Instance, params: Params, moq: dict[str, float] | None = None,
               objective: str = "code", time_limit: int = 600, log_path: Path | None = None,
               msg: bool = False) -> dict:
    """MILP complet. ``objective`` : ``code`` (coûts de l'implémentation, coût fixe K
    réel par commande) ou ``paper`` (w_i·(K·x + h·I + p·s), p = (1+pénalité)·prix)."""
    T = list(range(1, params.horizon + 1))
    P = inst.products
    moq = moq or {}
    M = big_m(inst, moq)
    model = pulp.LpProblem("zenith_orders_milp", pulp.LpMinimize)
    q = pulp.LpVariable.dicts("q", (P, T), lowBound=0, cat="Integer")
    x = pulp.LpVariable.dicts("x", (P, T), cat="Binary")
    I = pulp.LpVariable.dicts("I", (P, T), lowBound=0, cat="Continuous")
    s = pulp.LpVariable.dicts("s", (P, T), lowBound=0, cat="Continuous")

    if objective == "code":
        co = code_coefficients(inst, params)
        model += pulp.lpSum(params.order_cost * x[p][t] + co["storage_cost"][p] * I[p][t]
                            + co["stockout_cost"][p] * s[p][t] for p in P for t in T)
    elif objective == "paper":
        co = paper_coefficients(inst, params)
        model += pulp.lpSum(co["weight"][p] * (params.order_cost * x[p][t] + co["storage_cost"][p] * I[p][t]
                                               + co["stockout_cost"][p] * s[p][t]) for p in P for t in T)
    else:
        raise ValueError(objective)

    for p in P:
        L = inst.lead[p]
        for t in T:
            arrivals = (q[p][t - L] if (t - L) in T else 0.0)
            rho = inst.rho(p, t)
            if rho:
                arrivals = arrivals + rho
            prev = I[p][t - 1] if t > 1 else inst.stock0[p]
            model += I[p][t] == prev + arrivals - inst.demand[p][t - 1] + s[p][t]
            model += q[p][t] <= M[p] * x[p][t]
            if moq.get(p, 0) > 0:
                model += q[p][t] >= moq[p] * x[p][t]
    for t in T:
        model += pulp.lpSum(inst.cost[p] * q[p][t] for p in P) <= params.budget
        model += pulp.lpSum(params.volume * I[p][t] for p in P) <= params.capacity

    log_path = Path(log_path) if log_path else None
    # Pas d'option ``threads`` : CBC reste en branch-and-bound séquentiel
    # (déterministe). Avec ``-threads 1`` le binaire CBC fourni par PuLP se
    # bloque sous Windows dès qu'il faut brancher (deux threads en attente).
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit,
                               logPath=str(log_path) if log_path else None)
    tic = time.perf_counter()
    status = model.solve(solver)
    seconds = time.perf_counter() - tic
    log = parse_cbc_log(log_path) if log_path else {}
    qv = {p: np.array([int(round(pulp.value(q[p][t]) or 0.0)) for t in T]) for p in P}
    obj = float(pulp.value(model.objective) or 0.0)
    bound = log.get("bound")
    if bound is None and log.get("result", "").startswith("Optimal"):
        bound = obj
    gap = (obj - bound) / abs(obj) if (bound is not None and obj) else None
    return {
        "status": pulp.LpStatus[status],
        "cbc_result": log.get("result"),
        "objective": obj,
        "best_bound": bound,
        "gap_relative": gap,
        "cbc_gap": log.get("gap"),
        "seconds": seconds,
        "nodes": log.get("nodes"),
        "n_vars": len(model.variables()),
        "n_integer_vars": sum(1 for v in model.variables() if v.cat == "Integer"),
        "n_constraints": len(model.constraints),
        "time_limit_s": time_limit,
        "cbc_version": log.get("cbc_version"),
        "q": qv,
    }


# --------------------------------------------------------------------- #
# Évaluation d'un plan entier (conséquences réelles vs la prévision)
# --------------------------------------------------------------------- #
def evaluate_plan(inst: Instance, params: Params, q: dict[str, np.ndarray],
                  moq: dict[str, float] | None = None) -> dict:
    """Propage un plan entier dans le bilan de stock et vérifie chaque contrainte.

    I_t = max(0, I_{t-1} + a_t − d̂_t), s_t = max(0, d̂_t − I_{t-1} − a_t),
    a_t = q_{t−L} + ρ_t. Coûts sous les deux objectifs (code et article).
    """
    T = list(range(1, params.horizon + 1))
    moq = moq or {}
    co, pa = code_coefficients(inst, params), paper_coefficients(inst, params)
    spend = np.zeros(len(T)); vol = np.zeros(len(T))
    obj_code = obj_paper = 0.0
    n_orders = 0; moq_viol = 0
    lost = 0.0; dem = 0.0
    for p in inst.products:
        L = inst.lead[p]; inv = inst.stock0[p]
        for t in T:
            qt = int(q[p][t - 1])
            arr = (q[p][t - L - 1] if t - L >= 1 else 0) + inst.rho(p, t)
            avail = inv + arr
            d = inst.demand[p][t - 1]
            short = max(0.0, d - avail)
            inv = max(0.0, avail - d)
            order = qt > 0
            n_orders += order
            if order and moq.get(p, 0) > 0 and qt < moq[p]:
                moq_viol += 1
            spend[t - 1] += inst.cost[p] * qt
            vol[t - 1] += params.volume * inv
            obj_code += params.order_cost * order + co["storage_cost"][p] * inv + co["stockout_cost"][p] * short
            obj_paper += pa["weight"][p] * (params.order_cost * order + pa["storage_cost"][p] * inv
                                            + pa["stockout_cost"][p] * short)
            lost += short; dem += d
    budget_excess = np.maximum(spend - params.budget, 0.0)
    cap_excess = np.maximum(vol - params.capacity, 0.0)
    return {
        "objective_code": obj_code,
        "objective_paper": obj_paper,
        "n_orders": int(n_orders),
        "units_ordered": int(sum(int(v.sum()) for v in q.values())),
        "spend_by_month": [round(float(v), 2) for v in spend],
        "budget_violation_months": int((budget_excess > 1e-6).sum()),
        "budget_excess_usd_by_month": [round(float(v), 2) for v in budget_excess],
        "budget_excess_pct_by_month": [round(float(100 * v / params.budget), 4) for v in budget_excess],
        "capacity_use_by_month": [round(float(v), 3) for v in vol],
        "capacity_violation_months": int((cap_excess > 1e-6).sum()),
        "moq_violations": int(moq_viol),
        "planned_fill_rate": float(1 - lost / dem) if dem > 0 else None,
    }
