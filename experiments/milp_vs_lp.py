"""P0-5 — MILP exact vs relaxation LP arrondie, et contrôle de faisabilité.

Instance : celle de l'article (plan de commandes août → octobre 2025,
produits non obsolètes, prévisions ``previsions_complet.csv``, stock
``stock_courant``), reconstruite exactement comme ``scripts/run_optimization.py``.

1. MILP complet résolu par CBC (limite 600 s, 1 thread) : statut, objectif,
   meilleure borne, gap, temps, taille.
2. Relaxation LP **telle que codée** (coût fixe linéarisé K/d̄ par unité),
   arrondie comme le code (``np.round``, au plus proche) et, en variante, à
   l'entier supérieur. Le plan arrondi est propagé dans le bilan de stock et
   chaque contrainte est vérifiée : budget, capacité, MOQ.
3. Écart de coût : plan LP arrondi vs MILP, sous le même objectif.

Données absentes du dépôt, posées en hypothèses explicites :
- MOQ : aucune donnée. Deux variantes : ``none`` (m = 0) et ``1_month``
  (m_i = max(1, ⌈moyenne de d̂_i⌉), un mois de demande prévue).
- Commandes en transit ρ : aucune donnée → ρ = 0 (comme le code).
- Volume unitaire 0,01 m³ et capacité 5 000 m³ : constantes du code.
Budgets testés : 500 k$ (nominal) puis 150 / 100 / 50 k$ pour rendre la
contrainte budgétaire active (le plan nominal ne dépense que 179 k$ au mois 1).

Sortie : ``paper_results/milp_vs_lp.json``.
"""
from __future__ import annotations

import importlib.util
import json
import time

import numpy as np
import pandas as pd

from experiments.config import CACHE_DIR, PAPER_DIR, ROOT, ensure_dirs
from experiments.lp_model import (
    Params, evaluate_plan, make_instance, round_like_code, round_up, solve_lp, solve_milp,
)
from src.optimization import financial_kpis, optimize_orders
from src.utils import TAB_DIR

BUDGETS = (500_000.0, 150_000.0, 100_000.0, 50_000.0)
MOQ_VARIANTS = ("none", "1_month")
TIME_LIMIT = 600


def _load_run_optimization():
    spec = importlib.util.spec_from_file_location("run_optimization", ROOT / "scripts" / "run_optimization.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paper_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    products = _load_run_optimization().build_products_input()
    forecasts = pd.read_csv(TAB_DIR / "previsions_complet.csv", parse_dates=["date"])
    return products, forecasts


def paper_instance(params: Params):
    products, forecasts = paper_inputs()
    active = products[products["a_risque_obsolescence"] == 0].reset_index(drop=True)
    fc = forecasts.sort_values(["produit_id", "date"])
    fdict = {p: g["prevision"].to_numpy(dtype=float)[:params.horizon] for p, g in fc.groupby("produit_id")}
    stock0 = dict(zip(active["produit_id"], active["stock_courant"].astype(float)))
    return make_instance(active, fdict, stock0, params), products, forecasts


def moq_for(inst, variant: str) -> dict[str, float]:
    if variant == "none":
        return {}
    return {p: float(max(1.0, np.ceil(inst.demand[p].mean()))) for p in inst.products}


def reproduction_check(products: pd.DataFrame, forecasts: pd.DataFrame, inst, params: Params) -> dict:
    """optimize_orders (code) vs réplique vs plan publié commandes_centrales.csv."""
    plan, kpis = optimize_orders(products, forecasts)
    lp = solve_lp(inst, params)
    rep = round_like_code(lp["Q"])
    code_q = plan.pivot(index="produit_id", columns="mois_offset", values="quantite_commandee")
    same_as_code = all(np.array_equal(code_q.loc[p].to_numpy(), rep[p]) for p in inst.products)
    published = pd.read_csv(TAB_DIR / "commandes_centrales.csv")
    pub_q = published.pivot(index="produit_id", columns="mois_offset", values="quantite_commandee")
    same_as_published = all(np.array_equal(pub_q.loc[p].to_numpy(), rep[p]) for p in inst.products)
    return {
        "optimize_orders_status": kpis["statut"],
        "replica_matches_optimize_orders": bool(same_as_code),
        "replica_matches_published_commandes_centrales": bool(same_as_published),
        "cout_total_simule_usd_recomputed": financial_kpis(plan)["cout_total_simule_usd"],
        "note": "cout_total_simule_usd = coût de l'article (financial_kpis) sur ce plan, évalué "
                "contre la demande PRÉVUE",
    }


def run() -> dict:
    ensure_dirs()
    log_dir = CACHE_DIR / "cbc_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    base = Params()
    inst, products, forecasts = paper_instance(base)
    out: dict = {
        "instance": {
            "n_products_active": len(inst.products),
            "horizon_months": base.horizon,
            "months": sorted(forecasts["date"].dt.date.astype(str).unique().tolist())[:base.horizon],
            "rho_in_transit": "0 (aucune donnée de commandes en transit)",
            "big_M": "M_i = max(m_i, ⌈Σ_t d̂_i,t⌉, 1) — commander plus que la demande de l'horizon est dominé",
            "rounding_in_code": "int(np.round(q)) — au plus proche (src/optimization.py:184), pas à l'entier supérieur",
        },
        "reproduction": reproduction_check(products, forecasts, inst, base),
        "scenarios": [],
    }

    lp_cache: dict[float, dict] = {}
    jobs = [(b, m, "code") for b in BUDGETS for m in MOQ_VARIANTS] + \
           [(BUDGETS[0], m, "paper") for m in MOQ_VARIANTS]
    for budget, moq_name, objective in jobs:
        params = base.with_(budget=budget)
        moq = moq_for(inst, moq_name)
        if budget not in lp_cache:
            lp_cache[budget] = solve_lp(inst, params)
        lp = lp_cache[budget]
        tag = f"B{int(budget)}_{moq_name}_{objective}"
        tic = time.perf_counter()
        milp = solve_milp(inst, params, moq=moq, objective=objective, time_limit=TIME_LIMIT,
                          log_path=log_dir / f"{tag}.log")
        print(f"{tag}: MILP {milp['status']} ({milp['cbc_result']}) obj={milp['objective']:.2f} "
              f"bound={milp['best_bound']} t={time.perf_counter() - tic:.1f}s", flush=True)
        key = f"objective_{objective}"
        ev_milp = evaluate_plan(inst, params, milp["q"], moq)
        ev_near = evaluate_plan(inst, params, round_like_code(lp["Q"]), moq)
        ev_ceil = evaluate_plan(inst, params, round_up(lp["Q"]), moq)

        def _gap(ev):
            return 100 * (ev[key] - ev_milp[key]) / ev_milp[key] if ev_milp[key] else None

        out["scenarios"].append({
            "budget_usd": budget, "moq": moq_name, "objective": objective,
            "milp": {k: v for k, v in milp.items() if k != "q"},
            "milp_plan": ev_milp,
            "lp_relaxation_as_coded": {"status": lp["status"], "objective_lp": lp["objective"],
                                       "n_vars": lp["n_vars"], "n_constraints": lp["n_constraints"],
                                       "seconds": lp["seconds"]},
            "lp_rounded_nearest (code)": ev_near,
            "lp_rounded_up": ev_ceil,
            "cost_gap_lp_rounded_nearest_vs_milp_pct": _gap(ev_near),
            "cost_gap_lp_rounded_up_vs_milp_pct": _gap(ev_ceil),
        })
    (PAPER_DIR / "milp_vs_lp.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                                               encoding="utf-8")
    return out


def main() -> int:
    out = run()
    print(json.dumps(out["reproduction"], indent=2, ensure_ascii=False))
    for s in out["scenarios"]:
        k = f"objective_{s['objective']}"
        near = s["lp_rounded_nearest (code)"]
        print(f"B={s['budget_usd']:>9.0f} moq={s['moq']:<8s} obj={s['objective']:<5s} | "
              f"MILP {s['milp']['status']:<10s} {s['milp_plan'][k]:>10.1f} gap={s['milp']['gap_relative']} "
              f"t={s['milp']['seconds']:.1f}s | LP arrondi {near[k]:>10.1f} "
              f"(+{s['cost_gap_lp_rounded_nearest_vs_milp_pct']:.1f} %) budget_viol={near['budget_violation_months']} "
              f"moq_viol={near['moq_violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
