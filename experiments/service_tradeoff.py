"""P1-7 — Compromis coût / service.

Politique : pipeline complet (prévision différenciée + obsolescence + LP en
horizon glissant) sur le backtest à demande réalisée
(:mod:`experiments.policy_baselines`). Trois définitions du service, globales
et par classe ABC (classe connue à l'origine du mois) :
- ``unit_fill_rate``      : unités servies / unités demandées ;
- ``order_fill_rate``     : lignes de vente entièrement servies / lignes ;
- ``cycle_service_level`` : part des (produit, mois) avec demande > 0 sans
  aucune unité perdue (cycle = période de révision d'un mois).

Balayages
---------
1. Budget mensuel B ∈ {25k, 50k, 100k, 200k, 300k, 500k, 750k, 1M} $ (même
   B pour le LP et pour le contrôle budgétaire de la simulation).
2. Levier de service à budget nominal : multiplicateur de la pénalité de
   rupture de l'objectif LP (1+π) ∈ {1,0 ; 1,2 ; 1,5 ; 2 ; 3 ; 5 ; 10}.
3. Frontière de référence classique : (s, S) avec un z unique ∈ grille.

Pour chaque cible 70 / 80 / 90 / 95 % et chaque définition : plus petit
budget du balayage qui l'atteint sur la période de test, sinon
``infeasible`` (non atteinte même à 1 M$ avec les paramètres actuels).

Sorties : ``service_tradeoff.csv``, ``service_targets.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.config import PAPER_DIR, ensure_dirs
from experiments.lp_model import Params
from experiments.policies import LPPolicy, StockLevelPolicy, Z_GRID
from experiments.policy_baselines import N_DEV, prepare_data
from experiments.simulation import SimConfig, simulate, summarize

BUDGETS = (25_000, 50_000, 100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000)
PENALTY_FACTORS = (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0)
TARGETS = (0.70, 0.80, 0.90, 0.95)
SERVICE = ("unit_fill_rate", "order_fill_rate", "cycle_service_level")


def _rows(policy, data, cfg, sweep: str, value: float) -> list[dict]:
    res = simulate(policy, data, cfg)
    fr = res.frame(policy.name)
    K = len(data.months)
    fr["period"] = np.where(np.repeat(np.arange(K), data.n) < N_DEV, "dev", "test")
    fr["abc"] = np.concatenate([data.abc(k) for k in range(K)])
    rows = []
    for per in ("test", "full"):
        sub = fr if per == "full" else fr[fr["period"] == per]
        for abc, g in [("all", sub)] + list(sub.groupby("abc")):
            rows.append({"sweep": sweep, "value": value, "policy": policy.name, "period": per, "abc": abc,
                         **summarize(g), "budget_clips": res.budget_clips,
                         "infeasible_decisions": res.infeasible_decisions})
    return rows


def run() -> pd.DataFrame:
    ensure_dirs()
    data = prepare_data(["pipeline"])
    rows: list[dict] = []
    for b in BUDGETS:
        params = Params(budget=float(b))
        rows += _rows(LPPolicy("pipeline", "pipeline_lp", params), data, SimConfig(budget=float(b)), "budget", b)
        print(f"budget {b} done", flush=True)
    for f in PENALTY_FACTORS:
        params = Params(penalty=f - 1.0)
        rows += _rows(LPPolicy("pipeline", "pipeline_lp", params), data, SimConfig(), "penalty_factor", f)
        print(f"penalty {f} done", flush=True)
    for z in Z_GRID:
        rows += _rows(StockLevelPolicy("sS", {"A": z, "B": z, "C": z}), data, SimConfig(), "sS_z", z)
    res = pd.DataFrame(rows)
    res.to_csv(PAPER_DIR / "service_tradeoff.csv", index=False)

    targets = []
    base = res[(res["period"] == "test") & (res["abc"] == "all")]
    for metric in SERVICE:
        for tgt in TARGETS:
            row = {"service_definition": metric, "target": tgt}
            for sweep, label in (("budget", "min_budget_usd"), ("penalty_factor", "min_penalty_factor")):
                ok = base[(base["sweep"] == sweep) & (base[metric] >= tgt)].sort_values("value")
                row[label] = float(ok["value"].iloc[0]) if len(ok) else "infeasible"
            best = base[base["sweep"] == "budget"][metric].max()
            row["max_achieved_over_budget_sweep"] = float(best)
            ok = base[(base["sweep"] == "sS_z") & (base[metric] >= tgt)].sort_values("total_cost")
            row["sS_cheapest_z_reaching_target"] = float(ok["value"].iloc[0]) if len(ok) else "infeasible"
            row["sS_cost_at_that_z"] = float(ok["total_cost"].iloc[0]) if len(ok) else None
            targets.append(row)
    pd.DataFrame(targets).to_csv(PAPER_DIR / "service_targets.csv", index=False)
    return res


def main() -> int:
    res = run()
    sub = res[(res["period"] == "test") & (res["abc"] == "all")]
    cols = ["sweep", "value", "total_cost", "order_value", "unit_fill_rate", "order_fill_rate",
            "cycle_service_level", "budget_clips"]
    print(sub[cols].round(4).to_string(index=False))
    print(pd.read_csv(PAPER_DIR / "service_targets.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
