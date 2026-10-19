"""P1-6 — Baselines de politique d'inventaire, à conditions strictement identiques.

Toutes les politiques sont simulées sur la même demande réalisée (ventes
mensuelles et lignes de vente février 2024 → juillet 2025), le même stock
initial, les mêmes prix, coûts, délais, budget et capacité
(:mod:`experiments.simulation`). Les politiques classiques ont leur
coefficient de sécurité z réglé **par classe ABC sur la période de
développement uniquement** (févr. → déc. 2024, grille complète 12³) ;
les résultats de référence sont ceux de la période de **test**
(janv. → juil. 2025).

Politiques
----------
1. ``incumbent`` : règle empirique de l'article (``simulate_baseline_policy``)
   appliquée mois par mois avec les prévisions du pipeline (= ablation « −LP »).
2. ``sS_tuned`` ; 3. ``order_up_to_tuned`` ; 4. ``ma3_order_up_to_tuned``.
5. ``best_baseline_lp`` : meilleure baseline de prévision (MASE m=1 sur des
   cibles ≤ déc. 2024) + même optimiseur LP que le pipeline.
6. ``pipeline_no_obso_lp`` ; 7. ``pipeline_no_class_lp`` (seasonal naive pour
   tous, obsolescence conservée) ; 8. ``pipeline_lp`` (pipeline complet).
Variante : ``pipeline_mlp12_lp`` (MLP classe A à fenêtre 12 mois).

Sorties : ``policy_comparison.csv``, ``policy_service_by_class.csv``,
``policy_tuning.csv``, ``policy_sim_detail.csv``, ``policy_meta.json``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.config import DEV_END, DEV_MONTHS, PAPER_DIR, ensure_dirs
from experiments.forecasters import MODEL_BASELINES, SIMPLE_BASELINES
from experiments.lp_model import Params
from experiments.metrics import compute_metrics
from experiments.policies import IncumbentPolicy, LPPolicy, StockLevelPolicy, tune_z
from experiments.simulation import SimConfig, SimData, attach_forecasts, load_sim_data, simulate, summarize

N_DEV = len(DEV_MONTHS)
COLUMNS = ["policy", "n_orders", "total_units", "order_value", "avg_inventory_value", "ordering_cost",
           "holding_cost", "stockout_cost", "total_cost", "revenue", "unit_fill_rate", "order_fill_rate",
           "cycle_service_level"]


def choose_best_forecast_baseline(prefix: str = "rolling_origin") -> tuple[str, pd.DataFrame]:
    """Meilleure baseline (MASE m=1) sur les seules cibles de la période de développement."""
    err = pd.read_csv(PAPER_DIR / f"{prefix}_errors.csv")
    err = err[err["y_true"].notna() & (pd.to_datetime(err["target_month"]) <= DEV_END)]
    err = err.merge(pd.read_csv(PAPER_DIR / f"{prefix}_scales.csv"), on=["origin", "product_id"], how="left")
    rows = [{"model": m, **compute_metrics(g)} for m, g in err.groupby("method")
            if m in SIMPLE_BASELINES + MODEL_BASELINES]
    table = pd.DataFrame(rows).sort_values("MASE_m1")
    return str(table.iloc[0]["model"]), table


def prepare_data(methods: list[str], horizon: int = 3, errors_file: str = "rolling_origin_errors.csv") -> SimData:
    data = load_sim_data()
    attach_forecasts(data, errors_file, sorted(set(methods)), horizon)
    return data


def build_policies(best: str, z: dict[str, dict], params: Params) -> list:
    H = params.horizon
    return [
        IncumbentPolicy("pipeline", H),
        StockLevelPolicy("sS", z["sS"]),
        StockLevelPolicy("OUT", z["OUT"]),
        StockLevelPolicy("MA3", z["MA3"]),
        LPPolicy(best, "best_baseline_lp", params),
        LPPolicy("pipeline_no_obso", "pipeline_no_obso_lp", params, use_obsolescence=False),
        LPPolicy("seasonal_naive", "pipeline_no_class_lp", params),
        LPPolicy("pipeline", "pipeline_lp", params),
        LPPolicy("pipeline_mlp12", "pipeline_mlp12_lp", params),
    ]


def evaluate(policies: list, data: SimData, cfg: SimConfig, extra: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simule chaque politique ; renvoie (KPI par période, détail produit × mois)."""
    months = [m.date() for m in data.months]
    period = {m: ("dev" if k < N_DEV else "test") for k, m in enumerate(months)}
    # Classe ABC et drapeau d'obsolescence connus à l'origine de chaque mois (ordre mois-majeur)
    abc = np.concatenate([data.abc(k) for k in range(len(months))])
    obs = np.concatenate([data.obsolete(k) for k in range(len(months))]).astype(int)
    frames, rows = [], []
    for pol in policies:
        res = simulate(pol, data, cfg)
        fr = res.frame(pol.name)
        fr["period"] = fr["month"].map(period)
        fr["abc"] = abc
        fr["obsolete_flag_t0"] = obs
        frames.append(fr)
        for per in ("test", "dev", "full"):
            sub = fr if per == "full" else fr[fr["period"] == per]
            rows.append({"policy": pol.name, "period": per, **summarize(sub),
                         "n_months": int(sub["month"].nunique()),
                         "budget_clips": res.budget_clips, "capacity_violations": res.capacity_violations,
                         "infeasible_decisions": res.infeasible_decisions,
                         "z": json.dumps(getattr(pol, "z", None)),
                         "forecast_method": getattr(pol, "method", None), **(extra or {})})
    kpi = pd.DataFrame(rows)
    inc = kpi[kpi["policy"] == "incumbent"].set_index("period")["total_cost"]
    kpi["total_cost_vs_incumbent_pct"] = [100 * (c - inc[p]) / inc[p] if p in inc else np.nan
                                          for c, p in zip(kpi["total_cost"], kpi["period"])]
    return kpi, pd.concat(frames, ignore_index=True)


def tune_all(data: SimData, cfg: SimConfig) -> tuple[dict, pd.DataFrame]:
    z, grids = {}, []
    for kind in ("sS", "OUT", "MA3"):
        z[kind], grid = tune_z(kind, data, cfg, N_DEV)
        grids.append(grid)
    return z, pd.concat(grids, ignore_index=True)


def run() -> pd.DataFrame:
    ensure_dirs()
    best, ranking = choose_best_forecast_baseline()
    params, cfg = Params(), SimConfig()
    data = prepare_data(["pipeline", "pipeline_no_obso", "pipeline_mlp12", "seasonal_naive", best])
    z, grid = tune_all(data, cfg)
    grid.to_csv(PAPER_DIR / "policy_tuning.csv", index=False)
    kpi, detail = evaluate(build_policies(best, z, params), data, cfg)

    cols = COLUMNS + [c for c in kpi.columns if c not in COLUMNS]
    kpi = kpi[["period"] + [c for c in cols if c != "period"]]
    kpi.to_csv(PAPER_DIR / "policy_comparison.csv", index=False)
    detail.to_csv(PAPER_DIR / "policy_sim_detail.csv", index=False)

    by_class = []
    for (pol, per, abc), g in pd.concat([detail, detail.assign(period="full")]).groupby(["policy", "period", "abc"]):
        by_class.append({"policy": pol, "period": per, "abc": abc, **summarize(g)})
    pd.DataFrame(by_class).to_csv(PAPER_DIR / "policy_service_by_class.csv", index=False)

    meta = {
        "best_forecast_baseline": best,
        "best_baseline_ranking_dev": ranking[["model", "MASE_m1", "MAE", "n_pairs"]].to_dict("records"),
        "z_tuned_on_dev": z,
        "n_products": data.n,
        "dev_months": [str(m.date()) for m in data.months[:N_DEV]],
        "test_months": [str(m.date()) for m in data.months[N_DEV:]],
        "initial_stock_units": float(data.stock0.sum()),
        "cost_definitions": "commande 50 $/commande ; stockage stock_fin × coût × 0,1 %/j × 30 ; "
                            "rupture unités perdues × (prix − coût) (src.optimization.financial_kpis)",
        "avg_inventory_value": "moyenne mensuelle de Σ_i stock_fin × coût (valeur totale du stock)",
        "service_definitions": {
            "unit_fill_rate": "unités servies / unités demandées",
            "order_fill_rate": "lignes de vente entièrement servies / lignes de vente",
            "cycle_service_level": "part des (produit, mois) avec demande > 0 sans aucune unité perdue",
        },
    }
    (PAPER_DIR / "policy_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return kpi


def main() -> int:
    kpi = run()
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(kpi[kpi["period"] == "test"][COLUMNS + ["total_cost_vs_incumbent_pct", "budget_clips",
                                                      "infeasible_decisions", "z"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
