"""P1-8 — Analyses de sensibilité une-variable-à-la-fois.

Question : l'économie de coût du pipeline sur la politique empirique (−28,3 %
dans l'article) survit-elle aux variations des paramètres ?

Deux méthodes d'évaluation, pour chaque scénario
------------------------------------------------
``paper_forecast_based`` — la méthode de l'article : un plan de H mois
(août → octobre 2025) jugé contre la demande **prévue**, avec les coûts de
``financial_kpis``. Variante ``as_published`` (LP sur les produits actifs vs
heuristique sur tous les produits, comme l'article) et ``paired`` (même
périmètre de produits actifs).

``backtest_realized`` — le backtest à demande réalisée de P1-6 (période de
test janv. → juil. 2025) : pipeline LP vs incumbent vs meilleure politique
classique (type choisi sur la période de développement au point nominal, z
re-réglé sur la période de développement pour chaque scénario).

Paramètres (planificateur = LP ; évaluation = coût mesuré)
---------------------------------------------------------
- budget 0,5× / 1× / 2× : planificateur + contrainte de simulation ;
- délai fournisseur : nominal, +1 mois (connu du planificateur), retard
  stochastique (+1 mois avec probabilité 0,3, planificateur non informé,
  10 réplications, nombres aléatoires communs entre politiques) ;
- pénalité de rupture (1+π) 1,0 / 1,2 / 1,5 / 2,0 : objectif LP seulement
  (le coût évalué reste la marge perdue, comme ``financial_kpis``) ;
- taux de stockage 0,05 / 0,1 / 0,2 %/jour : planificateur + évaluation ;
- pondérations ABC (4 ; 2,5 ; 1,5) / (1 ; 1 ; 1) / (3 ; 2 ; 1) : objectif LP ;
- seuil d'obsolescence 90 / 180 / 270 / 365 j (règles : seuil/30 mois) ;
- horizon 1 / 2 / 3 / 6 mois (H ≤ 3 : prévisions H = 3 tronquées ;
  H = 6 : run rolling-origin dédié) ;
- change USD/CDF ±10 / ±20 % — HYPOTHÈSE (aucune donnée de change dans le
  dépôt) : prix de vente fixés en CDF, coûts d'achat en USD ; une hausse δ
  du dollar multiplie les prix exprimés en USD par 1/(1+δ).

Sortie : ``paper_results/sensitivity.csv``.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from experiments.config import FORECAST_ORIGINS, PAPER_DIR, SEED, ensure_dirs
from experiments.data import clean_upto, detect_at, load_cat, load_raw, obsolescence_features_at
from experiments.lp_model import Params, make_instance, round_like_code, solve_lp
from experiments.milp_vs_lp import paper_inputs
from experiments.policies import IncumbentPolicy, LPPolicy, StockLevelPolicy, tune_z
from experiments.policy_baselines import N_DEV, prepare_data
from experiments.simulation import SimConfig, attach_forecasts, delay_matrix, simulate, summarize
from src.optimization import fournisseur_label, lead_time_months

N_REPLICATIONS = 10
P_DELAY = 0.3


# --------------------------------------------------------------------- #
# Méthode de l'article (plan unique jugé contre la prévision)
# --------------------------------------------------------------------- #
def paper_kpis(plan: pd.DataFrame, holding_rate: float, order_cost: float = 50.0) -> dict:
    """``financial_kpis`` avec un taux de stockage paramétrable (formules identiques)."""
    df = plan.copy()
    df["valeur_stock_immo"] = df["stock_final"] * df["cout_achat"]
    df["cout_stockage"] = df["valeur_stock_immo"] * holding_rate * 30
    df["marge_perdue"] = df["rupture"] * (df["prix_vente"] - df["cout_achat"])
    df["cout_commande"] = df["commande_passee"] * order_cost
    df["cout_total"] = df["cout_commande"] + df["cout_stockage"] + df["marge_perdue"]
    served = (df["demande_prevue"] - df["rupture"]).clip(lower=0)
    return {"total_cost": round(df["cout_total"].sum(), 2),
            "unit_fill_rate": 1 - df["rupture"].sum() / max(df["demande_prevue"].sum(), 1),
            "order_value": float((df["quantite_commandee"] * df["cout_achat"]).sum()),
            "avg_inventory_value": float(df.groupby("mois_offset")["valeur_stock_immo"].sum().mean()),
            "revenue": float((served * df["prix_vente"]).sum())}


def lp_plan_frame(inst, params: Params, lp: dict) -> pd.DataFrame:
    """Plan au format de ``optimize_orders`` (Q arrondi, S et R continus arrondis)."""
    q = round_like_code(lp["Q"])
    rows = []
    for p in inst.products:
        for t in range(params.horizon):
            rows.append({"produit_id": p, "mois_offset": t + 1, "quantite_commandee": int(q[p][t]),
                         "stock_final": round(float(lp["S"][p][t]), 1), "rupture": round(float(lp["R"][p][t]), 2),
                         "demande_prevue": round(float(inst.demand[p][t]), 2), "commande_passee": int(q[p][t] > 0),
                         "cout_achat": inst.cost[p], "prix_vente": inst.price[p]})
    return pd.DataFrame(rows)


def baseline_plan(products: pd.DataFrame, forecasts: pd.DataFrame, horizon: int, lead_extra: int = 0) -> pd.DataFrame:
    """Réplique de :func:`src.optimization.simulate_baseline_policy` avec délai paramétrable."""
    fc = forecasts.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    rows = []
    for _, p in products.iterrows():
        sub = fc[fc["produit_id"] == p["produit_id"]].sort_values("date").head(horizon)
        if sub.empty:
            continue
        forecast = sub["prevision"].to_numpy(dtype=float)
        stock = float(p.get("stock_courant", 0) or 0)
        L = lead_time_months(p.get("origine_fournisseur")) + lead_extra
        avg = forecast.mean() if len(forecast) else 0.0
        pipeline: dict[int, float] = {}
        for t in range(1, horizon + 1):
            d = float(forecast[t - 1])
            stock += pipeline.pop(t, 0.0)
            served = min(stock, d)
            rupture = max(0.0, d - served)
            stock -= served
            qty = 0
            if stock < avg:
                qty = int(np.ceil(2 * avg))
                pipeline[t + L] = pipeline.get(t + L, 0.0) + qty
            rows.append({"produit_id": p["produit_id"], "mois_offset": t, "quantite_commandee": qty,
                         "commande_passee": int(qty > 0), "stock_final": round(stock, 1),
                         "rupture": round(rupture, 2), "demande_prevue": round(d, 2),
                         "cout_achat": float(p.get("cout_achat_unitaire", 0) or 0),
                         "prix_vente": float(p.get("prix_vente_unitaire", 0) or 0),
                         "fournisseur": fournisseur_label(p.get("origine_fournisseur"))})
    return pd.DataFrame(rows)


def paper_method(params: Params, holding_rate: float, price_factor: float = 1.0) -> list[dict]:
    products, forecasts = paper_inputs()
    products = products.copy()
    products["prix_vente_unitaire"] = products["prix_vente_unitaire"] * price_factor
    active = products[products["a_risque_obsolescence"] == 0].reset_index(drop=True)
    fc = forecasts.sort_values(["produit_id", "date"])
    fdict = {p: g["prevision"].to_numpy(dtype=float)[:params.horizon] for p, g in fc.groupby("produit_id")}
    inst = make_instance(active, fdict, dict(zip(active["produit_id"], active["stock_courant"].astype(float))), params)
    lp = solve_lp(inst, params)
    k_lp = paper_kpis(lp_plan_frame(inst, params, lp), holding_rate)
    base_all = baseline_plan(products, forecasts, params.horizon, params.lead_extra)
    out = []
    for variant, base in (("as_published", base_all), ("paired", base_all[base_all["produit_id"].isin(inst.products)])):
        k_b = paper_kpis(base, holding_rate)
        for pol, k in (("pipeline_lp", k_lp), ("incumbent", k_b)):
            out.append({"method": f"paper_forecast_based_{variant}", "policy": pol, **k,
                        "infeasible_decisions": int(lp["status"] != "Optimal") if pol == "pipeline_lp" else 0,
                        "savings_vs_incumbent_pct": 100 * (k_b["total_cost"] - k_lp["total_cost"]) / k_b["total_cost"]
                        if pol == "pipeline_lp" else 0.0})
    return out


# --------------------------------------------------------------------- #
# Backtest à demande réalisée
# --------------------------------------------------------------------- #
def obsolescence_flags(rule_days: int) -> pd.DataFrame:
    """Drapeaux hybrides (IF c=0,10 + règles au seuil donné) recalculés à chaque origine."""
    raw, cat = load_raw(), load_cat()
    rows = []
    for t0 in FORECAST_ORIGINS:
        feats = obsolescence_features_at(clean_upto(raw, cat, t0), t0)
        iforest, rules = detect_at(feats, 0.10, rule_days)
        flag = iforest | rules
        rows += [{"origin": str(t0.date()), "produit_id": p, "obsolete_flag": int(v)} for p, v in flag.items()]
    return pd.DataFrame(rows)


def with_flags(data, flags: pd.DataFrame):
    """Copie de SimData avec d'autres drapeaux ; prévision pipeline = 0 si flagué, modèle de classe sinon."""
    d2 = copy.copy(data)
    st = data.states.drop(columns="obsolete_flag").merge(flags, on=["origin", "produit_id"], how="left")
    st["obsolete_flag"] = st["obsolete_flag"].fillna(0).astype(int)
    d2.states = st
    d2._states_by_k = {}
    d2.forecasts = dict(data.forecasts)
    base = data.forecasts[("pipeline_no_obso", 3)]
    flag_mat = np.stack([d2.obsolete(k) for k in range(len(data.origins))])
    d2.forecasts[("pipeline", 3)] = np.where(flag_mat[:, :, None], 0.0, base)
    return d2


def backtest(data, cfg: SimConfig, params: Params, conv_kind: str) -> tuple[list[dict], dict]:
    H = params.horizon
    z, _ = tune_z(conv_kind, data, cfg, N_DEV)
    pols = [LPPolicy("pipeline", "pipeline_lp", params), IncumbentPolicy("pipeline", H),
            StockLevelPolicy(conv_kind, z)]
    mask_test = np.repeat(np.arange(len(data.months)) >= N_DEV, data.n)
    out = {}
    for pol in pols:
        res = simulate(pol, data, cfg)
        fr = res.frame(pol.name)[mask_test]
        out[pol.name] = {**summarize(fr), "infeasible_decisions": res.infeasible_decisions,
                         "budget_clips": res.budget_clips, "capacity_violations": res.capacity_violations}
    return [{"method": "backtest_realized", "policy": k, **v} for k, v in out.items()], z


def _savings(rows: list[dict]) -> list[dict]:
    by_pol = {r["policy"]: r for r in rows}
    inc = by_pol.get("incumbent", {}).get("total_cost")
    conv = [r for r in rows if r["policy"].endswith("_tuned")]
    conv_cost = conv[0]["total_cost"] if conv else None
    for r in rows:
        if r["policy"] == "pipeline_lp":
            r["savings_vs_incumbent_pct"] = 100 * (inc - r["total_cost"]) / inc if inc else None
            r["savings_vs_conventional_pct"] = 100 * (conv_cost - r["total_cost"]) / conv_cost if conv_cost else None
    return rows


def scenarios() -> list[dict]:
    S = []
    for f in (0.5, 1.0, 2.0):
        S.append({"parameter": "budget", "value": f"{f}x", "params": {"budget": 500_000 * f}, "cfg": {"budget": 500_000 * f}})
    S.append({"parameter": "lead_time", "value": "nominal", "params": {}, "cfg": {}})
    S.append({"parameter": "lead_time", "value": "+1_month", "params": {"lead_extra": 1}, "cfg": {"lead_extra": 1}})
    S.append({"parameter": "lead_time", "value": f"stochastic_p{P_DELAY}", "params": {}, "cfg": {}, "stochastic": True})
    for f in (1.0, 1.2, 1.5, 2.0):
        S.append({"parameter": "stockout_penalty_factor", "value": f, "params": {"penalty": round(f - 1, 4)}, "cfg": {}})
    for r in (0.0005, 0.001, 0.002):
        S.append({"parameter": "holding_rate_per_day", "value": r, "params": {"holding_rate": r}, "cfg": {"holding_rate": r}})
    for w in ((4.0, 2.5, 1.5), (1.0, 1.0, 1.0), (3.0, 2.0, 1.0)):
        S.append({"parameter": "abc_weights", "value": str(w),
                  "params": {"class_weights": (("A", w[0]), ("B", w[1]), ("C", w[2]))}, "cfg": {}})
    for d in (90, 180, 270, 365):
        S.append({"parameter": "obsolescence_threshold_days", "value": d, "params": {}, "cfg": {}, "rule_days": d})
    for h in (1, 2, 3, 6):
        S.append({"parameter": "horizon_months", "value": h, "params": {"horizon": h}, "cfg": {}})
    for fx in (-0.20, -0.10, 0.10, 0.20):
        S.append({"parameter": "usd_cdf_change", "value": fx, "params": {}, "cfg": {"price_factor": 1 / (1 + fx)}})
    return S


def ensure_h6_forecasts(wait_minutes: int = 120) -> None:
    """Prévisions H = 6 du pipeline ; attend un calcul déjà en cours plutôt que de le dupliquer."""
    import time

    from experiments.config import CACHE_DIR
    from experiments.rolling_origin import run as ro_run

    target = PAPER_DIR / "rolling_origin_h6_errors.csv"
    cache = CACHE_DIR / "rolling_origin"
    deadline = time.time() + 60 * wait_minutes
    while not target.exists() and time.time() < deadline:
        recent = [p for p in cache.glob("h6_*.pkl") if time.time() - p.stat().st_mtime < 30 * 60] if cache.exists() else []
        if not recent:
            break                       # aucun calcul concurrent visible : on calcule nous-mêmes
        time.sleep(30)
    if not target.exists():
        ro_run(horizon=6, methods=("pipeline",), out_prefix="rolling_origin_h6")
    else:
        time.sleep(5)                   # laisse l'écriture du CSV se terminer


def run() -> pd.DataFrame:
    ensure_dirs()
    comp = pd.read_csv(PAPER_DIR / "policy_comparison.csv")
    dev = comp[comp["period"] == "dev"].set_index("policy")["total_cost"]
    conv_name = dev.loc[["sS_tuned", "order_up_to_tuned", "ma3_order_up_to_tuned"]].idxmin()
    conv_kind = {"sS_tuned": "sS", "order_up_to_tuned": "OUT", "ma3_order_up_to_tuned": "MA3"}[conv_name]

    data = prepare_data(["pipeline", "pipeline_no_obso"])
    for h in (1, 2):
        data.forecasts[("pipeline", h)] = data.forecasts[("pipeline", 3)][:, :, :h]

    flags_cache: dict[int, pd.DataFrame] = {}
    rows: list[dict] = []
    for sc in scenarios():
        if sc["params"].get("horizon") == 6 and ("pipeline", 6) not in data.forecasts:
            ensure_h6_forecasts()
            attach_forecasts(data, "rolling_origin_h6_errors.csv", ["pipeline"], 6)
        params = Params().with_(**sc["params"])
        holding = sc["cfg"].get("holding_rate", 0.001)
        pf = sc["cfg"].get("price_factor", 1.0)
        common = {"parameter": sc["parameter"], "value": str(sc["value"])}
        # --- méthode de l'article
        if sc.get("rule_days") or sc.get("stochastic"):
            reason = ("prévisions de déploiement figées (previsions_complet.csv) : non recalculables pour "
                      "un autre seuil" if sc.get("rule_days") else "plan déterministe jugé contre la prévision : "
                      "un retard aléatoire n'y a pas de sens")
            rows.append({**common, "method": "paper_forecast_based", "policy": "n/a", "note": reason})
        else:
            rows += [{**common, **r} for r in paper_method(params, holding, pf)]
        # --- backtest
        d = data
        if sc.get("rule_days"):
            if sc["rule_days"] not in flags_cache:
                flags_cache[sc["rule_days"]] = obsolescence_flags(sc["rule_days"])
            d = with_flags(data, flags_cache[sc["rule_days"]])
        if sc.get("stochastic"):
            reps = []
            for r in range(N_REPLICATIONS):
                cfg = SimConfig(**sc["cfg"], delays=delay_matrix(d.n, len(d.months), P_DELAY, seed=SEED + r))
                res, _ = backtest(d, cfg, params, conv_kind)
                reps.append(_savings(res))
            for i, pol_row in enumerate(reps[0]):
                vals = pd.DataFrame([rep[i] for rep in reps])
                agg = {k: float(vals[k].mean()) for k in vals.columns if k not in ("method", "policy")
                       and pd.api.types.is_numeric_dtype(vals[k])}
                rows.append({**common, "method": "backtest_realized", "policy": pol_row["policy"], **agg,
                             "total_cost_sd": float(vals["total_cost"].std(ddof=1)),
                             "replications": N_REPLICATIONS})
        else:
            res, z = backtest(d, SimConfig(**sc["cfg"]), params, conv_kind)
            rows += [{**common, **r, "z": json.dumps(z) if r["policy"].endswith("_tuned") else None}
                     for r in _savings(res)]
        print(f"{sc['parameter']}={sc['value']} done", flush=True)
        partial = pd.DataFrame(rows).assign(conventional_kind=conv_kind)
        partial.to_csv(PAPER_DIR / "sensitivity_partial.csv", index=False)
    out = pd.DataFrame(rows)
    out["conventional_kind"] = conv_kind
    out.to_csv(PAPER_DIR / "sensitivity.csv", index=False)
    (PAPER_DIR / "sensitivity_partial.csv").unlink(missing_ok=True)
    return out


def main() -> int:
    out = run()
    sub = out[out["policy"] == "pipeline_lp"]
    cols = ["parameter", "value", "method", "total_cost", "unit_fill_rate", "savings_vs_incumbent_pct",
            "savings_vs_conventional_pct", "infeasible_decisions"]
    with pd.option_context("display.width", 220):
        print(sub[[c for c in cols if c in sub.columns]].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
