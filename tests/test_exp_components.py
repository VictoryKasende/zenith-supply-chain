"""Tests des briques des expériences de révision (réplique LP, simulation, statistiques)."""
import experiments  # noqa: F401

import numpy as np
import pandas as pd
import pytest

from experiments.lp_model import Params, evaluate_plan, make_instance, round_like_code, solve_lp, solve_milp
from experiments.stats_tests import diebold_mariano_series, holm, wilcoxon_paired
from src.optimization import optimize_orders


def _toy_products(n=6):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "produit_id": [f"P{i}" for i in range(n)],
        "classe_abc": list("ABCABC")[:n],
        "a_risque_obsolescence": 0,
        "cout_achat_unitaire": rng.uniform(10, 100, n).round(1),
        "prix_vente_unitaire": 0,
        "stock_courant": rng.integers(0, 20, n).astype(float),
        "origine_fournisseur": ["Dubaï", "Chine"] * (n // 2),
    }).assign(prix_vente_unitaire=lambda d: (d["cout_achat_unitaire"] * 1.4).round(1))


def _toy_forecasts(products, H=3):
    rng = np.random.default_rng(1)
    rows = [{"produit_id": p, "date": d, "prevision": float(rng.uniform(0, 30))}
            for p in products["produit_id"] for d in pd.date_range("2025-08-01", periods=H, freq="MS")]
    return pd.DataFrame(rows)


def test_lp_replica_equals_optimize_orders():
    prods, fc = _toy_products(), None
    fc = _toy_forecasts(prods)
    plan, kpis = optimize_orders(prods, fc)
    params = Params()
    fdict = {p: g.sort_values("date")["prevision"].to_numpy() for p, g in fc.groupby("produit_id")}
    inst = make_instance(prods, fdict, dict(zip(prods["produit_id"], prods["stock_courant"])), params)
    lp = solve_lp(inst, params)
    q = round_like_code(lp["Q"])
    code_q = plan.pivot(index="produit_id", columns="mois_offset", values="quantite_commandee")
    for p in inst.products:
        assert np.array_equal(code_q.loc[p].to_numpy(), q[p])
    assert lp["status"] == kpis["statut"] == "Optimal"


def test_in_transit_reduces_orders():
    prods = _toy_products()
    fc = _toy_forecasts(prods)
    fdict = {p: g.sort_values("date")["prevision"].to_numpy() for p, g in fc.groupby("produit_id")}
    params = Params()
    stock0 = {p: 0.0 for p in prods["produit_id"]}
    base = solve_lp(make_instance(prods, fdict, stock0, params), params)
    rho = {p: np.array([0.0, 1000.0, 1000.0]) for p in prods["produit_id"]}
    with_rho = solve_lp(make_instance(prods, fdict, stock0, params, in_transit=rho), params)
    assert sum(v.sum() for v in with_rho["Q"].values()) <= sum(v.sum() for v in base["Q"].values())


def test_milp_plan_is_feasible_and_not_worse_than_rounded_lp():
    prods = _toy_products()
    fc = _toy_forecasts(prods)
    fdict = {p: g.sort_values("date")["prevision"].to_numpy() for p, g in fc.groupby("produit_id")}
    params = Params(budget=2_000.0)
    inst = make_instance(prods, fdict, dict(zip(prods["produit_id"], prods["stock_courant"])), params)
    milp = solve_milp(inst, params, time_limit=60)
    ev_milp = evaluate_plan(inst, params, milp["q"])
    ev_lp = evaluate_plan(inst, params, round_like_code(solve_lp(inst, params)["Q"]))
    assert milp["status"] == "Optimal"
    assert ev_milp["budget_violation_months"] == 0 and ev_milp["capacity_violation_months"] == 0
    assert ev_milp["objective_code"] == pytest.approx(milp["objective"], rel=1e-6)
    if ev_lp["budget_violation_months"] == 0:
        assert ev_milp["objective_code"] <= ev_lp["objective_code"] + 1e-6


def test_holm_matches_definition():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_wilcoxon_rank_biserial_sign():
    x = np.array([1.0, 2.0, 3.0, 4.0]); y = x + 1
    w = wilcoxon_paired(x, y, "less")
    assert w["effect_size"] == pytest.approx(1.0)


def test_dm_detects_constant_shift():
    rng = np.random.default_rng(0)
    d = 1.0 + rng.normal(0, 0.1, 30)
    res = diebold_mariano_series(d, h=3)
    assert res["p_raw"] < 1e-6 and res["ci_low"] > 0


def _toy_simdata():
    from experiments.simulation import SimData
    n, K = 3, 4
    demand = np.array([[5, 0, 7, 3], [2, 2, 2, 2], [0, 10, 0, 10]], dtype=float)
    cums = {}
    for i in range(n):
        for k in range(K):
            if demand[i, k] > 0:
                q = demand[i, k]
                cums[(i, k)] = np.cumsum([q / 2, q / 2])
    months = list(pd.date_range("2024-02-01", periods=K, freq="MS"))
    return SimData(products=["A", "B", "C"], cost=np.array([10.0, 20.0, 30.0]),
                   price=np.array([14.0, 28.0, 42.0]), lead=np.array([1, 2, 1]), demand=demand,
                   line_cumsum=cums, stock0=np.array([3.0, 10.0, 0.0]), months=months,
                   origins=[m - pd.offsets.MonthEnd(1) for m in months], states=pd.DataFrame(),
                   monthly_full=pd.DataFrame())


class _ConstantPolicy:
    name = "const"

    def reset(self, data, cfg):
        pass

    def decide(self, k, on_hand, future_arrivals):
        return np.array([4.0, 0.0, 6.0]), True


def test_simulation_conserves_stock():
    from experiments.simulation import SimConfig, simulate
    data = _toy_simdata()
    res = simulate(_ConstantPolicy(), data, SimConfig())
    np.testing.assert_allclose(res.served + res.lost, data.demand)
    # stock final = stock initial + arrivées reçues − servi
    received = res.available - np.concatenate([data.stock0[:, None], res.on_hand_end[:, :-1]], axis=1)
    lead = data.lead
    for i in range(3):
        for k in range(4):
            expected = 4.0 * (i == 0) + 6.0 * (i == 2) if k - lead[i] >= 0 and i != 1 else 0.0
            assert received[i, k] == pytest.approx(expected)
    assert (res.lines_full <= res.lines_total).all()


def test_paper_method_replicas_match_src():
    from experiments.sensitivity import baseline_plan, paper_kpis
    from src.optimization import financial_kpis, simulate_baseline_policy
    prods = _toy_products()
    fc = _toy_forecasts(prods)
    ref = simulate_baseline_policy(prods, fc)
    rep = baseline_plan(prods, fc, horizon=3)
    cols = ["produit_id", "mois_offset", "quantite_commandee", "stock_final", "rupture", "demande_prevue"]
    pd.testing.assert_frame_equal(ref[cols].reset_index(drop=True), rep[cols].reset_index(drop=True))
    assert paper_kpis(ref, 0.001)["total_cost"] == pytest.approx(financial_kpis(ref)["cout_total_simule_usd"])
