"""P0-3 — Tests statistiques avec tailles d'effet, IC et correction de multiplicité.

H1 (différenciation par classe) — Wilcoxon signé-rangé apparié sur le MAE par
produit (rolling-origin, panel équilibré), pipeline vs **meilleure baseline
simple** (seasonal naive, MA3, Croston, Croston-SBA ; choisie par MAE moyen
par produit) et, pour continuité avec l'article, vs seasonal naive. Holm sur
ces deux comparaisons. Taille d'effet : corrélation rang-bisériale appariée
r = (R⁻ − R⁺)/(R⁺ + R⁻) (> 0 : le pipeline a l'erreur la plus faible).
IC bootstrap 95 % (unité : produit, B = 10 000) sur la médiane et la moyenne
des écarts.

H2 (complémentarité) — Diebold-Mariano avec correction Harvey-Leybourne-
Newbold, perte quadratique. L'unité temporelle est l'**origine** : à chaque
origine o, d_o = moyenne sur (produits, h = 1..3) de e₁² − e₂² ; T = 16
origines, variance HAC jusqu'au retard h − 1 = 2, loi t(T − 1). (L'article
concaténait les erreurs de 177 produits en une seule « série », ce qui
n'est pas une série temporelle.) Holm sur les 3 paires.

H4 / H5 (politique) — sur le backtest à demande réalisée, période de test :
Wilcoxon apparié par produit (coût total de la période) et, pour continuité,
par (produit, mois) ; bootstrap par blocs de mois (1 mois ≈ 4 semaines,
comme l'article, puis 3 mois) et par grappes de produits. H5 = gain net
annualisé = Δ(marge brute servie − coûts de commande − coûts de stockage)
× 12 / nombre de mois de test ; il est identique à l'économie de coût total
car coût de rupture = marge perdue (vérifié numériquement).

Sortie : ``paper_results/stat_tests.json``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from experiments.config import PAPER_DIR, SEED, ensure_dirs
from experiments.forecasters import SIMPLE_BASELINES

B_BOOT = 10_000


# --------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------- #
def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def wilcoxon_paired(x: np.ndarray, y: np.ndarray, alternative: str = "less") -> dict:
    """Wilcoxon signé-rangé (zéros exclus, méthode « wilcox ») + rang-bisériale."""
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    r_plus, r_minus = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
    res = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
    return {"statistic": float(res.statistic), "p_raw": float(res.pvalue), "R_plus": r_plus,
            "R_minus": r_minus, "n": int(len(d)), "n_nonzero": int(len(nz)),
            "effect_size": (r_minus - r_plus) / (r_plus + r_minus) if len(nz) else np.nan,
            "effect_size_name": "rang-bisériale appariée (R⁻ − R⁺)/(R⁺ + R⁻)"}


def bootstrap_ci(values: np.ndarray, func, rng: np.random.Generator, B: int = B_BOOT) -> tuple[float, float]:
    """IC percentile ; ``func`` doit accepter ``axis`` (np.mean, np.median)."""
    v = np.asarray(values, float)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    boot = func(v[idx], axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def diebold_mariano_series(d: np.ndarray, h: int) -> dict:
    """DM-HLN sur une série de différentiels de perte d_t (H0 : E[d] = 0)."""
    d = np.asarray(d, float)
    T = len(d)
    dbar = d.mean()
    gam = [np.mean((d[k:] - dbar) * (d[:T - k] - dbar)) for k in range(h)]
    V = gam[0] + 2 * sum(gam[1:])
    if T < 4 or V <= 0:
        return {"statistic": np.nan, "p_raw": np.nan, "mean_loss_diff": float(dbar), "n": T}
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    se = np.sqrt(V / T) / corr
    dm = dbar / se
    p = 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))
    tcrit = stats.t.ppf(0.975, df=T - 1)
    return {"statistic": float(dm), "p_raw": float(p), "mean_loss_diff": float(dbar),
            "ci_low": float(dbar - tcrit * se), "ci_high": float(dbar + tcrit * se), "n": T,
            "hac_lags": h - 1, "hln_correction": float(corr)}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# --------------------------------------------------------------------- #
# H1
# --------------------------------------------------------------------- #
def test_h1(panel: pd.DataFrame, rng: np.random.Generator) -> dict:
    per_prod = panel.groupby(["method", "product_id"])["abs_error"].mean().unstack("method")
    simple_means = per_prod[list(SIMPLE_BASELINES)].mean().sort_values()
    best = str(simple_means.index[0])
    out, pvals, keys = {}, [], []
    for name, other in (("H1_pipeline_vs_best_simple", best), ("H1_pipeline_vs_seasonal_naive", "seasonal_naive")):
        sub = per_prod[["pipeline", other]].dropna()
        w = wilcoxon_paired(sub["pipeline"], sub[other], "less")
        d = (sub["pipeline"] - sub[other]).to_numpy()
        med_ci = bootstrap_ci(d, np.median, rng)
        mean_ci = bootstrap_ci(d, np.mean, rng)
        out[name] = {
            **w, "comparison": f"pipeline vs {other}", "alternative": "MAE pipeline < MAE baseline",
            "mae_pipeline_mean_per_product": float(sub["pipeline"].mean()),
            "mae_baseline_mean_per_product": float(sub[other].mean()),
            "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
            "ci_low": med_ci[0], "ci_high": med_ci[1], "ci_target": "médiane des écarts (pipeline − baseline)",
            "mean_diff_ci_low": mean_ci[0], "mean_diff_ci_high": mean_ci[1],
            "resampling_unit": "produit", "n_bootstrap": B_BOOT,
        }
        pvals.append(w["p_raw"]); keys.append(name)
    for k, p in zip(keys, holm(pvals)):
        out[k]["p_adjusted"] = p
        out[k]["p_adjustment"] = "Holm sur la famille H1 (2 comparaisons)"
    out["H1_best_simple_baseline"] = best
    out["H1_simple_baseline_mean_mae"] = simple_means.to_dict()
    # Variante exploratoire (hors famille) : MLP classe A à fenêtre 12
    sub = per_prod[["pipeline_mlp12", best]].dropna()
    w = wilcoxon_paired(sub["pipeline_mlp12"], sub[best], "less")
    d = (sub["pipeline_mlp12"] - sub[best]).to_numpy()
    lo, hi = bootstrap_ci(d, np.median, rng)
    out["H1_exploratory_mlp12_vs_best_simple"] = {
        **w, "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
        "ci_low": lo, "ci_high": hi,
        "p_adjusted": None, "resampling_unit": "produit", "n_bootstrap": B_BOOT,
        "note": "exploratoire, non inclus dans la correction de Holm",
    }
    return out


# --------------------------------------------------------------------- #
# H2
# --------------------------------------------------------------------- #
H2_PAIRS = (("sarima_uniform", "seasonal_naive"), ("lightgbm_uniform", "seasonal_naive"),
            ("lightgbm_uniform", "sarima_uniform"))


def test_h2(panel: pd.DataFrame) -> dict:
    out, pvals, keys = {}, [], []
    sq = panel.pivot_table(index=["origin", "product_id", "horizon"], columns="method", values="sq_error")
    scaled = panel.assign(ssq=panel["sq_error"] / panel["rmsse_denom_m1"].where(panel["rmsse_denom_m1"] > 0))
    ssq = scaled.pivot_table(index=["origin", "product_id", "horizon"], columns="method", values="ssq")
    for a, b in H2_PAIRS:
        key = f"H2_DM_{a}_vs_{b}"
        diff = (sq[a] - sq[b]).groupby(level="origin").mean()
        res = diebold_mariano_series(diff.to_numpy(), h=3)
        mse_a, mse_b = float(sq[a].mean()), float(sq[b].mean())
        res.update({
            "comparison": f"{a} vs {b}", "loss": "erreur quadratique, moyenne transversale par origine",
            "mse_a": mse_a, "mse_b": mse_b,
            "effect_size": (mse_a - mse_b) / mse_b, "effect_size_name": "(MSE_a − MSE_b) / MSE_b",
            "ci_target": "différentiel de perte moyen E[e_a² − e_b²] (IC t, variance HAC/HLN)",
            "resampling_unit": "aucun (asymptotique HAC sur T origines)", "n_bootstrap": 0,
        })
        by_h = {}
        for h in (1, 2, 3):
            dh = (sq[a] - sq[b]).xs(h, level="horizon").groupby(level="origin").mean()
            by_h[f"h{h}"] = diebold_mariano_series(dh.to_numpy(), h=h)
        res["secondary_by_horizon"] = by_h
        dsc = (ssq[a] - ssq[b]).dropna().groupby(level="origin").mean()
        res["secondary_scaled_loss"] = diebold_mariano_series(dsc.to_numpy(), h=3)
        out[key] = res
        pvals.append(res["p_raw"]); keys.append(key)
    valid = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    adj = holm([pvals[i] for i in valid])
    for j, i in enumerate(valid):
        out[keys[i]]["p_adjusted"] = adj[j]
        out[keys[i]]["p_adjustment"] = "Holm-Bonferroni sur les 3 paires"
    return out


# --------------------------------------------------------------------- #
# H4 / H5
# --------------------------------------------------------------------- #
def _block_bootstrap_sum(series: np.ndarray, block: int, rng: np.random.Generator, B: int = B_BOOT) -> np.ndarray:
    """Bootstrap par blocs mobiles (Künsch) de la somme d'une série mensuelle."""
    n = len(series)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return series[idx].sum(axis=1)


def test_policies(detail: pd.DataFrame, comparison: pd.DataFrame, rng: np.random.Generator) -> dict:
    test = detail[detail["period"] == "test"]
    n_months = int(test["month"].nunique())
    annual = 12 / n_months
    dev = comparison[comparison["period"] == "dev"].set_index("policy")["total_cost"]
    conventional = ["sS_tuned", "order_up_to_tuned", "ma3_order_up_to_tuned"]
    best_conv = str(dev.loc[conventional].idxmin())
    out = {"H4_H5_context": {"test_months": sorted(test["month"].astype(str).unique().tolist()),
                             "annualisation_factor": annual,
                             "best_conventional_policy_selected_on_dev": best_conv}}
    for name, base in (("vs_incumbent", "incumbent"), ("vs_best_conventional", best_conv)):
        a = test[test["policy"] == "pipeline_lp"].set_index(["product_id", "month"])
        b = test[test["policy"] == base].set_index(["product_id", "month"])
        prod_a = a.groupby(level="product_id")["total_cost"].sum()
        prod_b = b.groupby(level="product_id")["total_cost"].sum()
        w = wilcoxon_paired(prod_a.to_numpy(), prod_b.loc[prod_a.index].to_numpy(), "less")
        wpm = wilcoxon_paired(a["total_cost"].to_numpy(), b.loc[a.index, "total_cost"].to_numpy(), "less")
        savings_prod = (prod_b - prod_a).to_numpy()
        monthly = (b.groupby(level="month")["total_cost"].sum() - a.groupby(level="month")["total_cost"].sum()).to_numpy()
        total = float(savings_prod.sum())
        boot_prod = savings_prod[rng.integers(0, len(savings_prod), (B_BOOT, len(savings_prod)))].sum(axis=1)
        boot_b1 = _block_bootstrap_sum(monthly, 1, rng)
        boot_b3 = _block_bootstrap_sum(monthly, 3, rng)
        cost_b = float(prod_b.sum())
        out[f"H4_{name}"] = {
            **w, "comparison": f"pipeline_lp vs {base} (coût total par produit sur la période de test)",
            "alternative": "coût pipeline < coût référence",
            "cost_pipeline": float(prod_a.sum()), "cost_reference": cost_b,
            "savings_total": total, "savings_pct": 100 * total / cost_b if cost_b else np.nan,
            "savings_annualised": total * annual,
            "ci_low": float(np.percentile(boot_prod, 2.5)) * annual,
            "ci_high": float(np.percentile(boot_prod, 97.5)) * annual,
            "ci_target": "économie annualisée (bootstrap par grappes de produits)",
            "resampling_unit": "produit (grappe : tous ses mois)", "n_bootstrap": B_BOOT,
            "n_resampling_units": int(len(savings_prod)),
            "block_bootstrap_1_month": {"block_length_months": 1, "n_blocks": n_months,
                                        "ci_low": float(np.percentile(boot_b1, 2.5)) * annual,
                                        "ci_high": float(np.percentile(boot_b1, 97.5)) * annual},
            "block_bootstrap_3_months": {"block_length_months": 3, "n_blocks": int(np.ceil(n_months / 3)),
                                         "ci_low": float(np.percentile(boot_b3, 2.5)) * annual,
                                         "ci_high": float(np.percentile(boot_b3, 97.5)) * annual},
            "secondary_wilcoxon_product_month": {k: wpm[k] for k in ("statistic", "p_raw", "n", "effect_size")},
        }
        # H5 : gain net = Δ(marge servie − commande − stockage)
        def _profit(df):
            return (df["gross_margin"] - df["ordering_cost"] - df["holding_cost"])
        pa = _profit(a).groupby(level="product_id").sum()
        pb = _profit(b).groupby(level="product_id").sum()
        gain_prod = (pa - pb.loc[pa.index]).to_numpy()
        boot_g = gain_prod[rng.integers(0, len(gain_prod), (B_BOOT, len(gain_prod)))].sum(axis=1) * annual
        # profit référence − profit pipeline < 0 ⇔ le pipeline gagne plus
        wg = wilcoxon_paired(pb.loc[pa.index].to_numpy(), pa.to_numpy(), "less")
        out[f"H5_{name}"] = {
            "comparison": f"pipeline_lp vs {base} (gain net = marge servie − commande − stockage, par produit)",
            "statistic": wg["statistic"], "p_raw": wg["p_raw"], "n": wg["n"], "effect_size": wg["effect_size"],
            "effect_size_name": "rang-bisériale (R⁻ − R⁺)/(R⁺ + R⁻) sur (profit référence − profit pipeline) ; > 0 = pipeline meilleur",
            "net_gain_annualised": float(gain_prod.sum()) * annual,
            "ci_low": float(np.percentile(boot_g, 2.5)), "ci_high": float(np.percentile(boot_g, 97.5)),
            "p_bootstrap_one_sided": float((boot_g <= 0).mean()),
            "identity_check_gain_minus_savings": float(gain_prod.sum() - total),
            "resampling_unit": "produit (grappe)", "n_bootstrap": B_BOOT,
        }
    keys = [k for k in out if k.startswith(("H4_", "H5_")) and "context" not in k]
    adj = holm([out[k]["p_raw"] for k in keys])
    for k, p in zip(keys, adj):
        out[k]["p_adjusted"] = p
        out[k]["p_adjustment"] = f"Holm sur {len(keys)} tests H4/H5"
    return out


# --------------------------------------------------------------------- #
def run() -> dict:
    ensure_dirs()
    from experiments.metrics import load_panel

    rng = np.random.default_rng(SEED)
    panel = load_panel()
    panel = panel[panel["horizon"] <= 3]
    result = {"H1": test_h1(panel, rng), "H2": test_h2(panel)}
    # Robustesse (hors famille Holm principale) : même test sur les produits non flagués à t0
    result["H1_active_subset"] = test_h1(panel[panel["obsolete_flag"] == 0], rng)
    detail_path = PAPER_DIR / "policy_sim_detail.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        comparison = pd.read_csv(PAPER_DIR / "policy_comparison.csv")
        result["H4_H5"] = test_policies(detail, comparison, rng)
    else:
        result["H4_H5"] = {"status": "non calculé : lancer d'abord experiments.policy_baselines"}
    result = _clean(result)
    (PAPER_DIR / "stat_tests.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    res = run()
    for fam, tests in res.items():
        for k, v in tests.items():
            if isinstance(v, dict) and "p_raw" in v:
                print(f"{k:<45s} stat={v.get('statistic')!s:<22.10} p_raw={v.get('p_raw')!s:<12.6} "
                      f"p_adj={v.get('p_adjusted')!s:<12.6} effect={v.get('effect_size')!s:<10.6} "
                      f"CI=[{v.get('ci_low')!s:.10}, {v.get('ci_high')!s:.10}] n={v.get('n')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
