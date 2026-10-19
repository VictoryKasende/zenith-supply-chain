"""P0-1 — Backtest rolling-origin sans fuite temporelle.

Pour chaque origine mensuelle t0 (≥ 18 mois d'historique), tout ce que le
pipeline apprend est recalculé sur les seules données brutes ≤ t0 :
nettoyage, ABC (CA cumulé ≤ t0), XYZ (CV ≤ t0), K-Means et son scaler,
Isolation Forest et son scaler, règles métier, affectation modèle ↔ produit,
ajustement des modèles. On prédit t0+1 … t0+H.

Univers : produits ayant au moins une vente ≤ t0 (les autres n'existent pas
encore pour Zenith à cette date). Toutes les méthodes sont évaluées sur
exactement les mêmes (origine, produit, horizon).

Sorties
-------
- ``paper_results/rolling_origin_errors.csv`` : une ligne par
  (origine, produit, méthode, horizon).
- ``paper_results/rolling_origin_states.csv`` : classes, drapeaux
  d'obsolescence et stock central reconstruits à chaque origine.
- ``paper_results/rolling_origin_scales.csv`` : dénominateurs MASE / RMSSE
  calculés sur l'historique ≤ t0.
- ``paper_results/rolling_origin_meta.json``.

Usage : ``python -m experiments.rolling_origin [--fresh] [--jobs N]``
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import (
    CACHE_DIR, EVAL_ORIGINS, FORECAST_ORIGINS, HORIZON, PAPER_DIR, SEED, ensure_dirs,
)
from experiments.data import (
    OriginState, build_state, full_clean, history_since_launch, load_cat, load_raw,
    monthly_matrix, state_table,
)
from experiments.forecasters import (
    ALL_METHODS, PIPELINE_METHODS, SIMPLE_BASELINES,
    baseline_forecast, lightgbm_global_forecasts, mlp12_forecast, pipeline_forecast,
)

CACHE_VERSION = "v1"


# --------------------------------------------------------------------- #
# Une origine
# --------------------------------------------------------------------- #
def scale_denominators(history: pd.Series) -> dict:
    """Dénominateurs MASE / RMSSE sur l'historique ≤ t0 (depuis la première vente)."""
    y = history_since_launch(history).to_numpy(dtype=float)
    out = {"n_train": len(y)}
    for m in (1, 12):
        if len(y) > m:
            d = y[m:] - y[:-m]
            out[f"mase_denom_m{m}"] = float(np.mean(np.abs(d)))
            out[f"rmsse_denom_m{m}"] = float(np.mean(d ** 2))
        else:
            out[f"mase_denom_m{m}"] = np.nan
            out[f"rmsse_denom_m{m}"] = np.nan
    return out


def forecast_from_state(state: OriginState, catalogue: pd.DataFrame, horizon: int = HORIZON,
                        methods: tuple[str, ...] = ALL_METHODS,
                        products: list[str] | None = None) -> pd.DataFrame:
    """Prévisions de toutes les méthodes pour les produits connus à t0."""
    products = state.products if products is None else [p for p in products if p in state.products]
    t0 = state.t0
    targets = [t0.to_period("M").to_timestamp() + pd.offsets.MonthBegin(h) for h in range(1, horizon + 1)]
    rows: list[dict] = []

    def _emit(pid, method, preds, assigned, effective):
        for h, (tm, yhat) in enumerate(zip(targets, preds), start=1):
            rows.append({"origin": t0.date(), "product_id": pid, "method": method, "horizon": h,
                         "target_month": tm.date(), "model_assigned": assigned,
                         "model_effective": effective, "y_pred": float(yhat)})

    glob = lightgbm_global_forecasts(state, horizon, products) if "lightgbm_global" in methods else {}
    for pid in products:
        history = state.monthly[pid].astype(float)
        pipe = None
        if any(m in methods for m in PIPELINE_METHODS):
            pipe = pipeline_forecast(state, catalogue, pid, horizon)
        if "pipeline" in methods:
            _emit(pid, "pipeline", *pipe)
        if "pipeline_mlp12" in methods:
            preds, assigned, eff = pipe
            if assigned == "lstm":
                preds, eff = mlp12_forecast(history, horizon)
                assigned = "mlp12"
            _emit(pid, "pipeline_mlp12", preds, assigned, eff)
        if "pipeline_no_obso" in methods:
            if pid in state.obsolete:
                _emit(pid, "pipeline_no_obso", *pipeline_forecast(state, catalogue, pid, horizon,
                                                                  ignore_obsolescence=True))
            else:
                _emit(pid, "pipeline_no_obso", *pipe)
        for method in SIMPLE_BASELINES + ("sarima_uniform", "lightgbm_uniform"):
            if method in methods:
                _emit(pid, method, baseline_forecast(method, history, horizon), method, method)
        if "lightgbm_global" in methods:
            _emit(pid, "lightgbm_global", glob[pid], "lightgbm_global", "lightgbm_global")
    return pd.DataFrame(rows)


def run_origin(t0: pd.Timestamp, horizon: int, methods: tuple[str, ...],
               raw: pd.DataFrame | None = None, catalogue: pd.DataFrame | None = None,
               products: list[str] | None = None) -> dict:
    """Reconstruit l'état à t0 puis prévoit. ``raw`` peut être modifié (test de fuite)."""
    raw = load_raw() if raw is None else raw
    catalogue = load_cat() if catalogue is None else catalogue
    tic = time.perf_counter()
    state = build_state(raw, catalogue, t0)
    fc = forecast_from_state(state, catalogue, horizon, methods, products)
    scales = pd.DataFrame([{"origin": state.t0.date(), "product_id": pid,
                            **scale_denominators(state.monthly[pid])} for pid in state.products])
    return {"forecasts": fc, "states": state_table(state), "scales": scales,
            "seconds": time.perf_counter() - tic}


def _cached_run_origin(t0: pd.Timestamp, horizon: int, methods: tuple[str, ...], cache_dir: Path) -> dict:
    path = cache_dir / f"h{horizon}_{t0:%Y-%m}_{'-'.join(methods)}_{CACHE_VERSION}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return {**pickle.load(fh), "from_cache": True}
    res = run_origin(t0, horizon, methods)
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return {**res, "from_cache": False}


# --------------------------------------------------------------------- #
# Vérité terrain
# --------------------------------------------------------------------- #
def attach_truth(fc: pd.DataFrame) -> pd.DataFrame:
    truth = monthly_matrix(full_clean(), pd.Timestamp("2025-07-01"))
    long = truth.stack().rename("y_true").reset_index()
    long.columns = ["target_month", "product_id", "y_true"]
    long["target_month"] = long["target_month"].dt.date
    out = fc.merge(long, on=["target_month", "product_id"], how="left")
    # Produits connus mais sans vente sur le mois cible : la demande observée vaut 0
    in_window = pd.to_datetime(out["target_month"]) <= pd.Timestamp("2025-07-01")
    out.loc[in_window & out["y_true"].isna(), "y_true"] = 0.0
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = out["error"] ** 2
    return out


# --------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------- #
def run(horizon: int = HORIZON, methods: tuple[str, ...] = ALL_METHODS,
        origins: list[pd.Timestamp] = FORECAST_ORIGINS, out_prefix: str = "rolling_origin",
        fresh: bool = False, jobs: int = 6) -> dict:
    from joblib import Parallel, delayed

    ensure_dirs()
    cache_dir = CACHE_DIR / "rolling_origin"
    if fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tic = time.perf_counter()
    results = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
        delayed(_cached_run_origin)(t0, horizon, tuple(methods), cache_dir) for t0 in origins
    )
    fc = pd.concat([r["forecasts"] for r in results], ignore_index=True)
    states = pd.concat([r["states"] for r in results], ignore_index=True)
    scales = pd.concat([r["scales"] for r in results], ignore_index=True)

    fc = fc.merge(states[["origin", "produit_id", "classe_abc", "classe_xyz", "obsolete_flag"]]
                  .rename(columns={"produit_id": "product_id"}), on=["origin", "product_id"], how="left")
    fc = fc.rename(columns={"classe_abc": "abc", "classe_xyz": "xyz"})
    fc = attach_truth(fc)
    eval_dates = {t.date() for t in EVAL_ORIGINS}
    fc["eval_set"] = fc["origin"].isin(eval_dates) & (fc["horizon"] <= HORIZON) & fc["y_true"].notna()

    cols = ["origin", "product_id", "horizon", "model_assigned", "abc", "xyz", "y_true", "y_pred",
            "abs_error", "sq_error", "method", "model_effective", "target_month", "obsolete_flag",
            "error", "eval_set"]
    fc = fc[cols].sort_values(["method", "origin", "product_id", "horizon"]).reset_index(drop=True)

    fc.to_csv(PAPER_DIR / f"{out_prefix}_errors.csv", index=False)
    states.to_csv(PAPER_DIR / f"{out_prefix}_states.csv", index=False)
    scales.to_csv(PAPER_DIR / f"{out_prefix}_scales.csv", index=False)

    meta = {
        "horizon": horizon,
        "methods": list(methods),
        "forecast_origins": [str(t.date()) for t in origins],
        "eval_origins": [str(t.date()) for t in EVAL_ORIGINS if t in origins],
        "n_rows": int(len(fc)),
        "n_products_by_origin": states.groupby("origin")["produit_id"].nunique().astype(int)
                                      .rename(index=str).to_dict(),
        "seconds_per_origin": {str(t.date()): round(r["seconds"], 1) for t, r in zip(origins, results)},
        "seconds_per_origin_note": "temps de calcul mesuré lors du calcul de l'origine (processus parallèles "
                                   "en concurrence) ; non représentatif d'une exécution isolée",
        "n_origins_from_cache": int(sum(r["from_cache"] for r in results)),
        "wall_seconds_total": round(time.perf_counter() - tic, 1),
        "seed": SEED,
        "python": platform.python_version(),
        "universe": "produits avec ≥ 1 vente ≤ t0",
        "y_true": "ventes mensuelles nettoyées (somme des 7 magasins) ; 0 si aucune vente le mois cible",
    }
    (PAPER_DIR / f"{out_prefix}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fresh", action="store_true", help="ignore le cache par origine")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    meta = run(fresh=args.fresh, jobs=args.jobs)
    print(json.dumps({k: meta[k] for k in ("n_rows", "wall_seconds_total")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
