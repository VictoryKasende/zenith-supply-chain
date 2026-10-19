"""P0-2 — Métriques de prévision indépendantes de l'échelle.

Calculées sur le panel équilibré du rolling-origin (16 origines × produits
connus × h = 1..3, colonne ``eval_set``). Convention : ``e = y_pred − y_true``
(biais positif = sur-prévision).

- MAE  = mean |e|
- MASE_m = mean sur les paires (produit, origine) de [ mean_h |e| / D_m ],
  D_m = (1/(n−m)) Σ_{t=m+1..n} |Y_t − Y_{t−m}| sur l'historique ≤ t0 du
  produit (depuis sa première vente). m = 1 et m = 12.
- RMSSE = mean sur les paires de sqrt( mean_h e² / D²_m ), m = 1 (M5) ; m = 12
  en variante.
- WAPE = Σ|e| / Σ|y| ; biais = mean(e) ; biais % = Σe / Σy × 100.

Cas dégénérés : si D_m vaut 0 (série plate) ou n'est pas défini (n ≤ m), la
paire est exclue de la métrique concernée et comptée dans ``n_excluded_*``.
Aucune valeur de remplacement.

MASE retenu : **m = 1**, choisi a priori (avant de voir les résultats) car
(i) c'est le choix standard de Hyndman & Koehler (2006) et du M5 pour des
séries courtes et intermittentes, (ii) m = 12 exige ≥ 13 mois depuis le
lancement et exclut davantage de séries. Les deux sont publiés.

Sorties : ``forecast_metrics.csv`` (global), ``forecast_metrics_by_class.csv``
(classe ABC à l'origine), ``forecast_metrics_by_horizon.csv``. Deux
sous-populations : ``all`` (tous les produits connus à t0) et ``active``
(produits non flagués obsolètes à t0, ce qui reproduit le périmètre de
l'article sans conditionner sur le futur).

Usage : ``python -m experiments.metrics``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.config import PAPER_DIR

RETAINED_MASE = "MASE_m1"


def load_panel(prefix: str = "rolling_origin") -> pd.DataFrame:
    err = pd.read_csv(PAPER_DIR / f"{prefix}_errors.csv")
    err = err[err["eval_set"]].copy()
    scales = pd.read_csv(PAPER_DIR / f"{prefix}_scales.csv")
    return err.merge(scales, on=["origin", "product_id"], how="left")


def _scaled(group: pd.DataFrame, denom_col: str, squared: bool) -> tuple[float, int, int]:
    """Moyenne des erreurs mises à l'échelle par paire (produit, origine)."""
    per_pair = group.groupby(["product_id", "origin"]).agg(
        num=("sq_error" if squared else "abs_error", "mean"),
        den=(denom_col, "first"),
    )
    ok = per_pair["den"].notna() & (per_pair["den"] > 0)
    vals = per_pair.loc[ok, "num"] / per_pair.loc[ok, "den"]
    if squared:
        vals = np.sqrt(vals)
    return (float(vals.mean()) if ok.any() else np.nan), int((~ok).sum()), int(len(per_pair))


def compute_metrics(df: pd.DataFrame) -> dict:
    mase12, ex12, n_pairs = _scaled(df, "mase_denom_m12", squared=False)
    mase1, ex1, _ = _scaled(df, "mase_denom_m1", squared=False)
    rmsse1, exr1, _ = _scaled(df, "rmsse_denom_m1", squared=True)
    rmsse12, exr12, _ = _scaled(df, "rmsse_denom_m12", squared=True)
    sum_y = float(df["y_true"].abs().sum())
    return {
        "n_series": int(df["product_id"].nunique()),
        "n_pairs": n_pairs,
        "n_obs": int(len(df)),
        "n_excluded": ex1,               # exclusions de la métrique retenue (MASE_m1)
        "MAE": float(df["abs_error"].mean()),
        "MASE_m12": mase12,
        "MASE_m1": mase1,
        "RMSSE": rmsse1,
        "WAPE": float(df["abs_error"].sum() / sum_y) if sum_y > 0 else np.nan,
        "bias": float(df["error"].mean()),
        "bias_pct": float(100 * df["error"].sum() / df["y_true"].sum()) if df["y_true"].sum() > 0 else np.nan,
        "RMSE": float(np.sqrt(df["sq_error"].mean())),
        "RMSSE_m12": rmsse12,
        "n_excluded_MASE_m12": ex12,
        "n_excluded_MASE_m1": ex1,
        "n_excluded_RMSSE_m1": exr1,
        "n_excluded_RMSSE_m12": exr12,
    }


def metrics_table(panel: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for subset, sub in (("all", panel), ("active", panel[panel["obsolete_flag"] == 0])):
        for keys, g in sub.groupby(["method"] + by):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append({"subset": subset, **dict(zip(["model"] + by, keys)), **compute_metrics(g)})
    out = pd.DataFrame(rows)
    return out.sort_values(["subset"] + by + [RETAINED_MASE]).reset_index(drop=True)


def run(prefix: str = "rolling_origin") -> pd.DataFrame:
    panel = load_panel(prefix)
    glob = metrics_table(panel, [])
    glob.to_csv(PAPER_DIR / "forecast_metrics.csv", index=False)
    metrics_table(panel, ["abc"]).to_csv(PAPER_DIR / "forecast_metrics_by_class.csv", index=False)
    metrics_table(panel, ["horizon"]).to_csv(PAPER_DIR / "forecast_metrics_by_horizon.csv", index=False)
    return glob


def main() -> int:
    glob = run()
    cols = ["subset", "model", "n_series", "n_excluded", "MAE", "MASE_m12", "MASE_m1", "RMSSE", "WAPE",
            "bias", "bias_pct"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(glob[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
