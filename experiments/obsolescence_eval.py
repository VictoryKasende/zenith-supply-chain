"""P0-4 — Évaluation non circulaire de la détection d'obsolescence.

Vérité terrain prospective : à une origine t0, un produit est **obsolète** si
ses ventes sur la fenêtre future ]t0, t0+6 mois] sont nulles (définition
principale) ou < 3 unités (variante), alors qu'il restait du stock à t0.
L'étiquette vient exclusivement du futur ; tous les détecteurs sont calculés
exclusivement sur les données ≤ t0 (nettoyage, features, scaler, Isolation
Forest refaits à chaque origine). La circularité disparaît par construction.

Population principale : produits connus à t0 dont le stock central
(Σ magasins du dernier ``stock_apres_vente`` ≤ t0) est > 0 — c'est là que la
décision « ne plus réapprovisionner / déstocker » a un enjeu. Variante : tous
les produits connus.

Détecteurs : Isolation Forest seul, règles métier seules, hybride (Étape 4 :
union des deux) et baseline « inactivité simple » (jours depuis la dernière
vente ≥ seuil). Balayages : seuil d'inactivité 90/180/270/365 j (les règles
utilisent seuil/30 mois consécutifs) et contamination 0,05/0,10/0,15/0,20.

Délai d'alerte : pour chaque début de dormance observé (premier mois m d'une
série d'au moins 6 mois sans vente, précédé d'un mois avec vente,
m ∈ [févr. 2024, févr. 2025]), délai = (première alerte à une origine
t0 ≥ m − 6 mois) − (1er jour de m), en jours. Négatif = alerte avant la
dormance.

IC 95 % : bootstrap par grappe de produits (B = 2000, seed 42) — un produit
apparaît à plusieurs origines, ses observations ne sont pas indépendantes.

Sorties : ``paper_results/obsolescence_eval.csv``,
``paper_results/obsolescence_confusion.json``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.config import PAPER_DIR, SEED, ensure_dirs, month_end
from experiments.data import (
    central_stock, clean_upto, detect_at, full_clean, load_cat, load_raw, monthly_matrix,
    obsolescence_features_at,
)

LABEL_HORIZON = 6
ALERT_ORIGINS = [month_end(m) for m in pd.date_range("2024-01-01", "2025-07-01", freq="MS")]
LABEL_ORIGINS = [t for t in ALERT_ORIGINS if t + pd.offsets.MonthBegin(LABEL_HORIZON) <= pd.Timestamp("2025-07-01")]
CONTAMINATIONS = (0.05, 0.10, 0.15, 0.20)
DAYS = (90, 180, 270, 365)
B_BOOT = 2000
CURRENT = {"contamination": 0.10, "rule_days": 180}


# --------------------------------------------------------------------- #
# Construction des observations (origine × produit)
# --------------------------------------------------------------------- #
def detector_flags(obs_feats: pd.DataFrame) -> dict[tuple, pd.Series]:
    """Tous les détecteurs × paramètres pour une origine (index produit)."""
    flags: dict[tuple, pd.Series] = {}
    iforest = {}
    for c in CONTAMINATIONS:
        iforest[c], _ = detect_at(obs_feats, contamination=c)
        flags[("iforest", c, np.nan)] = iforest[c]
    for d in DAYS:
        _, rules = detect_at(obs_feats, contamination=0.10, rule_days=d)
        flags[("rules", np.nan, d)] = rules
        for c in CONTAMINATIONS:
            flags[("hybrid", c, d)] = iforest[c] | rules
        inact = pd.Series((obs_feats["jours_depuis_derniere_vente"] >= d).to_numpy(),
                          index=obs_feats["produit_id"])
        flags[("inactivity", np.nan, d)] = inact
    return flags


def build_observations() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, cat = load_raw(), load_cat()
    truth = monthly_matrix(full_clean(), pd.Timestamp("2025-07-01"))
    label_dates = {t for t in LABEL_ORIGINS}
    rows = []
    for t0 in ALERT_ORIGINS:
        clean = clean_upto(raw, cat, t0)
        feats = obsolescence_features_at(clean, t0)
        stock = central_stock(clean)
        flags = detector_flags(feats)
        start = t0 + pd.offsets.MonthBegin(1)
        future = truth.loc[start:start + pd.offsets.MonthBegin(LABEL_HORIZON - 1)]
        for pid in feats["produit_id"]:
            base = {"origin": t0.date(), "produit_id": pid, "stock_t0": float(stock.get(pid, 0.0)),
                    "labelled": t0 in label_dates}
            if t0 in label_dates:
                fut = float(future[pid].sum()) if pid in future.columns else 0.0
                base["future_sales_6m"] = fut
            for (det, c, d), s in flags.items():
                base[f"{det}|{c}|{d}"] = bool(s.get(pid, False))
            rows.append(base)
        # Décomposition des règles métier : composante « ratio 3m/12m < 0,1 et âge > 365 j »
        # vs composantes « jours sans vente ≥ d ou mois consécutifs ≥ d/30 »
        f = feats.set_index("produit_id")
        ratio = (f["ratio_ventes_3m_vs_12m"] < 0.1) & (f["age_produit_jours"] > 365)
        for r in rows[-len(feats):]:
            r["rule_ratio_component"] = bool(ratio[r["produit_id"]])
            for d in DAYS:
                r[f"rule_inactivity_component_{d}"] = bool(
                    (f.loc[r["produit_id"], "jours_depuis_derniere_vente"] >= d)
                    | (f.loc[r["produit_id"], "nombre_mois_consecutifs_sans_vente"] >= round(d / 30)))
    obs = pd.DataFrame(rows)
    return obs, truth


# --------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------- #
def confusion(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    # Indéfini (NaN) quand le dénominateur est nul : jamais remplacé par 0.
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec, "F1": f1}


def cluster_bootstrap(df: pd.DataFrame, ycol: str, pcol: str, rng: np.random.Generator) -> dict:
    groups = {pid: idx for pid, idx in df.groupby("produit_id").indices.items()}
    keys = list(groups)
    y_all = df[ycol].to_numpy(bool); p_all = df[pcol].to_numpy(bool)
    stats = {"precision": [], "recall": [], "F1": []}
    for _ in range(B_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        m = confusion(y_all[idx], p_all[idx])
        for k in stats:
            stats[k].append(m[k])
    out = {}
    for k, v in stats.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        out[f"{k}_ci_low"] = float(np.percentile(v, 2.5)) if len(v) else np.nan
        out[f"{k}_ci_high"] = float(np.percentile(v, 97.5)) if len(v) else np.nan
    return out


def dormancy_onsets(truth: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Premier mois m ∈ [févr. 2024, févr. 2025] débutant ≥ 6 mois sans vente après un mois vendu."""
    onsets = {}
    months = truth.index
    candidates = [m for m in months if pd.Timestamp("2024-02-01") <= m <= pd.Timestamp("2025-02-01")]
    for pid in truth.columns:
        s = truth[pid]
        for m in candidates:
            i = months.get_loc(m)
            if i + LABEL_HORIZON <= len(s) and s.iloc[i - 1] > 0 and (s.iloc[i:i + LABEL_HORIZON] == 0).all():
                onsets[pid] = m
                break
    return onsets


def alert_delays(obs: pd.DataFrame, col: str, onsets: dict[str, pd.Timestamp]) -> dict:
    flagged = obs[obs[col]][["origin", "produit_id"]].copy()
    flagged["origin"] = pd.to_datetime(flagged["origin"])
    delays, missed = [], 0
    for pid, onset in onsets.items():
        f = flagged[(flagged["produit_id"] == pid) & (flagged["origin"] >= onset - pd.DateOffset(months=LABEL_HORIZON))]
        if f.empty:
            missed += 1
            continue
        delays.append((f["origin"].min() - onset).days)
    d = np.asarray(delays, dtype=float)
    return {
        "n_onsets": len(onsets), "n_alerted": int(len(d)), "n_missed": missed,
        "delay_median_days": float(np.median(d)) if len(d) else np.nan,
        "delay_q25_days": float(np.percentile(d, 25)) if len(d) else np.nan,
        "delay_q75_days": float(np.percentile(d, 75)) if len(d) else np.nan,
        "share_alert_before_onset": float((d < 0).mean()) if len(d) else np.nan,
    }


# --------------------------------------------------------------------- #
# Reproduction de l'évaluation circulaire de l'article
# --------------------------------------------------------------------- #
def _circular(feats: pd.DataFrame) -> dict:
    iforest, rules = detect_at(feats, 0.10, 180)
    pred = (iforest | rules).reindex(feats["produit_id"]).to_numpy(bool)
    y = (feats["nombre_mois_consecutifs_sans_vente"] >= 6).to_numpy(bool)
    return {**confusion(y, pred), "N": int(len(y)), "n_flagged": int(pred.sum())}


def retrospective_circular() -> dict:
    """Reproduit l'évaluation H3 de l'article (features sur 36 mois, vérité = ≥ 6 mois sans vente).

    Variante ``csv_order`` : exactement ``scripts/run_obsolescence.py`` sur
    ``zenith_clean.csv``. Variante ``deterministic_order`` : même calcul avec
    l'ordre total (date, transaction_id) utilisé par les expériences. Elles ne
    diffèrent que par le « dernier stock » retenu entre transactions du même
    jour (tri non stable dans ``build_obsolescence_features``).
    """
    from src.utils import PROCESSED_DIR

    csv = pd.read_csv(PROCESSED_DIR / "zenith_clean.csv", parse_dates=["date"])
    feats_csv = obsolescence_features_at(csv, csv["date"].max())
    clean = full_clean()
    feats_det = obsolescence_features_at(clean, clean["date"].max())
    merged = feats_csv.merge(feats_det, on="produit_id", suffixes=("_csv", "_det"))
    return {
        "ref_date": "2025-07-31",
        "note": "évaluation de l'article : la règle « ≥ 6 mois sans vente » du détecteur est aussi "
                "la définition de la vérité terrain (circulaire)",
        "csv_order": _circular(feats_csv),
        "deterministic_order": _circular(feats_det),
        "n_products_stock_courant_differs": int((merged["stock_courant_csv"] != merged["stock_courant_det"]).sum()),
    }


# --------------------------------------------------------------------- #
def run() -> pd.DataFrame:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    obs, truth = build_observations()
    onsets = dormancy_onsets(truth)
    det_cols = [c for c in obs.columns if c.count("|") == 2]
    lab = obs[obs["labelled"]].copy()
    lab["y_zero"] = lab["future_sales_6m"] == 0
    lab["y_lt3"] = lab["future_sales_6m"] < 3

    rows = []
    for pop_name, pop in (("stock_pos", lab[lab["stock_t0"] > 0]), ("all_known", lab)):
        for ycol, ydef in (("y_zero", "ventes_6m_nulles"), ("y_lt3", "ventes_6m_<3")):
            y = pop[ycol].to_numpy(bool)
            for col in det_cols:
                det, c, d = col.split("|")
                m = confusion(y, pop[col].to_numpy(bool))
                row = {"detector": det, "contamination": float(c), "inactivity_days": float(d),
                       "population": pop_name, "label": ydef, "n_obs": int(len(pop)),
                       "n_products": int(pop["produit_id"].nunique()), "n_origins": int(pop["origin"].nunique()),
                       "n_pos": int(y.sum()), "n_neg": int((~y).sum()), **m}
                if pop_name == "stock_pos" and ydef == "ventes_6m_nulles":
                    row.update(cluster_bootstrap(pop, ycol, col, rng))
                    row.update(alert_delays(obs, col, onsets))
                rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(PAPER_DIR / "obsolescence_eval.csv", index=False)

    main = res[(res["population"] == "stock_pos") & (res["label"] == "ventes_6m_nulles")]

    def _pick(det, c=np.nan, d=np.nan):
        sel = main[main["detector"] == det]
        if not np.isnan(c):
            sel = sel[np.isclose(sel["contamination"], c)]
        if not np.isnan(d):
            sel = sel[np.isclose(sel["inactivity_days"], d)]
        return {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in sel.iloc[0].to_dict().items()}

    summary = {
        "definition": {
            "label": "ventes nulles sur ]t0, t0+6 mois] ET stock central > 0 à t0",
            "label_origins": [str(t.date()) for t in LABEL_ORIGINS],
            "alert_origins": [str(t.date()) for t in ALERT_ORIGINS],
            "bootstrap": f"grappes de produits, B={B_BOOT}, seed={SEED}",
        },
        "article_circular_reproduction": retrospective_circular(),
        "rule_decomposition_label_origins": {
            "n_obs": int(len(lab)),
            "ratio_component_flags": int(lab["rule_ratio_component"].sum()),
            **{f"inactivity_component_{d}_flags": int(lab[f"rule_inactivity_component_{d}"].sum()) for d in DAYS},
            **{f"inactivity_component_{d}_also_flagged_by_ratio": int(
                (lab[f"rule_inactivity_component_{d}"] & lab["rule_ratio_component"]).sum()) for d in DAYS},
            "note": "si chaque drapeau d'inactivité est aussi levé par la composante ratio, le seuil d'inactivité "
                    "n'a aucun effet sur les règles métier",
        },
        "prospective": {
            "hybrid_current (IF c=0.10 + règles 180 j)": _pick("hybrid", 0.10, 180),
            "iforest_only (c=0.10)": _pick("iforest", 0.10),
            "rules_only (180 j)": _pick("rules", d=180),
            "inactivity_only (180 j)": _pick("inactivity", d=180),
        },
    }
    (PAPER_DIR / "obsolescence_confusion.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return res


def main() -> int:
    res = run()
    main_ = res[(res["population"] == "stock_pos") & (res["label"] == "ventes_6m_nulles")]
    cols = ["detector", "contamination", "inactivity_days", "n_pos", "n_neg", "TP", "FP", "FN", "TN",
            "precision", "recall", "F1", "F1_ci_low", "F1_ci_high", "delay_median_days", "n_missed"]
    print(main_[cols].round(3).to_string(index=False))
    print(json.dumps(json.loads((PAPER_DIR / "obsolescence_confusion.json").read_text(encoding="utf-8"))
                     ["article_circular_reproduction"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
