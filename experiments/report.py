"""Génère ``paper_results/RESULTS_FOR_PAPER.md`` à partir des seuls fichiers de résultats.

Chaque chiffre du rapport est lu dans un fichier de ``paper_results/`` (cité
dans la colonne « Source »). Les valeurs « publiées » sont celles de
l'article soumis, telles que transmises pour la révision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import PAPER_DIR

PUB = {
    "mae": {"pipeline": 8.24, "moving_average_3": 7.52, "croston": 7.68, "seasonal_naive": 8.52,
            "lightgbm_uniform": 9.44, "sarima_uniform": 10.82},
    "mae_class": {"A": 6.47, "B": 7.55, "C": 9.94},
    "ablation": {"pipeline_lp": 55029, "pipeline_no_class_lp": 75801, "pipeline_no_obso_lp": 64972, "incumbent": 76980},
    "obso": {"precision": 0.795, "recall": 1.000, "F1": 0.885, "TP": 58, "FP": 15, "FN": 0, "TN": 177},
    "policy": {"pipeline": 55029, "incumbent": 76740, "pct": -28.3, "service_inc": 69.19, "service_pipe": 69.94},
    "h1": {"W": 3264, "p": 0.107}, "h2_lgbm_sarima_p": 0.059,
    "h4": {"n": 531, "diff": 19381, "annual": 77525}, "h5": 38622,
    "runtime_s": 213, "deps": 17,
}

CONV_LABEL = {"sS": "(s, S) réglé", "OUT": "order-up-to réglé", "MA3": "MA3 + order-up-to réglé"}

PAPER_NAMES = {
    "pipeline": "Pipeline différencié", "moving_average_3": "Moyenne mobile 3 mois", "croston": "Croston",
    "croston_sba": "Croston-SBA", "seasonal_naive": "Seasonal naive",
    "lightgbm_uniform": "« LightGBM global » de l'article (LightGBM par produit)",
    "sarima_uniform": "« SARIMA global » de l'article (SARIMA(1,1,1)(1,1,1)₁₂ par produit)",
    "lightgbm_global": "LightGBM global poolé (nouveau)", "pipeline_mlp12": "Pipeline, MLP classe A à fenêtre 12 (variante)",
    "pipeline_no_obso": "Pipeline sans obsolescence",
}


# --------------------------------------------------------------------- #
# Formatage
# --------------------------------------------------------------------- #
def f(x, nd: int = 2, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, str):
        return x
    v = 100 * x if pct else x
    s = f"{v:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s + (" %" if pct else "")


def delta(new, pub, nd: int = 2, rel: bool = True) -> str:
    if new is None or pub is None or (isinstance(new, float) and np.isnan(new)):
        return "—"
    d = new - pub
    s = f(d, nd)
    if rel and pub:
        s += f" ({f(100 * d / pub, 1)} %)"
    return ("+" if d > 0 else "") + s


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _read_json(name: str) -> dict | None:
    p = PAPER_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _read_csv(name: str) -> pd.DataFrame | None:
    p = PAPER_DIR / name
    return pd.read_csv(p) if p.exists() else None


# --------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------- #
def section_verifications() -> str:
    pr = _read_json("paper_method_reproduction.json") or {}
    a = pr.get("data_audit", {})
    lp, inc = pr.get("lp", {}), pr.get("incumbent_as_published_250", {})
    end_date = a.get("data_period", ["?", "?"])[1]
    return f"""\
## Vérifications préalables (V1 – V3)

**V1 — Taux de service (69,19 % → 69,94 %).** Le code calcule un *unit fill rate*
**contre la demande prévue** : `100 × (1 − Σ rupture / Σ demande_prevue)`
(`src/optimization.py:344`, demande ligne 332). Pour la politique LP, `rupture` est la
variable d'écart continue `R` du LP (ligne 186), non recalculée après l'arrondi des
quantités ; pour l'incumbent, `max(0, d − servi)` simulé (lignes 292-293).
`demande_prevue` porte sur août → octobre 2025, **après la fin des données ({end_date})** :
aucune demande réalisée n'intervient. Le code actuel donne
{f(inc.get("unit_fill_rate_vs_forecast_pct"))} % → {f(lp.get("unit_fill_rate_vs_forecast_pct"))} % ;
69,19 / 69,94 % proviennent d'une exécution antérieure recopiée dans
`outputs/tables/optimization_summary.md` (69,2 / 69,9). Les trois définitions (unités, lignes,
cycles), sur demande réalisée, sont désormais publiées (section Service).

**V2 — Ablation (Fig. 2).** `src/validation_hypotheses.py:548-557` :
« −classification » = `coût_LP × 1,4` et « −obsolescence » = `coût_LP × 1,2`, deux
**multiplicateurs codés en dur** (commentaire : « on suppose +20 % ») ; « −LP » = coût de
l'heuristique empirique `simulate_baseline_policy` sur les
**{pr.get("n_products_incumbent_as_published", "?")}** produits (le LP n'en couvre que
{pr.get("n_products_lp", "?")}). Aucune de ces trois barres n'est issue d'une expérience ; elles
sont remplacées par de vraies simulations (Fig. 2 ci-dessous).

**V3 — Demande censurée.** Aucune correction. Les séries de prévision sont les ventes
brutes (`src/forecasting.py:86-97`) ; `rupture_signalee` est calculée
(`src/preprocessing.py:361-371`) mais n'est lue nulle part. `stock_apres_vente` est suivi
**par magasin** : {f(a.get("share_consecutive_rows_stock_consistent_product_store"), 1, pct=True)}
des transactions successives vérifient stock(t) = stock(t−1) − quantité au niveau produit ×
magasin, contre {f(a.get("share_consecutive_rows_stock_consistent_product_only"), 1, pct=True)} au
niveau produit ; {a.get("sales_rows_leaving_store_at_zero_stock", "?")} ventes laissent un magasin
à 0 ({a.get("products_with_such_rows", "?")} produits). Une vente manquée faute de stock ne laisse
aucune ligne : la demande est sous-estimée là où il y a eu rupture. Toutes les politiques du
backtest sont évaluées sur cette même demande observée ; la limite doit figurer dans l'article.
Conséquence annexe : le `stock_courant` du pipeline est le dernier stock du magasin ayant vendu en
dernier, pas le stock central (le backtest utilise la somme des magasins).

**Autre constat — classe A.** Dans l'exécution de l'article,
{a.get("article_lstm_products_mae_identical_to_seasonal_naive", "?")}/{a.get("article_lstm_products", "?")}
produits « LSTM/MLP » ont un MAE identique au seasonal naive : le MLP exige 36 mois d'historique
(`src/forecasting.py:228`) et l'entraînement n'en compte que 32.

Source des chiffres de cette section : `paper_method_reproduction.json`.
"""


PROTOCOL = """\
## Protocole commun (à décrire dans l'article)

- **Rolling-origin sans fuite** (`experiments/rolling_origin.py`) : origines mensuelles
  t0 = 31/01/2024 … 30/06/2025 (≥ 18 mois d'historique). À chaque t0, à partir des seules
  lignes brutes datées ≤ t0 : nettoyage (médianes de prix par famille, interpolation des
  stocks), agrégats produit, ABC (CA cumulé ≤ t0), XYZ, K-Means et scaler, features
  d'obsolescence, Isolation Forest et scaler, règles métier, affectation modèle ↔ produit,
  ajustement des modèles ; prévision h = 1..3. Test d'absence de fuite :
  `tests/test_exp_no_leakage.py` (modifier / supprimer / ajouter des lignes postérieures
  à t0 ne change ni une prévision, ni une classe, ni un drapeau, ni le stock ; contrôle
  positif : une modification ≤ t0 les change).
- **Panel d'évaluation** : 16 origines (01/2024 → 04/2025) × produits connus à t0 × h = 1..3,
  identique pour toutes les méthodes. `all` = tous les produits connus ; `active` = produits
  non flagués obsolètes à t0 (même idée que le périmètre de l'article, sans conditionner sur
  le futur). Vérité = ventes mensuelles nettoyées, 0 si aucune vente.
- **Backtest de politique** (`experiments/simulation.py`) : demande réalisée, lignes de
  vente servies dans l'ordre chronologique, ventes perdues, horizon glissant mensuel
  (février 2024 → juillet 2025), commandes en transit suivies. Développement = févr. → déc.
  2024 (réglage des politiques classiques), **test = janv. → juil. 2025** (chiffres de
  référence). Coûts = définitions de `financial_kpis` (50 $/commande, 0,1 %/jour de stockage,
  marge perdue).
- Seed 42 partout, un thread BLAS/OpenMP, CBC séquentiel.
"""


def section_forecast() -> str:
    m = _read_csv("forecast_metrics.csv")
    bc = _read_csv("forecast_metrics_by_class.csv")
    if m is None:
        return "## Table II — Prévision\n\n_Non disponible : lancer `experiments.metrics`._\n"
    meta = _read_json("rolling_origin_meta.json") or {}
    out = ["## Table II — Prévision (rolling-origin, 16 origines × h = 1..3)", ""]
    for subset, label in (("all", "tous les produits connus à t0"), ("active", "produits non flagués à t0")):
        sub = m[m["subset"] == subset].set_index("model")
        order = sub.sort_values("MASE_m1").index.tolist()
        rows = []
        for mod in order:
            r = sub.loc[mod]
            pub = PUB["mae"].get(mod)
            rows.append([PAPER_NAMES.get(mod, mod), f(pub) if pub else "—", f(r["MAE"]),
                         delta(r["MAE"], pub) if pub else "—", f(r["MASE_m1"], 3), f(r["MASE_m12"], 3),
                         f(r["RMSSE"], 3), f(r["WAPE"], 3), f(r["bias_pct"], 1) + " %",
                         int(r["n_series"]), int(r["n_excluded"]), int(r["n_excluded_MASE_m12"])])
        out += [f"**Sous-population `{subset}`** ({label}) — trié par MASE (m = 1, métrique retenue a priori)", "",
                table(["Méthode", "MAE publié", "MAE", "Écart", "MASE m=1", "MASE m=12", "RMSSE", "WAPE",
                       "Biais %", "Séries", "Exclues m=1", "Exclues m=12"], rows), ""]
    a = m[(m["subset"] == "all")].set_index("model")
    simple = ["seasonal_naive", "moving_average_3", "croston", "croston_sba"]
    best_simple = a.loc[simple, "MASE_m1"].idxmin()
    rank = a["MASE_m1"].rank().astype(int)
    out += [f"Lecture : sur `all`, le pipeline est classé **{rank['pipeline']}ᵉ sur {len(a)}** en MASE ; la "
            f"meilleure baseline simple est **{PAPER_NAMES[best_simple]}** (MASE {f(a.loc[best_simple, 'MASE_m1'], 3)} "
            f"contre {f(a.loc['pipeline', 'MASE_m1'], 3)} pour le pipeline). Les MAE publiés proviennent d'une "
            "seule fenêtre (avril → juillet 2025), de 177 produits filtrés sur des ventes > 0 dans le test et de "
            "classes calculées sur les 36 mois : les écarts mêlent changement de protocole et suppression des fuites.",
            "", f"Source : `forecast_metrics.csv`, `rolling_origin_errors.csv` "
            f"({meta.get('n_rows', '?')} lignes).", ""]
    if bc is not None:
        p = bc[(bc["model"] == "pipeline")].set_index(["subset", "abc"])
        rows = []
        for c in ("A", "B", "C"):
            na = p.loc[("all", c)] if ("all", c) in p.index else None
            ac = p.loc[("active", c)] if ("active", c) in p.index else None
            rows.append([c, f(PUB["mae_class"][c]), f(na["MAE"]) if na is not None else "—",
                         f(ac["MAE"]) if ac is not None else "—",
                         delta(ac["MAE"], PUB["mae_class"][c]) if ac is not None else "—",
                         f(ac["MASE_m1"], 3) if ac is not None else "—"])
        out += ["**MAE du pipeline par classe ABC (classe connue à t0)**", "",
                table(["Classe", "MAE publié", "MAE `all`", "MAE `active`", "Écart (`active`)", "MASE `active`"], rows),
                "", "Source : `forecast_metrics_by_class.csv`.", ""]
        eff = _read_csv("rolling_origin_errors.csv")
        if eff is not None:
            e = eff[(eff["method"] == "pipeline") & (eff["horizon"] == 1)]
            share = e.groupby("model_assigned")["model_effective"].apply(
                lambda s: (s == "naive_seasonal_fallback").mean())
            out += ["**Replis effectifs du pipeline sur le seasonal naive** (part des (produit, origine) où le "
                    "modèle affecté retombe sur le seasonal naive, 18 origines) : "
                    + ", ".join(f"`{k}` {f(v, 1, pct=True)}" for k, v in share.items()
                                if k in ("lstm", "sarima", "lightgbm")) + ". Dans l'article, la classe A "
                    "« LSTM/MLP » est le seasonal naive à 100 % (le MLP exige 36 mois, le train en a 32) ; la "
                    "variante `pipeline_mlp12` mesure ce que donnerait un MLP effectivement entraîné.", ""]
    return "\n".join(out)


def section_policies() -> str:
    c = _read_csv("policy_comparison.csv")
    meta = _read_json("policy_meta.json") or {}
    mv = _read_json("milp_vs_lp.json") or {}
    pr = _read_json("paper_method_reproduction.json")
    out = ["## Table III — Politiques d'inventaire", ""]
    rep = mv.get("reproduction", {})
    if pr:
        rows = [
            ["Coût total pipeline (USD)", f(55029, 0), f(pr["lp"]["total_cost"]), delta(pr["lp"]["total_cost"], 55029)],
            ["Coût total incumbent (USD), 250 produits", f(76740, 0), f(pr["incumbent_as_published_250"]["total_cost"]),
             delta(pr["incumbent_as_published_250"]["total_cost"], 76740)],
            ["Économie, périmètres différents (177 vs 250)", "−28,3 %", f(pr["savings_pct_as_published"], 2) + " %", "—"],
            ["Coût incumbent, mêmes 177 produits (USD)", "73 525 (H4)", f(pr["incumbent_paired_177"]["total_cost"]), "—"],
            ["Économie appariée", "—", f(pr["savings_pct_paired"], 2) + " %", "—"],
            ["Écart apparié sur 3 mois / × 4 (USD)", "19 381 / 77 525", f"{f(pr['paired_difference_usd_3_months'])} / "
             f"{f(pr['paired_difference_annualised_x4'])}", "—"],
            ["Fill rate unités incumbent → pipeline (vs prévision)", "69,19 % → 69,94 %",
             f"{f(pr['incumbent_as_published_250']['unit_fill_rate_vs_forecast_pct'])} % → "
             f"{f(pr['lp']['unit_fill_rate_vs_forecast_pct'])} %", "—"],
        ]
        out += ["**Méthode de l'article reproduite** (plan août → oct. 2025 jugé contre la demande *prévue*, code "
                "d'origine sur les entrées versionnées)", "",
                table(["Chiffre", "Publié", "Recalculé", "Écart"], rows), "",
                "Les valeurs publiées 55 029 / 76 740 / 69,19 / 69,94 % ne sont plus produites par le code : elles "
                "viennent d'une exécution antérieure recopiée dans `outputs/tables/optimization_summary.md`. Le plan "
                "LP recalculé est identique au plan versionné `commandes_centrales.csv` : "
                f"{rep.get('replica_matches_published_commandes_centrales')}. Source : "
                "`paper_method_reproduction.json`, `milp_vs_lp.json`.", ""]
    if c is None:
        return "\n".join(out + ["_Backtest non disponible._", ""])
    t = c[c["period"] == "test"].set_index("policy")
    order = ["incumbent", "sS_tuned", "order_up_to_tuned", "ma3_order_up_to_tuned", "best_baseline_lp",
             "pipeline_no_obso_lp", "pipeline_no_class_lp", "pipeline_lp", "pipeline_mlp12_lp"]
    rows = []
    for p in order:
        if p not in t.index:
            continue
        r = t.loc[p]
        rows.append([p, f(r["total_cost"], 0), f(r["total_cost_vs_incumbent_pct"], 1) + " %", f(r["ordering_cost"], 0),
                     f(r["holding_cost"], 0), f(r["stockout_cost"], 0), f(r["order_value"], 0),
                     f(r["avg_inventory_value"], 0), f(r["unit_fill_rate"], 1, pct=True),
                     f(r["order_fill_rate"], 1, pct=True), f(r["cycle_service_level"], 1, pct=True),
                     int(r["n_orders"]), int(r["budget_clips"]), int(r["infeasible_decisions"])])
    out += ["**Backtest à demande réalisée — période de test (janv. → juil. 2025)**", "",
            table(["Politique", "Coût total", "vs incumbent", "Commande", "Stockage", "Rupture", "Achats",
                   "Stock moyen", "Fill rate unités", "Fill rate lignes", "CSL", "Commandes", "Écrêt. budget",
                   "LP non optimaux"], rows), ""]
    pipe, inc = t.loc["pipeline_lp"], t.loc["incumbent"]
    convs = ["sS_tuned", "order_up_to_tuned", "ma3_order_up_to_tuned"]
    conv = t.loc[convs]
    best_test = conv["total_cost"].idxmin()
    dev = c[c["period"] == "dev"].set_index("policy").loc[convs, "total_cost"]
    best_dev = dev.idxmin()
    beaten = [p for p in convs + ["best_baseline_lp"] if t.loc[p, "total_cost"] < pipe["total_cost"]]
    out += [f"Lecture : pipeline LP {f(pipe['total_cost'], 0)} USD contre {f(inc['total_cost'], 0)} USD pour "
            f"l'incumbent, soit **{f(pipe['total_cost_vs_incumbent_pct'], 1)} %** (publié −28,3 % contre la "
            f"prévision). Politique classique choisie sur le développement : `{best_dev}` "
            f"({f(t.loc[best_dev, 'total_cost'], 0)} USD sur le test) ; la moins coûteuse a posteriori sur le test est "
            f"`{best_test}` ({f(conv.loc[best_test, 'total_cost'], 0)} USD, "
            f"{f(conv.loc[best_test, 'total_cost_vs_incumbent_pct'], 1)} % vs incumbent). Politiques moins coûteuses que "
            f"le pipeline complet sur le test : {', '.join(f'`{p}`' for p in beaten) or 'aucune'}. Taux de service en "
            f"unités : incumbent {f(inc['unit_fill_rate'], 2, pct=True)}, pipeline {f(pipe['unit_fill_rate'], 2, pct=True)} "
            "(publié 69,19 % → 69,94 %, contre la prévision). Le LP planifie exactement la prévision ponctuelle : il "
            "n'a pas de stock de sécurité, d'où un coût de rupture élevé face à la demande réelle.", "",
            f"Paramètres réglés sur le développement : z = {json.dumps(meta.get('z_tuned_on_dev'), ensure_ascii=False)} ; "
            f"meilleure baseline de prévision (MASE sur cibles ≤ déc. 2024) : `{meta.get('best_forecast_baseline')}`.",
            "", "Sources : `policy_comparison.csv` (périodes test / dev / full), `policy_sim_detail.csv`, "
            "`policy_tuning.csv`, `policy_meta.json`.", ""]
    return "\n".join(out)


def section_hypotheses() -> str:
    s = _read_json("stat_tests.json")
    o = _read_json("obsolescence_confusion.json")
    if s is None:
        return "## Table IV — Hypothèses\n\n_Non disponible._\n"
    out = ["## Table IV — Hypothèses", ""]
    rows = []
    h1 = s["H1"]
    for k in ("H1_pipeline_vs_best_simple", "H1_pipeline_vs_seasonal_naive"):
        r = h1[k]
        rows.append([k, r["comparison"], f(r["statistic"], 1), f(r["p_raw"], 4), f(r.get("p_adjusted"), 4),
                     f(r["effect_size"], 3), f"médiane Δ {f(r['median_diff'], 3)} [{f(r['ci_low'], 3)} ; {f(r['ci_high'], 3)}]",
                     r["n"], r["resampling_unit"]])
    for k in ("H1_pipeline_vs_best_simple", "H1_pipeline_vs_seasonal_naive"):
        r = s.get("H1_active_subset", {}).get(k)
        if r:
            rows.append([k + " (robustesse, `active`)", r["comparison"], f(r["statistic"], 1), f(r["p_raw"], 4),
                         "— (hors famille)", f(r["effect_size"], 3),
                         f"médiane Δ {f(r['median_diff'], 3)} [{f(r['ci_low'], 3)} ; {f(r['ci_high'], 3)}]",
                         r["n"], r["resampling_unit"]])
    for k, r in s["H2"].items():
        rows.append([k, r["comparison"], f(r["statistic"], 3), f(r["p_raw"], 4), f(r.get("p_adjusted"), 4),
                     f(r["effect_size"], 3) + " (ΔMSE rel.)", f"E[Δ] {f(r['mean_loss_diff'], 2)} "
                     f"[{f(r.get('ci_low'), 2)} ; {f(r.get('ci_high'), 2)}]", r["n"], "origines (HAC)"])
    hp = s.get("H4_H5", {})
    for k, r in hp.items():
        if not isinstance(r, dict) or "p_raw" not in r:
            continue
        val = r.get("savings_annualised", r.get("net_gain_annualised"))
        rows.append([k, r.get("comparison", ""), f(r["statistic"], 1), f(r["p_raw"], 4), f(r.get("p_adjusted"), 4),
                     f(r["effect_size"], 3), f"{f(val, 0)} USD/an [{f(r['ci_low'], 0)} ; {f(r['ci_high'], 0)}]",
                     r["n"], r["resampling_unit"]])
    out += [table(["Test", "Comparaison", "Statistique", "p brut", "p ajusté (Holm)", "Taille d'effet",
                   "Estimation [IC 95 %]", "n", "Unité"], rows), ""]
    b = h1["H1_pipeline_vs_seasonal_naive"]
    out += [f"Publié : H1 W = 3264, p = 0,107 (vs seasonal naive) → ici W = {f(b['statistic'], 1)}, p = "
            f"{f(b['p_raw'], 4)} ; meilleure baseline simple = `{h1['H1_best_simple_baseline']}`. "
            f"Publié : H2 LightGBM vs SARIMA p = 0,059 (test invalide : erreurs de produits concaténées) → ici "
            f"p brut {f(s['H2']['H2_DM_lightgbm_uniform_vs_sarima_uniform']['p_raw'], 4)}, ajusté "
            f"{f(s['H2']['H2_DM_lightgbm_uniform_vs_sarima_uniform'].get('p_adjusted'), 4)}.", ""]
    if "H4_vs_incumbent" in hp:
        h4, h5 = hp["H4_vs_incumbent"], hp["H5_vs_incumbent"]
        out += [f"Publié : H4 n = 531 paires, économie 19 381 USD × 4 = 77 525 USD/an (bootstrap i.i.d. des "
                f"paires produit × mois, malgré la mention « par blocs ») → ici, sur demande réalisée : économie "
                f"{f(h4['savings_total'], 0)} USD sur {len(hp['H4_H5_context']['test_months'])} mois de test, "
                f"{f(h4['savings_annualised'], 0)} USD/an, IC grappes produits [{f(h4['ci_low'], 0)} ; {f(h4['ci_high'], 0)}], "
                f"IC blocs 1 mois [{f(h4['block_bootstrap_1_month']['ci_low'], 0)} ; "
                f"{f(h4['block_bootstrap_1_month']['ci_high'], 0)}] ({h4['block_bootstrap_1_month']['n_blocks']} blocs). "
                f"Publié : H5 gain net 38 622 USD/an (ΔCA + Δmarge comptait la marge deux fois) → ici "
                f"{f(h5['net_gain_annualised'], 0)} USD/an (identique à l'économie de coût par construction, écart "
                f"numérique {f(h5['identity_check_gain_minus_savings'], 6)}).", ""]
        hc = hp.get("H4_vs_best_conventional")
        if hc:
            out += [f"Contre la meilleure politique classique **choisie sur la période de développement** "
                    f"(`{hp['H4_H5_context']['best_conventional_policy_selected_on_dev']}`) : économie "
                    f"{f(hc['savings_total'], 0)} USD ({f(hc['savings_pct'], 1)} %), soit {f(hc['savings_annualised'], 0)} "
                    f"USD/an, IC grappes produits [{f(hc['ci_low'], 0)} ; {f(hc['ci_high'], 0)}] (valeur négative = le "
                    "pipeline coûte plus). Bootstrap par blocs de 3 mois : "
                    f"{hc['block_bootstrap_3_months']['n_blocks']} blocs seulement sur 7 mois de test, IC indicatif.", ""]
    if o:
        circ = o["article_circular_reproduction"]["csv_order"]
        rows = [["Évaluation de l'article (circulaire, reproduite)", circ["TP"], circ["FP"], circ["FN"], circ["TN"],
                 f(circ["precision"], 3), f(circ["recall"], 3), f(circ["F1"], 3), "—", "—"]]
        for name, r in o["prospective"].items():
            rows.append([name, r["TP"], r["FP"], r["FN"], r["TN"], f(r["precision"], 3), f(r["recall"], 3),
                         f(r["F1"], 3), f"[{f(r.get('F1_ci_low'), 3)} ; {f(r.get('F1_ci_high'), 3)}]",
                         f(r.get("delay_median_days"), 0)])
        det = o["article_circular_reproduction"]["deterministic_order"]
        out += ["**H3 — Obsolescence, vérité terrain prospective** (ventes nulles sur les 6 mois suivant t0 ET stock "
                "> 0 à t0 ; 13 origines ; détecteurs sur ≤ t0)", "",
                table(["Détecteur", "TP", "FP", "FN", "TN", "Précision", "Rappel", "F1", "IC 95 % F1",
                       "Délai médian (j)"], rows), "",
                f"Publié : 58/15/0/177, P = 0,795, R = 1,000, F1 = 0,885 — reproduit exactement, mais circulaire "
                f"(la règle « ≥ 6 mois sans vente » du détecteur est la vérité terrain). Fragilité : avec un ordre "
                f"déterministe des transactions du même jour, {det['TP']}/{det['FP']}/{det['FN']}/{det['TN']} "
                f"({o['article_circular_reproduction']['n_products_stock_courant_differs']} stocks courants changent). "
                "Délai < 0 = alerte avant le début de la dormance. Balayages contamination × seuil : "
                "`obsolescence_eval.csv`.", ""]
        rd = o.get("rule_decomposition_label_origins")
        if rd:
            subsumed = all(rd[f"inactivity_component_{d}_also_flagged_by_ratio"] == rd[f"inactivity_component_{d}_flags"]
                           for d in (90, 180, 270, 365))
            out += [f"Décomposition des règles métier ({rd['n_obs']} observations produit × origine) : la composante "
                    f"« ratio 3 m / 12 m < 0,1 et âge > 365 j » lève {rd['ratio_component_flags']} drapeaux ; la "
                    f"composante d'inactivité en lève {rd['inactivity_component_90_flags']} (seuil 90 j) à "
                    f"{rd['inactivity_component_365_flags']} (365 j), "
                    + ("**tous déjà levés par la composante ratio** : le seuil d'inactivité n'a aucun effet sur les "
                       "règles, la règle effective est celle du ratio." if subsumed else
                       "dont une partie seulement est couverte par le ratio.") , ""]
        best = _read_csv("obsolescence_eval.csv")
        if best is not None:
            m = best[(best["population"] == "stock_pos") & (best["label"] == "ventes_6m_nulles")]
            top = m.loc[m["F1"].idxmax()]
            out += [f"Meilleur F1 du balayage (a posteriori, optimiste) : `{top['detector']}` "
                    f"(contamination {f(top['contamination'], 2)}, seuil {f(top['inactivity_days'], 0)} j) — "
                    f"P = {f(top['precision'], 3)}, R = {f(top['recall'], 3)}, F1 = {f(top['F1'], 3)} "
                    f"[{f(top['F1_ci_low'], 3)} ; {f(top['F1_ci_high'], 3)}], délai médian "
                    f"{f(top['delay_median_days'], 0)} j.", ""]
    out += ["Figure : `fig_obsolescence_pr.png` (précision vs rappel, vérité prospective).", "",
            "Sources : `stat_tests.json`, `obsolescence_confusion.json`, `obsolescence_eval.csv`.", ""]
    return "\n".join(out)


def section_ablation() -> str:
    c = _read_csv("policy_comparison.csv")
    if c is None:
        return ""
    t = c[c["period"] == "test"].set_index("policy")
    labels = {"pipeline_lp": "Pipeline complet", "pipeline_no_class_lp": "− classification (seasonal naive partout)",
              "pipeline_no_obso_lp": "− obsolescence", "incumbent": "− LP (heuristique empirique)"}
    rows = []
    for p, lab in labels.items():
        r = t.loc[p]
        rows.append([lab, f(PUB["ablation"][p], 0), "multiplicateur codé en dur" if p in ("pipeline_no_class_lp",
                     "pipeline_no_obso_lp") else ("exécution antérieure" if p == "pipeline_lp" else "heuristique, 250 produits"),
                     f(r["total_cost"], 0), f(r["total_cost_vs_incumbent_pct"], 1) + " %", f(r["unit_fill_rate"], 1, pct=True)])
    return "\n".join(["## Fig. 2 — Ablation (backtest, période de test)", "",
                      table(["Configuration", "Publié (USD)", "Origine du chiffre publié", "Mesuré (USD)",
                             "vs incumbent", "Fill rate unités"], rows), "",
                      "Les chiffres publiés et mesurés ne sont pas sur la même base (3 mois contre la prévision vs "
                      "7 mois de demande réalisée) : seule la comparaison interne de chaque colonne a un sens.", "",
                      "Source : `policy_comparison.csv`.", ""])


def section_milp() -> str:
    mv = _read_json("milp_vs_lp.json")
    if mv is None:
        return ""
    def _gap_cell(gap, ev):
        infeasible = ev["budget_violation_months"] or ev["capacity_violation_months"] or ev["moq_violations"]
        return f(gap, 2) + " %" + (" (plan infaisable)" if infeasible else "")

    rows = []
    for s in mv["scenarios"]:
        k = f"objective_{s['objective']}"
        near, up = s["lp_rounded_nearest (code)"], s["lp_rounded_up"]
        rows.append([f(s["budget_usd"], 0), s["moq"], s["objective"], s["milp"]["status"],
                     f(s["milp"]["objective"], 0), f(s["milp"]["best_bound"], 0),
                     f(100 * s["milp"]["gap_relative"], 3) + " %" if s["milp"]["gap_relative"] is not None else "—",
                     f(s["milp"]["seconds"], 1), f(near[k], 0),
                     _gap_cell(s["cost_gap_lp_rounded_nearest_vs_milp_pct"], near),
                     near["budget_violation_months"], f(max(near["budget_excess_pct_by_month"]), 2) + " %",
                     near["capacity_violation_months"], near["moq_violations"],
                     f"{near['n_orders']} / {s['milp_plan']['n_orders']}",
                     _gap_cell(s["cost_gap_lp_rounded_up_vs_milp_pct"], up), up["budget_violation_months"],
                     f(max(up["budget_excess_pct_by_month"]), 2) + " %"])
    first = mv["scenarios"][0]["milp"]
    nom = next(s for s in mv["scenarios"] if s["budget_usd"] == 500_000 and s["moq"] == "none" and s["objective"] == "code")
    moq = next(s for s in mv["scenarios"] if s["budget_usd"] == 500_000 and s["moq"] == "1_month" and s["objective"] == "code")
    near_viol = [s for s in mv["scenarios"] if s["moq"] == "none" and s["lp_rounded_nearest (code)"]["budget_violation_months"]]
    up_viol = [s for s in mv["scenarios"] if s["moq"] == "none" and s["lp_rounded_up"]["budget_violation_months"]]
    n_opt = sum(s["milp"]["status"] == "Optimal" for s in mv["scenarios"])
    lecture = (
        f"Lecture : {n_opt}/{len(mv['scenarios'])} MILP résolus à l'optimalité prouvée. Au point nominal (budget "
        f"non contraignant), le plan LP arrondi comme le code est faisable mais coûte "
        f"{f(nom['cost_gap_lp_rounded_nearest_vs_milp_pct'], 2)} % de plus que le MILP "
        f"({nom['lp_rounded_nearest (code)']['n_orders']} commandes contre {nom['milp_plan']['n_orders']}) ; arrondi à "
        f"l'entier supérieur : {f(nom['cost_gap_lp_rounded_up_vs_milp_pct'], 2)} %. Avec un MOQ d'un mois, le plan LP "
        f"viole {moq['lp_rounded_nearest (code)']['moq_violations']} fois la contrainte : il n'est plus réalisable. "
        f"Budget serré : l'arrondi au plus proche dépasse le budget pour "
        f"{', '.join(f(s['budget_usd'], 0) for s in near_viol) or 'aucun budget testé'} $/mois ; l'arrondi supérieur "
        f"pour {', '.join(f(s['budget_usd'], 0) for s in up_viol) or 'aucun'} $/mois. Conclusion : l'équivalence "
        "« LP arrondi ≈ MILP » ne tient qu'à quelques % près et seulement quand budget et MOQ ne contraignent pas ; "
        f"le MILP se résout ici en {f(min(s['milp']['seconds'] for s in mv['scenarios']), 0)} à "
        f"{f(max(s['milp']['seconds'] for s in mv['scenarios']), 0)} s, il peut donc remplacer la relaxation sur "
        "une instance de cette taille.")
    return "\n".join([
        "## MILP exact vs relaxation LP arrondie", "",
        f"Instance de l'article : {mv['instance']['n_products_active']} produits actifs × "
        f"{mv['instance']['horizon_months']} mois ; MILP {first['n_vars']} variables "
        f"({first['n_integer_vars']} entières/binaires), {first['n_constraints']} contraintes ; CBC "
        f"{first.get('cbc_version')}, limite 600 s. Arrondi du code : {mv['instance']['rounding_in_code']}. "
        "ρ (en transit) = 0 ; MOQ absente des données : variantes `none` et `1_month` (hypothèse).", "",
        table(["Budget/mois", "MOQ", "Objectif", "Statut MILP", "Obj. MILP", "Borne", "Gap", "Temps (s)",
               "Obj. LP arrondi (code)", "Surcoût vs MILP", "Mois budget dépassé", "Dépassement max", "Capacité",
               "Viol. MOQ", "Commandes LP / MILP", "Surcoût (arrondi sup.)", "Mois dépassés (arrondi sup.)",
               "Dépassement max (arrondi sup.)"], rows), "",
        "Objectif `code` : coûts de l'implémentation avec un vrai coût fixe K par commande (le LP le linéarise en "
        "K/d̄ par unité) ; objectif `paper` : w_i·(K·x + h·I + p·s) avec p = 1,2 × prix, tel que formulé dans "
        "l'article. Les deux plans sont propagés dans le même bilan de stock (demande prévue) avant évaluation. "
        "Un surcoût négatif n'existe que pour un plan qui viole une contrainte du MILP (budget ou MOQ) : il n'est "
        "pas comparable.", "", lecture, "",
        "Source : `milp_vs_lp.json`.", ""])


def section_service() -> str:
    st = _read_csv("service_tradeoff.csv")
    tg = _read_csv("service_targets.csv")
    bc = _read_csv("policy_service_by_class.csv")
    if st is None:
        return ""
    out = ["## Service — définitions, compromis coût / service, cibles", ""]
    if bc is not None:
        sub = bc[(bc["period"] == "test") & bc["policy"].isin(["incumbent", "pipeline_lp"])]
        rows = [[r["policy"], r["abc"], f(r["unit_fill_rate"], 1, pct=True), f(r["order_fill_rate"], 1, pct=True),
                 f(r["cycle_service_level"], 1, pct=True), f(r["total_cost"], 0)] for _, r in sub.iterrows()]
        out += ["**Par classe ABC (test)**", "", table(["Politique", "Classe", "Fill rate unités", "Fill rate lignes",
                                                        "CSL", "Coût total"], rows), ""]
    b = st[(st["period"] == "test") & (st["abc"] == "all")]
    rows = [[r["sweep"], f(r["value"], 2), f(r["total_cost"], 0), f(r["order_value"], 0),
             f(r["unit_fill_rate"], 1, pct=True), f(r["order_fill_rate"], 1, pct=True),
             f(r["cycle_service_level"], 1, pct=True), int(r["budget_clips"])] for _, r in b.iterrows()]
    out += ["**Balayages (pipeline LP ; frontière (s, S) à z unique en référence)**", "",
            table(["Balayage", "Valeur", "Coût total", "Achats", "Fill rate unités", "Fill rate lignes", "CSL",
                   "Écrêt. budget"], rows), ""]
    def _num(v, nd):
        try:
            return f(float(v), nd)
        except (TypeError, ValueError):
            return str(v)

    if tg is not None:
        rows = [[r["service_definition"], f(r["target"], 0, pct=True), _num(r["min_budget_usd"], 0),
                 _num(r["min_penalty_factor"], 1), f(r["max_achieved_over_budget_sweep"], 1, pct=True),
                 _num(r["sS_cheapest_z_reaching_target"], 3),
                 f(r["sS_cost_at_that_z"], 0) if pd.notna(r["sS_cost_at_that_z"]) else "—"]
                for _, r in tg.iterrows()]
        out += ["**Cibles de service (test)**", "",
                table(["Définition", "Cible", "Budget minimal (pipeline)", "Pénalité minimale (1+π)",
                       "Max atteint (balayage budget)", "z le moins coûteux (s, S)", "Coût (s, S) à ce z"], rows), ""]
    bud = b[b["sweep"] == "budget"]
    pen = b[b["sweep"] == "penalty_factor"]
    ss = b[b["sweep"] == "sS_z"]
    cheapest = ss.loc[ss["total_cost"].idxmin()]
    plateau = bud.loc[bud["unit_fill_rate"] >= bud["unit_fill_rate"].max() - 0.001, "value"].min()
    out += [f"Lecture : à pénalité nominale, le fill rate (unités) du pipeline plafonne à "
            f"{f(bud['unit_fill_rate'].max(), 1, pct=True)} ; dès {f(plateau, 0)} $/mois il est à moins de 0,1 point de "
            f"ce maximum (le budget ne contraint plus), et multiplier la pénalité de rupture de 1 à {f(pen['value'].max(), 0)} ne le "
            f"fait varier que de {f(pen['unit_fill_rate'].min(), 1, pct=True)} à {f(pen['unit_fill_rate'].max(), 1, pct=True)}. "
            "Le LP déterministe vise la prévision ponctuelle : sans stock de sécurité, ni le budget ni la pénalité ne "
            f"relèvent le service. À titre de repère, (s, S) avec z = {f(cheapest['value'], 2)} coûte "
            f"{f(cheapest['total_cost'], 0)} $ pour un fill rate de {f(cheapest['unit_fill_rate'], 1, pct=True)}.", ""]
    out += ["Figure : `fig_service_tradeoff.png` (coût total vs fill rate, période de test).", "",
            "Sources : `service_tradeoff.csv`, `service_targets.csv`, `policy_service_by_class.csv`.", ""]
    return "\n".join(out)


def section_sensitivity() -> str:
    s = _read_csv("sensitivity.csv")
    if s is None:
        return ""
    p = s[s["policy"] == "pipeline_lp"]
    rows = []
    for (param, val), g in p.groupby(["parameter", "value"], sort=False):
        bt = g[g["method"] == "backtest_realized"]
        pu = g[g["method"] == "paper_forecast_based_as_published"]
        pa = g[g["method"] == "paper_forecast_based_paired"]
        rows.append([param, val,
                     f(pu["savings_vs_incumbent_pct"].iloc[0], 1) + " %" if len(pu) else "n/a",
                     f(pa["savings_vs_incumbent_pct"].iloc[0], 1) + " %" if len(pa) else "n/a",
                     f(bt["savings_vs_incumbent_pct"].iloc[0], 1) + " %" if len(bt) else "—",
                     f(bt["savings_vs_conventional_pct"].iloc[0], 1) + " %" if len(bt) else "—",
                     f(bt["unit_fill_rate"].iloc[0], 1, pct=True) if len(bt) else "—",
                     f(bt["total_cost"].iloc[0], 0) if len(bt) else "—",
                     int(bt["infeasible_decisions"].iloc[0]) if len(bt) else "—"])
    conv = s["conventional_kind"].iloc[0]
    pu = p[p["method"] == "paper_forecast_based_as_published"]["savings_vs_incumbent_pct"].dropna()
    pa = p[p["method"] == "paper_forecast_based_paired"]["savings_vs_incumbent_pct"].dropna()
    bt = p[p["method"] == "backtest_realized"]
    nominal = {"budget": "1.0x", "lead_time": "nominal", "stockout_penalty_factor": "1.2",
               "holding_rate_per_day": "0.001", "abc_weights": "(4.0, 2.5, 1.5)",
               "obsolescence_threshold_days": "180", "horizon_months": "3"}
    is_nom = bt.apply(lambda r: nominal.get(r["parameter"]) == str(r["value"]), axis=1)
    distinct = pd.concat([bt[~is_nom], bt[is_nom].head(1)])
    n_beat_inc = int((distinct["savings_vs_incumbent_pct"] > 0).sum())
    n_beat_conv = int((distinct["savings_vs_conventional_pct"] > 0).sum())
    weights = p[(p["parameter"] == "abc_weights") & (p["method"] == "paper_forecast_based_as_published")]
    lecture = (
        f"Lecture (économie > 0 : le pipeline coûte moins que la référence) : avec la méthode de l'article, "
        f"l'économie du pipeline sur l'incumbent va de "
        f"{f(pu.min(), 1)} % à {f(pu.max(), 1)} % (non apparié) et de {f(pa.min(), 1)} % à {f(pa.max(), 1)} % (apparié) "
        f"selon les scénarios ; pondérations ABC : "
        + ", ".join(f"{r['value']} → {f(r['savings_vs_incumbent_pct'], 1)} %" for _, r in weights.iterrows())
        + f". Sur demande réalisée, sur {len(distinct)} configurations distinctes (le point nominal, répété une fois "
        f"par paramètre, n'est compté qu'une fois), le pipeline bat l'incumbent dans {n_beat_inc} et la politique "
        f"classique ({CONV_LABEL.get(conv, conv)}, z re-réglé sur le développement) dans {n_beat_conv}.")
    return "\n".join([
        "## Sensibilité (une variable à la fois)", "",
        table(["Paramètre", "Valeur", "Économie (méthode article, non apparié)", "Économie (méthode article, apparié)",
               "Économie backtest vs incumbent", f"Économie backtest vs {CONV_LABEL.get(conv, conv)}", "Fill rate backtest",
               "Coût backtest", "LP non optimaux"], rows), "",
        "Taux de change : hypothèse prix en CDF / coûts en USD (aucune donnée de change dans le dépôt).", "",
        "Propriété de la formulation : une commande passée en période t n'est reçue dans l'horizon que si "
        "t + L ≤ H. Quand le délai L d'un produit atteint l'horizon H (délai +1 mois : L = 3 pour la Chine avec "
        "H = 3 ; horizon H = 1 : L ≥ 1 pour tous), le LP ne commande plus jamais ce produit. Les scénarios "
        "correspondants mesurent cet effet, pas un défaut de simulation.", "",
        lecture, "", "Source : `sensitivity.csv`.", ""])


def section_repro() -> str:
    env = _read_json("environment.json")
    rt = _read_csv("runtime_summary.csv")
    if env is None or rt is None:
        return "## Reproductibilité\n\n_Benchmark non exécuté._\n"
    tot = rt[rt["module"] == "TOTAL_pipeline_6_modules"].iloc[0]
    rows = [[r["module"], f(r["seconds_mean"], 1), f(r["seconds_sd"], 1), f(r["peak_memory_mb_max"], 0),
             int(r["failures"])] for _, r in rt.iterrows()]
    return "\n".join([
        "## Reproductibilité", "",
        f"Machine : {env['cpu_model']}, {env['cpu_physical_cores']} cœurs physiques / {env['cpu_logical_cores']} "
        f"logiques, {f(env['ram_total_gb'], 1)} Go ; {env['os']} ; Python {env['python']} ; CBC {env['cbc_version']} ; "
        f"commit {env['git_commit']} (copie de travail modifiée : {env['git_worktree_dirty']}).", "",
        table(["Module", "Temps moyen (s)", "Écart-type", "Pic mémoire (Mo)", "Échecs"], rows), "",
        f"Publié : 213 s (somme codée en dur dans `scripts/run_evaluation.py:63-72`, EDA comprise) → mesuré : "
        f"{f(tot['seconds_mean'], 1)} ± {f(tot['seconds_sd'], 1)} s pour les 6 modules du pipeline (5 exécutions ; "
        "EDA et évaluation mesurées à part ; tableaux de bord exclus). Publié : 17 dépendances → "
        f"{env['requirements_txt_direct_dependencies']} dans `requirements.txt` après la révision (21 au moment de "
        "l'article, plus `shap` importé sans être déclaré ; `shap` et `psutil` ont été ajoutés), dépendances "
        f"importées mais non déclarées : {', '.join(env['undeclared_but_imported']) or 'aucune'}. Publié : « < 4 Go » "
        f"(jamais mesuré) → pic mesuré {f(tot['peak_memory_mb_max'], 0)} Mo par processus pour le pipeline "
        f"({f(rt['peak_memory_mb_max'].max(), 0)} Mo tous scripts confondus, EDA comprise). Les temps varient "
        f"fortement d'une exécution à l'autre (min {f(tot['seconds_min'], 0)} s, max {f(tot['seconds_max'], 0)} s) "
        "sur ce portable : publier la moyenne, l'écart-type et la machine.", "",
        "Sources : `environment.json`, `runtime.csv`, `runtime_summary.csv`.", ""])


NOT_DONE = """\
## Expériences non réalisées ou partielles

- **Correction de la demande censurée** : non réalisée. Les données ne contiennent que des
  ventes ; une rupture sans vente ne laisse aucune trace et les réceptions ne sont pas
  enregistrées. Un estimateur (p. ex. EM sur périodes à stock nul) supposerait des
  hypothèses non vérifiables ici. Limite à déclarer.
- **MOQ, volumes unitaires, commandes en transit réelles** : absents des données ; le MILP
  utilise des hypothèses explicites (MOQ `none` / `1_month`, 0,01 m³, ρ = 0 à l'instance de
  l'article). Le backtest, lui, suit les commandes en transit qu'il a lui-même passées.
- **Taux de change USD/CDF** : aucune donnée ni aucun paramètre de change dans le code ; la
  sensibilité repose sur une hypothèse (prix de vente en CDF, coûts en USD).
- **Sensibilité au seuil d'obsolescence selon la méthode de l'article** : non calculable, les
  prévisions de déploiement (`previsions_complet.csv`) sont figées pour le seuil de 180 j ;
  calculée dans le backtest.
- **LSTM** : le dépôt n'en contient pas (MLP scikit-learn à la place, cf. `src/forecasting.py`),
  et ce MLP retombe toujours sur le seasonal naive avec une fenêtre de 30 mois ; seule la
  variante à fenêtre de 12 mois est évaluée.
- **Puissance statistique des tests de politique (H4/H5)** : la période de test ne compte que
  7 mois ; le bootstrap par blocs de mois repose sur 7 blocs (1 mois) ou 3 blocs (3 mois) et
  reste indicatif. Allonger l'historique est la seule correction possible.
- **Délai d'alerte d'obsolescence** : calculé sur les seuls débuts de dormance observables dans
  la fenêtre (cf. `n_onsets` dans `obsolescence_eval.csv`) ; petit échantillon.
- **Temps par origine du rolling-origin** (`rolling_origin_meta.json`) : mesurés avec 7 processus
  en concurrence, non représentatifs ; les temps de référence sont ceux de `runtime.csv`.
"""


def run() -> Path:
    parts = ["# Résultats pour la révision de l'article", "",
             "Fichier généré par `python -m experiments.report` à partir de `paper_results/` "
             "(ne pas éditer à la main). Les valeurs « publiées » sont celles de l'article soumis.", "",
             section_verifications(), PROTOCOL, section_forecast(), section_policies(), section_hypotheses(),
             section_ablation(), section_milp(), section_service(), section_sensitivity(), section_repro(), NOT_DONE]
    path = PAPER_DIR / "RESULTS_FOR_PAPER.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> int:
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
