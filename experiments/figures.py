"""Figures de l'article, tracées à partir des seuls fichiers de ``paper_results/``.

- ``fig_service_tradeoff.png`` : coût total vs fill rate (période de test) — frontière
  (s, S) selon z, pipeline LP selon le budget, et les politiques de P1-6.
- ``fig_obsolescence_pr.png`` : précision vs rappel (vérité prospective) pour
  l'Isolation Forest et l'hybride (balayage de contamination), la baseline
  d'inactivité (balayage du seuil), les règles métier et l'évaluation circulaire de
  l'article.

Palette : trois premières teintes de la palette de référence (bleu, orange, aqua),
validées toutes paires en mode clair ; tout le reste en gris neutre avec étiquette
directe, pour que l'identité ne repose jamais sur la couleur seule.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.config import PAPER_DIR

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _line(ax, x, y, color, label):
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label,
            marker="o", markersize=6, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)


def service_tradeoff() -> None:
    st = pd.read_csv(PAPER_DIR / "service_tradeoff.csv")
    pc = pd.read_csv(PAPER_DIR / "policy_comparison.csv")
    b = st[(st["period"] == "test") & (st["abc"] == "all")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200, facecolor=SURFACE)
    _style(ax)
    ss = b[b["sweep"] == "sS_z"].sort_values("unit_fill_rate")
    _line(ax, 100 * ss["unit_fill_rate"], ss["total_cost"] / 1000, S1, "(s, S), z unique de 0 à 3")
    bud = b[(b["sweep"] == "budget") & (b["value"] >= 200_000)].sort_values("value")
    _line(ax, 100 * bud["unit_fill_rate"], bud["total_cost"] / 1000, S2, r"Pipeline LP, budget 200 k\$ → 1 M\$ (identique dès 300 k\$)")
    t = pc[pc["period"] == "test"].set_index("policy")
    labels = {"incumbent": "incumbent", "sS_tuned": "(s,S) réglé", "order_up_to_tuned": "order-up-to réglé",
              "ma3_order_up_to_tuned": "MA3 + order-up-to", "best_baseline_lp": "MA3 + LP",
              "pipeline_no_obso_lp": "pipeline − obsolescence", "pipeline_no_class_lp": "pipeline − classification",
              "pipeline_lp": "pipeline complet", "pipeline_mlp12_lp": "pipeline MLP-12"}
    for p, lab in labels.items():
        if p not in t.index:
            continue
        x, y = 100 * t.loc[p, "unit_fill_rate"], t.loc[p, "total_cost"] / 1000
        ax.plot(x, y, marker="D", markersize=6, color=MUTED, markeredgecolor=SURFACE, markeredgewidth=2,
                linestyle="none", zorder=4)
        dx, dy, ha = (-6, -11, "right") if p == "ma3_order_up_to_tuned" else (5, 3, "left")
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK2, ha=ha)
    ax.plot([], [], marker="D", color=MUTED, linestyle="none", label="Politiques de P1-6")
    ax.set_xlabel("Fill rate en unités sur demande réalisée (%), test janv. → juil. 2025")
    ax.set_ylabel("Coût total sur la période de test (k$)")
    ax.set_title("Coût total vs taux de service", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig_service_tradeoff.png", facecolor=SURFACE)
    plt.close(fig)


def obsolescence_pr() -> None:
    ev = pd.read_csv(PAPER_DIR / "obsolescence_eval.csv")
    conf = json.loads((PAPER_DIR / "obsolescence_confusion.json").read_text(encoding="utf-8"))
    m = ev[(ev["population"] == "stock_pos") & (ev["label"] == "ventes_6m_nulles")]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200, facecolor=SURFACE)
    _style(ax)
    hyb = m[(m["detector"] == "hybrid") & (m["inactivity_days"] == 180)].sort_values("contamination")
    _line(ax, hyb["recall"], hyb["precision"], S1, "Hybride (actuel), contamination 0,05 → 0,20")
    iso = m[m["detector"] == "iforest"].sort_values("contamination")
    _line(ax, iso["recall"], iso["precision"], S2, "Isolation Forest seul, contamination 0,05 → 0,20")
    ina = m[m["detector"] == "inactivity"].sort_values("inactivity_days")
    _line(ax, ina["recall"], ina["precision"], S3, "Inactivité simple, seuil 90 → 365 j")
    for _, r in ina.iterrows():
        ax.annotate(f"{int(r['inactivity_days'])} j", (r["recall"], r["precision"]), xytext=(5, -10),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    rules = m[(m["detector"] == "rules") & (m["inactivity_days"] == 180)].iloc[0]
    ax.plot(rules["recall"], rules["precision"], marker="s", markersize=7, color=MUTED, markeredgecolor=SURFACE,
            markeredgewidth=2, linestyle="none", zorder=4)
    ax.annotate("règles métier seules", (rules["recall"], rules["precision"]), xytext=(-6, 8),
                textcoords="offset points", fontsize=7.5, color=INK2, ha="right")
    circ = conf["article_circular_reproduction"]["csv_order"]
    ax.plot(circ["recall"], circ["precision"], marker="*", markersize=11, color=INK2, linestyle="none", zorder=4)
    ax.annotate("article (vérité circulaire)", (circ["recall"], circ["precision"]), xytext=(-6, 6),
                textcoords="offset points", fontsize=7.5, color=INK2, ha="right")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Détection d'obsolescence, vérité prospective (6 mois)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig_obsolescence_pr.png", facecolor=SURFACE)
    plt.close(fig)


def run() -> None:
    if (PAPER_DIR / "service_tradeoff.csv").exists() and (PAPER_DIR / "policy_comparison.csv").exists():
        service_tradeoff()
    if (PAPER_DIR / "obsolescence_eval.csv").exists():
        obsolescence_pr()


if __name__ == "__main__":
    run()
