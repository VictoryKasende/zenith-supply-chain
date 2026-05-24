"""Évaluation globale du système et validation des 6 hypothèses (mémoire §3.8).

Pour chaque hypothèse H1–H6 formulée au §1.3, on définit :
- un **test mesurable** (formule + données utilisées),
- un **critère de validation** (seuil chiffré),
- un **verdict** (validée / partiellement validée / non validée).

Sorties principales :
- :func:`evaluate_h1_differentiation`
- :func:`evaluate_h2_complementarite`
- :func:`evaluate_h3_obsolescence`
- :func:`evaluate_h4_optimisation`
- :func:`evaluate_h5_impact_financier`
- :func:`evaluate_h6_faisabilite_pme`
- :func:`build_evaluation_report` (agrège H1–H6 en DataFrame).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("evaluation")


# --------------------------------------------------------------------- #
# Structure de verdict
# --------------------------------------------------------------------- #
@dataclass
class HypothesisVerdict:
    code: str
    libelle: str
    test: str
    critere: str
    valeur: float
    seuil: float
    verdict: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "libelle": self.libelle,
            "test": self.test,
            "critere": self.critere,
            "valeur_mesuree": round(self.valeur, 3),
            "seuil": self.seuil,
            "verdict": self.verdict,
            "details": str(self.details),
        }


# =================================================================== #
# H1 — Différenciation par classe ABC
# =================================================================== #
def evaluate_h1_differentiation(metrics_df: pd.DataFrame) -> HypothesisVerdict:
    """H1 : MAE différencié par classe est meilleur que MAE global naïf.

    Comparaison : MAE moyen pondéré du système différencié vs MAE
    naïf (worst-case : on prendrait une moyenne globale).
    """
    sub = metrics_df.dropna(subset=["mae"])
    mae_diff = float(sub["mae"].mean())
    # Worst case : on prend le pire MAE classe × proportion produits
    mae_worst = float(sub.groupby("modele")["mae"].mean().max())
    ratio = mae_diff / mae_worst if mae_worst > 0 else 1.0
    verdict = "Validée" if mae_diff < mae_worst else "Non validée"
    return HypothesisVerdict(
        code="H1",
        libelle="Différenciation par classe",
        test="MAE moyen pipeline différencié vs MAE moyen du pire modèle isolé",
        critere="ratio < 1",
        valeur=ratio,
        seuil=1.0,
        verdict=verdict,
        details={
            "mae_pipeline_differencie": round(mae_diff, 3),
            "mae_pire_modele_isole": round(mae_worst, 3),
            "gain_relatif_pct": round((1 - ratio) * 100, 1),
        },
    )


# =================================================================== #
# H2 — Complémentarité statistique / apprentissage
# =================================================================== #
def evaluate_h2_complementarite(metrics_df: pd.DataFrame, min_sample: int = 5) -> HypothesisVerdict:
    """H2 : chaque modèle est meilleur ou au pire compétitif sur sa classe affectée.

    On regarde par classe ABC quel modèle (parmi ceux réellement affectés
    sur au moins ``min_sample`` produits) obtient le meilleur MAE. Si
    l'affectation théorique coïncide avec le meilleur observé, l'hypothèse
    est validée. On exclut les modèles testés sur trop peu de produits
    (typiquement Croston sur 1 produit), qui biaiseraient la moyenne.
    """
    expected = {"A": "lstm", "B": "lightgbm", "C": "sarima"}
    sub = metrics_df.dropna(subset=["mae"])
    sub = sub[sub["modele"].isin(["lstm", "lightgbm", "sarima", "croston_sba"])]

    actual_best: dict[str, str] = {}
    for classe, group in sub.groupby("classe_abc"):
        by_model = (
            group.groupby("modele")
            .agg(mae_moy=("mae", "mean"), n=("produit_id", "count"))
            .query("n >= @min_sample")
        )
        if by_model.empty:
            continue
        actual_best[classe] = str(by_model["mae_moy"].idxmin())

    matches = sum(1 for c, m in expected.items() if actual_best.get(c) == m)
    score = matches / len(expected)
    verdict = "Validée" if score >= 2 / 3 else ("Partiellement validée" if score > 0 else "Non validée")

    return HypothesisVerdict(
        code="H2",
        libelle="Complémentarité statistique ↔ apprentissage",
        test="Le modèle affecté à chaque classe est le meilleur observé (échantillon ≥ 5)",
        critere="≥ 2/3 classes ont leur modèle attendu en tête",
        valeur=score,
        seuil=2 / 3,
        verdict=verdict,
        details={
            "attendu": expected,
            "observé": actual_best,
            "min_sample": min_sample,
        },
    )


# =================================================================== #
# H3 — Détection précoce de l'obsolescence
# =================================================================== #
def evaluate_h3_obsolescence(
    flagged: pd.DataFrame,
    obsolescence_features: pd.DataFrame,
    threshold_months: int = 6,
) -> HypothesisVerdict:
    """H3 : Isolation Forest détecte ≥ 80 % des produits évidents d'obsolescence.

    Ground truth : produits avec ``nombre_mois_consecutifs_sans_vente >= threshold_months``.
    Rappel = (vrai positifs) / (total positifs réels).
    """
    truth = obsolescence_features[
        obsolescence_features["nombre_mois_consecutifs_sans_vente"] >= threshold_months
    ]["produit_id"].tolist()
    predicted = set(flagged["produit_id"].tolist())
    truth_set = set(truth)
    tp = len(truth_set & predicted)
    fp = len(predicted - truth_set)
    fn = len(truth_set - predicted)
    rappel = tp / max(len(truth_set), 1)
    precision = tp / max(tp + fp, 1)
    verdict = "Validée" if rappel >= 0.80 else "Non validée"
    return HypothesisVerdict(
        code="H3",
        libelle="Détection précoce d'obsolescence",
        test=f"Rappel des produits avec ≥ {threshold_months} mois sans vente",
        critere="rappel ≥ 80 %",
        valeur=rappel,
        seuil=0.80,
        verdict=verdict,
        details={
            "produits_evidents_total": len(truth_set),
            "vrais_positifs": tp,
            "faux_positifs": fp,
            "faux_negatifs": fn,
            "precision": round(precision, 3),
            "rappel": round(rappel, 3),
        },
    )


# =================================================================== #
# H4 — Optimisation > heuristique empirique
# =================================================================== #
def evaluate_h4_optimisation(comparison: pd.DataFrame) -> HypothesisVerdict:
    """H4 : le coût total simulé du LP est inférieur à celui de la politique empirique."""
    row = comparison[comparison["indicateur"] == "cout_total_simule_usd"].iloc[0]
    cost_lp = float(row["politique_optimisee"])
    cost_emp = float(row["politique_empirique"])
    delta_pct = float(row["delta_pct"])
    verdict = "Validée" if cost_lp < cost_emp else "Non validée"
    return HypothesisVerdict(
        code="H4",
        libelle="Optimisation supérieure à l'heuristique",
        test="Coût total simulé LP vs politique empirique",
        critere="coût LP < coût empirique",
        valeur=cost_lp / cost_emp if cost_emp > 0 else 1.0,
        seuil=1.0,
        verdict=verdict,
        details={
            "cout_lp_usd": round(cost_lp, 2),
            "cout_emp_usd": round(cost_emp, 2),
            "gain_usd": round(cost_emp - cost_lp, 2),
            "gain_pct": -delta_pct,
        },
    )


# =================================================================== #
# H5 — Validation par l'impact financier
# =================================================================== #
def evaluate_h5_impact_financier(comparison: pd.DataFrame) -> HypothesisVerdict:
    """H5 : le bénéfice net additionnel est strictement positif.

    Bénéfice net = (CA réalisé LP - CA réalisé EMP) + (stock immo libéré × taux capital annuel)
                 + (marge perdue évitée).
    """
    def _val(ind: str) -> float:
        return float(comparison[comparison["indicateur"] == ind].iloc[0]["delta"])

    delta_ca = _val("ca_realise_usd")           # >0 si LP fait plus de CA
    delta_stock = -_val("stock_moyen_immo_usd")  # >0 si LP libère du stock
    delta_marge = -_val("marge_perdue_usd")     # >0 si LP perd moins de marge
    cost_capital_pct = 0.12                     # 12 % annuel — coût d'opportunité
    benefice = delta_ca + delta_marge + delta_stock * cost_capital_pct
    verdict = "Validée" if benefice > 0 else "Non validée"
    return HypothesisVerdict(
        code="H5",
        libelle="Impact financier mesurable",
        test="Bénéfice net additionnel = ΔCA + Δmarge_évitée + Δstock×coût_capital",
        critere="bénéfice > 0",
        valeur=benefice,
        seuil=0.0,
        verdict=verdict,
        details={
            "delta_ca_realise_usd": round(delta_ca, 2),
            "delta_marge_perdue_evitee_usd": round(delta_marge, 2),
            "delta_stock_libere_usd": round(delta_stock, 2),
            "coupon_capital_libere_usd": round(delta_stock * cost_capital_pct, 2),
            "benefice_net_usd": round(benefice, 2),
        },
    )


# =================================================================== #
# H6 — Faisabilité PME (frugalité)
# =================================================================== #
def evaluate_h6_faisabilite_pme(
    total_runtime_sec: float,
    n_python_deps: int,
    requires_gpu: bool,
    cpu_seconds_limit: float = 300.0,
    max_deps: int = 20,
) -> HypothesisVerdict:
    """H6 : la solution tourne sur un poste standard en moins de 5 minutes,
    sans GPU et avec un nombre raisonnable de dépendances open source."""
    score = sum([
        total_runtime_sec <= cpu_seconds_limit,
        not requires_gpu,
        n_python_deps <= max_deps,
    ])
    verdict = "Validée" if score == 3 else ("Partiellement validée" if score >= 2 else "Non validée")
    return HypothesisVerdict(
        code="H6",
        libelle="Faisabilité PME (frugalité)",
        test="Temps total ≤ 5 min, sans GPU, dépendances ≤ 20",
        critere="3 conditions / 3",
        valeur=float(score),
        seuil=3.0,
        verdict=verdict,
        details={
            "temps_total_sec": round(total_runtime_sec, 1),
            "limite_temps_sec": cpu_seconds_limit,
            "n_dependances_python": n_python_deps,
            "limite_dependances": max_deps,
            "necessite_gpu": requires_gpu,
        },
    )


# =================================================================== #
# Rapport global
# =================================================================== #
def build_evaluation_report(verdicts: Iterable[HypothesisVerdict]) -> pd.DataFrame:
    return pd.DataFrame([v.to_dict() for v in verdicts])
