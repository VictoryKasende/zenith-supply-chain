"""Exécute l'Étape 4 — Détection d'obsolescence.

Lit :
- ``data/processed/zenith_clean.csv``      (transactions nettoyées Étape 2)
- ``outputs/tables/classification_produits.csv`` (Étape 3 — pour croiser ABC)

Produit :
- ``outputs/tables/obsolescence_features.csv``  (features par produit)
- ``outputs/tables/produits_obsoletes.csv``     (liste détaillée des produits flagués)
- ``outputs/tables/obsolescence_sensitivity.csv`` (sensibilité contamination)
- ``outputs/figures/obs_01..06.png``            (6 figures)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.obsolescence import (
    DEFAULT_FEATURES,
    build_obsolescence_features,
    detect_obsolescence,
    sensitivity_analysis,
)
from src.utils import FIG_DIR, PROCESSED_DIR, TAB_DIR, setup_logger

logger = setup_logger("pipeline.obsolescence")
sns.set_theme(style="whitegrid", context="talk")


def fig_anomaly_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df["score_obsolescence"], bins=40, color="#1f4e79", edgecolor="black")
    threshold = df.loc[df["a_risque_obsolescence"] == 1, "score_obsolescence"].max()
    if pd.notna(threshold):
        ax.axvline(threshold, color="red", linestyle="--", label=f"Seuil de coupure ≈ {threshold:.3f}")
    ax.set_title("Distribution des scores d'anomalie (Isolation Forest)")
    ax.set_xlabel("Score d'obsolescence (plus bas = plus suspect)")
    ax.set_ylabel("Nombre de produits"); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "obs_01_distribution_score.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_feature_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = df["a_risque_obsolescence"].map({0: "tab:blue", 1: "tab:red"})
    ax.scatter(df["jours_depuis_derniere_vente"], df["ratio_ventes_3m_vs_12m"],
               c=colors, alpha=0.7, s=45, edgecolors="black", linewidth=0.3)
    ax.set_xlabel("Jours depuis la dernière vente")
    ax.set_ylabel("Ratio ventes 3 mois / 12 mois")
    ax.set_title("Plan jours_sans_vente × ratio 3m/12m — points rouges = à risque")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", label="Actif", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", label="À risque", markersize=10),
    ])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "obs_02_scatter_features.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_cross_abc(df: pd.DataFrame) -> None:
    cross = (
        df.groupby(["classe_abc", "a_risque_obsolescence"]).size().unstack(fill_value=0)
        .rename(columns={0: "actif", 1: "à_risque"})
        .reindex(["A", "B", "C"], fill_value=0)
    )
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cross.plot(kind="bar", stacked=True, ax=axes[0], color=["#1f4e79", "#ef476f"])
    axes[0].set_title("Croisement Isolation Forest × classe ABC (effectifs)"); axes[0].tick_params(axis="x", rotation=0)
    cross_pct.plot(kind="bar", stacked=True, ax=axes[1], color=["#1f4e79", "#ef476f"])
    axes[1].set_title("Croisement (%)"); axes[1].set_ylabel("% des produits"); axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "obs_03_cross_abc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(sens: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sens["contamination"] * 100, sens["n_flagged_iforest"], marker="o", linewidth=2, color="#1f4e79")
    for _, row in sens.iterrows():
        ax.text(row["contamination"] * 100, row["n_flagged_iforest"] + 1,
                f"{int(row['n_flagged_iforest'])} ({row['pct_catalogue']}%)",
                ha="center", fontsize=11)
    ax.set_xlabel("Contamination (%)"); ax.set_ylabel("Produits flagués")
    ax.set_title("Sensibilité d'Isolation Forest au paramètre contamination")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "obs_04_sensitivity_contamination.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_sales_curves(
    df: pd.DataFrame, transactions: pd.DataFrame, n: int = 8
) -> None:
    """Trace les courbes de ventes mensuelles des n produits flagués les plus dormants."""
    flagged = (
        df[df["a_risque_obsolescence"] == 1]
        .sort_values("score_obsolescence")
        .head(n)
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=False)
    axes = axes.flatten()
    for i, pid in enumerate(flagged["produit_id"]):
        ts = (
            transactions[transactions["produit_id"] == pid]
            .assign(mois=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
            .groupby("mois")["quantite_vendue"].sum()
        )
        ax = axes[i]
        ax.plot(ts.index, ts.values, marker="o", color="#ef476f", linewidth=1.5)
        ax.fill_between(ts.index, ts.values, alpha=0.2, color="#ef476f")
        ax.set_title(f"{pid} — score {flagged.iloc[i]['score_obsolescence']:.2f}")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Courbes de ventes mensuelles des 8 produits les plus suspects", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "obs_05_courbes_ventes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_top_dormant_value(df: pd.DataFrame) -> None:
    flagged = df[df["a_risque_obsolescence"] == 1].sort_values("valeur_stock_dormant", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=flagged["valeur_stock_dormant"], y=flagged["produit_id"],
                hue=flagged["produit_id"], palette="rocket", ax=ax, legend=False)
    ax.set_title("Top 15 produits dormants par valeur de stock immobilisée")
    ax.set_xlabel("Valeur stock dormant (USD)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "obs_06_top_dormant_value.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logger.info("Lancement de la détection d'obsolescence")
    transactions = pd.read_csv(PROCESSED_DIR / "zenith_clean.csv", parse_dates=["date"])
    classes = pd.read_csv(TAB_DIR / "classification_produits.csv")

    # 1. Features
    feats = build_obsolescence_features(transactions)
    # On enrichit avec l'âge produit pour la règle métier
    age = (
        transactions.groupby("produit_id")["date"].min()
        .pipe(lambda s: (transactions["date"].max() - s).dt.days)
        .rename("age_produit_jours")
    )
    feats = feats.merge(age.reset_index(), on="produit_id", how="left")
    feats.to_csv(TAB_DIR / "obsolescence_features.csv", index=False)

    # 2. Détection
    result = detect_obsolescence(feats)

    # 3. Croisement avec ABC
    df_full = result.df.merge(classes[["produit_id", "classe_abc", "classe_xyz", "libelle_cluster"]],
                              on="produit_id", how="left")

    # 4. Export liste détaillée
    flagged = df_full[df_full["a_risque_obsolescence"] == 1].copy()
    flagged = flagged.sort_values("valeur_stock_dormant", ascending=False)
    flagged.to_csv(TAB_DIR / "produits_obsoletes.csv", index=False)
    logger.info(
        "Produits à risque : %d / %d (%.1f %%)",
        len(flagged), len(df_full), 100 * len(flagged) / len(df_full),
    )

    # 5. Analyse de sensibilité
    sens = sensitivity_analysis(feats)
    sens.to_csv(TAB_DIR / "obsolescence_sensitivity.csv", index=False)
    logger.info("Sensibilité contamination :\n%s", sens.to_string(index=False))

    # 6. Figures
    fig_anomaly_distribution(df_full)
    fig_feature_scatter(df_full)
    fig_cross_abc(df_full)
    fig_sensitivity(sens)
    fig_sales_curves(df_full, transactions)
    fig_top_dormant_value(df_full)

    # 7. Récap croisement
    cross = (
        df_full.groupby(["classe_abc", "a_risque_obsolescence"]).size()
        .unstack(fill_value=0).rename(columns={0: "actif", 1: "a_risque"})
        .reindex(["A", "B", "C"], fill_value=0)
    )
    logger.info("Croisement obsolescence × ABC :\n%s", cross.to_string())
    logger.info(
        "Valeur totale du stock dormant : %.0f USD (%d produits)",
        flagged["valeur_stock_dormant"].sum(), len(flagged),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
