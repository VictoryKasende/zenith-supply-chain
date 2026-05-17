"""Visualisations pour le rapport / tableau de bord."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
PALETTE = sns.color_palette("deep")


def fig_abc_distribution(products: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = products["classe_abc"].value_counts().reindex(["A", "B", "C"])
    axes[0].bar(counts.index, counts.values, color=PALETTE[:3])
    axes[0].set_title("Répartition des produits par classe ABC")
    axes[0].set_ylabel("Nombre de produits")
    for x, y in zip(counts.index, counts.values):
        axes[0].text(x, y, str(int(y)), ha="center", va="bottom")

    # Pareto
    sorted_ = products.sort_values("ca_total_36mois", ascending=False)
    cum = sorted_["ca_total_36mois"].cumsum() / sorted_["ca_total_36mois"].sum()
    axes[1].plot(range(1, len(cum) + 1), cum.values * 100, color=PALETTE[3])
    axes[1].axhline(70, color="green", ls="--", label="Seuil A (70%)")
    axes[1].axhline(90, color="orange", ls="--", label="Seuil B (90%)")
    axes[1].set_xlabel("Produits triés par CA décroissant")
    axes[1].set_ylabel("CA cumulé (%)")
    axes[1].set_title("Courbe de Pareto")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_obsolescence(products: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    p = products.copy()
    ax.scatter(
        p["jours_depuis_derniere_vente"],
        p["ratio_ventes_3m_vs_12m"],
        c=p["a_risque_obsolescence"].map({0: "tab:blue", 1: "tab:red"}),
        alpha=0.6,
        s=35,
    )
    ax.set_xlabel("Jours depuis la dernière vente")
    ax.set_ylabel("Ratio ventes 3 mois / 12 mois")
    ax.set_title("Détection d'obsolescence — Isolation Forest")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", label="Actif", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", label="À risque", markersize=8),
    ])
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_kmeans(products: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    p = products.copy()
    palette = sns.color_palette("tab10", n_colors=int(p["cluster_kmeans"].nunique()))
    for cid, sub in p.groupby("cluster_kmeans"):
        label = sub["cluster_label"].iloc[0]
        ax.scatter(
            sub["ca_total_36mois"],
            sub["coefficient_variation"].clip(upper=3),
            label=f"{int(cid)} — {label}",
            color=palette[int(cid) % len(palette)],
            alpha=0.7,
            s=45,
        )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("CA total 36 mois (USD, log)")
    ax.set_ylabel("Coefficient de variation (clippé à 3)")
    ax.set_title("Clusters K-Means de produits")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_metrics_by_class(metrics_df: pd.DataFrame, out: Path) -> Path:
    if metrics_df.empty:
        return out
    fig, ax = plt.subplots(figsize=(9, 5))
    sub = metrics_df.dropna(subset=["mae"])
    sns.boxplot(
        data=sub,
        x="classe_abc",
        y="mae",
        hue="modele",
        order=["A", "B", "C"],
        ax=ax,
    )
    ax.set_title("MAE par classe ABC et par modèle de prévision")
    ax.set_ylabel("MAE (unités/mois)")
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_financial_comparison(compare_df: pd.DataFrame, out: Path) -> Path:
    if compare_df.empty:
        return out
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    x = np.arange(len(compare_df))
    ax.bar(x - width / 2, compare_df["politique_empirique"], width, label="Empirique", color=PALETTE[1])
    ax.bar(x + width / 2, compare_df["politique_optimisee"], width, label="Optimisée", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["indicateur"], rotation=25, ha="right")
    ax.set_title("Politique empirique vs optimisée")
    ax.set_ylabel("Valeur")
    ax.legend()
    ax.set_yscale("symlog", linthresh=10)
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
