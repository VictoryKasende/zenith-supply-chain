"""Exécute l'Étape 3 — Classification ABC × XYZ × K-Means.

Lit ``data/features/product_features.csv`` (Étape 2) et produit :
- outputs/tables/classification_produits.csv : produit_id, classe_abc,
  classe_xyz, classe_abc_xyz, cluster_kmeans, libelle_cluster, ...
- outputs/tables/abc_xyz_matrix.csv : matrice croisée 3 × 3.
- outputs/tables/kmeans_diagnostics.csv : inertie + silhouette par k.
- outputs/tables/cluster_profile.csv : médianes par cluster.
- outputs/figures/cls_01..07.png : Pareto, distribution ABC, XYZ, heatmap
  ABC×XYZ, coude, silhouette, PCA 2D, profils clusters.
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

from src.classification import abc_xyz_matrix, classify_pipeline
from src.utils import FEATURES_DIR, FIG_DIR, TAB_DIR, setup_logger

logger = setup_logger("pipeline.classification")
sns.set_theme(style="whitegrid", context="talk")

ZENITH = {"A": "#1f4e79", "B": "#ff9f1c", "C": "#6c757d"}


def fig_pareto(df: pd.DataFrame) -> None:
    s = df.sort_values("ca_total_36mois", ascending=False).reset_index(drop=True)
    total = s["ca_total_36mois"].sum()
    cum = s["ca_total_36mois"].cumsum() / total * 100
    colors = s["classe_abc"].map(ZENITH).tolist()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(s)), s["ca_total_36mois"], color=colors, edgecolor="black", linewidth=0.2)
    ax2 = ax.twinx()
    ax2.plot(range(len(s)), cum, color="black", linewidth=2)
    ax2.axhline(70, color="green", linestyle="--", label="Seuil A (70 %)")
    ax2.axhline(90, color="orange", linestyle="--", label="Seuil B (90 %)")
    ax2.set_ylim(0, 105); ax2.set_ylabel("CA cumulé (%)")
    ax.set_xlabel("Produits triés par CA décroissant"); ax.set_ylabel("CA produit (USD)")
    ax.set_title("Courbe de Pareto — Classification ABC")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v, label=f"Classe {k}") for k, v in ZENITH.items()]
    ax.legend(handles=handles, loc="upper right"); ax2.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cls_01_pareto_abc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_xyz_distribution(df: pd.DataFrame) -> None:
    counts = df["classe_xyz"].value_counts().reindex(["X", "Y", "Z"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette={"X": "#2d6a4f", "Y": "#ff9f1c", "Z": "#ef476f"}, ax=ax, legend=False)
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=14)
    ax.set_title("Répartition XYZ — variabilité de la demande")
    ax.set_ylabel("Nombre de produits"); ax.set_xlabel("Classe XYZ")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cls_02_distribution_xyz.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_abc_xyz_heatmap(df: pd.DataFrame) -> None:
    nb = df.groupby(["classe_abc", "classe_xyz"]).size().unstack(fill_value=0).reindex(
        index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0
    )
    ca = df.groupby(["classe_abc", "classe_xyz"])["ca_total_36mois"].sum().unstack(fill_value=0).reindex(
        index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0
    )
    ca_pct = (ca / ca.values.sum() * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(nb, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar_kws={"label": "Produits"})
    axes[0].set_title("Matrice ABC × XYZ — nombre de produits")
    sns.heatmap(ca_pct, annot=True, fmt=".1f", cmap="OrRd", ax=axes[1], cbar_kws={"label": "% du CA total"})
    axes[1].set_title("Matrice ABC × XYZ — part du CA (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cls_03_matrice_abc_xyz.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_elbow_silhouette(diag) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ks = sorted(diag.inertia.keys())
    axes[0].plot(ks, [diag.inertia[k] for k in ks], marker="o", linewidth=2, color="#1f4e79")
    axes[0].axvline(diag.best_k, ls="--", color="red", label=f"k* = {diag.best_k}")
    axes[0].set_title("Méthode du coude — inertie intra-cluster")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertie"); axes[0].legend()

    axes[1].plot(ks, [diag.silhouette[k] for k in ks], marker="o", linewidth=2, color="#ff6b6b")
    axes[1].axvline(diag.best_k, ls="--", color="red", label=f"k* = {diag.best_k} ({diag.best_silhouette:.3f})")
    axes[1].set_title("Score de silhouette")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette moyenne"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cls_04_elbow_silhouette.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_pca(df: pd.DataFrame, diag) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    palette = sns.color_palette("tab10", n_colors=diag.best_k)
    for cid in sorted(df["cluster_kmeans"].unique()):
        mask = df["cluster_kmeans"] == cid
        label = df.loc[mask, "libelle_cluster"].iloc[0]
        ax.scatter(
            diag.pca_components[mask.values, 0],
            diag.pca_components[mask.values, 1],
            color=palette[int(cid) % len(palette)],
            label=f"{int(cid)} — {label}",
            alpha=0.7, s=50, edgecolors="black", linewidth=0.3,
        )
    ax.set_xlabel("Composante principale 1"); ax.set_ylabel("Composante principale 2")
    ax.set_title("Projection PCA 2D des clusters K-Means")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cls_05_pca_2d.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_cluster_profile(diag) -> None:
    # Normalisation min-max par feature pour mettre tous les indicateurs à la même échelle
    p = diag.cluster_profile.copy()
    p_norm = (p - p.min()) / (p.max() - p.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(p_norm, annot=p.round(1), fmt="g", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Valeur normalisée"})
    ax.set_title("Profil médian des clusters (annotations = valeurs brutes)")
    ax.set_xlabel("Feature"); ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cls_06_cluster_profile.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_cluster_distribution(df: pd.DataFrame) -> None:
    counts = df["libelle_cluster"].value_counts()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="viridis", ax=ax, legend=False)
    for i, (lbl, v) in enumerate(counts.items()):
        ax.text(v, i, f" {v}", va="center", fontsize=12)
    ax.set_title("Répartition des produits par libellé de cluster")
    ax.set_xlabel("Nombre de produits")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cls_07_distribution_clusters.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logger.info("Lancement de la classification Zenith")
    product_feats = pd.read_csv(FEATURES_DIR / "product_features.csv")

    final, diag = classify_pipeline(product_feats)

    # Tables
    cols = [
        "produit_id", "classe_abc", "classe_xyz", "classe_abc_xyz",
        "cluster_kmeans", "libelle_cluster",
        "ca_total_36mois", "ventes_totales_36mois", "coefficient_variation",
        "tendance_3_mois", "jours_depuis_derniere_vente",
    ]
    final[cols].to_csv(TAB_DIR / "classification_produits.csv", index=False)
    abc_xyz_matrix(final).to_csv(TAB_DIR / "abc_xyz_matrix.csv")
    pd.DataFrame({
        "k": list(diag.inertia.keys()),
        "inertia": list(diag.inertia.values()),
        "silhouette": list(diag.silhouette.values()),
    }).to_csv(TAB_DIR / "kmeans_diagnostics.csv", index=False)
    diag.cluster_profile.to_csv(TAB_DIR / "cluster_profile.csv")

    # Figures
    fig_pareto(final)
    fig_xyz_distribution(final)
    fig_abc_xyz_heatmap(final)
    fig_elbow_silhouette(diag)
    fig_pca(final, diag)
    fig_cluster_profile(diag)
    fig_cluster_distribution(final)

    # Récap console
    abc = final["classe_abc"].value_counts().reindex(["A", "B", "C"]).fillna(0).astype(int)
    xyz = final["classe_xyz"].value_counts().reindex(["X", "Y", "Z"]).fillna(0).astype(int)
    abc_xyz = final["classe_abc_xyz"].value_counts().sort_index()
    logger.info("ABC : %s", abc.to_dict())
    logger.info("XYZ : %s", xyz.to_dict())
    logger.info("ABC × XYZ : %s", abc_xyz.to_dict())
    logger.info("Libellés clusters :\n%s", final["libelle_cluster"].value_counts().to_string())
    logger.info("Sorties écrites dans outputs/tables/ et outputs/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
