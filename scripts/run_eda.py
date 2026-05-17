"""Exploration des données (EDA) — Étape 1 du pipeline Zenith.

Ce script reproduit, en mode standalone, le contenu du notebook
`notebooks/01_exploration.ipynb`. Il génère toutes les figures et tables
attendues pour le Chapitre 4 du mémoire.

Sorties :
- outputs/figures/eda_*.png      (15+ figures)
- outputs/tables/eda_summary.csv (statistiques clés)
- outputs/tables/eda_topN_*.csv  (tops produits / clients)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "raw" / "zenith_dataset_brut.csv"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eda")

sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("deep")
ZENITH_COLORS = {"primary": "#1f4e79", "accent": "#ff6b6b", "neutral": "#6c757d"}


# -------------------------------------------------------------------- #
# 1. Chargement & typage
# -------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    dtypes = {
        "transaction_id": "string",
        "magasin": "string",
        "ville": "string",
        "produit_id": "string",
        "produit_nom": "string",
        "famille": "string",
        "marque": "string",
        "origine_fournisseur": "string",
        "client_id": "string",
        "client_nom": "string",
        "type_client": "string",
        "mode_paiement": "string",
    }
    df = pd.read_csv(RAW, dtype=dtypes, low_memory=False)
    # Date parsing tolérant aux formats mixtes
    iso = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    mask = iso.isna() & df["date"].notna()
    if mask.any():
        alt = pd.to_datetime(df.loc[mask, "date"], format="%d/%m/%Y", errors="coerce")
        iso.loc[mask] = alt
    df["date"] = iso
    return df


# -------------------------------------------------------------------- #
# 2. Statistiques générales
# -------------------------------------------------------------------- #
def general_stats(df: pd.DataFrame) -> dict:
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    dup_pct = round(df.duplicated().sum() / len(df) * 100, 2)
    stats = {
        "nb_lignes": len(df),
        "nb_colonnes": df.shape[1],
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "nb_mois_couverts": int((df["date"].max() - df["date"].min()).days // 30),
        "nb_produits": df["produit_id"].nunique(),
        "nb_clients": df["client_id"].nunique(),
        "nb_magasins": df["magasin"].nunique(),
        "nb_villes": df["ville"].nunique(),
        "nb_familles": df["famille"].nunique(),
        "doublons_pct": dup_pct,
    }
    for col, pct in missing_pct.items():
        stats[f"missing_{col}_pct"] = float(pct)
    return stats


# -------------------------------------------------------------------- #
# 3. Distributions
# -------------------------------------------------------------------- #
def fig_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(df["prix_vente_unitaire"].dropna(), bins=60, color=PALETTE[0], edgecolor="black")
    axes[0].set_yscale("log")
    axes[0].set_title("Distribution prix_vente_unitaire (log)")
    axes[0].set_xlabel("USD")

    axes[1].hist(df["quantite_vendue"].dropna(), bins=40, color=PALETTE[1], edgecolor="black")
    axes[1].set_title("Distribution quantite_vendue")
    axes[1].set_xlabel("Unités")

    axes[2].hist(df["montant_total"].dropna(), bins=60, color=PALETTE[2], edgecolor="black")
    axes[2].set_yscale("log")
    axes[2].set_title("Distribution montant_total (log)")
    axes[2].set_xlabel("USD")
    plt.tight_layout()
    plt.savefig(FIG / "eda_01_distributions_numeriques.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_categoricals(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    cols = ["magasin", "ville", "famille", "marque", "type_client", "mode_paiement"]
    for ax, c in zip(axes.flatten(), cols):
        s = df[c].value_counts().head(15)
        sns.barplot(x=s.values, y=s.index, hue=s.index, ax=ax, palette="viridis", legend=False)
        ax.set_title(f"Top modalités — {c}")
        ax.set_xlabel("Transactions")
    plt.tight_layout()
    plt.savefig(FIG / "eda_02_distributions_categorielles.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_outliers(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=df, x="famille", y="prix_vente_unitaire", ax=axes[0], showfliers=True)
    axes[0].set_yscale("log")
    axes[0].set_title("Boxplot prix par famille — outliers visibles")
    axes[0].tick_params(axis="x", rotation=70)

    sns.boxplot(data=df, x="famille", y="quantite_vendue", ax=axes[1], showfliers=True)
    axes[1].set_title("Boxplot quantité par famille")
    axes[1].tick_params(axis="x", rotation=70)
    plt.tight_layout()
    plt.savefig(FIG / "eda_03_outliers_boxplots.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- #
# 4. Analyse temporelle
# -------------------------------------------------------------------- #
def fig_temporal(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["date", "montant_total"]).copy()
    df["annee"] = df["date"].dt.year
    df["mois"] = df["date"].dt.month
    df["mois_periode"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["jour_semaine"] = df["date"].dt.day_name()

    # CA mensuel global
    ca_mensuel = df.groupby("mois_periode")["montant_total"].sum()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ca_mensuel.index, ca_mensuel.values, color=ZENITH_COLORS["primary"], linewidth=2)
    ax.fill_between(ca_mensuel.index, ca_mensuel.values, alpha=0.2, color=ZENITH_COLORS["primary"])
    ax.set_title("Évolution mensuelle du chiffre d'affaires global")
    ax.set_ylabel("CA (USD)")
    ax.set_xlabel("Mois")
    plt.tight_layout()
    plt.savefig(FIG / "eda_04_ca_mensuel_global.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # CA mensuel par magasin
    ca_mag = df.groupby(["mois_periode", "magasin"])["montant_total"].sum().unstack()
    fig, ax = plt.subplots(figsize=(15, 6))
    ca_mag.plot(ax=ax, linewidth=1.8)
    ax.set_title("Évolution mensuelle du CA par magasin")
    ax.set_ylabel("CA (USD)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "eda_05_ca_mensuel_par_magasin.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Heatmap saisonnalité mois × année
    pivot = df.groupby(["annee", "mois"])["montant_total"].sum().unstack()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "CA USD"})
    ax.set_title("Saisonnalité du CA (mois × année)")
    plt.tight_layout()
    plt.savefig(FIG / "eda_06_saisonnalite_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Distribution par jour de la semaine
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    js = df.groupby("jour_semaine")["montant_total"].sum().reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=js.index, y=js.values, hue=js.index, ax=ax, palette="crest", legend=False)
    ax.set_title("CA total par jour de la semaine")
    ax.set_ylabel("CA (USD)")
    plt.tight_layout()
    plt.savefig(FIG / "eda_07_ca_par_jour_semaine.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Dynamique Lomami vs Mobutu 2 par année
    dyn = (
        df[df["magasin"].isin(["Lomami", "Mobutu 2"])]
        .groupby([df["date"].dt.year.rename("annee"), "magasin"])["montant_total"]
        .sum()
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    dyn.plot(kind="bar", ax=ax, color=[ZENITH_COLORS["accent"], ZENITH_COLORS["primary"]])
    ax.set_title("Bascule du leadership : Lomami vs Mobutu 2 par année")
    ax.set_ylabel("CA (USD)")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "eda_08_dynamique_lomami_mobutu2.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "ca_mensuel_moy": float(ca_mensuel.mean()),
        "ca_mensuel_max_mois": str(ca_mensuel.idxmax().date()),
        "ca_mensuel_min_mois": str(ca_mensuel.idxmin().date()),
        "jour_semaine_top": str(js.idxmax()),
    }


# -------------------------------------------------------------------- #
# 5. Analyse client
# -------------------------------------------------------------------- #
def fig_clients(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["type_client", "montant_total", "client_id"]).copy()

    type_ca = df.groupby("type_client").agg(
        ca=("montant_total", "sum"),
        n_transactions=("transaction_id", "count"),
        n_clients=("client_id", "nunique"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    type_ca["ca"].plot(kind="bar", ax=axes[0], color=[ZENITH_COLORS["primary"], ZENITH_COLORS["accent"]])
    axes[0].set_title("CA total par type de client")
    axes[0].set_ylabel("CA (USD)")
    axes[0].tick_params(axis="x", rotation=0)
    type_ca["n_transactions"].plot(kind="bar", ax=axes[1], color=[ZENITH_COLORS["primary"], ZENITH_COLORS["accent"]])
    axes[1].set_title("Nombre de transactions par type de client")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "eda_09_clients_par_type.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Top 20 clients B2B
    b2b = df[df["type_client"] == "Entreprise"]
    top20 = (
        b2b.groupby("client_nom")["montant_total"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=top20.values, y=top20.index, hue=top20.index, ax=ax, palette="rocket", legend=False)
    ax.set_title("Top 20 clients B2B par CA")
    ax.set_xlabel("CA total (USD)")
    plt.tight_layout()
    plt.savefig(FIG / "eda_10_top20_clients_b2b.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    top20.to_csv(TAB / "eda_top20_clients_b2b.csv", header=["ca_usd"])
    return {"part_ca_b2b": float(type_ca.loc["Entreprise", "ca"] / type_ca["ca"].sum() * 100)}


# -------------------------------------------------------------------- #
# 6. Analyse produit
# -------------------------------------------------------------------- #
def fig_products(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["produit_id", "montant_total"]).copy()

    # Top 20 par CA
    top20 = (
        df.groupby(["produit_id", "produit_nom"])["montant_total"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    labels = [f"{a} — {b[:35]}" for a, b in top20.index]
    sns.barplot(x=top20.values, y=labels, hue=labels, ax=ax, palette="flare", legend=False)
    ax.set_title("Top 20 produits par chiffre d'affaires")
    ax.set_xlabel("CA total (USD)")
    plt.tight_layout()
    plt.savefig(FIG / "eda_11_top20_produits_ca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    top20.to_csv(TAB / "eda_top20_produits.csv")

    # Distribution du nombre de ventes par produit
    n_tx = df.groupby("produit_id")["transaction_id"].count()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(n_tx.values, bins=40, color=PALETTE[4], edgecolor="black")
    ax.axvline(n_tx.median(), color="red", linestyle="--", label=f"Médiane = {int(n_tx.median())}")
    ax.set_title("Distribution du nombre de transactions par produit")
    ax.set_xlabel("Nombre de transactions")
    ax.set_ylabel("Nombre de produits")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "eda_12_distribution_transactions_par_produit.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Saisonnalité par famille (CA mensuel)
    fam_mois = (
        df.assign(mois=df["date"].dt.month)
        .groupby(["famille", "mois"])["montant_total"]
        .sum()
        .unstack()
        .fillna(0)
    )
    # Normaliser par ligne pour visualiser le profil saisonnier
    fam_norm = fam_mois.div(fam_mois.sum(axis=1), axis=0)
    # Top 12 familles par CA
    top_fam = fam_mois.sum(axis=1).sort_values(ascending=False).head(12).index
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(fam_norm.loc[top_fam], cmap="YlGnBu", annot=False, cbar_kws={"label": "Part du CA annuel"})
    ax.set_title("Profil saisonnier des 12 familles principales")
    ax.set_xlabel("Mois de l'année")
    plt.tight_layout()
    plt.savefig(FIG / "eda_13_saisonnalite_familles.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Boxplot des marges par famille
    marges = df.assign(marge_pct=(df["prix_vente_unitaire"] - df["cout_achat_unitaire"]) / df["prix_vente_unitaire"] * 100)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=marges, x="famille", y="marge_pct", hue="famille", ax=ax, palette="Set3", showfliers=False, legend=False)
    ax.set_title("Taux de marge par famille (%) — fliers exclus")
    ax.tick_params(axis="x", rotation=70)
    ax.set_ylabel("Taux de marge (%)")
    plt.tight_layout()
    plt.savefig(FIG / "eda_14_marges_par_famille.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "nb_produits_actifs": int((n_tx > 0).sum()),
        "transactions_median_par_produit": int(n_tx.median()),
        "transactions_max_par_produit": int(n_tx.max()),
    }


# -------------------------------------------------------------------- #
# 7. Hot-spots qualité données (imperfections)
# -------------------------------------------------------------------- #
def fig_data_quality(df: pd.DataFrame) -> dict:
    missing = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(x=missing.values, y=missing.index, hue=missing.index, ax=ax, palette="rocket_r", legend=False)
    ax.set_title("Pourcentage de valeurs manquantes par colonne")
    ax.set_xlabel("% de NaN")
    plt.tight_layout()
    plt.savefig(FIG / "eda_15_valeurs_manquantes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Prix aberrants (>5× médiane famille)
    med = df.groupby("famille")["prix_vente_unitaire"].transform("median")
    n_aberrants = int(((df["prix_vente_unitaire"] > 5 * med) & med.notna()).sum())
    n_neg = int((df["quantite_vendue"] <= 0).sum())
    n_dup = int(df.duplicated().sum())

    # Date formats: comptage des dates non-ISO
    raw_dates = pd.read_csv(RAW, usecols=["date"])["date"].dropna().astype(str)
    n_iso = int(raw_dates.str.match(r"^\d{4}-\d{2}-\d{2}$").sum())
    n_eu = int(raw_dates.str.match(r"^\d{2}/\d{2}/\d{4}$").sum())

    summary = {
        "doublons_exacts": n_dup,
        "prix_aberrants_x5_median": n_aberrants,
        "quantites_negatives_nulles": n_neg,
        "dates_format_iso": n_iso,
        "dates_format_dd_mm_yyyy": n_eu,
    }
    return summary


# -------------------------------------------------------------------- #
# Orchestration
# -------------------------------------------------------------------- #
def main() -> None:
    logger.info("EDA — chargement du dataset brut")
    df = load_raw()
    logger.info("Dataset chargé : %d lignes × %d colonnes", *df.shape)

    stats = general_stats(df)
    logger.info("Période : %s → %s", stats["date_min"], stats["date_max"])

    fig_distributions(df)
    fig_categoricals(df)
    fig_outliers(df)
    temporal = fig_temporal(df)
    clients = fig_clients(df)
    products = fig_products(df)
    quality = fig_data_quality(df)

    summary = {**stats, **temporal, **clients, **products, **quality}
    # Export CSV
    pd.Series(summary).rename("valeur").to_csv(TAB / "eda_summary.csv", header=True)
    logger.info("Synthèse écrite : %s", (TAB / "eda_summary.csv").relative_to(ROOT))

    # Compte des figures
    n_figs = len(list(FIG.glob("eda_*.png")))
    logger.info("Figures générées : %d", n_figs)


if __name__ == "__main__":
    main()
