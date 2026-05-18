"""Préparation des datasets pour les tableaux de bord Power BI (Étape 9).

Construit une étoile classique :

::

                       dim_temps
                          │
   dim_magasins ─┐        ▼        ┌─ dim_produits
                 └── fact_ventes ──┘
                          │
                   dim_clients
                          │
               (jointures previsions, commandes,
                alertes via produit_id / date)

Chaque table renvoyée est un :class:`pandas.DataFrame` prêt à être exporté en
CSV (encodage UTF-8, séparateur virgule, dates ISO).
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# --------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------- #
def build_dim_temps(date_min: str | pd.Timestamp, date_max: str | pd.Timestamp) -> pd.DataFrame:
    """Calendrier complet entre ``date_min`` et ``date_max`` (clé : date)."""
    dates = pd.date_range(date_min, date_max, freq="D")
    df = pd.DataFrame({"date": dates})
    df["annee"] = df["date"].dt.year
    df["mois"] = df["date"].dt.month
    df["mois_nom"] = df["date"].dt.strftime("%B")
    df["trimestre"] = df["date"].dt.quarter
    df["semaine"] = df["date"].dt.isocalendar().week.astype(int)
    df["jour_semaine"] = df["date"].dt.dayofweek
    df["jour_nom"] = df["date"].dt.day_name()
    df["jour_mois"] = df["date"].dt.day
    df["est_weekend"] = (df["jour_semaine"] >= 5).astype(int)
    df["periode_yyyymm"] = df["date"].dt.strftime("%Y-%m")
    df["est_rentree_scolaire"] = df["mois"].isin([8, 9]).astype(int)
    df["est_rentree_academique"] = df["mois"].isin([10, 11]).astype(int)
    df["est_periode_pic_b2b"] = df["mois"].isin([1, 2, 3, 11, 12]).astype(int)
    return df


def build_dim_produits(
    catalogue: pd.DataFrame,
    classification: pd.DataFrame,
    obsolescence_features: pd.DataFrame,
    obsoletes_ids: set[str],
) -> pd.DataFrame:
    """Dimension produits enrichie (ABC, XYZ, cluster, statut obsolescence)."""
    df = catalogue.copy()
    df = df.merge(
        classification[
            [
                "produit_id", "classe_abc", "classe_xyz", "classe_abc_xyz",
                "cluster_kmeans", "libelle_cluster", "ca_total_36mois",
                "ventes_totales_36mois", "coefficient_variation",
            ]
        ],
        on="produit_id", how="left",
    )
    df = df.merge(
        obsolescence_features[
            [
                "produit_id", "jours_depuis_derniere_vente",
                "nombre_mois_consecutifs_sans_vente", "valeur_stock_dormant",
            ]
        ],
        on="produit_id", how="left",
    )
    df["a_risque_obsolescence"] = df["produit_id"].isin(obsoletes_ids).astype(int)
    df["statut"] = df["a_risque_obsolescence"].map({0: "Actif", 1: "À risque"})
    return df


def build_dim_clients(transactions: pd.DataFrame) -> pd.DataFrame:
    """Dimension clients : type, ville, CA total, nb transactions, dates extrêmes."""
    df = (
        transactions
        .groupby(["client_id", "client_nom"], dropna=False)
        .agg(
            type_client=("type_client", lambda x: x.mode().iloc[0] if not x.mode().empty else "Inconnu"),
            ville=("ville", lambda x: x.mode().iloc[0] if not x.mode().empty else "Inconnue"),
            ca_total_usd=("montant_total", "sum"),
            nb_transactions=("transaction_id", "count"),
            date_premiere_visite=("date", "min"),
            date_derniere_visite=("date", "max"),
        )
        .reset_index()
    )
    df["fidele"] = (df["nb_transactions"] >= 3).astype(int)
    return df


def build_dim_magasins(transactions: pd.DataFrame) -> pd.DataFrame:
    """Dimension magasins : ville, part du CA total, transactions."""
    ca_total = transactions["montant_total"].sum()
    df = (
        transactions
        .groupby(["magasin", "ville"])
        .agg(
            ca_total_usd=("montant_total", "sum"),
            nb_transactions=("transaction_id", "count"),
            nb_clients=("client_id", "nunique"),
        )
        .reset_index()
    )
    df["part_ca_pct"] = (df["ca_total_usd"] / ca_total * 100).round(2)
    df = df.sort_values("ca_total_usd", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------- #
# Table de faits
# --------------------------------------------------------------------- #
def build_fact_ventes(transactions: pd.DataFrame) -> pd.DataFrame:
    """Table de faits — 1 ligne = 1 transaction enrichie.

    Conserve toutes les clés étrangères (date, produit_id, client_id, magasin)
    plus les mesures (quantité, montant, marge, rupture).
    """
    df = transactions.copy()
    df["marge_unitaire"] = df["prix_vente_unitaire"] - df["cout_achat_unitaire"]
    df["benefice_transaction"] = df["marge_unitaire"] * df["quantite_vendue"]
    cols = [
        "transaction_id", "date", "produit_id", "magasin", "ville", "client_id",
        "type_client", "mode_paiement", "quantite_vendue", "prix_vente_unitaire",
        "cout_achat_unitaire", "marge_unitaire", "benefice_transaction",
        "montant_total", "stock_apres_vente",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


# --------------------------------------------------------------------- #
# Tables de mesures secondaires
# --------------------------------------------------------------------- #
def build_previsions_powerbi(previsions: pd.DataFrame) -> pd.DataFrame:
    """Renomme pour cohérence avec le datamodel Power BI."""
    return previsions.rename(columns={"prevision": "qte_prevue"})


def build_commandes_powerbi(commandes: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les colonnes utiles + types propres."""
    out = commandes.copy()
    if "date_decision" in out.columns:
        out["date_decision"] = pd.to_datetime(out["date_decision"])
    return out


def build_alertes_powerbi(obsoletes: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Liste des alertes obsolescence prête à être consommée par Power BI."""
    out = obsoletes.copy()
    if "produit_nom" not in out.columns:
        out = out.merge(catalogue[["produit_id", "produit_nom"]], on="produit_id", how="left")
    severity = pd.cut(
        out["valeur_stock_dormant"].fillna(0),
        bins=[-1, 100, 500, np.inf],
        labels=["Faible", "Modérée", "Élevée"],
    )
    out["severite"] = severity.astype(str)
    return out
