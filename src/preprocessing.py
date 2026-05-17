"""Prétraitement et feature engineering du dataset Zenith.

Ce module reproduit fidèlement la méthodologie du chapitre 3 du mémoire :
§3.3.1 Nettoyage des données
§3.3.2 Feature engineering
§3.3.3 Partitionnement des données

Chaque étape de nettoyage et de construction de features est encapsulée dans
une fonction testable, qui renvoie le DataFrame transformé et alimente un
``ReportBuilder`` pour produire un rapport ligne-à-ligne avant/après.

Exemple d'utilisation
---------------------
>>> from src.preprocessing import preprocess_pipeline
>>> clean, features, report = preprocess_pipeline()
>>> clean.to_csv("data/processed/zenith_clean.csv", index=False)

Sorties produites par ``preprocess_pipeline`` :
- ``clean`` : DataFrame nettoyé (1 ligne = 1 transaction).
- ``features`` : DataFrame enrichi (1 ligne = 1 transaction + variables dérivées).
- ``report`` : DataFrame résumant l'évolution du dataset à chaque étape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import (
    RAW_TRANSACTIONS,
    RAW_CATALOGUE,
    coefficient_of_variation,
    linear_slope,
    load_catalogue,
    load_raw_transactions,
    parse_mixed_dates,
    setup_logger,
)

logger = setup_logger("preprocessing")

# --------------------------------------------------------------------- #
# Constantes de nettoyage
# --------------------------------------------------------------------- #
#: Correspondances pour normaliser les libellés famille (fautes de frappe
#: observées dans le dataset brut).
FAMILLE_FIXES: dict[str, str] = {
    "Cartouches": "Cartouche",
    "Accessoires": "Accessoire",
    "Imprimente": "Imprimante",
    "Ordinator": "Ordinateur",
    "Réseau": "Reseau",
}

#: Seuil multiplicatif sur la médiane famille pour détecter les prix aberrants.
PRICE_ABERRANT_FACTOR = 5.0

#: Seuil au-delà duquel on considère qu'un prix résulte d'une erreur de saisie
#: (typiquement une virgule oubliée → prix multiplié par 10).
PRICE_X10_FACTOR = 8.0

#: Tolérance relative sur la cohérence ``montant_total ≈ prix × quantité`` (1 %).
MONTANT_TOLERANCE = 0.01

#: Proportion maximale de valeurs manquantes par ligne avant suppression.
ROW_DROP_MISSING_RATIO = 0.5


# --------------------------------------------------------------------- #
# Construction du rapport de transformation
# --------------------------------------------------------------------- #
@dataclass
class ReportBuilder:
    """Accumulateur d'évolution du dataset à chaque étape de prétraitement."""

    rows: list[dict] = field(default_factory=list)

    def log(self, etape: str, df: pd.DataFrame, **extras) -> None:
        entry = {"etape": etape, "lignes": int(len(df)), "colonnes": int(df.shape[1])}
        entry.update(extras)
        self.rows.append(entry)
        logger.info(
            "[%s] %d lignes × %d colonnes %s",
            etape,
            entry["lignes"],
            entry["colonnes"],
            ", ".join(f"{k}={v}" for k, v in extras.items()) if extras else "",
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# =================================================================== #
# §3.3.1 — NETTOYAGE
# =================================================================== #

def parse_dates_iso(df: pd.DataFrame) -> pd.DataFrame:
    """1) Uniformise toutes les dates au format ISO ``YYYY-MM-DD``."""
    out = df.copy()
    out["date"] = parse_mixed_dates(out["date"])
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """2) Supprime les doublons exacts (toutes colonnes identiques)."""
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, n_before - len(out)


def normalize_famille(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """3) Corrige les fautes de frappe sur la colonne ``famille``."""
    out = df.copy()
    before = out["famille"].astype("string").copy()
    out["famille"] = before.str.strip().replace(FAMILLE_FIXES)
    n_fixes = int((before != out["famille"]).sum())
    return out, n_fixes


def fix_aberrant_prices(
    df: pd.DataFrame,
    factor: float = PRICE_ABERRANT_FACTOR,
    x10_factor: float = PRICE_X10_FACTOR,
) -> tuple[pd.DataFrame, dict]:
    """4) Détecte et corrige les prix aberrants.

    - Médiane et IQR calculés par famille.
    - Un prix est dit aberrant s'il dépasse ``factor × médiane famille``.
    - Si le prix dépasse ``x10_factor × médiane`` ⇒ on divise par 10
      (erreur de saisie probable, virgule oubliée).
    - Sinon ⇒ on impute par la médiane famille.
    """
    out = df.copy()
    med = out.groupby("famille")["prix_vente_unitaire"].transform("median")
    above_factor = (out["prix_vente_unitaire"] > factor * med) & med.notna()
    x10_mask = (out["prix_vente_unitaire"] > x10_factor * med) & med.notna()
    impute_mask = above_factor & ~x10_mask

    # Correction × 10
    n_x10 = int(x10_mask.sum())
    out.loc[x10_mask, "prix_vente_unitaire"] = out.loc[x10_mask, "prix_vente_unitaire"] / 10.0
    # Imputation médiane
    n_imp = int(impute_mask.sum())
    out.loc[impute_mask, "prix_vente_unitaire"] = med[impute_mask]

    # Recalcul du montant_total cohérent là où on a corrigé un prix
    touched = x10_mask | impute_mask
    out.loc[touched, "montant_total"] = (
        out.loc[touched, "prix_vente_unitaire"] * out.loc[touched, "quantite_vendue"]
    )
    return out, {"prix_aberrants_x10_corriges": n_x10, "prix_aberrants_imputes_mediane": n_imp}


def flag_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """5) Marque les retours (quantité ≤ 0) via ``est_retour`` et les écarte."""
    out = df.copy()
    out["est_retour"] = (out["quantite_vendue"] <= 0).fillna(False).astype(bool)
    n_retours = int(out["est_retour"].sum())
    out = out[~out["est_retour"]].reset_index(drop=True)
    return out, n_retours


def impute_from_catalogue(df: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """6a) Complète les manquants à partir du catalogue produit officiel."""
    out = df.copy()
    cat = catalogue.set_index("produit_id")
    cols = [
        "produit_nom", "famille", "marque", "origine_fournisseur",
        "cout_achat_unitaire", "prix_vente_unitaire",
    ]
    for col in cols:
        if col in cat.columns:
            ref = out["produit_id"].map(cat[col])
            out[col] = out[col].where(out[col].notna(), ref)
    return out


def impute_cout_from_margin(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """6b) Impute ``cout_achat_unitaire`` à partir de la marge moyenne par famille.

    Hypothèse : pour une famille, le ratio coût/prix est relativement stable.
    On calcule la médiane de ce ratio par famille, puis pour chaque transaction
    avec un coût manquant on retient ``prix_vente_unitaire × ratio_famille``.
    """
    out = df.copy()
    ratio_per_row = out["cout_achat_unitaire"] / out["prix_vente_unitaire"]
    med_ratio = ratio_per_row.groupby(out["famille"]).transform("median")
    mask = out["cout_achat_unitaire"].isna() & med_ratio.notna() & out["prix_vente_unitaire"].notna()
    n_fix = int(mask.sum())
    out.loc[mask, "cout_achat_unitaire"] = out.loc[mask, "prix_vente_unitaire"] * med_ratio[mask]
    return out, n_fix


def impute_brand(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """6c) Impute ``marque`` par la modalité dominante par produit."""
    out = df.copy()
    mode_by_prod = (
        out.groupby("produit_id")["marque"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode(dropna=True).empty else pd.NA)
    )
    mask = out["marque"].isna()
    out.loc[mask, "marque"] = out.loc[mask, "produit_id"].map(mode_by_prod)
    n_fix = int(mask.sum() - out["marque"].isna().sum())
    return out, n_fix


def impute_mode_paiement(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """6d) ``mode_paiement`` : ``Comptant`` par défaut pour B2C, ``Crédit`` pour B2B."""
    out = df.copy()
    mask = out["mode_paiement"].isna()
    n = int(mask.sum())
    is_b2b = out["type_client"].astype("string").str.lower() == "entreprise"
    out.loc[mask & is_b2b, "mode_paiement"] = "Crédit"
    out.loc[mask & ~is_b2b, "mode_paiement"] = "Comptant"
    return out, n


def impute_client_nom(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """6e) ``client_nom`` manquant pour B2C → ``Anonyme``."""
    out = df.copy()
    mask = out["client_nom"].isna()
    n = int(mask.sum())
    out.loc[mask, "client_nom"] = "Anonyme"
    return out, n


def impute_stock_apres_vente(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """6f) Interpolation linéaire par produit sur ``stock_apres_vente``."""
    out = df.copy().sort_values(["produit_id", "date"]).reset_index(drop=True)
    n_missing_before = int(out["stock_apres_vente"].isna().sum())
    out["stock_apres_vente"] = (
        out.groupby("produit_id")["stock_apres_vente"]
        .transform(lambda s: s.interpolate("linear", limit_direction="both"))
    )
    # Imputation finale : 0 (produit jamais vu en stock dans l'historique)
    out["stock_apres_vente"] = out["stock_apres_vente"].fillna(0)
    n_missing_after = int(out["stock_apres_vente"].isna().sum())
    return out, n_missing_before - n_missing_after


def drop_mostly_empty_rows(df: pd.DataFrame, ratio: float = ROW_DROP_MISSING_RATIO) -> tuple[pd.DataFrame, int]:
    """7) Supprime les lignes dont >``ratio`` des colonnes sont vides."""
    threshold = int((1 - ratio) * df.shape[1])
    out = df.dropna(axis=0, thresh=threshold).reset_index(drop=True)
    return out, len(df) - len(out)


def reconcile_amount(df: pd.DataFrame, tolerance: float = MONTANT_TOLERANCE) -> tuple[pd.DataFrame, int]:
    """8) Vérifie la cohérence ``montant_total ≈ prix × quantité`` (tolérance 1%)."""
    out = df.copy()
    expected = out["prix_vente_unitaire"] * out["quantite_vendue"]
    relative_err = (out["montant_total"] - expected).abs() / expected.replace(0, np.nan)
    mask = (relative_err > tolerance) | out["montant_total"].isna()
    n_fix = int(mask.sum())
    out.loc[mask, "montant_total"] = expected[mask]
    return out, n_fix


# --------------------------------------------------------------------- #
# Pipeline de nettoyage complet
# --------------------------------------------------------------------- #
def clean_dataset(
    df_raw: pd.DataFrame,
    catalogue: Optional[pd.DataFrame] = None,
    report: Optional[ReportBuilder] = None,
) -> pd.DataFrame:
    """Exécute toutes les étapes de §3.3.1 et alimente le rapport."""
    if report is None:
        report = ReportBuilder()
    df = df_raw.copy()
    report.log("00_donnees_brutes", df)

    df = parse_dates_iso(df)
    report.log("01_dates_iso", df, dates_nat=int(df["date"].isna().sum()))

    df, n_dup = drop_exact_duplicates(df)
    report.log("02_doublons_supprimes", df, doublons_supprimes=n_dup)

    df, n_fam = normalize_famille(df)
    report.log("03_famille_normalisee", df, libelles_corriges=n_fam)

    if catalogue is not None:
        df = impute_from_catalogue(df, catalogue)
        report.log("04_imputation_catalogue", df)

    df, prices = fix_aberrant_prices(df)
    report.log("05_prix_aberrants_corriges", df, **prices)

    df, n_ret = flag_returns(df)
    report.log("06_retours_isoles", df, retours_isoles=n_ret)

    df, n_cout = impute_cout_from_margin(df)
    report.log("07_cout_impute_marge_famille", df, lignes_imputees=n_cout)

    df, n_brand = impute_brand(df)
    report.log("08_marque_imputee_modale", df, lignes_imputees=n_brand)

    df, n_mode = impute_mode_paiement(df)
    report.log("09_mode_paiement_par_type_client", df, lignes_imputees=n_mode)

    df, n_nom = impute_client_nom(df)
    report.log("10_client_nom_anonyme", df, lignes_imputees=n_nom)

    df, n_stock = impute_stock_apres_vente(df)
    report.log("11_stock_interpole", df, valeurs_imputees=n_stock)

    df, n_drop = drop_mostly_empty_rows(df)
    report.log("12_lignes_vides_supprimees", df, lignes_supprimees=n_drop)

    df, n_amt = reconcile_amount(df)
    report.log("13_montant_recalcule", df, lignes_corrigees=n_amt)

    df = df.sort_values("date").reset_index(drop=True)
    return df


# =================================================================== #
# §3.3.2 — FEATURE ENGINEERING
# =================================================================== #

def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Variables temporelles (calendrier RDC + cyclique sin/cos)."""
    out = df.copy()
    d = out[date_col]
    out["annee"] = d.dt.year
    out["mois"] = d.dt.month
    out["trimestre"] = d.dt.quarter
    out["semaine"] = d.dt.isocalendar().week.astype(int)
    out["jour_semaine"] = d.dt.dayofweek
    out["jour_annee"] = d.dt.dayofyear
    out["est_weekend"] = (out["jour_semaine"] >= 5).astype(int)
    out["est_fin_de_mois"] = (d.dt.day >= 25).astype(int)
    out["mois_sin"] = np.sin(2 * np.pi * out["mois"] / 12)
    out["mois_cos"] = np.cos(2 * np.pi * out["mois"] / 12)
    out["jour_semaine_sin"] = np.sin(2 * np.pi * out["jour_semaine"] / 7)
    out["jour_semaine_cos"] = np.cos(2 * np.pi * out["jour_semaine"] / 7)
    out["est_rentree_scolaire"] = out["mois"].isin([8, 9]).astype(int)
    out["est_rentree_academique"] = out["mois"].isin([10, 11]).astype(int)
    out["est_saison_seche"] = out["mois"].isin([5, 6, 7, 8, 9]).astype(int)
    out["est_periode_pic_b2b"] = out["mois"].isin([1, 2, 3, 11, 12]).astype(int)
    return out


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables financières (marge, bénéfice, valeur stock immobilisé)."""
    out = df.copy()
    out["marge_unitaire"] = out["prix_vente_unitaire"] - out["cout_achat_unitaire"]
    out["benefice_transaction"] = out["marge_unitaire"] * out["quantite_vendue"]
    out["taux_marge_pct"] = np.where(
        out["prix_vente_unitaire"] > 0,
        100 * out["marge_unitaire"] / out["prix_vente_unitaire"],
        np.nan,
    )
    out["valeur_stock_immobilisee"] = out["stock_apres_vente"] * out["cout_achat_unitaire"]
    return out


def add_rupture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables rupture par produit/magasin (signalée + jours consécutifs)."""
    out = df.copy()
    out["rupture_signalee"] = (out["stock_apres_vente"] == 0).astype(int)
    out = out.sort_values(["produit_id", "magasin", "date"]).reset_index(drop=True)
    # Run-length encoding du booléen rupture par produit×magasin
    grp = out.groupby(["produit_id", "magasin"])["rupture_signalee"]
    out["jours_consecutifs_rupture"] = (
        grp.transform(lambda s: s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1))
    )
    return out


def build_product_aggregates(
    df: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Statistiques d'agrégation par produit (CV, tendance, fraîcheur, ...)."""
    if ref_date is None:
        ref_date = df["date"].max()

    g = df.groupby("produit_id")
    feats = pd.DataFrame({
        "produit_id": g.size().index,
        "ventes_totales_36mois": g["quantite_vendue"].sum().values,
        "ca_total_36mois": g["montant_total"].sum().values,
        "nb_transactions": g.size().values,
        "date_premiere_vente": g["date"].min().values,
        "date_derniere_vente": g["date"].max().values,
    })
    feats["age_produit_jours"] = (ref_date - pd.to_datetime(feats["date_premiere_vente"])).dt.days
    feats["jours_depuis_derniere_vente"] = (ref_date - pd.to_datetime(feats["date_derniere_vente"])).dt.days

    # Ventes mensuelles → CV, mois actifs, écart-type
    monthly = (
        df.assign(mois_periode=df["date"].dt.to_period("M"))
        .groupby(["produit_id", "mois_periode"])["quantite_vendue"]
        .sum()
        .reset_index()
    )
    cv_data = (
        monthly.groupby("produit_id")["quantite_vendue"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "ventes_moyennes_mensuelles",
            "std": "ecart_type_ventes_mensuelles",
            "count": "nombre_mois_avec_ventes",
        })
    )
    cv_data["coefficient_variation"] = np.where(
        cv_data["ventes_moyennes_mensuelles"] > 0,
        cv_data["ecart_type_ventes_mensuelles"].fillna(0) / cv_data["ventes_moyennes_mensuelles"],
        np.nan,
    )
    feats = feats.merge(cv_data, on="produit_id", how="left")

    # Tendances OLS sur 3 et 6 derniers mois
    monthly_sorted = monthly.sort_values(["produit_id", "mois_periode"])
    trend3, trend6 = {}, {}
    for pid, grp in monthly_sorted.groupby("produit_id"):
        v = grp["quantite_vendue"].to_numpy(dtype=float)
        trend3[pid] = linear_slope(v[-3:])
        trend6[pid] = linear_slope(v[-6:])
    feats["tendance_3_mois"] = feats["produit_id"].map(trend3).fillna(0.0)
    feats["tendance_6_mois"] = feats["produit_id"].map(trend6).fillna(0.0)

    # Prix médian par produit
    prix = df.groupby("produit_id")["prix_vente_unitaire"].median().rename("prix_vente_unitaire_moyen")
    feats = feats.merge(prix.reset_index(), on="produit_id", how="left")

    # Stock courant (dernière valeur observée)
    last_stock = df.sort_values("date").groupby("produit_id")["stock_apres_vente"].last().rename("stock_courant")
    feats = feats.merge(last_stock.reset_index(), on="produit_id", how="left")
    return feats


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering (§3.3.2)."""
    out = add_temporal_features(df_clean)
    out = add_financial_features(out)
    out = add_rupture_features(out)
    return out


# --------------------------------------------------------------------- #
# Orchestration globale
# --------------------------------------------------------------------- #
def preprocess_pipeline(
    raw_path: str | None = None,
    catalogue_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exécute nettoyage + feature engineering + agrégats produit.

    Renvoie ``(clean, features, product_features, report)``.
    """
    raw = load_raw_transactions(raw_path or RAW_TRANSACTIONS)
    catalogue = load_catalogue(catalogue_path or RAW_CATALOGUE)
    report = ReportBuilder()
    clean = clean_dataset(raw, catalogue=catalogue, report=report)
    features = engineer_features(clean)
    product_feats = build_product_aggregates(clean)
    return clean, features, product_feats, report.to_dataframe()
