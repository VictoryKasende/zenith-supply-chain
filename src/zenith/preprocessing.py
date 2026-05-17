"""Nettoyage des données brutes (cf. mémoire §3.3.1).

Opérations:
- Uniformisation des dates (mixte ISO + DD/MM/YYYY)
- Suppression des doublons
- Normalisation des libellés famille (fautes de frappe)
- Correction / suppression des prix aberrants
- Gestion des quantités négatives ou nulles
- Imputation des valeurs manquantes
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Dictionnaire de normalisation des libellés famille (cf. mémoire §3.3.1)
FAMILLE_FIXES = {
    "Cartouches": "Cartouche",
    "Accessoires": "Accessoire",
    "Imprimente": "Imprimante",
    "Ordinator": "Ordinateur",
    "Réseau": "Reseau",
}


def parse_dates(series: pd.Series) -> pd.Series:
    """Convertit une colonne date en datetime64, gère ISO + DD/MM/YYYY."""
    iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    fallback_mask = iso.isna() & series.notna()
    if fallback_mask.any():
        alt = pd.to_datetime(
            series[fallback_mask], format="%d/%m/%Y", errors="coerce"
        )
        iso.loc[fallback_mask] = alt
    return iso


def normalize_famille(series: pd.Series) -> pd.Series:
    """Normalise les libellés famille (fautes de frappe + casse)."""
    s = series.astype("string").str.strip()
    return s.replace(FAMILLE_FIXES)


def fix_aberrant_prices(df: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    """Corrige les prix aberrants: si prix > k * médiane famille, divise par 10.

    Hypothèse: erreur de saisie (virgule oubliée) — cf. mémoire §3.3.1.
    """
    df = df.copy()
    median_by_fam = df.groupby("famille")["prix_vente_unitaire"].transform("median")
    mask = (df["prix_vente_unitaire"] > k * median_by_fam) & median_by_fam.notna()
    n_fixes = int(mask.sum())
    df.loc[mask, "prix_vente_unitaire"] = df.loc[mask, "prix_vente_unitaire"] / 10.0
    # Recalcul du montant total cohérent si présent
    if "quantite_vendue" in df.columns:
        df.loc[mask, "montant_total"] = (
            df.loc[mask, "prix_vente_unitaire"] * df.loc[mask, "quantite_vendue"]
        )
    logger.info("Prix aberrants corrigés: %d", n_fixes)
    return df


def impute_from_catalogue(df: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Complète les valeurs manquantes en s'appuyant sur le catalogue produit."""
    df = df.copy()
    cat = catalogue.set_index("produit_id")
    for col in [
        "produit_nom",
        "famille",
        "marque",
        "origine_fournisseur",
        "cout_achat_unitaire",
        "prix_vente_unitaire",
    ]:
        if col in cat.columns:
            ref = df["produit_id"].map(cat[col])
            df[col] = df[col].where(df[col].notna(), ref)
    return df


def clean_dataset(
    df_raw: pd.DataFrame, catalogue: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Pipeline de nettoyage complet du dataset brut."""
    n0 = len(df_raw)
    df = df_raw.copy()

    # 1. Suppression des doublons exacts
    df = df.drop_duplicates()
    logger.info("Doublons supprimés: %d", n0 - len(df))

    # 2. Parsing des dates
    df["date"] = parse_dates(df["date"])

    # 3. Normalisation des libellés famille
    df["famille"] = normalize_famille(df["famille"])

    # 4. Imputation via catalogue
    if catalogue is not None:
        df = impute_from_catalogue(df, catalogue)

    # 5. Correction prix aberrants
    df = fix_aberrant_prices(df)

    # 6. Quantités: suppression des valeurs négatives ou nulles
    n_before = len(df)
    df = df[df["quantite_vendue"] > 0]
    logger.info("Quantités <= 0 supprimées: %d", n_before - len(df))

    # 7. Lignes incomplètes critiques (date, produit_id, quantite manquants)
    n_before = len(df)
    df = df.dropna(subset=["date", "produit_id", "quantite_vendue"])
    logger.info("Lignes sans clés essentielles supprimées: %d", n_before - len(df))

    # 8. Imputations restantes (médiane numérique, modalité dominante catégorielle)
    for col in ["prix_vente_unitaire", "cout_achat_unitaire", "stock_apres_vente"]:
        med = df.groupby("produit_id")[col].transform("median")
        df[col] = df[col].fillna(med)
        df[col] = df[col].fillna(df[col].median())

    for col in ["marque", "origine_fournisseur", "type_client", "mode_paiement"]:
        if col not in df.columns:
            continue
        mode_by_prod = (
            df.groupby("produit_id")[col]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        )
        df[col] = df[col].fillna(df["produit_id"].map(mode_by_prod))
        if df[col].isna().any():
            global_mode = df[col].mode()
            if not global_mode.empty:
                df[col] = df[col].fillna(global_mode.iloc[0])

    # 9. Recalcul du montant total si incohérent
    expected_montant = df["prix_vente_unitaire"] * df["quantite_vendue"]
    df["montant_total"] = df["montant_total"].where(
        (df["montant_total"] - expected_montant).abs() < 0.5, expected_montant
    )

    # 10. Reset index + tri chronologique
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Dataset nettoyé: %d transactions (avant %d)", len(df), n0)
    return df


def temporal_split(
    df: pd.DataFrame, train_end: str, val_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel chronologique (cf. mémoire §3.3.3)."""
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()
    return train, val, test
