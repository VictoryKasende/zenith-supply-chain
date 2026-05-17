"""Feature engineering (cf. mémoire §3.3.2).

Construit les variables dérivées :
- temporelles (annee, mois, sin/cos, est_rentree_*, etc.)
- agrégations par produit (CA, CV, tendances, jours depuis dernière vente, ...)
- financières (marge, taux marge, valeur stock immobilisée)
- rupture (rupture_signalee, jours consécutifs en rupture)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Variables temporelles
# ------------------------------------------------------------------ #
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables temporelles à partir de la colonne `date`."""
    df = df.copy()
    d = df["date"]
    df["annee"] = d.dt.year
    df["mois"] = d.dt.month
    df["trimestre"] = d.dt.quarter
    df["semaine"] = d.dt.isocalendar().week.astype(int)
    df["jour_semaine"] = d.dt.dayofweek
    df["jour_annee"] = d.dt.dayofyear
    df["est_weekend"] = (df["jour_semaine"] >= 5).astype(int)

    # Encodage cyclique
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)
    df["jsem_sin"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
    df["jsem_cos"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)

    df["est_rentree_scolaire"] = df["mois"].isin([8, 9]).astype(int)
    df["est_rentree_academique"] = df["mois"].isin([10, 11]).astype(int)
    df["est_saison_seche"] = df["mois"].isin([5, 6, 7, 8, 9]).astype(int)
    df["est_periode_pic_b2b"] = df["mois"].isin([1, 2, 3, 11, 12]).astype(int)
    return df


# ------------------------------------------------------------------ #
# Variables financières
# ------------------------------------------------------------------ #
def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["marge_unitaire"] = df["prix_vente_unitaire"] - df["cout_achat_unitaire"]
    df["benefice_transaction"] = df["marge_unitaire"] * df["quantite_vendue"]
    df["taux_marge_pct"] = np.where(
        df["prix_vente_unitaire"] > 0,
        100 * df["marge_unitaire"] / df["prix_vente_unitaire"],
        0,
    )
    df["valeur_stock_immobilisee"] = (
        df["stock_apres_vente"].fillna(0) * df["cout_achat_unitaire"]
    )
    df["rupture_signalee"] = (df["stock_apres_vente"].fillna(-1) == 0).astype(int)
    return df


# ------------------------------------------------------------------ #
# Agrégations produit
# ------------------------------------------------------------------ #
def build_product_features(
    df: pd.DataFrame, ref_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Statistiques par produit (CA, CV, tendances, fraîcheur, ...).

    `ref_date` : date d'analyse à partir de laquelle calculer
    `jours_depuis_derniere_vente`. Par défaut = max(date) du dataset.
    """
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
        "cout_achat_unitaire": g["cout_achat_unitaire"].median().values,
        "prix_vente_unitaire": g["prix_vente_unitaire"].median().values,
        "marge_unitaire_moy": (g["prix_vente_unitaire"].median() - g["cout_achat_unitaire"].median()).values,
        "famille": g["famille"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).values,
        "marque": g["marque"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).values,
        "origine_fournisseur": g["origine_fournisseur"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        ).values,
    })
    feats["age_produit_jours"] = (
        ref_date - pd.to_datetime(feats["date_premiere_vente"])
    ).dt.days
    feats["jours_depuis_derniere_vente"] = (
        ref_date - pd.to_datetime(feats["date_derniere_vente"])
    ).dt.days

    # Ventes mensuelles → CV, mois actifs, tendances
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
    )
    cv_data["coefficient_variation"] = np.where(
        cv_data["mean"] > 0, cv_data["std"].fillna(0) / cv_data["mean"], np.nan
    )
    cv_data = cv_data.rename(
        columns={
            "mean": "ventes_moyennes_mensuelles",
            "std": "ecart_type_ventes_mensuelles",
            "count": "nombre_mois_avec_ventes",
        }
    )
    feats = feats.merge(cv_data, on="produit_id", how="left")

    # Tendance (pente OLS) sur les 3 et 6 derniers mois
    def _slope(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        return float(np.polyfit(x, values, 1)[0])

    monthly_sorted = monthly.sort_values(["produit_id", "mois_periode"])
    trend3, trend6 = {}, {}
    for pid, grp in monthly_sorted.groupby("produit_id"):
        v = grp["quantite_vendue"].to_numpy(dtype=float)
        trend3[pid] = _slope(v[-3:])
        trend6[pid] = _slope(v[-6:])
    feats["tendance_3_mois"] = feats["produit_id"].map(trend3).fillna(0.0)
    feats["tendance_6_mois"] = feats["produit_id"].map(trend6).fillna(0.0)

    # Ratio ventes 3m / 12m (indicateur de déclin)
    end = ref_date
    last_3m = df[df["date"] >= end - pd.Timedelta(days=90)]
    last_12m = df[df["date"] >= end - pd.Timedelta(days=365)]
    s3 = last_3m.groupby("produit_id")["quantite_vendue"].sum()
    s12 = last_12m.groupby("produit_id")["quantite_vendue"].sum()
    ratio = (s3 / s12.replace(0, np.nan)).rename("ratio_ventes_3m_vs_12m")
    feats = feats.merge(ratio.reset_index(), on="produit_id", how="left")
    feats["ratio_ventes_3m_vs_12m"] = feats["ratio_ventes_3m_vs_12m"].fillna(0.0)

    # Mois consécutifs sans vente
    def _consec_zero_tail(group: pd.DataFrame) -> int:
        all_months = pd.period_range(
            df["date"].min().to_period("M"), end.to_period("M"), freq="M"
        )
        present = set(group["mois_periode"])
        count = 0
        for m in reversed(all_months):
            if m in present and group.loc[group["mois_periode"] == m, "quantite_vendue"].sum() > 0:
                break
            count += 1
        return count

    consec = monthly.groupby("produit_id").apply(_consec_zero_tail, include_groups=False)
    feats["nombre_mois_consecutifs_sans_vente"] = feats["produit_id"].map(consec).fillna(0)

    # Stock courant et valeur stock dormant (dernier stock observé)
    last_stock = df.sort_values("date").groupby("produit_id")["stock_apres_vente"].last()
    feats["stock_courant"] = feats["produit_id"].map(last_stock).fillna(0)
    feats["valeur_stock_dormant"] = feats["stock_courant"] * feats["cout_achat_unitaire"]
    return feats


# ------------------------------------------------------------------ #
# Séries journalières par produit
# ------------------------------------------------------------------ #
def daily_series(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Reconstruit des séries journalières (ou mensuelles) par produit.

    Les jours sans vente sont remplis à zéro pour préparer la prévision.
    """
    pivot = (
        df.groupby(["produit_id", "date"])["quantite_vendue"]
        .sum()
        .reset_index()
    )
    out = []
    for pid, grp in pivot.groupby("produit_id"):
        s = grp.set_index("date")["quantite_vendue"].sort_index()
        if freq == "D":
            idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        elif freq == "M":
            idx = pd.period_range(s.index.min().to_period("M"), s.index.max().to_period("M"), freq="M").to_timestamp()
            s = s.resample("MS").sum()
        else:
            raise ValueError(f"freq inconnu: {freq}")
        s = s.reindex(idx, fill_value=0)
        out.append(s.rename(pid))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=1).fillna(0)


def add_lag_features(
    series: pd.Series, lags: tuple[int, ...] = (1, 7, 30), windows: tuple[int, ...] = (7, 30)
) -> pd.DataFrame:
    """Construit des features lag et fenêtres mobiles pour une série."""
    df = pd.DataFrame({"y": series})
    for L in lags:
        df[f"lag_{L}"] = series.shift(L)
    for W in windows:
        df[f"roll_mean_{W}"] = series.shift(1).rolling(W).mean()
        df[f"roll_std_{W}"] = series.shift(1).rolling(W).std()
    return df
