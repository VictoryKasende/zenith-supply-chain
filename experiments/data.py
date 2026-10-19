"""Chargement des données et reconstruction de l'état du système à une origine t0.

Règle de non-fuite : :func:`build_state` ne reçoit que les transactions brutes
datées ≤ t0. Nettoyage (médianes de prix par famille, interpolation de stock),
agrégats produit, ABC/XYZ, K-Means, scaler et Isolation Forest sont tous
recalculés sur ce seul sous-ensemble. Le catalogue (prix et coûts de liste,
origine fournisseur) est une donnée de référence statique.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from experiments.config import DATA_START_MONTH, quiet_pipeline_logs
from src.classification import classify_pipeline
from src.forecasting import monthly_series_by_product
from src.obsolescence import (
    business_rule_mask,
    build_obsolescence_features,
    detect_obsolescence,
)
from src.preprocessing import build_product_aggregates, clean_dataset
from src.utils import load_catalogue, load_raw_transactions

quiet_pipeline_logs()


# --------------------------------------------------------------------- #
# Données brutes et nettoyées
# --------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    return load_raw_transactions()


@lru_cache(maxsize=1)
def load_cat() -> pd.DataFrame:
    return load_catalogue()


def clean_upto(raw: pd.DataFrame, catalogue: pd.DataFrame, t0: pd.Timestamp | None) -> pd.DataFrame:
    """Nettoyage de l'Étape 2 appliqué aux seules lignes brutes datées ≤ t0."""
    sub = raw if t0 is None else raw[raw["date"] <= t0]
    clean = clean_dataset(sub, catalogue=catalogue)
    # Ordre total déterministe (le tri final de clean_dataset n'est pas stable).
    return clean.sort_values(["date", "transaction_id"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def full_clean() -> pd.DataFrame:
    """Données nettoyées complètes : sert uniquement de vérité terrain."""
    return clean_upto(load_raw(), load_cat(), None)


def monthly_matrix(clean: pd.DataFrame, last_month: pd.Timestamp) -> pd.DataFrame:
    """Matrice mois × produit des ventes, de août 2022 à ``last_month`` inclus."""
    m = monthly_series_by_product(clean)
    idx = pd.date_range(DATA_START_MONTH, last_month, freq="MS")
    return m.reindex(idx, fill_value=0.0).astype(float)


def central_stock(clean: pd.DataFrame) -> pd.Series:
    """Stock central = Σ magasins du dernier ``stock_apres_vente`` observé.

    ``stock_apres_vente`` est suivi par magasin (84 % des transactions
    successives vérifient stock_t = stock_{t-1} − qté au niveau produit ×
    magasin, contre 25 % au niveau produit seul).
    """
    last = clean.groupby(["produit_id", "magasin"])["stock_apres_vente"].last()
    return last.groupby(level=0).sum().astype(float)


def realized_lines(clean: pd.DataFrame) -> pd.DataFrame:
    """Lignes de vente (demande réalisée) avec leur mois, en ordre chronologique."""
    out = clean[["produit_id", "date", "transaction_id", "quantite_vendue"]].dropna(subset=["quantite_vendue"]).copy()
    out["mois"] = out["date"].values.astype("datetime64[M]")
    return out.sort_values(["produit_id", "date", "transaction_id"]).reset_index(drop=True)


# --------------------------------------------------------------------- #
# État à l'origine t0
# --------------------------------------------------------------------- #
@dataclass
class OriginState:
    t0: pd.Timestamp
    clean: pd.DataFrame
    monthly: pd.DataFrame
    classes: pd.DataFrame
    obs_features: pd.DataFrame
    iforest_flag: pd.Series
    rule_flag: pd.Series
    stock: pd.Series
    contamination: float = 0.10
    rule_days: int = 180

    @property
    def products(self) -> list[str]:
        return sorted(self.classes["produit_id"].tolist())

    @property
    def obsolete(self) -> set[str]:
        flag = self.iforest_flag | self.rule_flag
        return set(flag[flag].index)

    def class_of(self, pid: str) -> tuple[str, str]:
        row = self.classes.set_index("produit_id").loc[pid]
        return row["classe_abc"], row["classe_xyz"]


def obsolescence_features_at(clean: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Features de l'Étape 4 à la date t0 (âge produit compris, cf. run_obsolescence)."""
    feats = build_obsolescence_features(clean, ref_date=t0)
    age = (t0 - clean.groupby("produit_id")["date"].min()).dt.days.rename("age_produit_jours")
    return feats.merge(age.reset_index(), on="produit_id", how="left")


def detect_at(obs_features: pd.DataFrame, contamination: float = 0.10,
              rule_days: int = 180) -> tuple[pd.Series, pd.Series]:
    """Drapeaux Isolation Forest seul et règles métier seules (index produit).

    L'hybride de l'Étape 4 est leur union : c'est exactement ce que fait
    :func:`src.obsolescence.detect_obsolescence` (IF puis règles en OU).
    """
    res = detect_obsolescence(obs_features, contamination=contamination, apply_business_rules=False)
    iforest = pd.Series(res.df["a_risque_obsolescence"].to_numpy() == 1, index=obs_features["produit_id"])
    rules = business_rule_mask(obs_features, days_threshold=rule_days,
                               months_threshold=int(round(rule_days / 30)))
    rules = pd.Series(rules.to_numpy(), index=obs_features["produit_id"])
    return iforest, rules


def build_state(raw: pd.DataFrame, catalogue: pd.DataFrame, t0: pd.Timestamp,
                contamination: float = 0.10, rule_days: int = 180) -> OriginState:
    """Reconstruit tout ce que le pipeline apprend, à partir des seules données ≤ t0."""
    t0 = pd.Timestamp(t0)
    clean = clean_upto(raw, catalogue, t0)
    last_month = t0.to_period("M").to_timestamp()
    monthly = monthly_matrix(clean, last_month)

    feats = build_product_aggregates(clean, ref_date=t0)
    classes, _ = classify_pipeline(feats)
    classes = classes[["produit_id", "classe_abc", "classe_xyz", "cluster_kmeans", "libelle_cluster",
                       "ca_total_36mois", "coefficient_variation"]].copy()

    obs_feats = obsolescence_features_at(clean, t0)
    iforest, rules = detect_at(obs_feats, contamination, rule_days)
    stock = central_stock(clean).reindex(classes["produit_id"]).fillna(0.0)
    return OriginState(t0=t0, clean=clean, monthly=monthly, classes=classes,
                       obs_features=obs_feats, iforest_flag=iforest, rule_flag=rules,
                       stock=stock, contamination=contamination, rule_days=rule_days)


def state_table(state: OriginState) -> pd.DataFrame:
    """Résumé tabulaire d'un état (une ligne par produit) pour paper_results."""
    df = state.classes[["produit_id", "classe_abc", "classe_xyz", "cluster_kmeans", "libelle_cluster"]].copy()
    df.insert(0, "origin", state.t0.date())
    df["iforest_flag"] = df["produit_id"].map(state.iforest_flag).astype(int)
    df["rule_flag"] = df["produit_id"].map(state.rule_flag).astype(int)
    df["obsolete_flag"] = (df["iforest_flag"] | df["rule_flag"]).astype(int)
    df["central_stock"] = df["produit_id"].map(state.stock)
    first = state.monthly.ne(0).idxmax()
    df["first_sale_month"] = df["produit_id"].map(first).dt.date
    return df


def history_since_launch(series: pd.Series) -> pd.Series:
    """Série à partir du premier mois avec vente (pour les dénominateurs MASE)."""
    nz = np.flatnonzero(series.to_numpy() > 0)
    return series.iloc[nz[0]:] if len(nz) else series.iloc[0:0]
