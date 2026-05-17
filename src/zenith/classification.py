"""Classification ABC × XYZ et clustering K-Means (cf. mémoire §3.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from .config import (
    ABC_A_THRESHOLD,
    ABC_B_THRESHOLD,
    RANDOM_STATE,
    XYZ_X_CV,
    XYZ_Y_CV,
)


def classify_abc(
    features: pd.DataFrame,
    revenue_col: str = "ca_total_36mois",
    a_thresh: float = ABC_A_THRESHOLD,
    b_thresh: float = ABC_B_THRESHOLD,
) -> pd.DataFrame:
    """Classification ABC (Pareto) sur le chiffre d'affaires."""
    df = features.copy()
    df = df.sort_values(revenue_col, ascending=False).reset_index(drop=True)
    total = df[revenue_col].sum()
    df["ca_cumul"] = df[revenue_col].cumsum()
    df["ca_cumul_pct"] = df["ca_cumul"] / total

    def _label(p: float) -> str:
        if p <= a_thresh:
            return "A"
        if p <= b_thresh:
            return "B"
        return "C"

    df["classe_abc"] = df["ca_cumul_pct"].apply(_label)
    return df


def classify_xyz(
    features: pd.DataFrame,
    cv_col: str = "coefficient_variation",
    x_cv: float = XYZ_X_CV,
    y_cv: float = XYZ_Y_CV,
) -> pd.DataFrame:
    """Classification XYZ sur le coefficient de variation des ventes mensuelles."""
    df = features.copy()
    cv = df[cv_col].fillna(np.inf)

    def _label(v: float) -> str:
        if v < x_cv:
            return "X"
        if v < y_cv:
            return "Y"
        return "Z"

    df["classe_xyz"] = cv.apply(_label)
    df["classe_abc_xyz"] = df["classe_abc"] + df["classe_xyz"]
    return df


KMEANS_FEATURES = (
    "ventes_totales_36mois",
    "ca_total_36mois",
    "coefficient_variation",
    "nombre_mois_avec_ventes",
    "tendance_3_mois",
    "jours_depuis_derniere_vente",
    "prix_vente_unitaire",
)


def kmeans_clustering(
    features: pd.DataFrame,
    feature_cols: tuple[str, ...] = KMEANS_FEATURES,
    k_candidates: tuple[int, ...] = (3, 4, 5, 6, 7),
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """K-Means avec sélection automatique de k (silhouette + coude)."""
    df = features.copy()
    X = df[list(feature_cols)].fillna(0).to_numpy(dtype=float)
    scaler = MinMaxScaler()
    Xn = scaler.fit_transform(X)

    diag = {"inertia": {}, "silhouette": {}}
    best_k, best_score = None, -np.inf
    best_model = None

    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(Xn)
        diag["inertia"][k] = float(km.inertia_)
        try:
            sil = silhouette_score(Xn, labels) if k > 1 else 0.0
        except Exception:
            sil = 0.0
        diag["silhouette"][k] = float(sil)
        if sil > best_score:
            best_score, best_k, best_model = sil, k, km

    final = best_model.fit_predict(Xn) if best_model is not None else np.zeros(len(df))
    df["cluster_kmeans"] = final
    diag["best_k"] = best_k
    diag["best_silhouette"] = best_score

    # Profil métier de chaque cluster
    profile = (
        df.groupby("cluster_kmeans")[list(feature_cols)]
        .median()
        .round(2)
    )
    diag["cluster_profile"] = profile

    df["cluster_label"] = df["cluster_kmeans"].map(_label_clusters(profile))
    return df, diag


def _label_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Attribue un libellé métier interprétable à chaque cluster."""
    labels = {}
    for cid, row in profile.iterrows():
        ca = row["ca_total_36mois"]
        jours_sans = row["jours_depuis_derniere_vente"]
        cv = row["coefficient_variation"]
        tendance = row["tendance_3_mois"]
        if jours_sans > 180:
            labels[cid] = "Dormant / risque obsolescence"
        elif ca >= profile["ca_total_36mois"].quantile(0.75):
            labels[cid] = "Forte rotation stable" if cv < 0.7 else "Forte rotation volatile"
        elif tendance > 0:
            labels[cid] = "En croissance"
        elif tendance < 0:
            labels[cid] = "En déclin"
        else:
            labels[cid] = "Rotation modérée"
    return labels
