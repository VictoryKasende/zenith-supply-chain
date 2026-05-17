"""Classification des produits — ABC (Pareto), XYZ (variabilité) et K-Means.

Méthodologie (mémoire §3.4) :
- ABC : tri du CA cumulé, seuils 70 % / 90 %.
- XYZ : coefficient de variation des ventes mensuelles, seuils 0.5 / 1.0,
  avec repli sur les quantiles 33 / 66 si la classe X est vide.
- K-Means : clustering non supervisé sur 7 features standardisées, choix de k
  par silhouette (validation par méthode du coude).
- Chaque cluster est étiqueté par une règle métier interprétable.

Sortie principale : DataFrame avec `produit_id, classe_abc, classe_xyz,
classe_abc_xyz, cluster_kmeans, libelle_cluster, ...`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import RANDOM_STATE, setup_logger

logger = setup_logger("classification")


# --------------------------------------------------------------------- #
# Constantes (cf. mémoire §3.4)
# --------------------------------------------------------------------- #
ABC_A_THRESHOLD = 0.70
ABC_B_THRESHOLD = 0.90

XYZ_X_CV = 0.5
XYZ_Y_CV = 1.0

KMEANS_FEATURES: tuple[str, ...] = (
    "ventes_totales_36mois",
    "ca_total_36mois",
    "coefficient_variation",
    "nombre_mois_avec_ventes",
    "tendance_3_mois",
    "jours_depuis_derniere_vente",
    "prix_vente_unitaire_moyen",
)

K_CANDIDATES: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10)

#: Features fortement asymétriques pour lesquelles on applique ``log1p`` avant
#: standardisation, afin d'éviter qu'un bestseller unique n'écrase la
#: variabilité des autres produits.
LOG_FEATURES: tuple[str, ...] = (
    "ventes_totales_36mois",
    "ca_total_36mois",
    "prix_vente_unitaire_moyen",
)


# =================================================================== #
# Classification ABC
# =================================================================== #
def classify_abc(
    df: pd.DataFrame,
    revenue_col: str = "ca_total_36mois",
    a_thresh: float = ABC_A_THRESHOLD,
    b_thresh: float = ABC_B_THRESHOLD,
) -> pd.DataFrame:
    """Affecte une classe ABC selon le CA cumulé (Pareto).

    - Tri produits par CA décroissant.
    - Classe A : produits dont le cumul ≤ ``a_thresh`` du CA total.
    - Classe B : produits dont le cumul ≤ ``b_thresh`` du CA total.
    - Classe C : reste.
    """
    out = df.copy()
    out = out.sort_values(revenue_col, ascending=False).reset_index(drop=True)
    total = out[revenue_col].sum()
    out["ca_cumul"] = out[revenue_col].cumsum()
    out["ca_cumul_pct"] = out["ca_cumul"] / total

    def _label(p: float) -> str:
        if p <= a_thresh:
            return "A"
        if p <= b_thresh:
            return "B"
        return "C"

    out["classe_abc"] = out["ca_cumul_pct"].apply(_label)
    return out


# =================================================================== #
# Classification XYZ
# =================================================================== #
def classify_xyz(
    df: pd.DataFrame,
    cv_col: str = "coefficient_variation",
    x_cv: float = XYZ_X_CV,
    y_cv: float = XYZ_Y_CV,
    quantile_fallback: bool = True,
) -> pd.DataFrame:
    """Affecte une classe XYZ selon le coefficient de variation des ventes.

    Si la classe X (CV faible) est vide ou minoritaire (< 5 % du catalogue), on
    bascule sur les quantiles 33 / 66 % du CV (paramètre ``quantile_fallback``).
    """
    out = df.copy()
    cv = out[cv_col].fillna(np.inf)

    def _label_thresh(v: float) -> str:
        if v < x_cv:
            return "X"
        if v < y_cv:
            return "Y"
        return "Z"

    initial = cv.apply(_label_thresh)
    pct_x = (initial == "X").mean()
    if quantile_fallback and pct_x < 0.05:
        logger.info(
            "Classe X très peu peuplée (%.1f %%) — bascule sur quantiles 33/66 du CV.",
            pct_x * 100,
        )
        q33, q66 = cv.replace(np.inf, np.nan).quantile([0.33, 0.66])

        def _label_q(v: float) -> str:
            if v <= q33:
                return "X"
            if v <= q66:
                return "Y"
            return "Z"

        out["classe_xyz"] = cv.apply(_label_q)
    else:
        out["classe_xyz"] = initial
    out["classe_abc_xyz"] = out["classe_abc"] + out["classe_xyz"]
    return out


# =================================================================== #
# Matrice ABC × XYZ
# =================================================================== #
def abc_xyz_matrix(df: pd.DataFrame, revenue_col: str = "ca_total_36mois") -> pd.DataFrame:
    """Tableau croisé 3 × 3 (nb produits, part de CA en %)."""
    grouped = df.groupby(["classe_abc", "classe_xyz"]).agg(
        n_produits=("produit_id", "nunique"),
        ca=(revenue_col, "sum"),
    )
    total_ca = grouped["ca"].sum()
    grouped["pct_ca"] = (grouped["ca"] / total_ca * 100).round(2)
    return grouped.reset_index().pivot_table(
        index="classe_abc", columns="classe_xyz",
        values=["n_produits", "pct_ca"], fill_value=0,
    )


# =================================================================== #
# K-Means clustering
# =================================================================== #
@dataclass
class KMeansDiagnostics:
    inertia: dict[int, float]
    silhouette: dict[int, float]
    best_k: int
    best_silhouette: float
    cluster_profile: pd.DataFrame
    pca_components: np.ndarray
    feature_cols: tuple[str, ...]


def kmeans_pipeline(
    df: pd.DataFrame,
    feature_cols: Iterable[str] = KMEANS_FEATURES,
    k_candidates: Iterable[int] = K_CANDIDATES,
    random_state: int = RANDOM_STATE,
    min_k: int = 3,
) -> tuple[pd.DataFrame, KMeansDiagnostics]:
    """Pipeline complet : log-transform + standardisation, choix de k, fit final.

    Parameters
    ----------
    min_k : int, default 3
        Empêche le pipeline de retenir un découpage binaire dégénéré (k=2).
        On retient le k qui maximise la silhouette parmi les k ≥ ``min_k``.

    Returns
    -------
    df_out : DataFrame produits avec ``cluster_kmeans`` et ``libelle_cluster``.
    diag : ``KMeansDiagnostics`` (inertie, silhouette, profils, PCA).
    """
    feature_cols = tuple(feature_cols)
    out = df.copy()
    X = out[list(feature_cols)].fillna(0).to_numpy(dtype=float).copy()
    # Log-transform sur les colonnes asymétriques pour ne pas écraser la variabilité
    for i, c in enumerate(feature_cols):
        if c in LOG_FEATURES:
            X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    inertia, silhouette = {}, {}
    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(Xn)
        inertia[k] = float(km.inertia_)
        silhouette[k] = float(silhouette_score(Xn, labels)) if k > 1 else 0.0

    eligible = {k: s for k, s in silhouette.items() if k >= min_k}
    if not eligible:
        eligible = silhouette
    best_k = max(eligible, key=eligible.get)
    logger.info(
        "K-Means : k* = %d (silhouette = %.3f, inertie = %.1f, min_k=%d)",
        best_k, silhouette[best_k], inertia[best_k], min_k,
    )

    km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    out["cluster_kmeans"] = km.fit_predict(Xn)

    profile = out.groupby("cluster_kmeans")[list(feature_cols)].median().round(2)
    out["libelle_cluster"] = out["cluster_kmeans"].map(_label_clusters(profile, df=out))

    # Projection PCA 2D pour visualisation
    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(Xn)

    diag = KMeansDiagnostics(
        inertia=inertia,
        silhouette=silhouette,
        best_k=best_k,
        best_silhouette=silhouette[best_k],
        cluster_profile=profile,
        pca_components=components,
        feature_cols=feature_cols,
    )
    return out, diag


def _label_clusters(profile: pd.DataFrame, df: pd.DataFrame) -> dict[int, str]:
    """Attribue un libellé métier interprétable à chaque cluster."""
    labels: dict[int, str] = {}
    ca_q75 = profile["ca_total_36mois"].quantile(0.75)
    ca_q25 = profile["ca_total_36mois"].quantile(0.25)
    for cid, row in profile.iterrows():
        jours_sans = row["jours_depuis_derniere_vente"]
        cv = row["coefficient_variation"]
        ca = row["ca_total_36mois"]
        tendance = row["tendance_3_mois"]
        if jours_sans > 180:
            label = "Dormant — risque obsolescence"
        elif ca >= ca_q75 and cv < 0.7:
            label = "Bestseller stable"
        elif ca >= ca_q75:
            label = "Bestseller volatile"
        elif tendance > 0 and ca > ca_q25:
            label = "En croissance"
        elif tendance < 0:
            label = "En déclin"
        else:
            label = "Rotation modérée"
        labels[cid] = label
    return labels


# =================================================================== #
# Orchestration globale
# =================================================================== #
def classify_pipeline(product_features: pd.DataFrame) -> tuple[pd.DataFrame, KMeansDiagnostics]:
    """Exécute ABC → XYZ → K-Means sur le DataFrame produit agrégé."""
    abc = classify_abc(product_features)
    abc_xyz = classify_xyz(abc)
    final, diag = kmeans_pipeline(abc_xyz)
    return final, diag
