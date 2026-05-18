"""Détection automatique des produits à risque d'obsolescence (mémoire §3.5).

Méthodologie
------------
1. **Construction des features d'obsolescence** par produit :
   - ``jours_depuis_derniere_vente`` : plus la valeur est élevée, plus le produit est suspect.
   - ``tendance_3_mois`` / ``tendance_6_mois`` : pente OLS des ventes mensuelles
     récentes (négatif = déclin).
   - ``ratio_ventes_3m_vs_12m`` : ratio des ventes 3 derniers mois / 12 derniers mois.
   - ``nombre_mois_consecutifs_sans_vente`` : run-length de zéros depuis la fin.
   - ``valeur_stock_dormant`` : ``stock_courant × cout_achat`` (USD immobilisés).
   - ``variation_relative_prix`` : pente du prix moyen (déstockage probable).

2. **Isolation Forest** (Liu et al., 2008) :
   - ``n_estimators=100, contamination=0.10, max_samples=256``.
   - Score d'anomalie : plus négatif = produit suspect.
   - Drapeau binaire ``a_risque_obsolescence``.

3. **Règles métier de filet** combinées avec Isolation Forest pour ne pas
   manquer les cas évidents (produits sans vente depuis ≥ 6 mois, ratio
   3m/12m < 0.1 et produit ancien, …).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils import linear_slope, setup_logger

logger = setup_logger("obsolescence")

# --------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------- #
DEFAULT_FEATURES: tuple[str, ...] = (
    "jours_depuis_derniere_vente",
    "tendance_3_mois",
    "tendance_6_mois",
    "ratio_ventes_3m_vs_12m",
    "nombre_mois_consecutifs_sans_vente",
    "valeur_stock_dormant",
    "variation_relative_prix",
)

IFOREST_N_ESTIMATORS = 100
IFOREST_MAX_SAMPLES = 256
IFOREST_RANDOM_STATE = 42
IFOREST_DEFAULT_CONTAMINATION = 0.10


# =================================================================== #
# Construction des features d'obsolescence
# =================================================================== #
def build_obsolescence_features(
    transactions: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Construit la table de features (1 ligne = 1 produit) attendue par
    ``detect_obsolescence``.

    Parameters
    ----------
    transactions : DataFrame nettoyé issu de l'Étape 2 (``zenith_clean.csv``)
        avec colonnes ``produit_id, date, quantite_vendue, montant_total,
        stock_apres_vente, cout_achat_unitaire, prix_vente_unitaire``.
    ref_date : Timestamp, optionnel
        Date de référence pour calculer ``jours_depuis_derniere_vente``
        (par défaut ``transactions['date'].max()``).
    """
    if ref_date is None:
        ref_date = transactions["date"].max()

    tmp = transactions.copy()
    tmp["mois_periode"] = tmp["date"].dt.to_period("M")
    monthly = (
        tmp.groupby(["produit_id", "mois_periode"], observed=True)["quantite_vendue"]
        .sum()
        .astype(float)
        .reset_index()
    )

    rows: list[dict] = []
    end_period = ref_date.to_period("M")
    all_months = pd.period_range(monthly["mois_periode"].min(), end_period, freq="M")
    for pid, grp in monthly.groupby("produit_id"):
        ts = grp.set_index("mois_periode")["quantite_vendue"].astype(float)
        ts = ts.reindex(all_months, fill_value=0.0)
        last_sale = tmp.loc[tmp["produit_id"] == pid, "date"].max()
        days_since = int((ref_date - last_sale).days) if pd.notna(last_sale) else 9999

        # Tendances
        slope3 = linear_slope(ts.tail(3).to_numpy())
        slope6 = linear_slope(ts.tail(6).to_numpy())

        # Ratios
        s3 = float(ts.tail(3).sum())
        s12 = float(ts.tail(12).sum())
        ratio_3_12 = s3 / s12 if s12 > 0 else 0.0

        # Mois consécutifs sans vente (run-length en partant de la fin)
        consec = 0
        for v in reversed(ts.values):
            if v == 0:
                consec += 1
            else:
                break

        rows.append({
            "produit_id": pid,
            "jours_depuis_derniere_vente": days_since,
            "tendance_3_mois": float(slope3),
            "tendance_6_mois": float(slope6),
            "ratio_ventes_3m_vs_12m": float(ratio_3_12),
            "nombre_mois_consecutifs_sans_vente": int(consec),
        })

    feats = pd.DataFrame(rows)

    # Stock dormant : (dernier stock × coût d'achat)
    last_stock = (
        transactions.sort_values("date")
        .groupby("produit_id")["stock_apres_vente"]
        .last()
        .rename("stock_courant")
    )
    cout = transactions.groupby("produit_id")["cout_achat_unitaire"].median().rename("cout_achat_median")
    feats = feats.merge(last_stock.reset_index(), on="produit_id", how="left")
    feats = feats.merge(cout.reset_index(), on="produit_id", how="left")
    feats["valeur_stock_dormant"] = feats["stock_courant"].fillna(0) * feats["cout_achat_median"].fillna(0)

    # Variation relative du prix : pente OLS du prix mensuel moyen / prix moyen
    prix_mensuel = (
        transactions
        .assign(mois_periode=transactions["date"].dt.to_period("M"))
        .groupby(["produit_id", "mois_periode"])["prix_vente_unitaire"]
        .mean()
        .reset_index()
    )
    var_prix: dict[str, float] = {}
    for pid, grp in prix_mensuel.groupby("produit_id"):
        prices = grp["prix_vente_unitaire"].to_numpy(dtype=float)
        if len(prices) < 3:
            var_prix[pid] = 0.0
            continue
        mean_price = prices.mean()
        slope = linear_slope(prices[-6:]) if len(prices) >= 6 else linear_slope(prices)
        var_prix[pid] = float(slope / mean_price) if mean_price > 0 else 0.0
    feats["variation_relative_prix"] = feats["produit_id"].map(var_prix).fillna(0.0)

    return feats


# =================================================================== #
# Isolation Forest
# =================================================================== #
@dataclass
class ObsolescenceResult:
    df: pd.DataFrame
    contamination: float
    feature_cols: tuple[str, ...]
    n_flagged: int
    n_rules_only: int


def detect_obsolescence(
    features: pd.DataFrame,
    feature_cols: Iterable[str] = DEFAULT_FEATURES,
    contamination: float = IFOREST_DEFAULT_CONTAMINATION,
    apply_business_rules: bool = True,
    random_state: int = IFOREST_RANDOM_STATE,
) -> ObsolescenceResult:
    """Applique Isolation Forest + règles métier sur les features fournies.

    Parameters
    ----------
    apply_business_rules : bool, default True
        Si vrai, on flagge en plus tout produit vérifiant l'une des règles :
        - ``jours_depuis_derniere_vente >= 180``
        - ``nombre_mois_consecutifs_sans_vente >= 6``
        - ``ratio_ventes_3m_vs_12m < 0.1`` ET produit > 365 jours
    """
    feature_cols = tuple(feature_cols)
    out = features.copy()

    X = out[list(feature_cols)].fillna(0).to_numpy(dtype=float)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    iforest = IsolationForest(
        n_estimators=IFOREST_N_ESTIMATORS,
        max_samples=min(IFOREST_MAX_SAMPLES, len(out)),
        contamination=contamination,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )
    iforest.fit(Xn)
    out["score_obsolescence"] = iforest.decision_function(Xn)
    out["a_risque_obsolescence"] = (iforest.predict(Xn) == -1).astype(int)

    iforest_flag = out["a_risque_obsolescence"].copy()
    n_rules_only = 0
    if apply_business_rules:
        age_proxy = out.get("age_produit_jours", pd.Series(365, index=out.index))
        rule_mask = (
            (out["jours_depuis_derniere_vente"] >= 180)
            | (out["nombre_mois_consecutifs_sans_vente"] >= 6)
            | ((out["ratio_ventes_3m_vs_12m"] < 0.1) & (age_proxy > 365))
        )
        out.loc[rule_mask, "a_risque_obsolescence"] = 1
        n_rules_only = int(((out["a_risque_obsolescence"] == 1) & (iforest_flag == 0)).sum())

    n_flagged = int(out["a_risque_obsolescence"].sum())
    logger.info(
        "Obsolescence : %d produits à risque (Isolation Forest=%d, règles seules=%d) — contamination=%.2f",
        n_flagged, int(iforest_flag.sum()), n_rules_only, contamination,
    )
    return ObsolescenceResult(
        df=out,
        contamination=contamination,
        feature_cols=feature_cols,
        n_flagged=n_flagged,
        n_rules_only=n_rules_only,
    )


def sensitivity_analysis(
    features: pd.DataFrame,
    contaminations: Iterable[float] = (0.05, 0.10, 0.15, 0.20),
    feature_cols: Iterable[str] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """Compare le nombre de produits flagués selon la contamination."""
    rows: list[dict] = []
    for c in contaminations:
        res = detect_obsolescence(features, feature_cols=feature_cols, contamination=c, apply_business_rules=False)
        rows.append({
            "contamination": c,
            "n_flagged_iforest": res.n_flagged,
            "pct_catalogue": round(res.n_flagged / len(features) * 100, 1),
            "score_seuil_anomalie": float(res.df.loc[res.df["a_risque_obsolescence"] == 1, "score_obsolescence"].max()) if res.n_flagged else float("nan"),
        })
    return pd.DataFrame(rows)
