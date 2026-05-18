"""Exécute l'Étape 9 — export des datasets pour Power BI.

Génère 8 fichiers CSV dans ``outputs/powerbi/`` prêts à être importés
dans Power BI Desktop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.powerbi_export import (
    build_alertes_powerbi,
    build_commandes_powerbi,
    build_dim_clients,
    build_dim_magasins,
    build_dim_produits,
    build_dim_temps,
    build_fact_ventes,
    build_previsions_powerbi,
)
from src.utils import PROCESSED_DIR, POWERBI_DIR, RAW_CATALOGUE, TAB_DIR, setup_logger

logger = setup_logger("pipeline.powerbi")


def main() -> int:
    logger.info("Étape 9 — Export des datasets pour Power BI")

    # ---- Chargement ----
    transactions = pd.read_csv(PROCESSED_DIR / "zenith_clean.csv", parse_dates=["date"])
    catalogue = pd.read_csv(RAW_CATALOGUE)
    classes = pd.read_csv(TAB_DIR / "classification_produits.csv")
    obs_feats = pd.read_csv(TAB_DIR / "obsolescence_features.csv")
    obs_list = pd.read_csv(TAB_DIR / "produits_obsoletes.csv")
    previsions = pd.read_csv(TAB_DIR / "previsions_complet.csv", parse_dates=["date"])
    commandes = pd.read_csv(TAB_DIR / "commandes_recommandees.csv")
    obsoletes_ids = set(obs_list["produit_id"].tolist())

    # ---- Construction ----
    fact = build_fact_ventes(transactions)
    dim_produits = build_dim_produits(catalogue, classes, obs_feats, obsoletes_ids)
    dim_clients = build_dim_clients(transactions)
    dim_magasins = build_dim_magasins(transactions)
    # Calendrier élargi jusqu'à la fin de l'horizon de prévision
    date_min = transactions["date"].min()
    date_max = max(transactions["date"].max(), pd.to_datetime(previsions["date"].max()))
    dim_temps = build_dim_temps(date_min, date_max + pd.offsets.MonthEnd(0))
    previsions_pbi = build_previsions_powerbi(previsions)
    commandes_pbi = build_commandes_powerbi(commandes)
    alertes_pbi = build_alertes_powerbi(obs_list, catalogue)

    # ---- Export CSV ----
    files = {
        "fact_ventes.csv": fact,
        "dim_produits.csv": dim_produits,
        "dim_clients.csv": dim_clients,
        "dim_magasins.csv": dim_magasins,
        "dim_temps.csv": dim_temps,
        "previsions.csv": previsions_pbi,
        "commandes_recommandees.csv": commandes_pbi,
        "alertes_obsolescence.csv": alertes_pbi,
    }
    for name, df in files.items():
        path = POWERBI_DIR / name
        df.to_csv(path, index=False)
        logger.info("→ %s : %d lignes × %d colonnes", path.relative_to(ROOT), *df.shape)

    logger.info("8 fichiers exportés dans outputs/powerbi/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
