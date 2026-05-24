"""Exécute l'Étape 2 — Prétraitement & feature engineering.

Produit :
- data/processed/zenith_clean.csv      (transactions nettoyées, 1 ligne = 1 vente)
- data/features/zenith_features.csv    (transactions + variables dérivées)
- data/features/product_features.csv   (agrégats par produit)
- outputs/tables/preprocessing_report.csv (avant/après par étape)

Le module ``src.preprocessing`` est appelé tel quel ; ce script joue le rôle
d'orchestrateur et de point d'entrée CLI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_pipeline
from src.utils import FEATURES_DIR, PROCESSED_DIR, TAB_DIR, setup_logger, temporal_split

logger = setup_logger("pipeline.preprocessing")


def main() -> int:
    logger.info("Lancement du prétraitement Zenith")
    clean, features, product_feats, report = preprocess_pipeline()

    clean_path = PROCESSED_DIR / "zenith_clean.csv"
    feat_path = FEATURES_DIR / "zenith_features.csv"
    prod_path = FEATURES_DIR / "product_features.csv"
    report_path = TAB_DIR / "preprocessing_report.csv"

    clean.to_csv(clean_path, index=False)
    features.to_csv(feat_path, index=False)
    product_feats.to_csv(prod_path, index=False)
    report.to_csv(report_path, index=False)

    # Partitionnement temporel (information seule, non sauvegardée car les
    # consommateurs en aval refont leur propre split à la demande).
    train, val, test = temporal_split(clean)
    logger.info(
        "Split temporel — train: %d (%.1f%%) | val: %d (%.1f%%) | test: %d (%.1f%%)",
        len(train), 100 * len(train) / len(clean),
        len(val), 100 * len(val) / len(clean),
        len(test), 100 * len(test) / len(clean),
    )

    n_tx = len(clean)
    n_features = features.shape[1]
    n_products = len(product_feats)
    logger.info(
        "Sorties produites : %s, %s, %s, %s (%d transactions, %d features, %d produits)",
        clean_path.relative_to(ROOT), feat_path.relative_to(ROOT),
        prod_path.relative_to(ROOT), report_path.relative_to(ROOT),
        n_tx, n_features, n_products,
    )

    # Affichage récap rapide
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    logger.info("Rapport avant/après :\n%s", report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
