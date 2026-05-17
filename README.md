# Zenith Supply Chain — Planification intelligente des approvisionnements

> Implémentation complète du mémoire :
> **« Prédiction et optimisation des commandes pour éviter ruptures et surstocks :
> planification intelligente des approvisionnements — cas de Zenith Informatique et Bureautique »**
> KASENDE NGELEKA Victoire — Master 2 Data Science orientée Supply Chain (UDBL, 2026).

Ce dépôt implémente, en Python pur et open source, le pipeline décisionnel
décrit dans le **chapitre 3 du mémoire** :

```
Données brutes  →  Prétraitement  →  Feature engineering  →  Classification (ABC × XYZ + K-Means)
                                                          →  Détection d'obsolescence (Isolation Forest)
                                                          →  Prévision adaptée par classe
                                                              (SARIMA / LightGBM / LSTM / Croston)
                                                          →  Optimisation linéaire (PuLP / CBC)
                                                          →  Évaluation (technique + financière)
```

## Architecture du dépôt

```
zenith-supply-chain/
├── data/
│   ├── raw/                          # données brutes (transactions + catalogue)
│   ├── processed/                    # données nettoyées + features (.parquet)
│   └── results/                      # sorties de chaque étape (.csv)
├── src/zenith/
│   ├── config.py                     # constantes (seuils ABC, budget, etc.)
│   ├── preprocessing.py              # §3.3.1 nettoyage
│   ├── feature_engineering.py        # §3.3.2 features temporelles / produit / financières
│   ├── classification.py             # §3.4 ABC × XYZ + K-Means
│   ├── obsolescence.py               # §3.5 Isolation Forest
│   ├── forecasting.py                # §3.6 SARIMA / LightGBM / LSTM / Croston
│   ├── optimization.py               # §3.7 MILP via PuLP
│   ├── evaluation.py                 # §3.8 métriques techniques + financières
│   └── viz.py                        # figures pour rapport / dashboard
├── scripts/
│   └── run_pipeline.py               # orchestrateur end-to-end
├── notebooks/
│   └── 01_exploration.ipynb          # EDA + lecture des résultats
├── reports/
│   ├── figures/                      # PNG pour le mémoire
│   ├── tables/
│   └── summary.md                    # synthèse exécution
├── requirements.txt
└── setup.py
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Exécution

```bash
python scripts/run_pipeline.py
```

Le script exécute les sept étapes du pipeline et produit :

- `data/processed/transactions_clean.parquet` — données nettoyées + features
- `data/processed/products_features.parquet` — table produit enrichie
- `data/results/classification.csv` — ABC × XYZ + cluster K-Means
- `data/results/obsolescence.csv` — score Isolation Forest + flag risque
- `data/results/forecasts.csv` — prévisions mensuelles par produit
- `data/results/forecast_metrics.csv` — MAE / RMSE / MAPE par produit
- `data/results/forecast_metrics_by_class.csv` — synthèse par classe ABC × modèle
- `data/results/optimization_plan.csv` — plan de commandes optimisé (MILP)
- `data/results/baseline_plan.csv` — plan empirique (politique actuelle simulée)
- `data/results/financial_comparison.csv` — KPI financiers comparés
- `reports/figures/*.png` — visualisations
- `reports/summary.md` — synthèse exécutive

## Correspondance code ↔ mémoire

| Chapitre / section | Module Python |
|--------------------|---------------|
| §3.3.1 Nettoyage   | `zenith.preprocessing.clean_dataset` |
| §3.3.2 Features    | `zenith.feature_engineering.*` |
| §3.3.3 Split temporel | `zenith.preprocessing.temporal_split` |
| §3.4.1 Classification ABC | `zenith.classification.classify_abc` |
| §3.4.2 Analyse XYZ | `zenith.classification.classify_xyz` |
| §3.4.3 K-Means     | `zenith.classification.kmeans_clustering` |
| §3.5 Isolation Forest | `zenith.obsolescence.detect_obsolescence` |
| §3.6.2 SARIMA      | `zenith.forecasting.sarima_forecast` |
| §3.6.3 LightGBM    | `zenith.forecasting.lightgbm_forecast` |
| §3.6.4 LSTM        | `zenith.forecasting.lstm_forecast` |
| §3.7 MILP          | `zenith.optimization.optimize_orders` |
| §3.8 Évaluation    | `zenith.evaluation.*` |

## Hypothèses de recherche validées

- **H1 — Différenciation par classe** : `forecasting.choose_model_for_class` applique
  un modèle distinct par couple (ABC, XYZ), conforme à la table 3.3 du mémoire.
- **H2 — Complémentarité statistique / apprentissage** : SARIMA pour
  produits saisonniers, LightGBM pour le gros du catalogue, LSTM (via MLP
  séquentiel CPU) pour la classe A.
- **H3 — Détection précoce d'obsolescence** : Isolation Forest +
  règles métier dans `obsolescence.detect_obsolescence`.
- **H4 — Optimisation supérieure à l'heuristique** : comparaison
  `optimize_orders` vs `simulate_baseline_policy` dans `evaluation.compare_policies`.
- **H5 — Validation par l'impact financier** : KPI `cout_total_simule_usd`,
  `marge_perdue_usd`, `stock_moyen_immo_usd`, `taux_service_pct`.
- **H6 — Faisabilité PME** : 100 % open source, CPU only, ~3 minutes
  d'exécution sur un poste standard.

## Paramètres clés (modifiables dans `src/zenith/config.py`)

| Paramètre | Valeur | Source |
|-----------|--------|--------|
| Seuil ABC A / B | 70 % / 90 % | §3.4.1 |
| Seuils XYZ (CV) | 0.5 / 1.0 | §3.4.2 |
| Contamination Isolation Forest | 10 % | §3.5.3 |
| Délai livraison Dubaï / Chine | 35 / 55 jours | §3.7 |
| Coût commande fixe | 50 USD | §3.7.2 |
| Taux stockage | 0.1 % du coût d'achat / jour | §3.7.2 |
| Budget mensuel par défaut | 500 000 USD | calibré sur historique |
| Horizon de planification | 6 mois | couvre un cycle d'importation |

## Auteur

KASENDE NGELEKA Victoire — `victorykasende@gmail.com`
