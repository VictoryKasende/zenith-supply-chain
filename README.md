# Zenith Supply Chain — Planification intelligente des approvisionnements

> Implémentation complète du mémoire :
> **« Prédiction et optimisation des commandes pour éviter ruptures et surstocks :
> planification intelligente des approvisionnements — cas de Zenith Informatique et Bureautique »**
> KASENDE NGELEKA Victoire — Master 2 Data Science orientée Supply Chain (UDBL, 2026).

Ce dépôt livre :
1. un **système opérationnel** (pipeline Python + tests + outil interactif Streamlit) que Zenith peut utiliser au quotidien ;
2. les **datasets et tableaux de bord Power BI** prêts à l'emploi ;
3. l'intégralité des **résultats expérimentaux** pour le **Chapitre 4 du mémoire**.

## Verdict de l'évaluation (6/6 hypothèses validées)

| Hypothèse | Verdict | Valeur mesurée |
|---|---|---:|
| **H1** — Différenciation par classe | ✅ Validée | MAE -57.5 % vs pire modèle isolé |
| **H2** — Complémentarité stat / ML | ✅ Validée | 2/3 classes ont leur modèle attendu en tête |
| **H3** — Détection précoce d'obsolescence | ✅ Validée | Rappel 100 % (58/58) |
| **H4** — Optimisation > heuristique | ✅ Validée | Coût total LP -28 % vs empirique |
| **H5** — Impact financier mesurable | ✅ Validée | Bénéfice net +5 259 USD sur 3 mois |
| **H6** — Faisabilité PME | ✅ Validée | Pipeline complet en 213 s, sans GPU, 20 dépendances |

## Pipeline en 9 étapes

```
Étape 1 — Exploration (EDA)           → 15 figures + eda_summary.csv
Étape 2 — Prétraitement & features    → zenith_clean.csv + zenith_features.csv
Étape 3 — Classification ABC × XYZ × K-Means → classification_produits.csv
Étape 4 — Détection d'obsolescence    → produits_obsoletes.csv (73 produits)
Étape 5 — Prévisions adaptées         → previsions_complet.csv (1 500 lignes)
Étape 6 — Optimisation LP             → commandes_recommandees.csv (866 lignes)
Étape 7 — Évaluation globale          → rapport_evaluation.md
Étape 8 — Outil Streamlit             → app/zenith_tool.py (6 pages)
Étape 9 — Datasets Power BI           → outputs/powerbi/ (8 CSV + guide)
```

## Architecture du dépôt

```
zenith-supply-chain/
├── data/
│   ├── raw/                # zenith_dataset_brut.csv, catalogue_produits_250.csv
│   ├── processed/          # zenith_clean.csv (nettoyé)
│   └── features/           # zenith_features.csv, product_features.csv
├── src/                    # modules Python
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocessing.py    # §3.3 nettoyage + feature engineering
│   ├── classification.py   # §3.4 ABC × XYZ + K-Means
│   ├── obsolescence.py     # §3.5 Isolation Forest + règles métier
│   ├── forecasting.py      # §3.6 SARIMA / LightGBM / LSTM-like / Croston
│   ├── optimization.py     # §3.7 LP avec PuLP/CBC
│   ├── evaluation.py       # §3.8 validation H1–H6
│   └── powerbi_export.py   # construction du datamodel Power BI
├── scripts/                # orchestrateurs ligne de commande
│   ├── run_eda.py
│   ├── run_preprocessing.py
│   ├── run_classification.py
│   ├── run_obsolescence.py
│   ├── run_forecasting.py
│   ├── run_optimization.py
│   ├── run_evaluation.py
│   └── run_powerbi_export.py
├── notebooks/              # narratifs Jupyter (1 par étape)
│   ├── 01_exploration.ipynb
│   ├── 02_pretraitement.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_obsolescence.ipynb
│   ├── 05_previsions.ipynb
│   ├── 06_optimisation.ipynb
│   └── 07_evaluation.ipynb
├── app/
│   └── zenith_tool.py      # application Streamlit interactive (6 pages)
├── tests/                  # 55 tests pytest passants
│   ├── test_preprocessing.py
│   ├── test_classification.py
│   ├── test_obsolescence.py
│   ├── test_forecasting.py
│   ├── test_optimization.py
│   ├── test_evaluation.py
│   ├── test_app.py
│   └── test_powerbi_export.py
├── outputs/
│   ├── figures/            # 40+ figures publication-ready
│   ├── tables/             # 25+ CSV résultats + synthèses Markdown
│   ├── models/             # signatures SARIMA, etc.
│   ├── powerbi/            # 8 CSV pour Power BI + DASHBOARD_GUIDE.md
│   └── rapport_evaluation.md  # rapport final pour Chapitre 4
├── .streamlit/config.toml  # thème Zenith
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Exécution du pipeline complet

```bash
# Une étape à la fois (recommandé) :
python scripts/run_eda.py
python scripts/run_preprocessing.py
python scripts/run_classification.py
python scripts/run_obsolescence.py
python scripts/run_forecasting.py        # ~3 min
python scripts/run_optimization.py
python scripts/run_evaluation.py
python scripts/run_powerbi_export.py
```

## Lancement de l'application interactive

```bash
streamlit run app/zenith_tool.py
# puis ouvrir http://localhost:8501
```

**6 pages** dans la sidebar :
1. 📊 Tableau de bord global
2. 📦 Classification produits
3. ⚠️ Alertes obsolescence
4. 🔮 Prévisions de demande
5. 🛒 Recommandations de commande
6. 🧪 Simulation what-if (slider budget / niveau service)

## Tableaux de bord Power BI

1. Lancer `python scripts/run_powerbi_export.py` pour générer les 8 CSV dans
   `outputs/powerbi/`.
2. Ouvrir Power BI Desktop → **Accueil → Texte/CSV** → charger les 8 fichiers.
3. Suivre **`outputs/powerbi/DASHBOARD_GUIDE.md`** pour créer les
   4 dashboards : pilotage commercial, ABC × XYZ, alertes obsolescence,
   optimisation des commandes.

## Lancement des tests

```bash
PYTHONPATH=. pytest tests/ -v
# 55 passed
```

## Correspondance code ↔ mémoire

| Section du mémoire | Module / script |
|---|---|
| §3.3.1 Nettoyage | `src/preprocessing.py::clean_dataset` |
| §3.3.2 Feature engineering | `src/preprocessing.py::engineer_features` |
| §3.3.3 Split temporel | `src/utils.py::temporal_split` |
| §3.4 Classification ABC × XYZ | `src/classification.py` |
| §3.5 Isolation Forest | `src/obsolescence.py::detect_obsolescence` |
| §3.6.2 SARIMA auto | `src/forecasting.py::sarima_auto` |
| §3.6.3 LightGBM | `src/forecasting.py::lightgbm_forecast` |
| §3.6.4 LSTM | `src/forecasting.py::lstm_like_forecast` |
| §3.7 MILP / LP commandes | `src/optimization.py::optimize_orders` |
| §3.8 Évaluation H1–H6 | `src/evaluation.py` |

## Paramètres clés (modifiables dans `src/optimization.py`)

| Paramètre | Valeur | Source |
|---|---:|---|
| Seuils ABC A / B | 70 % / 90 % | §3.4.1 |
| Seuils XYZ (CV) | 0.5 / 1.0 | §3.4.2 |
| Contamination Isolation Forest | 10 % | §3.5.3 |
| Délais Dubaï / Chine | 1 mois / 2 mois | §3.7 |
| Coût fixe par commande | 50 USD | §3.7.2 |
| Taux stockage | 0.1 % / jour | §3.7.2 |
| Pénalité rupture client | +20 % | §3.7.2 |
| Pondération service classe A | × 4.0 | §3.7.4 (soft) |
| Budget mensuel par défaut | 500 000 USD | calibré sur historique |
| Capacité de stockage | 5 000 m³ | hypothèse PME |
| Horizon de planification | 3 mois | §3.7 |

## Auteur

**KASENDE NGELEKA Victoire** — `victorykasende@gmail.com`
Master 2 Data Science orientée Supply Chain — Université Don Bosco de
Lubumbashi — Juin 2026.
