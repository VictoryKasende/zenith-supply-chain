# Zenith Supply Chain — Planification intelligente des approvisionnements

> Implémentation complète du mémoire :
> **« Prédiction et optimisation des commandes pour éviter ruptures et surstocks :
> planification intelligente des approvisionnements — cas de Zenith Informatique et Bureautique »**
> KASENDE NGELEKA Victoire — Master 2 Data Science orientée Supply Chain (UDBL, 2026).

Ce dépôt livre :
1. un **système opérationnel** (pipeline Python + tests + outil interactif Streamlit) que Zenith peut utiliser au quotidien ;
2. les **datasets et tableaux de bord Power BI** prêts à l'emploi ;
3. l'intégralité des **résultats expérimentaux** pour le **Chapitre 4 du mémoire**.

## Verdict de l'évaluation (protocole de révision, sans fuite temporelle)

Prévisions évaluées en rolling-origin (16 origines mensuelles, tout réappris sur les
seules données passées) ; politiques de commande simulées sur la demande réellement
observée (période de test janvier → juillet 2025) et comparées à des politiques
classiques réglées sur une période de développement distincte. Chiffres, protocole et
sources : [`paper_results/RESULTS_FOR_PAPER.md`](paper_results/RESULTS_FOR_PAPER.md).

| Hypothèse | Verdict | Résultat mesuré |
|---|---|---|
| **H1** — Différenciation par classe | ❌ Non validée | MASE 0,836 contre 0,749 pour une moyenne mobile 3 mois, qui reste meilleure (Wilcoxon, p = 1,00). Le pipeline bat seulement le seasonal naive (p ajusté = 0,006). |
| **H2** — Complémentarité stat / ML | ⚠️ Partielle | LightGBM bat SARIMA (Diebold-Mariano, p ajusté = 0,029) mais pas le seasonal naive (p = 0,155) ; SARIMA fait moins bien que le seasonal naive. |
| **H3** — Détection précoce d'obsolescence | ✅ Validée, avec réserve | Vérité prospective (6 mois) : rappel 0,931, précision 0,507, F1 = 0,657 [0,532 ; 0,743], alerte médiane 62 jours avant la dormance. Une simple règle « 90 jours sans vente » atteint F1 = 0,857, mais alerte 91 jours après. |
| **H4** — Optimisation > heuristique | ❌ Non validée | Coût −6,4 % face à la politique empirique, non significatif (p ajusté = 0,39). Une politique (s, S) bien réglée fait −27,6 % avec 96,9 % de service, contre 82,8 % pour le pipeline. |
| **H5** — Impact financier mesurable | ❌ Non validée | Gain net estimé à 25 707 USD/an, IC 95 % [−17 915 ; 67 188] : l'intervalle contient 0. |
| **H6** — Faisabilité PME | ❌ Non validée (critères ≤ 5 min et ≤ 20 dépendances) | 780 ± 231 s sur un portable 4 cœurs sans GPU, 375 Mo de mémoire, 23 dépendances déclarées. |

Ce qui est établi : la détection d'obsolescence a une vraie valeur (la retirer fait
passer le coût de 217 597 à 254 049 USD sur la période de test) ; le MILP de commande se
résout exactement en moins de 10 minutes, la relaxation arrondie n'est donc pas
nécessaire. Limite principale du pipeline : le LP planifie exactement la prévision, sans
stock de sécurité, ce qui plafonne le taux de service vers 83 %.

La version initiale annonçait 6/6 hypothèses validées : elle reposait sur des coûts
jugés contre la prévision plutôt que contre la demande, sur des multiplicateurs codés en
dur pour l'ablation et sur une vérité terrain d'obsolescence circulaire.

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
# 78 passed
```

## Expériences de révision (article IEEE)

Le package `experiments/` rejoue l'évaluation avec un protocole sans fuite
temporelle et écrit tous ses résultats dans `paper_results/` :

```bash
python -m experiments.run_all            # tout depuis zéro (≈ 4-5 h sur 4 cœurs, dont ≈ 1 h 15 de chronométrage)
python -m experiments.run_all --keep-cache --skip-benchmark   # réutilise les prévisions en cache
python -m experiments.rolling_origin     # ou une expérience seule (P0-1)
```

| Script | Expérience | Sortie principale |
|---|---|---|
| `paper_reproduction.py` | chiffres publiés recalculés par le code d'origine | `paper_method_reproduction.json` |
| `rolling_origin.py` | backtest rolling-origin, tout réappris sur ≤ t0 | `rolling_origin_errors.csv` |
| `metrics.py` | MAE, MASE, RMSSE, WAPE, biais | `forecast_metrics.csv` |
| `stats_tests.py` | Wilcoxon, Diebold-Mariano HLN, Holm, bootstrap | `stat_tests.json` |
| `obsolescence_eval.py` | vérité terrain prospective, 4 détecteurs | `obsolescence_eval.csv` |
| `milp_vs_lp.py` | MILP exact vs LP arrondi, faisabilité | `milp_vs_lp.json` |
| `policy_baselines.py` | politiques sur demande réalisée | `policy_comparison.csv` |
| `service_tradeoff.py` | coût / service, cibles de service | `service_tradeoff.csv` |
| `sensitivity.py` | sensibilité une-variable-à-la-fois | `sensitivity.csv` |
| `benchmark_env.py` | environnement, temps et mémoire mesurés | `runtime.csv` |
| `report.py` | synthèse chiffrée pour l'article | `RESULTS_FOR_PAPER.md` |

L'absence de fuite est vérifiée par `tests/test_exp_no_leakage.py`. Seed 42,
un thread BLAS/OpenMP ; `paper_results/_cache/` est régénérable et non versionné.

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
