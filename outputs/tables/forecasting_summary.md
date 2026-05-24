# Synthèse de l'Étape 5 — Prévision de la demande

## Stratégie d'affectation modèle / classe

| Classe / profil | Modèle | Justification |
|---|---|---|
| A non-intermittent | **LSTM-like** (MLP séquentiel scikit-learn) | Fort enjeu financier, séquences de 30 jours |
| B | **LightGBM** | Bon compromis perf/coût, features riches |
| C non-intermittent | **SARIMA auto_arima** (s=12) | Méthode statistique légère et interprétable |
| Intermittent (Z ou ≤30 % mois actifs) | **Croston SBA** | Demande sporadique |
| Obsolète (Étape 4) | **Prévision = 0** | Exclus du réapprovisionnement |
| Cold-start (<6 mois d'historique) | **Analogie famille** | Moyenne pondérée des produits frères |

> *Note* : faute de TensorFlow disponible (principe de frugalité §3.2.1), le LSTM est implémenté via un MLP scikit-learn sur fenêtres glissantes de 30 points — équivalence opérationnelle pour des séries mensuelles courtes.

## Volumétrie d'exécution

- 250 produits traités en **192.6 s** (poste standard, mode séquentiel).
- 6 mois de prévision opérationnelle écrits dans `previsions_complet.csv` (≈ 1 500 lignes).
- 4 mois de prévision test (avril–juillet 2025) pour évaluation MAE/RMSE/MAPE.

## Répartition des modèles retenus (250 produits)

| Modèle | Produits |
|---|---:|
| SARIMA | 80 |
| LSTM-like | 49 |
| LightGBM | 41 |
| Croston SBA | 7 |
| Obsolète | 12 |
| Analogie famille (cold-start) | 61 |

> Le nombre de produits "obsolètes" comptés ici (12) diffère du total Étape 4 (73) car certains produits flagués obsolètes sont aussi cold-start (sans historique suffisant à fin 2024 pour entraînement) — ils basculent alors sur l'analogie famille.

## Performance (MAE moyen sur la fenêtre test)

| Classe | Modèle | n_produits | MAE moy | RMSE moy | MAPE moy |
|---|---|---:|---:|---:|---:|
| A | LSTM-like | 49 | **6.5** | 7.7 | 110.6 % |
| A | Croston SBA | 1 | 5.9 | 6.5 | 75.8 % |
| A | Obsolète | 4 | 10.8 | 13.5 | 100 % |
| B | LightGBM | 41 | **7.6** | 9.0 | 76.6 % |
| B | Obsolète | 4 | 25.3 | 30.4 | 100 % |
| C | SARIMA | 80 | **9.9** | 11.8 | 74.9 % |
| C | Croston SBA | 6 | 5.0 | 5.8 | 156.0 % |
| C | Obsolète | 4 | 27.8 | 32.0 | 100 % |

## Top 5 des features les plus utiles (LightGBM, gain moyen)

1. `mois_sin` — 2 107 (saisonnalité cyclique)
2. `ma_12` — 2 071 (moyenne mobile 12 mois)
3. `lag_12` — 2 063 (valeur il y a 1 an)
4. `lag_1` — 2 033 (dernière valeur)
5. `lag_6` — 1 850 (valeur il y a 6 mois)

> La triple présence des features de saisonnalité confirme la forte composante annuelle des ventes Zenith (rentrée scolaire d'août-septembre).

## Sorties

- `outputs/tables/previsions_complet.csv` (1 500 lignes — produit × mois × prevision × modèle × IC)
- `outputs/tables/comparaison_modeles.csv` (MAE / RMSE / MAPE par produit)
- `outputs/tables/forecast_metrics_by_class.csv` (synthèse par classe ABC × modèle)
- `outputs/tables/lgbm_feature_importance.csv`
- `outputs/models/sarima_signatures.csv` (paramètres optimaux par produit)
- `outputs/figures/fc_01_mae_by_class.png` — boxplot MAE par classe / modèle
- `outputs/figures/fc_02_real_vs_pred.png` — courbes réel vs prévu
- `outputs/figures/fc_03_model_distribution.png` — répartition des modèles
- `outputs/figures/fc_04_lgbm_feature_importance.png` — importance moyenne

## Tests unitaires

10 tests dans `tests/test_forecasting.py` (sélection modèle obsolète/cold-start/intermittent, naïve, Croston SBA, LightGBM, métriques, séries mensuelles). Tous passants (`pytest tests/test_forecasting.py -v` ⇒ **10 passed**).
