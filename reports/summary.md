# Synthèse exécution pipeline Zenith Supply Chain

- **Période couverte** : 2022-08-01 → 2025-07-31
- **Transactions nettoyées** : 66 915
- **Produits analysés** : 250

## Classification ABC
- Classe A : 69 produits
- Classe B : 59 produits
- Classe C : 122 produits

## Répartition ABC × XYZ
- AX : 20 produits
- AY : 46 produits
- AZ : 3 produits
- BX : 25 produits
- BY : 29 produits
- BZ : 5 produits
- CX : 54 produits
- CY : 50 produits
- CZ : 18 produits

## Détection d'obsolescence
- Produits à risque : **73** / 250 (29.2%)

## Performance des modèles de prévision (MAE moyen, par classe ABC × modèle)

| classe_abc   | modele   |   n_produits |   mae_moy |   rmse_moy |   mape_moy |
|:-------------|:---------|-------------:|----------:|-----------:|-----------:|
| A            | croston  |            2 |    12.819 |     13.886 |    332.365 |
| A            | lstm     |           48 |     7.234 |      8.378 |    126.464 |
| A            | obsolete |           19 |   nan     |    nan     |    nan     |
| B            | lightgbm |           39 |     8.061 |      9.368 |     79.644 |
| B            | obsolete |           20 |   nan     |    nan     |    nan     |
| C            | croston  |            7 |     4.368 |      5.248 |    165.716 |
| C            | obsolete |           34 |   nan     |    nan     |    nan     |
| C            | sarima   |           81 |    16.11  |     18.996 |    119.274 |

## Comparaison politique empirique vs politique optimisée

| indicateur              |   politique_empirique |   politique_optimisee |      delta |   delta_pct |
|:------------------------|----------------------:|----------------------:|-----------:|------------:|
| nb_commandes            |         501           |         698           |     197    |       39.32 |
| qte_commandee_totale    |       24420           |       15726           |   -8694    |      -35.6  |
| valeur_commande_totale  |           1.66414e+06 |           1.17357e+06 | -490574    |      -29.48 |
| stock_moyen_immo_usd    |        1303.04        |         134.91        |   -1168.12 |      -89.65 |
| nb_ruptures_unites      |        4439.42        |        4251.45        |    -187.98 |       -4.23 |
| marge_perdue_usd        |       40699.6         |       37492.7         |   -3206.88 |       -7.88 |
| cout_commande_total_usd |       25050           |       34900           |    9850    |       39.32 |
| cout_stockage_total_usd |       58636.7         |        4298.31        |  -54338.4  |      -92.67 |
| cout_total_simule_usd   |      124386           |       76691           |  -47695.2  |      -38.34 |
| ca_realise_usd          |           1.80268e+06 |           1.81837e+06 |   15696.1  |        0.87 |
| taux_service_pct        |          81.83        |          82.6         |       0.77 |        0.94 |

## Fichiers générés
- `data/processed/transactions_clean.parquet`
- `data/processed/products_features.parquet`
- `data/results/classification.csv`
- `data/results/obsolescence.csv`
- `data/results/forecasts.csv`
- `data/results/forecast_metrics.csv`
- `data/results/forecast_metrics_by_class.csv`
- `data/results/optimization_plan.csv`
- `data/results/baseline_plan.csv`
- `data/results/financial_comparison.csv`
- `reports/figures/*.png`