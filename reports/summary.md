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
| B            | lightgbm |           39 |     7.754 |      9.143 |     75.447 |
| B            | obsolete |           20 |   nan     |    nan     |    nan     |
| C            | croston  |            7 |     4.368 |      5.248 |    165.716 |
| C            | obsolete |           34 |   nan     |    nan     |    nan     |
| C            | sarima   |           81 |    15.725 |     18.594 |    116.956 |

## Comparaison politique empirique vs politique optimisée

| indicateur              |   politique_empirique |   politique_optimisee |      delta |   delta_pct |
|:------------------------|----------------------:|----------------------:|-----------:|------------:|
| nb_commandes            |         501           |         696           |     195    |       38.92 |
| qte_commandee_totale    |       24433           |       15749           |   -8684    |      -35.54 |
| valeur_commande_totale  |           1.66199e+06 |           1.17634e+06 | -485645    |      -29.22 |
| stock_moyen_immo_usd    |        1295.47        |         134.44        |   -1161.03 |      -89.62 |
| nb_ruptures_unites      |        4427.51        |        4234.97        |    -192.54 |       -4.35 |
| marge_perdue_usd        |       40618.2         |       37050.1         |   -3568.09 |       -8.78 |
| cout_commande_total_usd |       25050           |       34800           |    9750    |       38.92 |
| cout_stockage_total_usd |       58295.9         |        4283.13        |  -54012.8  |      -92.65 |
| cout_total_simule_usd   |      123964           |       76133.2         |  -47830.9  |      -38.58 |
| ca_realise_usd          |           1.80265e+06 |           1.82135e+06 |   18706    |        1.04 |
| taux_service_pct        |          81.88        |          82.67        |       0.79 |        0.96 |

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