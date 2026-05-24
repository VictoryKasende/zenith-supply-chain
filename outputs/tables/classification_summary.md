# Synthèse de l'Étape 3 — Classification ABC × XYZ × K-Means

## Répartition ABC (seuils 70 % / 90 %)

| Classe | Produits | % du catalogue | Cumul CA |
|---|---:|---:|---:|
| A | 69 | 27.6 % | ≤ 70 % |
| B | 59 | 23.6 % | 70-90 % |
| C | 122 | 48.8 % | > 90 % |

> **Confirmation Pareto** : 28 % des références génèrent 70 % du CA. Concentration plus faible que dans les distributions classiques (souvent 20/80) mais typique d'un catalogue informatique avec plusieurs gammes équivalentes (HP/Dell/Canon, etc.).

## Répartition XYZ (CV des ventes mensuelles, seuils 0.5 / 1.0)

| Classe | CV | Produits | Profil de demande |
|---|---|---:|---|
| X | < 0.5 | 98 | Stable, prévisible |
| Y | 0.5 – 1.0 | 126 | Modérément variable |
| Z | ≥ 1.0 | 26 | Très irrégulière |

## Matrice ABC × XYZ (250 produits, % du CA)

| | X | Y | Z |
|---|---:|---:|---:|
| **A** | 20 prod / **20.9 % CA** | 46 prod / **46.5 % CA** | 3 prod / 2.1 % CA |
| **B** | 25 prod / 8.5 % | 29 prod / 10.2 % | 5 prod / 1.6 % |
| **C** | 53 prod / 4.3 % | 51 prod / 4.5 % | 18 prod / 1.4 % |

**Lecture stratégique** : les cases **AX (21 %) et AY (47 %)** concentrent à elles seules ~68 % du CA — l'effort de prévision et de réapprovisionnement doit y être priorisé.

## K-Means (k* = 3, silhouette = 0.32)

Features utilisées (avec `log1p` sur ventes, CA, prix) :
- ventes_totales_36mois, ca_total_36mois, coefficient_variation,
- nombre_mois_avec_ventes, tendance_3_mois,
- jours_depuis_derniere_vente, prix_vente_unitaire_moyen.

| Cluster | Libellé métier | Produits | CA médian | Volumes médians | CV médian | Fraîcheur (j) |
|---|---|---:|---:|---:|---:|---:|
| 0 | **Dormant — risque obsolescence** | 36 | 20 700 USD | 76 | 1.00 | 695 |
| 1 | **Rotation modérée** | 88 | 9 655 USD | 1 298 | 0.44 | 1 |
| 2 | **Bestseller stable** | 126 | 81 259 USD | 323 | 0.58 | 5 |

## Stratégie de prévision par classe (utilisée en Étape 5)

| Classe | Modèle | Justification |
|---|---|---|
| A non-intermittents | **LSTM** | Fort enjeu financier, dépendances temporelles longues |
| B non-intermittents | **LightGBM** | Bon compromis perf/coût pour la majorité des produits courants |
| C non-intermittents | **SARIMA** | Méthode statistique légère pour les produits moins stratégiques |
| AZ / BZ / CZ (intermittents) | **Croston / SBA** | Demande sporadique avec longs trous |
| Cluster 0 (Dormant) | **Pas de prévision** | Exclus du réapprovisionnement (Étape 4) |

## Sorties

- `outputs/tables/classification_produits.csv` (250 lignes)
- `outputs/tables/abc_xyz_matrix.csv`
- `outputs/tables/kmeans_diagnostics.csv`
- `outputs/tables/cluster_profile.csv`
- `outputs/figures/cls_01_pareto_abc.png` (courbe de Pareto colorée par classe)
- `outputs/figures/cls_02_distribution_xyz.png`
- `outputs/figures/cls_03_matrice_abc_xyz.png`
- `outputs/figures/cls_04_elbow_silhouette.png`
- `outputs/figures/cls_05_pca_2d.png`
- `outputs/figures/cls_06_cluster_profile.png`
- `outputs/figures/cls_07_distribution_clusters.png`

## Tests unitaires

6 tests dans `tests/test_classification.py` :
- Respect des seuils Pareto ABC ;
- Toutes les classes A, B, C présentes ;
- Repli quantile XYZ si X vide ;
- Création de la colonne croisée `classe_abc_xyz` ;
- Dimensions de la matrice ABC × XYZ ;
- Pipeline K-Means complet renvoie labels valides et silhouette dans [-1, 1].

Tous passent (`pytest tests/test_classification.py -v` ⇒ **6 passed**).
