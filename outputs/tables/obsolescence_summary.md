# Synthèse de l'Étape 4 — Détection d'obsolescence

## Résultat global

| Indicateur | Valeur |
|---|---:|
| Produits à risque | **73 / 250** (29.2 %) |
| Détectés par Isolation Forest seul | 25 |
| Ajoutés par règles métier seules | 48 |
| Valeur du stock dormant total | **38 387 USD** |

## Sensibilité au paramètre `contamination` (Isolation Forest pur)

| `contamination` | Produits flagués | % catalogue |
|---:|---:|---:|
| 0.05 | 13 | 5.2 % |
| **0.10** | **25** | **10.0 %** |
| 0.15 | 38 | 15.2 % |
| 0.20 | 50 | 20.0 % |

> Le réglage de référence (`contamination=0.10`, mémoire §3.5.3) sélectionne 25 produits, complété par 48 alertes via règles métier ⇒ 73 au total. Ce mix algorithme + heuristique permet d'éviter aussi bien les faux négatifs (produits évidemment dormants) que les faux positifs (produits récents avec faible historique).

## Croisement Isolation Forest × classe ABC

| Classe ABC | Actifs | À risque | % à risque |
|---|---:|---:|---:|
| A | 50 | 19 | 27.5 % |
| B | 41 | 18 | 30.5 % |
| C | 86 | 36 | 29.5 % |

L'obsolescence touche les trois classes de manière équilibrée — ce qui justifie une **vérification automatique** plutôt qu'une simple règle « seuls les C peuvent être obsolètes ».

## Features utilisées (par produit)

1. `jours_depuis_derniere_vente`
2. `tendance_3_mois`
3. `tendance_6_mois`
4. `ratio_ventes_3m_vs_12m`
5. `nombre_mois_consecutifs_sans_vente`
6. `valeur_stock_dormant` (= stock courant × coût d'achat médian)
7. `variation_relative_prix` (pente OLS du prix mensuel moyen / prix moyen)

## Paramétrage Isolation Forest

- `n_estimators` = 100
- `contamination` = 0.10
- `max_samples` = 256
- `bootstrap` = False
- `random_state` = 42

## Règles métier de filet (appliquées en complément)

Un produit est flagué également si **au moins une** condition est vraie :
1. `jours_depuis_derniere_vente >= 180`
2. `nombre_mois_consecutifs_sans_vente >= 6`
3. `ratio_ventes_3m_vs_12m < 0.1` ET `age_produit_jours > 365`

## Sorties

- `outputs/tables/obsolescence_features.csv` (250 lignes × 9 colonnes)
- `outputs/tables/produits_obsoletes.csv` (73 lignes flagués)
- `outputs/tables/obsolescence_sensitivity.csv`
- 6 figures : `outputs/figures/obs_01..06.png`

## Tests unitaires

5 tests dans `tests/test_obsolescence.py` (build_features, dormant > actif, détection produit dormant, règles métier captent les évidents, monotonie sensibilité). Tous passants.

## Impact pour la suite

- **Étape 5 (Prévisions)** : produits flagués → modèle = `obsolete` (prévisions à zéro, exclus de l'évaluation MAE).
- **Étape 6 (Optimisation)** : produits flagués → aucune commande recommandée, suivi en page « alerte » pour déstockage manuel.
- **Étape 8 (Streamlit)** : page dédiée « Alertes obsolescence ».
- **Étape 9 (Power BI)** : dimension `alerte_obsolescence` dans le datamodel.
