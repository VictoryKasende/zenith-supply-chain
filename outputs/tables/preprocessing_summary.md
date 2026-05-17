# Synthèse de l'Étape 2 — Prétraitement

## Volumétrie

| Étape | Lignes | Δ | Détail |
|---|---:|---:|---|
| Brut | 68 258 | — | 18 colonnes, période 2022-08-01 → 2025-07-31 |
| Après doublons | 67 586 | **–672** | doublons exacts |
| Après normalisation famille | 67 586 | 0 | **66 libellés** corrigés (Cartouches→Cartouche, Imprimente→Imprimante, Ordinator→Ordinateur, Accessoires→Accessoire, Réseau→Reseau) |
| Après correction prix aberrants | 67 586 | 0 | **2 276** prix divisés par 10 (saisies × 10), **1 033** prix imputés par médiane famille |
| Après isolation des retours | 67 250 | **–336** | retours mis dans `est_retour=True`, exclus de la modélisation |
| Après imputation coût | 67 250 | 0 | 0 lignes (catalogue déjà complet) |
| Après imputation marque | 67 250 | 0 | 0 lignes (catalogue déjà complet) |
| Après imputation mode_paiement | 67 250 | 0 | **1 680** lignes (B2B→Crédit, B2C→Comptant) |
| Après imputation client_nom | 67 250 | 0 | **2 531** B2C → "Anonyme" |
| Après interpolation stock | 67 250 | 0 | **673** valeurs interpolées linéairement par produit |
| Après lignes vides | 67 250 | 0 | aucune ligne avec >50 % de manquants |
| Après cohérence montant | 67 250 | 0 | **337** montants recalculés (prix × quantité, tolérance 1 %) |

## Features dérivées

`zenith_features.csv` contient désormais **41 colonnes** vs 18 brutes :

**Temporelles (12)** — `annee, mois, trimestre, semaine, jour_semaine, jour_annee, est_weekend, est_fin_de_mois, mois_sin, mois_cos, jour_semaine_sin, jour_semaine_cos`

**Événementielles RDC (4)** — `est_rentree_scolaire (août-sept), est_rentree_academique (oct-nov), est_saison_seche (mai-sept), est_periode_pic_b2b (jan-mars, nov-déc)`

**Financières (4)** — `marge_unitaire, benefice_transaction, taux_marge_pct, valeur_stock_immobilisee`

**Rupture (2)** — `rupture_signalee, jours_consecutifs_rupture`

**Conservation (18 + 1)** — toutes les colonnes brutes + `est_retour` (booléen pour les retours isolés).

## Agrégats produit (250 produits)

`product_features.csv` contient pour chaque produit :
`ventes_totales_36mois, ca_total_36mois, nb_transactions, date_premiere_vente, date_derniere_vente, age_produit_jours, jours_depuis_derniere_vente, ventes_moyennes_mensuelles, ecart_type_ventes_mensuelles, nombre_mois_avec_ventes, coefficient_variation, tendance_3_mois, tendance_6_mois, prix_vente_unitaire_moyen, stock_courant`.

## Partitionnement temporel

| Ensemble | Période | Lignes | Part |
|---|---|---:|---:|
| Train | 2022-08-01 → 2024-12-31 | 57 062 | 84.9 % |
| Validation | 2025-01-01 → 2025-03-31 | 4 456 | 6.6 % |
| Test | 2025-04-01 → 2025-07-31 | 5 732 | 8.5 % |

La répartition observée diffère légèrement de la cible 80/10/10 indiquée dans
le mémoire ; cela tient à la densité naturelle des transactions par mois.

## Tests unitaires

9 tests dans `tests/test_preprocessing.py` couvrent :
- Parsing de dates mixtes ;
- Correction des prix aberrants (×10 et imputation médiane) ;
- Recalcul cohérent du `montant_total` après correction ;
- Normalisation des libellés famille ;
- Isolation des retours ;
- Variables temporelles cycliques et drapeaux saisonniers ;
- Cohérence `montant_total ≈ prix × quantité` (tolérance 1 %) ;
- Pipeline complet sur un mini-dataset.

Tous les tests passent (`pytest tests/test_preprocessing.py -v` ⇒ **9 passed**).
