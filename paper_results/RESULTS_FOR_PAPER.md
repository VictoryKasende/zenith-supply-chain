# Résultats pour la révision de l'article

Fichier généré par `python -m experiments.report` à partir de `paper_results/` (ne pas éditer à la main). Les valeurs « publiées » sont celles de l'article soumis.

## Vérifications préalables (V1 – V3)

**V1 — Taux de service (69,19 % → 69,94 %).** Le code calcule un *unit fill rate*
**contre la demande prévue** : `100 × (1 − Σ rupture / Σ demande_prevue)`
(`src/optimization.py:344`, demande ligne 332). Pour la politique LP, `rupture` est la
variable d'écart continue `R` du LP (ligne 186), non recalculée après l'arrondi des
quantités ; pour l'incumbent, `max(0, d − servi)` simulé (lignes 292-293).
`demande_prevue` porte sur août → octobre 2025, **après la fin des données (2025-07-31)** :
aucune demande réalisée n'intervient. Le code actuel donne
69,49 % → 70,23 % ;
69,19 / 69,94 % proviennent d'une exécution antérieure recopiée dans
`outputs/tables/optimization_summary.md` (69,2 / 69,9). Les trois définitions (unités, lignes,
cycles), sur demande réalisée, sont désormais publiées (section Service).

**V2 — Ablation (Fig. 2).** `src/validation_hypotheses.py:548-557` :
« −classification » = `coût_LP × 1,4` et « −obsolescence » = `coût_LP × 1,2`, deux
**multiplicateurs codés en dur** (commentaire : « on suppose +20 % ») ; « −LP » = coût de
l'heuristique empirique `simulate_baseline_policy` sur les
**250** produits (le LP n'en couvre que
177). Aucune de ces trois barres n'est issue d'une expérience ; elles
sont remplacées par de vraies simulations (Fig. 2 ci-dessous).

**V3 — Demande censurée.** Aucune correction. Les séries de prévision sont les ventes
brutes (`src/forecasting.py:86-97`) ; `rupture_signalee` est calculée
(`src/preprocessing.py:361-371`) mais n'est lue nulle part. `stock_apres_vente` est suivi
**par magasin** : 86,5 %
des transactions successives vérifient stock(t) = stock(t−1) − quantité au niveau produit ×
magasin, contre 25,0 % au
niveau produit ; 802 ventes laissent un magasin
à 0 (194 produits). Une vente manquée faute de stock ne laisse
aucune ligne : la demande est sous-estimée là où il y a eu rupture. Toutes les politiques du
backtest sont évaluées sur cette même demande observée ; la limite doit figurer dans l'article.
Conséquence annexe : le `stock_courant` du pipeline est le dernier stock du magasin ayant vendu en
dernier, pas le stock central (le backtest utilise la somme des magasins).

**Autre constat — classe A.** Dans l'exécution de l'article,
49/49
produits « LSTM/MLP » ont un MAE identique au seasonal naive : le MLP exige 36 mois d'historique
(`src/forecasting.py:228`) et l'entraînement n'en compte que 32.

Source des chiffres de cette section : `paper_method_reproduction.json`.

## Protocole commun (à décrire dans l'article)

- **Rolling-origin sans fuite** (`experiments/rolling_origin.py`) : origines mensuelles
  t0 = 31/01/2024 … 30/06/2025 (≥ 18 mois d'historique). À chaque t0, à partir des seules
  lignes brutes datées ≤ t0 : nettoyage (médianes de prix par famille, interpolation des
  stocks), agrégats produit, ABC (CA cumulé ≤ t0), XYZ, K-Means et scaler, features
  d'obsolescence, Isolation Forest et scaler, règles métier, affectation modèle ↔ produit,
  ajustement des modèles ; prévision h = 1..3. Test d'absence de fuite :
  `tests/test_exp_no_leakage.py` (modifier / supprimer / ajouter des lignes postérieures
  à t0 ne change ni une prévision, ni une classe, ni un drapeau, ni le stock ; contrôle
  positif : une modification ≤ t0 les change).
- **Panel d'évaluation** : 16 origines (01/2024 → 04/2025) × produits connus à t0 × h = 1..3,
  identique pour toutes les méthodes. `all` = tous les produits connus ; `active` = produits
  non flagués obsolètes à t0 (même idée que le périmètre de l'article, sans conditionner sur
  le futur). Vérité = ventes mensuelles nettoyées, 0 si aucune vente.
- **Backtest de politique** (`experiments/simulation.py`) : demande réalisée, lignes de
  vente servies dans l'ordre chronologique, ventes perdues, horizon glissant mensuel
  (février 2024 → juillet 2025), commandes en transit suivies. Développement = févr. → déc.
  2024 (réglage des politiques classiques), **test = janv. → juil. 2025** (chiffres de
  référence). Coûts = définitions de `financial_kpis` (50 $/commande, 0,1 %/jour de stockage,
  marge perdue).
- Seed 42 partout, un thread BLAS/OpenMP, CBC séquentiel.

## Table II — Prévision (rolling-origin, 16 origines × h = 1..3)

**Sous-population `all`** (tous les produits connus à t0) — trié par MASE (m = 1, métrique retenue a priori)

| Méthode | MAE publié | MAE | Écart | MASE m=1 | MASE m=12 | RMSSE | WAPE | Biais % | Séries | Exclues m=1 | Exclues m=12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Moyenne mobile 3 mois | 7,52 | 7,37 | -0,15 (-2,0 %) | 0,749 | 0,751 | 0,672 | 0,473 | 3,8 % | 250 | 4 | 53 |
| « LightGBM global » de l'article (LightGBM par produit) | 9,44 | 8,16 | -1,28 (-13,6 %) | 0,833 | 0,813 | 0,744 | 0,524 | 11,2 % | 250 | 4 | 53 |
| Pipeline différencié | 8,24 | 8,01 | -0,23 (-2,8 %) | 0,836 | 0,821 | 0,752 | 0,515 | -1,7 % | 250 | 4 | 53 |
| Pipeline, MLP classe A à fenêtre 12 (variante) | — | 8,28 | — | 0,863 | 0,848 | 0,774 | 0,532 | 0,2 % | 250 | 4 | 53 |
| Seasonal naive | 8,52 | 8,86 | +0,34 (4,0 %) | 0,919 | 0,864 | 0,833 | 0,569 | 8,7 % | 250 | 4 | 53 |
| LightGBM global poolé (nouveau) | — | 7,85 | — | 0,943 | 0,803 | 0,730 | 0,504 | 15,0 % | 250 | 4 | 53 |
| « SARIMA global » de l'article (SARIMA(1,1,1)(1,1,1)₁₂ par produit) | 10,82 | 12,38 | +1,56 (14,5 %) | 1,340 | 1,250 | 1,174 | 0,795 | 22,6 % | 250 | 4 | 53 |
| Pipeline sans obsolescence | — | 10,18 | — | 1,743 | 1,248 | 1,015 | 0,654 | 22,3 % | 250 | 4 | 53 |
| Croston-SBA | — | 11,00 | — | 1,847 | 1,253 | 1,039 | 0,707 | 32,6 % | 250 | 4 | 53 |
| Croston | 7,68 | 11,45 | +3,77 (49,1 %) | 1,930 | 1,304 | 1,076 | 0,735 | 39,6 % | 250 | 4 | 53 |

**Sous-population `active`** (produits non flagués à t0) — trié par MASE (m = 1, métrique retenue a priori)

| Méthode | MAE publié | MAE | Écart | MASE m=1 | MASE m=12 | RMSSE | WAPE | Biais % | Séries | Exclues m=1 | Exclues m=12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM global poolé (nouveau) | — | 8,06 | — | 0,841 | 0,842 | 0,760 | 0,430 | 8,5 % | 226 | 2 | 40 |
| Croston-SBA | — | 8,33 | — | 0,877 | 0,862 | 0,779 | 0,444 | 6,8 % | 226 | 2 | 40 |
| Moyenne mobile 3 mois | 7,52 | 8,75 | +1,23 (16,4 %) | 0,898 | 0,908 | 0,806 | 0,466 | 3,7 % | 226 | 2 | 40 |
| Croston | 7,68 | 8,61 | +0,93 (12,2 %) | 0,906 | 0,890 | 0,799 | 0,459 | 12,5 % | 226 | 2 | 40 |
| « LightGBM global » de l'article (LightGBM par produit) | 9,44 | 9,19 | -0,25 (-2,6 %) | 0,945 | 0,950 | 0,854 | 0,490 | 8,6 % | 226 | 2 | 40 |
| Pipeline différencié | 8,24 | 8,99 | +0,75 (9,1 %) | 0,970 | 0,960 | 0,878 | 0,479 | 5,4 % | 226 | 2 | 40 |
| Pipeline sans obsolescence | — | 8,99 | — | 0,970 | 0,960 | 0,878 | 0,479 | 5,4 % | 226 | 2 | 40 |
| Seasonal naive | 8,52 | 9,25 | +0,73 (8,6 %) | 0,975 | 0,973 | 0,901 | 0,493 | 1,9 % | 226 | 2 | 40 |
| Pipeline, MLP classe A à fenêtre 12 (variante) | — | 9,34 | — | 1,005 | 0,994 | 0,906 | 0,498 | 7,4 % | 226 | 2 | 40 |
| « SARIMA global » de l'article (SARIMA(1,1,1)(1,1,1)₁₂ par produit) | 10,82 | 13,86 | +3,04 (28,1 %) | 1,449 | 1,443 | 1,320 | 0,739 | 16,6 % | 226 | 2 | 40 |

Lecture : sur `all`, le pipeline est classé **3ᵉ sur 10** en MASE ; la meilleure baseline simple est **Moyenne mobile 3 mois** (MASE 0,749 contre 0,836 pour le pipeline). Les MAE publiés proviennent d'une seule fenêtre (avril → juillet 2025), de 177 produits filtrés sur des ventes > 0 dans le test et de classes calculées sur les 36 mois : les écarts mêlent changement de protocole et suppression des fuites.

Source : `forecast_metrics.csv`, `rolling_origin_errors.csv` (134460 lignes).

**MAE du pipeline par classe ABC (classe connue à t0)**

| Classe | MAE publié | MAE `all` | MAE `active` | Écart (`active`) | MASE `active` |
|---|---|---|---|---|---|
| A | 6,47 | 7,64 | 8,21 | +1,74 (26,9 %) | 1,034 |
| B | 7,55 | 6,64 | 7,94 | +0,39 (5,2 %) | 0,936 |
| C | 9,94 | 8,88 | 9,99 | +0,05 (0,5 %) | 0,947 |

Source : `forecast_metrics_by_class.csv`.

**Replis effectifs du pipeline sur le seasonal naive** (part des (produit, origine) où le modèle affecté retombe sur le seasonal naive, 18 origines) : `lightgbm` 0,0 %, `lstm` 100,0 %, `sarima` 28,7 %. Dans l'article, la classe A « LSTM/MLP » est le seasonal naive à 100 % (le MLP exige 36 mois, le train en a 32) ; la variante `pipeline_mlp12` mesure ce que donnerait un MLP effectivement entraîné.

## Table III — Politiques d'inventaire

**Méthode de l'article reproduite** (plan août → oct. 2025 jugé contre la demande *prévue*, code d'origine sur les entrées versionnées)

| Chiffre | Publié | Recalculé | Écart |
|---|---|---|---|
| Coût total pipeline (USD) | 55 029 | 54 143,71 | -885,29 (-1,6 %) |
| Coût total incumbent (USD), 250 produits | 76 740 | 76 979,92 | +239,92 (0,3 %) |
| Économie, périmètres différents (177 vs 250) | −28,3 % | -29,67 % | — |
| Coût incumbent, mêmes 177 produits (USD) | 73 525 (H4) | 73 525,06 | — |
| Économie appariée | — | -26,36 % | — |
| Écart apparié sur 3 mois / × 4 (USD) | 19 381 / 77 525 | 19 381,35 / 77 525,40 | — |
| Fill rate unités incumbent → pipeline (vs prévision) | 69,19 % → 69,94 % | 69,49 % → 70,23 % | — |

Les valeurs publiées 55 029 / 76 740 / 69,19 / 69,94 % ne sont plus produites par le code : elles viennent d'une exécution antérieure recopiée dans `outputs/tables/optimization_summary.md`. Le plan LP recalculé est identique au plan versionné `commandes_centrales.csv` : True. Source : `paper_method_reproduction.json`, `milp_vs_lp.json`.

**Backtest à demande réalisée — période de test (janv. → juil. 2025)**

| Politique | Coût total | vs incumbent | Commande | Stockage | Rupture | Achats | Stock moyen | Fill rate unités | Fill rate lignes | CSL | Commandes | Écrêt. budget | LP non optimaux |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incumbent | 232 593 | 0,0 % | 21 850 | 146 613 | 64 130 | 1 534 239 | 698 156 | 85,4 % | 86,0 % | 82,4 % | 437 | 0 | 0 |
| sS_tuned | 168 412 | -27,6 % | 24 850 | 121 037 | 22 525 | 1 486 337 | 576 365 | 96,9 % | 96,8 % | 93,3 % | 497 | 0 | 0 |
| order_up_to_tuned | 175 717 | -24,5 % | 55 900 | 98 487 | 21 329 | 1 505 684 | 468 988 | 96,6 % | 96,5 % | 91,5 % | 1118 | 0 | 0 |
| ma3_order_up_to_tuned | 187 317 | -19,5 % | 42 000 | 129 179 | 16 139 | 1 341 405 | 615 136 | 96,6 % | 96,6 % | 93,1 % | 840 | 0 | 0 |
| best_baseline_lp | 199 231 | -14,3 % | 43 650 | 57 512 | 98 069 | 1 075 213 | 273 867 | 81,6 % | 81,7 % | 69,5 % | 873 | 0 | 0 |
| pipeline_no_obso_lp | 254 049 | 9,2 % | 52 250 | 122 810 | 78 989 | 1 481 703 | 584 810 | 85,6 % | 85,7 % | 72,2 % | 1045 | 0 | 0 |
| pipeline_no_class_lp | 221 029 | -5,0 % | 47 550 | 68 292 | 105 187 | 1 360 339 | 325 200 | 80,2 % | 80,7 % | 67,1 % | 951 | 0 | 0 |
| pipeline_lp | 217 597 | -6,4 % | 49 050 | 70 441 | 98 106 | 1 352 892 | 335 433 | 82,8 % | 83,2 % | 70,1 % | 981 | 0 | 0 |
| pipeline_mlp12_lp | 209 268 | -10,0 % | 48 400 | 79 037 | 81 831 | 1 338 233 | 376 367 | 83,8 % | 84,3 % | 73,6 % | 968 | 0 | 0 |

Lecture : pipeline LP 217 597 USD contre 232 593 USD pour l'incumbent, soit **-6,4 %** (publié −28,3 % contre la prévision). Politique classique choisie sur le développement : `ma3_order_up_to_tuned` (187 317 USD sur le test) ; la moins coûteuse a posteriori sur le test est `sS_tuned` (168 412 USD, -27,6 % vs incumbent). Politiques moins coûteuses que le pipeline complet sur le test : `sS_tuned`, `order_up_to_tuned`, `ma3_order_up_to_tuned`, `best_baseline_lp`. Taux de service en unités : incumbent 85,39 %, pipeline 82,83 % (publié 69,19 % → 69,94 %, contre la prévision). Le LP planifie exactement la prévision ponctuelle : il n'a pas de stock de sécurité, d'où un coût de rupture élevé face à la demande réelle.

Paramètres réglés sur le développement : z = {"sS": {"A": 1.28, "B": 1.645, "C": 2.0}, "OUT": {"A": 0.5, "B": 0.75, "C": 0.75}, "MA3": {"A": 1.0, "B": 1.0, "C": 1.0}} ; meilleure baseline de prévision (MASE sur cibles ≤ déc. 2024) : `moving_average_3`.

Sources : `policy_comparison.csv` (périodes test / dev / full), `policy_sim_detail.csv`, `policy_tuning.csv`, `policy_meta.json`.

## Table IV — Hypothèses

| Test | Comparaison | Statistique | p brut | p ajusté (Holm) | Taille d'effet | Estimation [IC 95 %] | n | Unité |
|---|---|---|---|---|---|---|---|---|
| H1_pipeline_vs_best_simple | pipeline vs moving_average_3 | 17 333,0 | 1,0000 | 1,0000 | -0,328 | médiane Δ 0,221 [0,046 ; 0,461] | 250 | produit |
| H1_pipeline_vs_seasonal_naive | pipeline vs seasonal_naive | 9 195,5 | 0,0030 | 0,0061 | 0,215 | médiane Δ -0,047 [-0,219 ; 0,000] | 250 | produit |
| H1_pipeline_vs_best_simple (robustesse, `active`) | pipeline vs moving_average_3 | 15 378,0 | 0,9953 | — (hors famille) | -0,199 | médiane Δ 0,241 [0,056 ; 0,475] | 226 | produit |
| H1_pipeline_vs_seasonal_naive (robustesse, `active`) | pipeline vs seasonal_naive | 5 918,0 | 0,0308 | — (hors famille) | 0,166 | médiane Δ 0,000 [0,000 ; 0,000] | 226 | produit |
| H2_DM_sarima_uniform_vs_seasonal_naive | sarima_uniform vs seasonal_naive | 2,490 | 0,0250 | 0,0500 | 1,798 (ΔMSE rel.) | E[Δ] 383,21 [55,21 ; 711,22] | 16 | origines (HAC) |
| H2_DM_lightgbm_uniform_vs_seasonal_naive | lightgbm_uniform vs seasonal_naive | -1,498 | 0,1549 | 0,1549 | -0,176 (ΔMSE rel.) | E[Δ] -37,50 [-90,85 ; 15,86] | 16 | origines (HAC) |
| H2_DM_lightgbm_uniform_vs_sarima_uniform | lightgbm_uniform vs sarima_uniform | -2,967 | 0,0096 | 0,0288 | -0,706 (ΔMSE rel.) | E[Δ] -420,71 [-722,99 ; -118,43] | 16 | origines (HAC) |
| H4_vs_incumbent | pipeline_lp vs incumbent (coût total par produit sur la période de test) | 9 269,0 | 0,0979 | 0,3917 | 0,105 | 25 707 USD/an [-17 393 ; 68 024] | 246 | produit (grappe : tous ses mois) |
| H5_vs_incumbent | pipeline_lp vs incumbent (gain net = marge servie − commande − stockage, par produit) | 9 269,0 | 0,0979 | 0,3917 | 0,105 | 25 707 USD/an [-17 915 ; 67 188] | 246 | produit (grappe) |
| H4_vs_best_conventional | pipeline_lp vs ma3_order_up_to_tuned (coût total par produit sur la période de test) | 13 820,0 | 0,9843 | 1,0000 | -0,169 | -51 908 USD/an [-99 539 ; -8 760] | 246 | produit (grappe : tous ses mois) |
| H5_vs_best_conventional | pipeline_lp vs ma3_order_up_to_tuned (gain net = marge servie − commande − stockage, par produit) | 13 820,0 | 0,9843 | 1,0000 | -0,169 | -51 908 USD/an [-100 291 ; -9 601] | 246 | produit (grappe) |

Publié : H1 W = 3264, p = 0,107 (vs seasonal naive) → ici W = 9 195,5, p = 0,0030 ; meilleure baseline simple = `moving_average_3`. Publié : H2 LightGBM vs SARIMA p = 0,059 (test invalide : erreurs de produits concaténées) → ici p brut 0,0096, ajusté 0,0288.

Publié : H4 n = 531 paires, économie 19 381 USD × 4 = 77 525 USD/an (bootstrap i.i.d. des paires produit × mois, malgré la mention « par blocs ») → ici, sur demande réalisée : économie 14 996 USD sur 7 mois de test, 25 707 USD/an, IC grappes produits [-17 393 ; 68 024], IC blocs 1 mois [-1 435 ; 48 982] (7 blocs). Publié : H5 gain net 38 622 USD/an (ΔCA + Δmarge comptait la marge deux fois) → ici 25 707 USD/an (identique à l'économie de coût par construction, écart numérique -0,000000).

Contre la meilleure politique classique **choisie sur la période de développement** (`ma3_order_up_to_tuned`) : économie -30 280 USD (-16,2 %), soit -51 908 USD/an, IC grappes produits [-99 539 ; -8 760] (valeur négative = le pipeline coûte plus). Bootstrap par blocs de 3 mois : 3 blocs seulement sur 7 mois de test, IC indicatif.

**H3 — Obsolescence, vérité terrain prospective** (ventes nulles sur les 6 mois suivant t0 ET stock > 0 à t0 ; 13 origines ; détecteurs sur ≤ t0)

| Détecteur | TP | FP | FN | TN | Précision | Rappel | F1 | IC 95 % F1 | Délai médian (j) |
|---|---|---|---|---|---|---|---|---|---|
| Évaluation de l'article (circulaire, reproduite) | 58 | 15 | 0 | 177 | 0,795 | 1,000 | 0,885 | — | — |
| hybrid_current (IF c=0.10 + règles 180 j) | 216 | 210 | 16 | 2490 | 0,507 | 0,931 | 0,657 | [0,532 ; 0,743] | -62 |
| iforest_only (c=0.10) | 66 | 93 | 166 | 2607 | 0,415 | 0,284 | 0,338 | [0,148 ; 0,507] | 58 |
| rules_only (180 j) | 214 | 118 | 18 | 2582 | 0,645 | 0,922 | 0,759 | [0,653 ; 0,830] | -1 |
| inactivity_only (180 j) | 132 | 6 | 100 | 2694 | 0,957 | 0,569 | 0,714 | [0,571 ; 0,811] | 180 |

Publié : 58/15/0/177, P = 0,795, R = 1,000, F1 = 0,885 — reproduit exactement, mais circulaire (la règle « ≥ 6 mois sans vente » du détecteur est la vérité terrain). Fragilité : avec un ordre déterministe des transactions du même jour, 58/16/0/176 (20 stocks courants changent). Délai < 0 = alerte avant le début de la dormance. Balayages contamination × seuil : `obsolescence_eval.csv`.

Décomposition des règles métier (3232 observations produit × origine) : la composante « ratio 3 m / 12 m < 0,1 et âge > 365 j » lève 585 drapeaux ; la composante d'inactivité en lève 444 (seuil 90 j) à 217 (365 j), **tous déjà levés par la composante ratio** : le seuil d'inactivité n'a aucun effet sur les règles, la règle effective est celle du ratio.

Meilleur F1 du balayage (a posteriori, optimiste) : `inactivity` (contamination —, seuil 90 j) — P = 0,938, R = 0,789, F1 = 0,857 [0,787 ; 0,908], délai médian 91 j.

Figure : `fig_obsolescence_pr.png` (précision vs rappel, vérité prospective).

Sources : `stat_tests.json`, `obsolescence_confusion.json`, `obsolescence_eval.csv`.

## Fig. 2 — Ablation (backtest, période de test)

| Configuration | Publié (USD) | Origine du chiffre publié | Mesuré (USD) | vs incumbent | Fill rate unités |
|---|---|---|---|---|---|
| Pipeline complet | 55 029 | exécution antérieure | 217 597 | -6,4 % | 82,8 % |
| − classification (seasonal naive partout) | 75 801 | multiplicateur codé en dur | 221 029 | -5,0 % | 80,2 % |
| − obsolescence | 64 972 | multiplicateur codé en dur | 254 049 | 9,2 % | 85,6 % |
| − LP (heuristique empirique) | 76 980 | heuristique, 250 produits | 232 593 | 0,0 % | 85,4 % |

Les chiffres publiés et mesurés ne sont pas sur la même base (3 mois contre la prévision vs 7 mois de demande réalisée) : seule la comparaison interne de chaque colonne a un sens.

Source : `policy_comparison.csv`.

## MILP exact vs relaxation LP arrondie

Instance de l'article : 177 produits actifs × 3 mois ; MILP 2124 variables (1062 entières/binaires), 1068 contraintes ; CBC 2.10.3, limite 600 s. Arrondi du code : int(np.round(q)) — au plus proche (src/optimization.py:184), pas à l'entier supérieur. ρ (en transit) = 0 ; MOQ absente des données : variantes `none` et `1_month` (hypothèse).

| Budget/mois | MOQ | Objectif | Statut MILP | Obj. MILP | Borne | Gap | Temps (s) | Obj. LP arrondi (code) | Surcoût vs MILP | Mois budget dépassé | Dépassement max | Capacité | Viol. MOQ | Commandes LP / MILP | Surcoût (arrondi sup.) | Mois dépassés (arrondi sup.) | Dépassement max (arrondi sup.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 000 | none | code | Optimal | 709 868 | 709 868 | 0,000 % | 9,7 | 732 507 | 3,19 % | 0 | 0,00 % | 0 | 0 | 178 / 153 | 0,17 % | 0 | 0,00 % |
| 500 000 | 1_month | code | Optimal | 711 502 | 711 502 | 0,000 % | 12,5 | 732 507 | 2,95 % (plan infaisable) | 0 | 0,00 % | 0 | 116 | 178 / 145 | -0,06 % (plan infaisable) | 0 | 0,00 % |
| 150 000 | none | code | Optimal | 815 226 | 815 226 | 0,000 % | 28,0 | 825 779 | 1,29 % | 0 | 0,00 % | 0 | 0 | 111 / 114 | -1,61 % (plan infaisable) | 1 | 4,67 % |
| 150 000 | 1_month | code | Optimal | 871 875 | 871 875 | 0,000 % | 41,6 | 825 779 | -5,29 % (plan infaisable) | 0 | 0,00 % | 0 | 69 | 111 / 100 | -8,00 % (plan infaisable) | 1 | 4,67 % |
| 100 000 | none | code | Optimal | 1 059 030 | 1 059 030 | 0,000 % | 47,7 | 1 064 605 | 0,53 % | 0 | 0,00 % | 0 | 0 | 69 / 74 | -1,49 % (plan infaisable) | 2 | 7,56 % |
| 100 000 | 1_month | code | Optimal | 1 158 946 | 1 158 946 | 0,000 % | 249,7 | 1 064 605 | -8,14 % (plan infaisable) | 0 | 0,00 % | 0 | 43 | 69 / 68 | -9,98 % (plan infaisable) | 2 | 7,56 % |
| 50 000 | none | code | Optimal | 1 612 169 | 1 612 169 | 0,000 % | 592,4 | 1 619 120 | 0,43 % (plan infaisable) | 1 | 0,34 % | 0 | 0 | 30 / 32 | -0,32 % (plan infaisable) | 2 | 6,26 % |
| 50 000 | 1_month | code | Optimal | 1 646 200 | 1 646 200 | 0,000 % | 20,1 | 1 619 120 | -1,64 % (plan infaisable) | 1 | 0,34 % | 0 | 18 | 30 / 20 | -2,38 % (plan infaisable) | 2 | 6,26 % |
| 500 000 | none | paper | Optimal | 613 113 | 613 113 | 0,000 % | 5,1 | 631 693 | 3,03 % | 0 | 0,00 % | 0 | 0 | 178 / 150 | 0,41 % | 0 | 0,00 % |
| 500 000 | 1_month | paper | Optimal | 618 516 | 618 516 | 0,000 % | 8,4 | 631 693 | 2,13 % (plan infaisable) | 0 | 0,00 % | 0 | 116 | 178 / 143 | -0,47 % (plan infaisable) | 0 | 0,00 % |

Objectif `code` : coûts de l'implémentation avec un vrai coût fixe K par commande (le LP le linéarise en K/d̄ par unité) ; objectif `paper` : w_i·(K·x + h·I + p·s) avec p = 1,2 × prix, tel que formulé dans l'article. Les deux plans sont propagés dans le même bilan de stock (demande prévue) avant évaluation. Un surcoût négatif n'existe que pour un plan qui viole une contrainte du MILP (budget ou MOQ) : il n'est pas comparable.

Lecture : 10/10 MILP résolus à l'optimalité prouvée. Au point nominal (budget non contraignant), le plan LP arrondi comme le code est faisable mais coûte 3,19 % de plus que le MILP (178 commandes contre 153) ; arrondi à l'entier supérieur : 0,17 %. Avec un MOQ d'un mois, le plan LP viole 116 fois la contrainte : il n'est plus réalisable. Budget serré : l'arrondi au plus proche dépasse le budget pour 50 000 $/mois ; l'arrondi supérieur pour 150 000, 100 000, 50 000 $/mois. Conclusion : l'équivalence « LP arrondi ≈ MILP » ne tient qu'à quelques % près et seulement quand budget et MOQ ne contraignent pas ; le MILP se résout ici en 5 à 592 s, il peut donc remplacer la relaxation sur une instance de cette taille.

Source : `milp_vs_lp.json`.

## Service — définitions, compromis coût / service, cibles

**Par classe ABC (test)**

| Politique | Classe | Fill rate unités | Fill rate lignes | CSL | Coût total |
|---|---|---|---|---|---|
| incumbent | A | 86,9 % | 87,5 % | 82,2 % | 128 634 |
| incumbent | B | 85,8 % | 85,1 % | 81,8 % | 54 595 |
| incumbent | C | 84,8 % | 85,8 % | 82,8 % | 49 365 |
| pipeline_lp | A | 78,3 % | 80,3 % | 65,1 % | 118 989 |
| pipeline_lp | B | 86,1 % | 84,6 % | 68,3 % | 45 507 |
| pipeline_lp | C | 82,9 % | 83,6 % | 73,7 % | 53 101 |

**Balayages (pipeline LP ; frontière (s, S) à z unique en référence)**

| Balayage | Valeur | Coût total | Achats | Fill rate unités | Fill rate lignes | CSL | Écrêt. budget |
|---|---|---|---|---|---|---|---|
| budget | 25 000,00 | 418 690 | 169 780 | 7,9 % | 6,6 % | 10,5 % | 8 |
| budget | 50 000,00 | 375 621 | 339 473 | 11,2 % | 10,6 % | 13,4 % | 6 |
| budget | 100 000,00 | 329 590 | 651 742 | 15,4 % | 17,1 % | 21,5 % | 9 |
| budget | 200 000,00 | 211 455 | 1 273 682 | 78,7 % | 79,1 % | 66,8 % | 4 |
| budget | 300 000,00 | 217 679 | 1 353 841 | 82,8 % | 83,2 % | 70,1 % | 0 |
| budget | 500 000,00 | 217 597 | 1 352 892 | 82,8 % | 83,2 % | 70,1 % | 0 |
| budget | 750 000,00 | 217 597 | 1 352 892 | 82,8 % | 83,2 % | 70,1 % | 0 |
| budget | 1 000 000,00 | 217 597 | 1 352 892 | 82,8 % | 83,2 % | 70,1 % | 0 |
| penalty_factor | 1,00 | 217 600 | 1 352 892 | 82,8 % | 83,2 % | 70,0 % | 0 |
| penalty_factor | 1,20 | 217 597 | 1 352 892 | 82,8 % | 83,2 % | 70,1 % | 0 |
| penalty_factor | 1,50 | 217 639 | 1 352 905 | 83,0 % | 83,3 % | 70,1 % | 0 |
| penalty_factor | 2,00 | 217 685 | 1 352 913 | 83,0 % | 83,4 % | 70,1 % | 0 |
| penalty_factor | 3,00 | 217 685 | 1 352 913 | 83,0 % | 83,4 % | 70,1 % | 0 |
| penalty_factor | 5,00 | 217 685 | 1 352 913 | 83,0 % | 83,4 % | 70,1 % | 0 |
| penalty_factor | 10,00 | 217 685 | 1 352 913 | 83,0 % | 83,4 % | 70,1 % | 0 |
| sS_z | 0,00 | 198 591 | 1 181 588 | 81,0 % | 81,2 % | 70,0 % | 0 |
| sS_z | 0,25 | 184 127 | 1 300 250 | 84,3 % | 84,0 % | 74,3 % | 0 |
| sS_z | 0,50 | 178 139 | 1 302 389 | 86,7 % | 87,0 % | 78,1 % | 0 |
| sS_z | 0,75 | 171 983 | 1 367 095 | 89,5 % | 89,9 % | 82,7 % | 0 |
| sS_z | 1,00 | 165 026 | 1 424 570 | 92,3 % | 92,1 % | 86,2 % | 0 |
| sS_z | 1,28 | 167 811 | 1 478 728 | 94,2 % | 94,2 % | 90,2 % | 0 |
| sS_z | 1,50 | 173 099 | 1 504 961 | 95,2 % | 95,0 % | 91,2 % | 0 |
| sS_z | 1,65 | 176 090 | 1 537 200 | 95,9 % | 95,9 % | 92,3 % | 0 |
| sS_z | 2,00 | 190 506 | 1 547 994 | 97,8 % | 97,7 % | 94,8 % | 0 |
| sS_z | 2,33 | 200 593 | 1 568 762 | 98,5 % | 98,3 % | 96,5 % | 1 |
| sS_z | 2,50 | 205 055 | 1 584 841 | 99,0 % | 98,7 % | 97,0 % | 1 |
| sS_z | 3,00 | 226 348 | 1 583 827 | 99,5 % | 99,4 % | 98,9 % | 1 |

**Cibles de service (test)**

| Définition | Cible | Budget minimal (pipeline) | Pénalité minimale (1+π) | Max atteint (balayage budget) | z le moins coûteux (s, S) | Coût (s, S) à ce z |
|---|---|---|---|---|---|---|
| unit_fill_rate | 70 % | 200 000 | 1,0 | 82,8 % | 1,000 | 165 026 |
| unit_fill_rate | 80 % | 300 000 | 1,0 | 82,8 % | 1,000 | 165 026 |
| unit_fill_rate | 90 % | infeasible | infeasible | 82,8 % | 1,000 | 165 026 |
| unit_fill_rate | 95 % | infeasible | infeasible | 82,8 % | 1,500 | 173 099 |
| order_fill_rate | 70 % | 200 000 | 1,0 | 83,2 % | 1,000 | 165 026 |
| order_fill_rate | 80 % | 300 000 | 1,0 | 83,2 % | 1,000 | 165 026 |
| order_fill_rate | 90 % | infeasible | infeasible | 83,2 % | 1,000 | 165 026 |
| order_fill_rate | 95 % | infeasible | infeasible | 83,2 % | 1,645 | 176 090 |
| cycle_service_level | 70 % | 300 000 | 1,0 | 70,1 % | 1,000 | 165 026 |
| cycle_service_level | 80 % | infeasible | infeasible | 70,1 % | 1,000 | 165 026 |
| cycle_service_level | 90 % | infeasible | infeasible | 70,1 % | 1,280 | 167 811 |
| cycle_service_level | 95 % | infeasible | infeasible | 70,1 % | 2,330 | 200 593 |

Lecture : à pénalité nominale, le fill rate (unités) du pipeline plafonne à 82,8 % ; dès 300 000 $/mois il est à moins de 0,1 point de ce maximum (le budget ne contraint plus), et multiplier la pénalité de rupture de 1 à 10 ne le fait varier que de 82,8 % à 83,0 %. Le LP déterministe vise la prévision ponctuelle : sans stock de sécurité, ni le budget ni la pénalité ne relèvent le service. À titre de repère, (s, S) avec z = 1,00 coûte 165 026 $ pour un fill rate de 92,3 %.

Figure : `fig_service_tradeoff.png` (coût total vs fill rate, période de test).

Sources : `service_tradeoff.csv`, `service_targets.csv`, `policy_service_by_class.csv`.

## Sensibilité (une variable à la fois)

| Paramètre | Valeur | Économie (méthode article, non apparié) | Économie (méthode article, apparié) | Économie backtest vs incumbent | Économie backtest vs MA3 + order-up-to réglé | Fill rate backtest | Coût backtest | LP non optimaux |
|---|---|---|---|---|---|---|---|---|
| budget | 0.5x | 29,7 % | 26,4 % | -5,7 % | -18,5 % | 82,9 % | 216 408 | 0 |
| budget | 1.0x | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| budget | 2.0x | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| lead_time | nominal | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| lead_time | +1_month | 20,2 % | 18,0 % | -5,1 % | -33,1 % | 30,8 % | 308 020 | 0 |
| lead_time | stochastic_p0.3 | n/a | n/a | 2,7 % | -22,7 % | 71,6 % | 249 565 | 0 |
| stockout_penalty_factor | 1.0 | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 82,8 % | 217 600 | 0 |
| stockout_penalty_factor | 1.2 | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| stockout_penalty_factor | 1.5 | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 83,0 % | 217 639 | 0 |
| stockout_penalty_factor | 2.0 | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 83,0 % | 217 685 | 0 |
| holding_rate_per_day | 0.0005 | 22,3 % | 20,2 % | -14,5 % | -43,5 % | 82,8 % | 182 377 | 0 |
| holding_rate_per_day | 0.001 | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| holding_rate_per_day | 0.002 | 39,3 % | 34,8 % | 24,0 % | -6,1 % | 82,8 % | 288 038 | 0 |
| abc_weights | (4.0, 2.5, 1.5) | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| abc_weights | (1.0, 1.0, 1.0) | 29,8 % | 26,6 % | 6,6 % | -16,0 % | 82,1 % | 217 333 | 0 |
| abc_weights | (3.0, 2.0, 1.0) | 29,8 % | 26,6 % | 6,6 % | -16,0 % | 82,1 % | 217 333 | 0 |
| obsolescence_threshold_days | 90 | n/a | n/a | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| obsolescence_threshold_days | 180 | n/a | n/a | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| obsolescence_threshold_days | 270 | n/a | n/a | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| obsolescence_threshold_days | 365 | n/a | n/a | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| horizon_months | 1 | 25,1 % | 22,2 % | -79,7 % | -148,3 % | 3,9 % | 465 197 | 0 |
| horizon_months | 2 | 25,9 % | 23,1 % | -17,4 % | -62,3 % | 30,0 % | 304 096 | 0 |
| horizon_months | 3 | 29,7 % | 26,4 % | 6,4 % | -16,2 % | 82,8 % | 217 597 | 0 |
| horizon_months | 6 | 34,8 % | 30,8 % | 4,9 % | -16,2 % | 82,9 % | 217 641 | 0 |
| usd_cdf_change | -0.2 | 19,7 % | 17,4 % | -9,1 % | -49,8 % | 83,0 % | 330 736 | 0 |
| usd_cdf_change | -0.1 | 24,1 % | 21,3 % | -1,5 % | -34,2 % | 82,9 % | 267 863 | 0 |
| usd_cdf_change | 0.1 | 36,9 % | 33,1 % | 14,7 % | -10,3 % | 82,8 % | 176 469 | 0 |
| usd_cdf_change | 0.2 | 46,7 % | 42,5 % | 22,1 % | -9,9 % | 82,8 % | 145 930 | 0 |

Taux de change : hypothèse prix en CDF / coûts en USD (aucune donnée de change dans le dépôt).

Propriété de la formulation : une commande passée en période t n'est reçue dans l'horizon que si t + L ≤ H. Quand le délai L d'un produit atteint l'horizon H (délai +1 mois : L = 3 pour la Chine avec H = 3 ; horizon H = 1 : L ≥ 1 pour tous), le LP ne commande plus jamais ce produit. Les scénarios correspondants mesurent cet effet, pas un défaut de simulation.

Lecture (économie > 0 : le pipeline coûte moins que la référence) : avec la méthode de l'article, l'économie du pipeline sur l'incumbent va de 19,7 % à 46,7 % (non apparié) et de 17,4 % à 42,5 % (apparié) selon les scénarios ; pondérations ABC : (4.0, 2.5, 1.5) → 29,7 %, (1.0, 1.0, 1.0) → 29,8 %, (3.0, 2.0, 1.0) → 29,8 %. Sur demande réalisée, sur 22 configurations distinctes (le point nominal, répété une fois par paramètre, n'est compté qu'une fois), le pipeline bat l'incumbent dans 15 et la politique classique (MA3 + order-up-to réglé, z re-réglé sur le développement) dans 0.

Source : `sensitivity.csv`.

## Reproductibilité

Machine : Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz, 4 cœurs physiques / 8 logiques, 11,8 Go ; Windows-11-10.0.26200-SP0 ; Python 3.14.5 ; CBC 2.10.3 ; commit a04d41d5a253d94ef43fc72161a9c57cab4a9e1d (copie de travail modifiée : True).

| Module | Temps moyen (s) | Écart-type | Pic mémoire (Mo) | Échecs |
|---|---|---|---|---|
| classification | 18,7 | 6,3 | 257 | 0 |
| eda_figures | 27,6 | 11,3 | 396 | 0 |
| evaluation_h1_h6 | 9,1 | 3,2 | 220 | 0 |
| export_powerbi | 13,1 | 6,1 | 130 | 0 |
| forecasting | 699,3 | 204,2 | 375 | 0 |
| obsolescence_detection | 21,2 | 6,8 | 285 | 0 |
| optimization | 15,2 | 7,3 | 261 | 0 |
| preprocessing | 12,6 | 4,6 | 202 | 0 |
| TOTAL_pipeline_6_modules | 780,0 | 231,1 | 375 | 0 |

Publié : 213 s (somme codée en dur dans `scripts/run_evaluation.py:63-72`, EDA comprise) → mesuré : 780,0 ± 231,1 s pour les 6 modules du pipeline (5 exécutions ; EDA et évaluation mesurées à part ; tableaux de bord exclus). Publié : 17 dépendances → 23 dans `requirements.txt` après la révision (21 au moment de l'article, plus `shap` importé sans être déclaré ; `shap` et `psutil` ont été ajoutés), dépendances importées mais non déclarées : aucune. Publié : « < 4 Go » (jamais mesuré) → pic mesuré 375 Mo par processus pour le pipeline (396 Mo tous scripts confondus, EDA comprise). Les temps varient fortement d'une exécution à l'autre (min 511 s, max 1 019 s) sur ce portable : publier la moyenne, l'écart-type et la machine.

Sources : `environment.json`, `runtime.csv`, `runtime_summary.csv`.

## Expériences non réalisées ou partielles

- **Correction de la demande censurée** : non réalisée. Les données ne contiennent que des
  ventes ; une rupture sans vente ne laisse aucune trace et les réceptions ne sont pas
  enregistrées. Un estimateur (p. ex. EM sur périodes à stock nul) supposerait des
  hypothèses non vérifiables ici. Limite à déclarer.
- **MOQ, volumes unitaires, commandes en transit réelles** : absents des données ; le MILP
  utilise des hypothèses explicites (MOQ `none` / `1_month`, 0,01 m³, ρ = 0 à l'instance de
  l'article). Le backtest, lui, suit les commandes en transit qu'il a lui-même passées.
- **Taux de change USD/CDF** : aucune donnée ni aucun paramètre de change dans le code ; la
  sensibilité repose sur une hypothèse (prix de vente en CDF, coûts en USD).
- **Sensibilité au seuil d'obsolescence selon la méthode de l'article** : non calculable, les
  prévisions de déploiement (`previsions_complet.csv`) sont figées pour le seuil de 180 j ;
  calculée dans le backtest.
- **LSTM** : le dépôt n'en contient pas (MLP scikit-learn à la place, cf. `src/forecasting.py`),
  et ce MLP retombe toujours sur le seasonal naive avec une fenêtre de 30 mois ; seule la
  variante à fenêtre de 12 mois est évaluée.
- **Puissance statistique des tests de politique (H4/H5)** : la période de test ne compte que
  7 mois ; le bootstrap par blocs de mois repose sur 7 blocs (1 mois) ou 3 blocs (3 mois) et
  reste indicatif. Allonger l'historique est la seule correction possible.
- **Délai d'alerte d'obsolescence** : calculé sur les seuls débuts de dormance observables dans
  la fenêtre (cf. `n_onsets` dans `obsolescence_eval.csv`) ; petit échantillon.
- **Temps par origine du rolling-origin** (`rolling_origin_meta.json`) : mesurés avec 7 processus
  en concurrence, non représentatifs ; les temps de référence sont ceux de `runtime.csv`.
