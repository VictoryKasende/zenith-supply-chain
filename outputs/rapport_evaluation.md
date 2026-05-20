# Rapport d'évaluation — Pipeline Zenith Supply Chain

Ce rapport synthétise la validation des six hypothèses de recherche
(H1–H6) formulées dans l'introduction du mémoire (§1.3).

## Verdicts globaux

| Code | Hypothèse | Verdict | Valeur mesurée | Seuil |
|---|---|---|---:|---:|
| H1 | Différenciation par classe | ✅ Validée | 0.426 | 1.0 |
| H2 | Complémentarité statistique ↔ apprentissage | ✅ Validée | 0.667 | 0.6666666666666666 |
| H3 | Détection précoce d'obsolescence | ✅ Validée | 1.0 | 0.8 |
| H4 | Optimisation supérieure à l'heuristique | ✅ Validée | 0.703 | 1.0 |
| H5 | Impact financier mesurable | ✅ Validée | 5392.082 | 0.0 |
| H6 | Faisabilité PME (frugalité) | ✅ Validée | 3.0 | 3.0 |

## Détails par hypothèse

### H1 — Différenciation par classe

- **Test** : MAE moyen pipeline différencié vs MAE moyen du pire modèle isolé
- **Critère** : ratio < 1
- **Valeur mesurée** : 0.426
- **Seuil** : 1.0
- **Verdict** : **Validée**
- **Détails** : {'mae_pipeline_differencie': 9.073, 'mae_pire_modele_isole': 21.312, 'gain_relatif_pct': 57.4}

### H2 — Complémentarité statistique ↔ apprentissage

- **Test** : Le modèle affecté à chaque classe est le meilleur observé (échantillon ≥ 5)
- **Critère** : ≥ 2/3 classes ont leur modèle attendu en tête
- **Valeur mesurée** : 0.667
- **Seuil** : 0.6666666666666666
- **Verdict** : **Validée**
- **Détails** : {'attendu': {'A': 'lstm', 'B': 'lightgbm', 'C': 'sarima'}, 'observé': {'A': 'lstm', 'B': 'lightgbm', 'C': 'croston_sba'}, 'min_sample': 5}

### H3 — Détection précoce d'obsolescence

- **Test** : Rappel des produits avec ≥ 6 mois sans vente
- **Critère** : rappel ≥ 80 %
- **Valeur mesurée** : 1.0
- **Seuil** : 0.8
- **Verdict** : **Validée**
- **Détails** : {'produits_evidents_total': 58, 'vrais_positifs': 58, 'faux_positifs': 15, 'faux_negatifs': 0, 'precision': 0.795, 'rappel': 1.0}

### H4 — Optimisation supérieure à l'heuristique

- **Test** : Coût total simulé LP vs politique empirique
- **Critère** : coût LP < coût empirique
- **Valeur mesurée** : 0.703
- **Seuil** : 1.0
- **Verdict** : **Validée**
- **Détails** : {'cout_lp_usd': 54143.71, 'cout_emp_usd': 76979.92, 'gain_usd': 22836.21, 'gain_pct': 29.67}

### H5 — Impact financier mesurable

- **Test** : Bénéfice net additionnel = ΔCA + Δmarge_évitée + Δstock×coût_capital
- **Critère** : bénéfice > 0
- **Valeur mesurée** : 5392.082
- **Seuil** : 0.0
- **Verdict** : **Validée**
- **Détails** : {'delta_ca_realise_usd': 4398.31, 'delta_marge_perdue_evitee_usd': 920.12, 'delta_stock_libere_usd': 613.77, 'coupon_capital_libere_usd': 73.65, 'benefice_net_usd': 5392.08}

### H6 — Faisabilité PME (frugalité)

- **Test** : Temps total ≤ 5 min, sans GPU, dépendances ≤ 20
- **Critère** : 3 conditions / 3
- **Valeur mesurée** : 3.0
- **Seuil** : 3.0
- **Verdict** : **Validée**
- **Détails** : {'temps_total_sec': 213, 'limite_temps_sec': 300.0, 'n_dependances_python': 20, 'limite_dependances': 20, 'necessite_gpu': False}

## Temps total de l'exécution complète

- Étapes 1–6 cumulées : **213 s** sur poste standard CPU (2 cœurs).
- Compatible avec la contrainte H6 (≤ 5 min) → la solution est exécutable
  quotidiennement sans dégrader l'usage métier.

## Files generated

- `outputs/tables/evaluation_hypotheses.csv` (synthèse machine-lisible)
- `outputs/figures/eval_01_verdicts.png` (verdicts H1-H6)
- `outputs/figures/eval_02_financial_kpis.png` (KPI financiers)