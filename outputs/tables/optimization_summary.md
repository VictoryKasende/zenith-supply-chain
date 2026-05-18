# Synthèse de l'Étape 6 — Optimisation linéaire des commandes

## Formulation LP (PuLP / CBC)

**Variables** (continues, arrondies à l'entier en post-traitement) :
- `Q[i,t]` ≥ 0 : quantité commandée du produit *i* au mois *t*.
- `S[i,t]` ≥ 0 : stock fin de période.
- `R[i,t]` ≥ 0 : ruptures (proxy de service).

**Objectif** : minimiser `Σ_i Σ_t (c_commande/avg_demande × Q + c_stockage × S + c_rupture × R)`

**Contraintes** :
- Conservation : `S_t = S_{t-1} + Q_{t-L} − D + R`
- Budget mensuel : `Σ_i c_achat × Q_{i,t} ≤ 500 000 USD`
- Capacité de stockage : `Σ_i v_i × S_{i,t} ≤ 5 000 m³`
- Obsolètes : `Q_{i,t} = 0`

## Paramètres économiques

| Paramètre | Valeur |
|---|---:|
| Coût fixe par commande | 50 USD |
| Coût stockage / jour | 0.1 % du coût d'achat |
| Pénalité rupture client | +20 % de la marge perdue |
| Délai Dubaï / Chine | 1 mois / 2 mois |
| Pondération service classe A | × 4.0 (vs B × 2.5, C × 1.5) |
| Horizon de planification | 3 mois |
| Budget mensuel | 500 000 USD |
| Capacité entrepôt | 5 000 m³ |

## Comparaison politique optimisée vs empirique (horizon 3 mois)

| Indicateur | Empirique | Optimisée | Δ | Δ % |
|---|---:|---:|---:|---:|
| Nb commandes passées | 285 | **177** | -108 | **-37.9 %** |
| Quantité totale | 14 508 | 3 824 | -10 684 | **-73.6 %** |
| Valeur commandes (USD) | 899 080 | 290 450 | -608 630 | **-67.7 %** |
| Stock immo moyen (USD) | 972 | 406 | -566 | **-58.3 %** |
| Ruptures (unités) | 3 455 | 3 371 | -83 | -2.4 % |
| Marge perdue (USD) | 40 630 | 39 718 | -911 | -2.2 % |
| Coût stockage (USD) | 21 861 | 6 461 | -15 400 | **-70.5 %** |
| Coût commandes (USD) | 14 250 | 8 850 | -5 400 | -37.9 % |
| **Coût total simulé** | **76 740** | **55 029** | **-21 711** | **-28.3 %** |
| CA réalisé (USD) | 714 877 | 719 157 | +4 280 | +0.6 % |
| Taux de service global | 69.2 % | 69.9 % | +0.8 pt | +1.1 % |

> **Lecture stratégique** :
> 1. Le LP **n'augmente pas significativement les ruptures** (-2.4 %) tout en commandant **73 % de quantités en moins**.
> 2. La **trésorerie immobilisée chute de 58 %** — c'est la marge de manœuvre principale dégagée par l'optimisation pour Zenith.
> 3. Le **CA réalisé est très légèrement supérieur** (+0.6 %) car les ruptures sont mieux ciblées (on évite les manques sur les produits A à forte marge).

## Distribution multi-magasins

Le plan central (177 commandes) est distribué sur les **7 magasins** Zenith
selon leur part historique de ventes pour chaque produit (`distribute_to_stores`).
Résultat : **866 lignes** dans `commandes_recommandees.csv` (colonnes
`produit_id, magasin, mois_offset, date_decision, fournisseur, classe_abc,
quantite_commandee, cout_achat, montant_total, demande_prevue, ...`).

## Sorties

- `outputs/tables/commandes_recommandees.csv` (866 lignes : plan magasin)
- `outputs/tables/commandes_centrales.csv` (531 lignes : plan central)
- `outputs/tables/baseline_policy_plan.csv` (politique empirique simulée)
- `outputs/tables/comparaison_avant_apres.csv` (KPI comparatifs)
- `outputs/tables/optimization_summary.md` (ce document)
- `outputs/figures/opt_01_par_classe_abc.png` (volumes par classe)
- `outputs/figures/opt_02_comparison_kpis.png` (politique vs politique)
- `outputs/figures/opt_03_budget_par_fournisseur.png`
- `outputs/figures/opt_04_service_par_classe.png`

## Tests unitaires

9 tests dans `tests/test_optimization.py` :
- `lead_time_months` Dubaï/Chine ;
- `fournisseur_label` dispatch ;
- Exclusion des produits obsolètes ;
- Génération `horizon × n_produits` lignes ;
- Baseline policy renvoie DataFrame valide ;
- `financial_kpis` contient les clés attendues ;
- `compare_policies` renvoie 4 colonnes attendues ;
- `distribute_to_stores` préserve les quantités totales.

Tous passants (`pytest tests/test_optimization.py -v` ⇒ **9 passed**).
