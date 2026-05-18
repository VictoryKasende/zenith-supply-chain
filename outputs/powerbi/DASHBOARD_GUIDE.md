# Guide de conception — Tableaux de bord Power BI Zenith

Ce guide détaille la mise en œuvre des **4 dashboards** Power BI à partir des
8 fichiers CSV exportés dans ce répertoire par le pipeline.

## 1. Import des données

Dans Power BI Desktop :

1. **Accueil → Obtenir des données → Texte/CSV** : charger les 8 fichiers
   présents dans `outputs/powerbi/`.
2. **Accueil → Transformer les données** : vérifier les types (Date, Nombre
   décimal, Texte) ; appliquer les modifications.

## 2. Modélisation (schéma en étoile)

```
                       dim_temps                 dim_magasins
                          │                           │
                          ▼                           ▼
        dim_produits ──► fact_ventes ◄── dim_clients
                          │
              ┌───────────┼────────────┐
              ▼           ▼            ▼
        previsions   commandes      alertes
                    recommandees   obsolescence
```

### Relations à créer dans **Modèle**

| De | Cardinalité | Vers | Clé |
|---|---|---|---|
| `fact_ventes` | n : 1 | `dim_produits` | produit_id |
| `fact_ventes` | n : 1 | `dim_clients` | client_id |
| `fact_ventes` | n : 1 | `dim_magasins` | magasin |
| `fact_ventes` | n : 1 | `dim_temps` | date |
| `previsions` | n : 1 | `dim_produits` | produit_id |
| `previsions` | n : 1 | `dim_temps` | date |
| `commandes_recommandees` | n : 1 | `dim_produits` | produit_id |
| `commandes_recommandees` | n : 1 | `dim_magasins` | magasin |
| `alertes_obsolescence` | n : 1 | `dim_produits` | produit_id |

> Activer **filtrage croisé bidirectionnel** entre `fact_ventes` et les autres
> tables de mesures (previsions, commandes) pour permettre la consolidation.

## 3. Mesures DAX recommandées

```dax
-- KPIs commerciaux
CA total          = SUM(fact_ventes[montant_total])
Marge totale      = SUMX(fact_ventes, fact_ventes[benefice_transaction])
Panier moyen      = DIVIDE([CA total], COUNTROWS(fact_ventes))
Nb transactions   = COUNTROWS(fact_ventes)

-- Comparaison année / année
CA mois précédent = CALCULATE([CA total], DATEADD(dim_temps[date], -1, MONTH))
Croissance CA %    = DIVIDE([CA total] - [CA mois précédent], [CA mois précédent])

-- Stock
Stock dormant     = SUM(alertes_obsolescence[valeur_stock_dormant])
Nb alertes        = COUNTROWS(alertes_obsolescence)

-- Prévisions
Demande prévue    = SUM(previsions[qte_prevue])

-- Commandes
Budget commandes  = SUM(commandes_recommandees[montant_total])
Nb commandes      = COUNTROWS(commandes_recommandees)
```

## 4. Dashboards à concevoir

### Dashboard 1 — Pilotage commercial

**Public** : direction commerciale, gestionnaires.

**Visuels** :
- **5 cartes KPI** en haut : CA total, Croissance YoY %, Nb transactions, Panier moyen, Marge totale.
- **Line chart** : évolution mensuelle du CA (axe X = `dim_temps[periode_yyyymm]`, axe Y = [CA total]).
- **Donut** : répartition CA par magasin.
- **Bar chart horizontal** : CA par famille (top 10).
- **Tableau** : Top 20 produits par CA (colonnes : produit_nom, famille, classe_abc, CA total, qté vendue).

**Slicers** (à placer en haut ou en bandeau latéral) :
- `dim_temps[annee]`, `dim_temps[trimestre]`
- `dim_magasins[ville]`
- `dim_clients[type_client]`

**Mise en page** : 1 ligne de 5 KPI, 2 colonnes en dessous (line chart à gauche, donut + bar à droite), tableau en bas.

---

### Dashboard 2 — Analyse produits ABC × XYZ

**Public** : responsable achats, contrôle de gestion.

**Visuels** :
- **Heatmap (matrice colorée)** : `dim_produits[classe_abc]` × `dim_produits[classe_xyz]`, valeurs = nb produits et part CA.
- **Tableau filtrable** : 250 produits avec `produit_id, produit_nom, famille, marque, classe_abc, classe_xyz, libelle_cluster, ca_total_36mois, coefficient_variation`.
- **Line chart** : évolution CA par classe ABC dans le temps.
- **Box plot / column chart** : distribution des marges par famille (axe X = `dim_produits[famille]`, axe Y = mesure `Marge moyenne` = AVERAGEX(fact_ventes, prix - cout)).
- **Carte ABC sur Pareto** : courbe CA cumulé % avec lignes 70 % / 90 %.

**Slicers** :
- `dim_produits[classe_abc]`, `[classe_xyz]`, `[libelle_cluster]`
- `dim_produits[famille]`, `[marque]`

---

### Dashboard 3 — Alertes obsolescence et ruptures

**Public** : responsable supply chain, gestionnaire des stocks.

**Visuels** :
- **3 cartes KPI** : Nb produits à risque, Stock dormant total (USD), % du catalogue.
- **Tableau** : produits flagués triés par `valeur_stock_dormant` desc — colonnes : produit_nom, famille, classe_abc, jours_depuis_derniere_vente, mois_consec_sans_vente, valeur_stock_dormant, **severite** (filaire colorée selon Faible/Modérée/Élevée).
- **Bar chart** : valeur de stock dormant par famille.
- **Carte (Map visual)** : ruptures par magasin (taille bulle = nb ruptures).
- **Line chart** : évolution mensuelle du taux de rupture global.

**Conditional formatting** :
- Couleur de fond rouge sur `severite = "Élevée"`, orange sur "Modérée", jaune sur "Faible".

**Slicers** :
- `dim_magasins[magasin]`, `dim_produits[famille]`, `alertes_obsolescence[severite]`.

---

### Dashboard 4 — Optimisation des commandes

**Public** : direction générale, contrôleur de gestion.

**Visuels** :
- **3 cartes KPI** : Nb commandes recommandées, Quantité totale, Budget total nécessaire.
- **Bar chart groupé** : commandes empiriques vs LP optimisé (politique côte à côte) sur 3 indicateurs (qté commandée, budget, ruptures évitées). Données : `outputs/tables/comparaison_avant_apres.csv` (à importer en table de mesures, **non lié au schéma étoile**).
- **Stacked column** : budget mensuel par fournisseur (Dubaï vs Chine), axe X = `mois_offset` (1 / 2 / 3).
- **Calendrier** (matrice mois × produit) : quantités à commander, conditional formatting sur la quantité.
- **Carte simulation** : zone de texte expliquant le gain (-28 % coût total simulé).

**Slicers** :
- `commandes_recommandees[magasin]`, `[fournisseur]`, `[classe_abc]`, `[mois_offset]`.

---

## 5. Tips Power BI

- **Couleurs thème Zenith** : Format → Format de page → Couleurs personnalisées
  - Primaire : `#1f4e79` (bleu)
  - Accent : `#ff6b6b` (corail)
  - Succès : `#2d6a4f` (vert)
  - Attention : `#ff9f1c` (orange)
- **Rafraîchissement automatique** : connecter via Power BI Service avec un
  passerelle de données pointant sur le dossier `outputs/powerbi/` partagé.
- **Performance** : importer les CSV via "Connecter" plutôt que "Lier" pour
  bénéficier du moteur VertiPaq.
- **Niveau de service** : créer une mesure DAX
  ```dax
  Taux service = 1 - DIVIDE(SUM(commandes_recommandees[rupture]),
                            SUM(commandes_recommandees[demande_prevue]))
  ```
  et l'afficher en KPI sur le Dashboard 4.

## 6. Modèle `.pbix` template

Le présent dépôt ne contient pas de fichier `.pbix` binaire (Power BI Desktop
est requis pour le créer). Si vous souhaitez un template prêt à l'emploi,
suivez la procédure ci-dessus dans Power BI Desktop puis enregistrez sous
`outputs/powerbi/zenith_dashboards.pbix`.

## 7. Vérification

Après import, vérifier les volumétries :

| Fichier | Lignes attendues |
|---|---:|
| `fact_ventes.csv` | ~67 250 |
| `dim_produits.csv` | 250 |
| `dim_clients.csv` | ~3 070 (ou ~4 600 avec dédoublonnage léger) |
| `dim_magasins.csv` | 7 |
| `dim_temps.csv` | ~1 280 (3 ans + 6 mois forecast) |
| `previsions.csv` | 1 500 |
| `commandes_recommandees.csv` | 866 |
| `alertes_obsolescence.csv` | 73 |
