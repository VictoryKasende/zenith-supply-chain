# Guide pas-à-pas — Construire les 4 dashboards Power BI Zenith

Ce guide vous accompagne **clic par clic**, en partant de zéro, pour
construire les 4 tableaux de bord à partir des 8 fichiers CSV exportés
dans `outputs/powerbi/`. Comptez **1 h 30 à 2 h** au total.

> **Prérequis** : Power BI Desktop installé (gratuit, Microsoft Store ou
> [powerbi.microsoft.com/desktop](https://powerbi.microsoft.com/desktop)).
> Les 8 CSV + le fichier `zenith_theme.json` dans le même dossier.

---

## ÉTAPE 0 — Préparer le dossier

1. Placez ces 9 fichiers dans un même dossier, par ex. `C:\Zenith\PowerBI\` :
   - `fact_ventes.csv`
   - `dim_produits.csv`
   - `dim_clients.csv`
   - `dim_magasins.csv`
   - `dim_temps.csv`
   - `previsions.csv`
   - `commandes_recommandees.csv`
   - `alertes_obsolescence.csv`
   - `zenith_theme.json`  ← le thème de couleurs

---

## ÉTAPE 1 — Appliquer le thème Zenith (à faire en TOUT PREMIER)

1. Ouvrez **Power BI Desktop** → nouveau rapport vierge.
2. Ruban **Affichage** (View) → bouton **Thèmes** → flèche déroulante →
   **Rechercher des thèmes** (Browse for themes).
3. Sélectionnez `zenith_theme.json` → **Ouvrir**.
4. Un bandeau confirme « *Le thème a été importé avec succès* ».

> À partir de maintenant, **tous les visuels** (cartes, graphiques,
> tableaux, slicers) adopteront automatiquement les couleurs du logo
> Zenith : bleu `#1D3B8A`, rouge `#E63946`, fond blanc, en-têtes de
> tableaux bleus à texte blanc, etc. Vous n'aurez **rien à recolorer
> manuellement**.

---

## ÉTAPE 2 — Importer les 8 fichiers CSV

Pour **chaque** fichier (répéter 8 fois) :

1. Ruban **Accueil** → **Obtenir des données** → **Texte/CSV**.
2. Choisissez le fichier → **Ouvrir**.
3. Dans l'aperçu, vérifiez que **« Délimiteur = Virgule »** et que
   l'encodage est **65001 : Unicode (UTF-8)** (important pour les
   accents français).
4. Cliquez **Charger** (Load).

> Astuce : vous pouvez tout importer d'un coup via **Obtenir des données
> → Plus → Dossier**, pointez sur `C:\Zenith\PowerBI\`, puis combinez.
> Mais l'import un par un est plus sûr pour un débutant.

### Vérification des volumétries (panneau « Données » à droite)

| Table | Lignes attendues |
|---|---:|
| fact_ventes | ~67 250 |
| dim_produits | 250 |
| dim_clients | ~4 625 |
| dim_magasins | 7 |
| dim_temps | ~1 280 |
| previsions | 1 500 |
| commandes_recommandees | 866 |
| alertes_obsolescence | 73 |

---

## ÉTAPE 3 — Vérifier les types de colonnes

1. Ruban **Accueil** → **Transformer les données** (ouvre Power Query).
2. Pour chaque table, vérifiez l'icône en tête de colonne :
   - **date** (dim_temps, fact_ventes, previsions) → type **Date**
     (icône calendrier). Si ce n'est pas le cas : clic-droit sur la
     colonne → **Changer le type → Date**.
   - **montant_total, cout_achat, prix_vente, valeur_stock_dormant,
     ca_total_36mois** → type **Nombre décimal**.
   - **quantite_vendue, quantite_commandee, mois_offset** → **Nombre entier**.
   - Tout le reste (produit_id, magasin, classe_abc…) → **Texte**.
3. Ruban **Accueil → Fermer et appliquer**.

---

## ÉTAPE 4 — Créer le modèle (relations en étoile)

1. Volet de gauche → icône **Modèle** (3e icône, schéma de tables).
2. Créez les **9 relations** en glissant-déposant une clé d'une table
   vers l'autre (ou **Gérer les relations → Nouvelle**) :

| Table source (côté « plusieurs ») | Colonne | → Table cible (côté « un ») | Colonne |
|---|---|---|---|
| fact_ventes | produit_id | dim_produits | produit_id |
| fact_ventes | client_id | dim_clients | client_id |
| fact_ventes | magasin | dim_magasins | magasin |
| fact_ventes | date | dim_temps | date |
| previsions | produit_id | dim_produits | produit_id |
| previsions | date | dim_temps | date |
| commandes_recommandees | produit_id | dim_produits | produit_id |
| commandes_recommandees | magasin | dim_magasins | magasin |
| alertes_obsolescence | produit_id | dim_produits | produit_id |

3. Pour chaque relation : **Cardinalité = Plusieurs à un (\*:1)**,
   **Direction de filtre croisé = Unique** (sauf exceptions ci-dessous).
4. **Exception** : pour `previsions` et `commandes_recommandees` vers
   `dim_produits`, mettez la **Direction = Les deux (bidirectionnel)**
   afin que les slicers produits filtrent ces tables.

> Si Power BI refuse une relation date (types différents), assurez-vous
> que les deux colonnes `date` sont bien de type **Date** (pas Date/Heure).

5. **Marquer dim_temps comme table de dates** : clic-droit sur
   `dim_temps` dans le volet Données → **Marquer comme table de dates**
   → colonne **date** → OK. (Active l'intelligence temporelle DAX.)

---

## ÉTAPE 5 — Créer les mesures DAX

1. Volet **Données** → clic-droit sur `fact_ventes` → **Nouvelle mesure**.
2. Collez chaque mesure ci-dessous (une par une), validez avec Entrée.

```dax
CA total = SUM(fact_ventes[montant_total])

Marge totale = SUM(fact_ventes[benefice_transaction])

Nb transactions = COUNTROWS(fact_ventes)

Panier moyen = DIVIDE([CA total], [Nb transactions])

CA mois précédent = CALCULATE([CA total], DATEADD(dim_temps[date], -1, MONTH))

Croissance CA % =
VAR prec = [CA mois précédent]
RETURN DIVIDE([CA total] - prec, prec)

Taux marge % = DIVIDE([Marge totale], [CA total])

Quantité vendue = SUM(fact_ventes[quantite_vendue])
```

Sur `alertes_obsolescence` :

```dax
Stock dormant total = SUM(alertes_obsolescence[valeur_stock_dormant])

Nb produits à risque = COUNTROWS(alertes_obsolescence)

% catalogue à risque = DIVIDE([Nb produits à risque], 250)
```

Sur `previsions` :

```dax
Demande prévue = SUM(previsions[qte_prevue])
```

Sur `commandes_recommandees` :

```dax
Budget commandes = SUM(commandes_recommandees[montant_total])

Nb commandes = COUNTROWS(commandes_recommandees)

Quantité commandée = SUM(commandes_recommandees[quantite_commandee])

Ruptures = SUM(commandes_recommandees[rupture])

Taux service = 1 - DIVIDE([Ruptures], SUM(commandes_recommandees[demande_prevue]))
```

> Astuce format : sélectionnez une mesure dans le volet Données, puis
> dans l'onglet **Outils de mesure** réglez le **Format** (Devise $,
> Pourcentage, ou Nombre) et les décimales. `CA total`, `Marge totale`,
> `Budget commandes`, `Stock dormant total` → **Devise ($ US, 0 déc.)** ;
> `Croissance CA %`, `Taux marge %`, `Taux service`, `% catalogue à
> risque` → **Pourcentage (1 déc.)**.

---

## ÉTAPE 6 — DASHBOARD 1 : Pilotage commercial

> Public : direction commerciale. Page nommée **« Pilotage commercial »**
> (double-clic sur l'onglet de page en bas pour renommer).

### 6.1 — Cinq cartes KPI (ligne du haut)

Pour chaque KPI : volet **Visualisations** → icône **Carte** (Card) →
glissez la mesure dans le champ **Champs**.

| Position | Visuel | Mesure |
|---|---|---|
| 1 | Carte | `CA total` |
| 2 | Carte | `Croissance CA %` |
| 3 | Carte | `Nb transactions` |
| 4 | Carte | `Panier moyen` |
| 5 | Carte | `Marge totale` |

Disposez-les côte à côte sur une bande horizontale (largeur ~ 220 px
chacune, hauteur ~ 110 px).

### 6.2 — Graphique en courbes (CA mensuel)

1. Visuel **Graphique en courbes** (Line chart).
2. **Axe X** : `dim_temps[date]` (descendez au niveau **Mois**).
3. **Axe Y** : `CA total`.
4. Placez-le en bas à gauche, ~ 700 × 320 px.

### 6.3 — Donut (CA par magasin)

1. Visuel **Anneau** (Donut chart).
2. **Légende** : `dim_magasins[magasin]`.
3. **Valeurs** : `CA total`.
4. Placez-le en bas à droite (haut).

### 6.4 — Barres horizontales (CA par famille, top 10)

1. Visuel **Graphique à barres groupées** (horizontal).
2. **Axe Y** : `dim_produits[famille]`.
3. **Axe X** : `CA total`.
4. Filtre visuel : volet **Filtres → famille → Filtrage Top N → Haut 10
   par CA total**.

### 6.5 — Tableau (top 20 produits)

1. Visuel **Tableau** (Table).
2. **Colonnes** : `dim_produits[produit_nom]`, `[famille]`,
   `[classe_abc]`, mesure `CA total`, mesure `Quantité vendue`.
3. Triez par `CA total` décroissant ; filtre **Top 20**.

### 6.6 — Slicers (bandeau supérieur ou latéral)

Ajoutez 4 **Segments** (Slicer) :
- `dim_temps[annee]`
- `dim_temps[trimestre]`
- `dim_magasins[ville]`
- `dim_clients[type_client]`

> Mise en page : 1 ligne de 5 KPI en haut, courbe à gauche + (donut au-dessus
> de barres) à droite, tableau en bas pleine largeur, slicers en bandeau
> supérieur.

---

## ÉTAPE 7 — DASHBOARD 2 : Analyse produits ABC × XYZ

> Nouvelle page **« Analyse ABC-XYZ »** (icône **+** en bas).

### 7.1 — Matrice ABC × XYZ

1. Visuel **Matrice**.
2. **Lignes** : `dim_produits[classe_abc]`.
3. **Colonnes** : `dim_produits[classe_xyz]`.
4. **Valeurs** : `CA total` (puis ajoutez `Nb transactions` ou un
   `COUNTROWS(dim_produits)` pour le nombre de produits).
5. **Mise en forme conditionnelle** : sélectionnez la mesure dans
   Valeurs → **Couleur d'arrière-plan → Format par échelle de couleurs**
   (min blanc `#F8FAFC` → max bleu `#1D3B8A`).

### 7.2 — Tableau filtrable des 250 produits

1. Visuel **Tableau**.
2. Colonnes : `produit_id, produit_nom, famille, marque, classe_abc,
   classe_xyz, libelle_cluster, ca_total_36mois, coefficient_variation`.

### 7.3 — Courbe CA par classe ABC

1. **Graphique en courbes** : Axe X `dim_temps[date]` (Mois),
   Axe Y `CA total`, **Légende** `dim_produits[classe_abc]`.

### 7.4 — Colonnes : marge par famille

1. **Graphique à colonnes groupées** : Axe X `dim_produits[famille]`,
   Axe Y mesure `Taux marge %`.

### 7.5 — Slicers

`classe_abc`, `classe_xyz`, `libelle_cluster`, `famille`, `marque`.

---

## ÉTAPE 8 — DASHBOARD 3 : Alertes obsolescence

> Nouvelle page **« Alertes obsolescence »**.

### 8.1 — Trois cartes KPI

| Visuel | Mesure |
|---|---|
| Carte | `Nb produits à risque` |
| Carte | `Stock dormant total` |
| Carte | `% catalogue à risque` |

### 8.2 — Tableau des produits flagués

1. Visuel **Tableau**, source `alertes_obsolescence`.
2. Colonnes : `produit_id, produit_nom, famille, classe_abc,
   jours_depuis_derniere_vente, nombre_mois_consecutifs_sans_vente,
   valeur_stock_dormant, severite`.
3. **Mise en forme conditionnelle** sur la colonne `severite` :
   sélectionnez la colonne → **Couleur d'arrière-plan → Format par
   règles** :
   - « Élevée » → fond rouge `#E63946`, texte blanc
   - « Modérée » → fond orange `#F59E0B`
   - « Faible » → fond jaune pâle `#FEF3C7`

### 8.3 — Barres : stock dormant par famille

1. **Barres horizontales** : Axe Y `dim_produits[famille]`,
   Axe X `Stock dormant total`.

### 8.4 — (Optionnel) Carte géographique des magasins

1. Visuel **Carte** (Map). Comme les villes sont connues (Lubumbashi,
   Kolwezi, Likasi), mettez `dim_magasins[ville]` en **Emplacement** et
   `Ruptures` (ou `Nb commandes`) en **Taille de bulle**.

### 8.5 — Slicers

`magasin`, `famille`, `severite`.

---

## ÉTAPE 9 — DASHBOARD 4 : Optimisation des commandes

> Nouvelle page **« Optimisation commandes »**.

### 9.1 — Importer la table de comparaison (hors étoile)

1. **Obtenir des données → Texte/CSV** → importez
   `comparaison_avant_apres.csv` (depuis `outputs/tables/`).
2. Cette table reste **isolée** (aucune relation) — c'est normal,
   elle sert uniquement aux visuels de comparaison.

### 9.2 — Trois cartes KPI

| Visuel | Mesure / champ |
|---|---|
| Carte | `Nb commandes` |
| Carte | `Quantité commandée` |
| Carte | `Budget commandes` |

### 9.3 — Barres groupées : empirique vs optimisé

1. **Graphique à colonnes groupées**.
2. Axe X : `comparaison_avant_apres[indicateur]`.
3. Valeurs : `comparaison_avant_apres[politique_empirique]` ET
   `comparaison_avant_apres[politique_optimisee]`.
4. Filtrez les indicateurs pertinents (quantite_commandee,
   valeur_commande_totale_usd, cout_total_simule_usd).

### 9.4 — Colonnes empilées : budget par fournisseur

1. **Graphique à colonnes empilées**.
2. Axe X : `commandes_recommandees[mois_offset]`.
3. Axe Y : `Budget commandes`.
4. Légende : `commandes_recommandees[fournisseur]` (Dubaï / Chine).

### 9.5 — Matrice calendrier (mois × produit)

1. Visuel **Matrice** : Lignes `produit_id`, Colonnes `mois_offset`,
   Valeurs `Quantité commandée`. Mise en forme conditionnelle sur la
   quantité (échelle blanc → bleu).

### 9.6 — Carte « gain »

1. Visuel **Carte de texte** (Zone de texte) : écrivez
   « **−28 % de coût total** simulé vs politique empirique ·
   **−58 % de trésorerie immobilisée** ».

### 9.7 — Slicers

`magasin`, `fournisseur`, `classe_abc`, `mois_offset`.

---

## ÉTAPE 10 — Finitions et publication

1. **Nom et logo** : sur chaque page, ajoutez en haut une **Zone de
   texte** « Zenith Informatique & Bureautique » + insérez le logo
   (**Insertion → Image**).
2. **Boutons de navigation** entre pages : **Insertion → Boutons →
   Navigateur de page**.
3. **Interactions** : **Format → Modifier les interactions** pour
   choisir quels visuels filtrent quels autres.
4. **Enregistrer** : `Fichier → Enregistrer sous` →
   `zenith_dashboards.pbix` (placez-le dans `outputs/powerbi/`).
5. **Publier** (optionnel) : `Accueil → Publier` vers Power BI Service
   pour un partage en ligne avec actualisation planifiée.

---

## Récapitulatif des couleurs (déjà dans le thème)

| Usage | Couleur | Hex |
|---|---|---|
| Primaire (bleu logo) | ![](https://placehold.co/12/1D3B8A/1D3B8A.png) | `#1D3B8A` |
| Accent (rouge logo) | ![](https://placehold.co/12/E63946/E63946.png) | `#E63946` |
| Bleu foncé (titres) | | `#152B66` |
| Succès | | `#16A34A` |
| Attention | | `#F59E0B` |
| Fond carte | | `#F8FAFC` |
| Bordure | | `#E2E8F0` |
| Texte | | `#0F172A` |

Le thème `zenith_theme.json` applique automatiquement ces couleurs à
**tous** les visuels — vous n'avez qu'à construire la structure.
