# Captures d'écran — outil interactif Zenith

Pour générer les **5 captures d'écran** demandées par le cahier des charges
(une par page Streamlit fonctionnelle), il suffit de :

1. Lancer l'application :
   ```bash
   streamlit run app/zenith_tool.py
   ```
2. Ouvrir <http://localhost:8501> dans le navigateur.
3. Naviguer dans chaque page de la sidebar et utiliser le raccourci de capture
   d'écran du système (Win + Shift + S sur Windows, Cmd + Shift + 4 sur macOS).
4. Sauvegarder les images sous les noms suivants dans ce répertoire :

| Fichier attendu | Page Streamlit |
|---|---|
| `screen_01_dashboard.png` | 📊 Tableau de bord |
| `screen_02_classification.png` | 📦 Classification produits |
| `screen_03_obsolescence.png` | ⚠️ Alertes obsolescence |
| `screen_04_previsions.png` | 🔮 Prévisions de demande |
| `screen_05_commandes.png` | 🛒 Recommandations de commande |
| `screen_06_simulation.png` | 🧪 Simulation what-if |

Les captures seront utilisées dans le **Chapitre 4** du mémoire (section
« Outil interactif ») pour illustrer la solution opérationnelle livrée à Zenith.

## Test rapide en environnement remote

L'app a été testée en mode `headless` et répond **HTTP 200** sur le port 8501 :

```
$ streamlit run app/zenith_tool.py --server.headless true --server.port 8501
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Toutes les pages chargent leurs CSV depuis `outputs/tables/` sans erreur.
