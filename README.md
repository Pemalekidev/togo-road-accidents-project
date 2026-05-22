#  Togo Road Accidents — Capstone Project

> **Analyse complète des accidents de la route au Togo (2022–2025)**  
> Pipeline Data Science de bout en bout : collecte → nettoyage → EDA → SQL → Cartographie → Dashboard → Modélisation prédictive

---
##  Contexte du projet

Ce projet analyse **50 500 accidents de la route au Togo** sur la période 2022–2025 à travers un pipeline Data Science complet. L'objectif est de comprendre les facteurs de risque, identifier les zones dangereuses et prédire la gravité des accidents afin d'orienter les politiques de sécurité routière.

##  Objectifs

| Objectif | Méthode |
|---|---|
| Analyser les tendances temporelles et spatiales | EDA + Visualisations |
| Identifier les causes et facteurs aggravants | SQL + Statistiques |
| Cartographier les zones à risque | Folium / Heatmaps |
| Prédire la gravité d'un accident | Machine Learning |
| Estimer le coût économique | Régression |
| Tableau de bord interactif | Dash / Plotly |

##  Dataset

| Attribut | Valeur |
|---|---|
| Lignes | 50 500 |
| Colonnes | 29 |
| Période | 2022 – 2025 |
| Source | Données simulées réalistes Togo |
| Format | CSV (UTF-8) |

### Colonnes principales

```
accident_id, date_accident, heure, jour_semaine, semestre, periode_journaliere,
region, ville, latitude, longitude, axe_routier, type_route, meteo, luminosite,
type_vehicule, nombre_vehicules, nombre_victimes, nombre_deces, gravite,
cause, etat_route, limitation_vitesse, vitesse_estimee, port_casque,
port_ceinture, intervention_secours, type_accident, accident_fete, cout_estime
```

##  Structure du projet

```
togo-road-accidents-capstone/
│
├── README.md                   ← Ce fichier
├── requirements.txt            ← Dépendances Python
├── .gitignore
│
├── data/
│   ├── raw/                    ← Données brutes originales
│   ├── processed/              ← Données nettoyées & features engineerées
│   └── external/               ← Données externes (shapefiles, etc.)
│
├── notebooks/
│   ├── 01_data_collection.ipynb        ← Exploration initiale & validation
│   ├── 02_data_cleaning.ipynb          ← Nettoyage & traitement des nulls
│   ├── 03_eda_visualization.ipynb      ← Analyse exploratoire approfondie
│   ├── 04_sql_analysis.ipynb           ← Requêtes SQL analytiques
│   ├── 05_folium_map.ipynb             ← Cartographie interactive
│   ├── 06_dash_dashboard.ipynb         ← Prototype du dashboard
│   └── 07_predictive_modeling.ipynb    ← Modélisation ML complète
│
├── src/
│   ├── data_preprocessing.py   ← Pipeline de nettoyage
│   ├── feature_engineering.py  ← Création de features
│   ├── sql_queries.py          ← Requêtes SQL analytiques
│   ├── visualization.py        ← Fonctions de visualisation
│   ├── modeling.py             ← Entraînement & évaluation des modèles
│   └── utils.py                ← Utilitaires communs
│
├── dashboard/
│   ├── app.py                  ← Application Dash principale
│   ├── assets/                 ← CSS, images
│   └── components/             ← Composants Dash réutilisables
│
├── reports/
│   ├── figures/                ← Graphiques exportés
│   ├── tables/                 ← Tableaux exportés
│   └── final_presentation.pptx
│
└── docs/
    └── project_summary.md      ← Résumé exécutif
```

##  Installation & Lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/votre-user/togo-road-accidents-capstone.git
cd togo-road-accidents-capstone
```

### 2. Environnement virtuel
```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

### 3. Lancer JupyterLab
```bash
jupyter lab
```

### 4. Lancer le Dashboard
```bash
cd dashboard
python app.py
# → http://localhost:8050
```

##  Pipeline ML

```
Raw Data → Cleaning → Feature Engineering → Train/Test Split
    → [RandomForest | XGBoost | LightGBM | LogisticRegression]
    → Cross-Validation → Hyperparameter Tuning (Optuna)
    → SHAP Explainability → Final Evaluation
```

**Tâche principale :** Classification multiclasse de la gravité (Léger / Grave / Mortel)  
**Métriques :** F1-macro, AUC-OvR, Balanced Accuracy

##  Résultats attendus

- Modèle de gravité : **F1-macro > 0.80**
- Variables les plus importantes : vitesse, heure, type de route, météo
- Zones à risque identifiées sur carte interactive
- Dashboard opérationnel avec filtres dynamiques

##  Auteur

Projet Capstone — Data Science  
**Togo Road Safety Analytics** | 2025

##  Licence

MIT License
