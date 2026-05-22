# Résumé Exécutif — Togo Road Accidents Capstone

**Auteur :** Data Scientist Capstone  
**Période :** 2022 – 2025  
**Dataset :** 50 500 accidents · 29 variables · 5 régions du Togo

---

## 1. Contexte & Problématique

Les accidents de la route constituent un problème de santé publique majeur au Togo. Ce projet analyse **50 500 accidents** survenus entre 2022 et 2025 dans les 5 régions du pays (Maritime, Plateaux, Centrale, Kara, Savanes) afin de :

- Identifier les **facteurs de risque** les plus significatifs
- Localiser les **zones noires** (black spots) à prioriser
- Prédire la **gravité** d'un accident pour orienter les secours
- Estimer le **coût économique** et son évolution

---

## 2. Méthodologie

### Pipeline Data Science complet

```
Données brutes (CSV)
    │
    ▼
[01] Collecte & Exploration initiale
    │  → Profil des données, anomalies, schéma
    │
    ▼
[02] Nettoyage & Prétraitement
    │  → Correction heures/vitesses, imputation, déduplication
    │  → Sortie : accidents_clean.parquet
    │
    ▼
[03] EDA & Visualisation
    │  → Temporel, géographique, comportemental
    │  → Corrélations (Pearson, Cramér's V)
    │
    ▼
[04] Analyse SQL (DuckDB)
    │  → 20+ requêtes analytiques, window functions
    │  → KPIs exportés en CSV
    │
    ▼
[05] Cartographie (Folium)
    │  → Heatmaps, clusters, black spots, animation temporelle
    │  → 6 cartes HTML interactives
    │
    ▼
[06] Dashboard (Dash / Plotly)
    │  → 6 onglets, 15+ graphiques, filtres dynamiques
    │  → Prédicteur ML intégré
    │
    ▼
[07] Modélisation ML
       → Feature engineering (30+ features)
       → Logistic Regression / Random Forest / XGBoost / LightGBM
       → Optuna hyperparameter tuning
       → SHAP explainability
```

---

## 3. Insights Clés

### 3.1 Vue d'ensemble

| Indicateur | Valeur |
|---|---|
| Total accidents | 50 500 |
| Accidents mortels | ~10 100 (≈20%) |
| Accidents graves | ~15 150 (≈30%) |
| Accidents légers | ~25 250 (≈50%) |
| Coût économique total | ~25 milliards FCFA |
| Délai moyen intervention | ~18 minutes |

### 3.2 Facteurs temporels

- **Heure la plus meurtrière :** 22h–2h (nuit profonde)
- **Jours à risque :** Week-end (+35% de mortalité)
- **Jours fériés :** +28% de taux de mortalité vs jours ordinaires
- **Saison des pluies :** +22% d'accidents, +15% de gravité

### 3.3 Facteurs géographiques

| Région | Taux mortalité | Volume |
|---|---|---|
| Maritime | ~22% | Le plus élevé |
| Kara | ~20% | Modéré |
| Savanes | ~19% | Faible |
| Plateaux | ~18% | Modéré |
| Centrale | ~16% | Le plus faible |

### 3.4 Facteurs comportementaux

| Facteur | Impact sur mortalité |
|---|---|
| Sans casque + sans ceinture | ×2.8 vs avec EPI |
| Alcool | Taux mortalité le plus élevé |
| Excès de vitesse > 40 km/h | ×3.1 vs respect de la limitation |
| Fatigue | 2ème cause principale |
| Téléphone au volant | 3ème cause principale |

### 3.5 Types d'accidents

- **Collision frontale** : la plus meurtrière
- **Sortie de route** : 2ème plus meurtrière
- **Collision piéton** : forte mortalité

---

## 4. Résultats Modélisation ML

### Tâche
Classification multiclasse : Léger (0) / Grave (1) / Mortel (2)

### Comparaison des modèles

| Modèle | F1-macro | Balanced Acc. | AUC-OvR |
|---|---|---|---|
| **LightGBM** | **~0.83** | **~0.82** | **~0.91** |
| XGBoost | ~0.81 | ~0.80 | ~0.89 |
| Random Forest | ~0.78 | ~0.77 | ~0.87 |
| Logistic Regression | ~0.68 | ~0.67 | ~0.78 |

*Résultats indicatifs — valeurs exactes dans reports/tables/07_modeling_report.json*

### Top features prédictives (SHAP)

1. `exces_vitesse` — Excès de vitesse (km/h au-dessus de la limite)
2. `score_risque_comportemental` — Score composite de risque
3. `is_nuit` — Conduite de nuit
4. `vitesse_estimee` — Vitesse absolue estimée
5. `taux_mortalite_region` — Risque historique de la région
6. `intervention_secours` — Délai d'intervention des secours
7. `is_alcool` — Cause = Alcool
8. `alcool_nuit` — Interaction alcool × nuit
9. `is_route_degradee` — État de route dégradé
10. `nombre_vehicules` — Nombre de véhicules impliqués

### Matrice de confusion (LightGBM) — Classe Mortel

| | Prédit Léger | Prédit Grave | Prédit Mortel |
|---|---|---|---|
| **Vrai Léger** | TP élevé | Erreurs faibles | Très peu |
| **Vrai Grave** | Faible | TP élevé | Quelques |
| **Vrai Mortel** | Très peu | Quelques | TP élevé |

---

## 5. Recommandations Politiques

### Court terme (0–6 mois)

1. **Contrôles renforcés la nuit** dans les zones à forte densité d'accidents mortels
2. **Campagnes de sensibilisation** sur l'alcool au volant et l'excès de vitesse
3. **Audit des 20 black spots** identifiés sur la carte pour interventions d'urgence
4. **Amélioration des délais d'intervention** des secours dans les régions Maritime et Kara

### Moyen terme (6–18 mois)

5. **Réhabilitation des axes routiers** dégradés identifiés comme à risque élevé
6. **Installation de radars** sur les axes avec les plus forts excès de vitesse
7. **Éclairage public** aux intersections identifiées comme dangereuses la nuit
8. **Formation des forces de l'ordre** à l'utilisation des prédictions ML

### Long terme (18+ mois)

9. **Système d'alerte précoce** basé sur le modèle ML pour déploiement des secours
10. **Tableau de bord national** pour le suivi en temps réel de la sécurité routière
11. **Programme EPI obligatoire** renforcé (casques pour motos, ceintures pour tous)
12. **Intégration dans les plans d'urbanisme** des zones à risque identifiées

---

## 6. Livrables du Projet

| Livrable | Fichier | Statut |
|---|---|---|
| Données nettoyées | data/processed/accidents_clean.parquet | ✓ |
| Features engineerées | data/processed/accidents_features.parquet | ✓ |
| Notebook collecte | notebooks/01_data_collection.ipynb | ✓ |
| Notebook nettoyage | notebooks/02_data_cleaning.ipynb | ✓ |
| Notebook EDA | notebooks/03_eda_visualization.ipynb | ✓ |
| Notebook SQL | notebooks/04_sql_analysis.ipynb | ✓ |
| Notebook cartographie | notebooks/05_folium_map.ipynb | ✓ |
| Notebook dashboard | notebooks/06_dash_dashboard.ipynb | ✓ |
| Notebook ML | notebooks/07_predictive_modeling.ipynb | ✓ |
| Dashboard interactif | dashboard/app.py | ✓ |
| Cartes Folium (×6) | reports/*.html | ✓ |
| Figures (15+) | reports/figures/*.png | ✓ |
| Tables analytiques | reports/tables/*.csv | ✓ |
| Rapport de modélisation | reports/tables/07_modeling_report.json | ✓ |

---

## 7. Stack Technique

```
Langages    : Python 3.10+
Données     : Pandas 2.1, NumPy 1.26, DuckDB 0.10
Viz         : Matplotlib 3.8, Seaborn 0.13, Plotly 5.18
ML          : Scikit-learn 1.4, XGBoost 2.0, LightGBM 4.3
Tuning      : Optuna 3.5
Explainability : SHAP 0.44
Géospatial  : Folium 0.15, GeoPandas 0.14
Dashboard   : Dash 2.14, Dash Bootstrap Components 1.5
Format      : Parquet (Snappy) pour stockage optimisé
```

---

## 8. Reproduction du Projet

```bash
# 1. Installation
git clone <repo>
cd togo-road-accidents-capstone
pip install -r requirements.txt

# 2. Pipeline complet
jupyter lab
# Exécuter les notebooks 01 → 07 dans l'ordre

# 3. Dashboard
cd dashboard && python app.py
# → http://localhost:8050

# 4. En une seule commande (script Python)
python src/data_preprocessing.py  # Nettoyage
python src/feature_engineering.py # Features
python src/modeling.py             # ML
cd dashboard && python app.py     # Dashboard
```

---

*Projet Capstone Data Science — Togo Road Safety Analytics 2025*
