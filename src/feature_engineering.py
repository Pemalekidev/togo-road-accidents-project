"""
feature_engineering.py
======================
Création de features pour la modélisation prédictive.
Transforme les données nettoyées en features ML-ready.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")

from utils import (
    get_logger, save_dataframe, load_dataframe,
    CLEAN_DATA_FILE, FEAT_DATA_FILE, Timer
)

logger = get_logger("feature_engineering")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURES TEMPORELLES
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des features temporelles riches depuis date_accident et heure.
    - Cycliques (sin/cos) pour heure, jour, mois
    - Périodes : rush hour, week-end, vacances
    """
    df = df.copy()

    # Extraction de base
    df["annee"]     = df["date_accident"].dt.year.astype("int16")
    df["mois"]      = df["date_accident"].dt.month.astype("int8")
    df["semaine"]   = df["date_accident"].dt.isocalendar().week.astype("int8")
    df["jour_mois"] = df["date_accident"].dt.day.astype("int8")
    df["trimestre"] = df["date_accident"].dt.quarter.astype("int8")

    # Encodage cyclique de l'heure (capture la circularité 23h→0h)
    df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)

    # Encodage cyclique du mois
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)

    # Encodage cyclique du jour de la semaine (lundi=0)
    jour_num = df["date_accident"].dt.dayofweek
    df["jour_sin"] = np.sin(2 * np.pi * jour_num / 7)
    df["jour_cos"] = np.cos(2 * np.pi * jour_num / 7)

    # Week-end
    df["is_weekend"] = (jour_num >= 5).astype("int8")

    # Heures de pointe (7h-9h, 17h-20h)
    df["is_rush_hour"] = (
        ((df["heure"] >= 7) & (df["heure"] <= 9)) |
        ((df["heure"] >= 17) & (df["heure"] <= 20))
    ).astype("int8")

    # Nuit profonde (22h-5h) — période à risque élevé
    df["is_nuit_profonde"] = (
        (df["heure"] >= 22) | (df["heure"] <= 5)
    ).astype("int8")

    # Mois de saison des pluies au Togo (mars-juillet, oct-nov)
    df["is_saison_pluies"] = df["mois"].isin([3, 4, 5, 6, 7, 10, 11]).astype("int8")

    logger.info("  Features temporelles créées ✓")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES DE RISQUE COMPORTEMENTAL
# ─────────────────────────────────────────────────────────────────────────────

def add_behavioral_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features capturant le comportement à risque du conducteur.
    """
    df = df.copy()

    # Excès de vitesse : vitesse estimée - limitation
    df["exces_vitesse"] = (df["vitesse_estimee"] - df["limitation_vitesse"]).clip(lower=0)
    df["ratio_vitesse"] = df["vitesse_estimee"] / df["limitation_vitesse"].replace(0, 1)

    # Score de risque comportemental (0-5)
    df["score_risque_comportemental"] = (
        (df["port_casque"] == "Non").astype("int8") +
        (df["port_ceinture"] == "Non").astype("int8") +
        (df["cause"].isin(["Alcool", "Téléphone", "Fatigue"])).astype("int8") +
        (df["exces_vitesse"] > 20).astype("int8") +
        (df["cause"] == "Alcool").astype("int8")
    )

    # Indicateur alcool / distracteur
    df["is_alcool"] = (df["cause"] == "Alcool").astype("int8")
    df["is_distracteur"] = (df["cause"] == "Téléphone").astype("int8")
    df["is_fatigue"] = (df["cause"] == "Fatigue").astype("int8")
    df["is_exces_vitesse"] = (df["cause"] == "Excès de vitesse").astype("int8")

    # Port des EPI combiné
    df["epi_double"] = (
        (df["port_casque"] == "Oui") & (df["port_ceinture"] == "Oui")
    ).astype("int8")
    df["no_epi"] = (
        (df["port_casque"] == "Non") & (df["port_ceinture"] == "Non")
    ).astype("int8")

    logger.info("  Features comportementales créées ✓")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURES ENVIRONNEMENTALES
# ─────────────────────────────────────────────────────────────────────────────

def add_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features liées à l'environnement routier et météorologique.
    """
    df = df.copy()

    # Conditions météo à risque
    df["is_meteo_risque"] = df["meteo"].isin(["Pluie", "Brouillard"]).astype("int8")

    # Conditions visuelles dégradées
    df["is_nuit"] = (df["luminosite"] == "Nuit").astype("int8")
    df["is_crepuscule"] = (df["luminosite"] == "Crépuscule").astype("int8")

    # Route dégradée
    df["is_route_degradee"] = df["etat_route"].isin(
        ["Dégradé", "Très dégradé"]
    ).astype("int8")

    # Score de conditions dangereuses (0-4)
    df["score_conditions_env"] = (
        df["is_meteo_risque"] +
        df["is_nuit"] +
        df["is_route_degradee"] +
        df["is_crepuscule"]
    )

    # Type de route encodé ordinal (Urbaine < Interurbaine < Rurale)
    route_order = {"Urbaine": 0, "Interurbaine": 1, "Rurale": 2}
    df["type_route_num"] = df["type_route"].map(route_order).fillna(1).astype("int8")

    # Limitation de vitesse catégorisée
    df["speed_zone"] = pd.cut(
        df["limitation_vitesse"],
        bins=[0, 50, 70, 90, 200],
        labels=["Zone30-50", "Zone70", "Zone90", "Zone+90"],
        right=True
    )

    logger.info("  Features environnementales créées ✓")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURES D'INTERACTION
# ─────────────────────────────────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features d'interaction entre variables (termes croisés pertinents).
    """
    df = df.copy()

    # Vitesse × Nuit : excès de vitesse la nuit est particulièrement dangereux
    df["exces_vitesse_nuit"] = df["exces_vitesse"] * df["is_nuit"]

    # Alcool × Nuit
    df["alcool_nuit"] = df["is_alcool"] * df["is_nuit"]

    # Mauvais temps × Route dégradée
    df["pluie_route_deg"] = df["is_meteo_risque"] * df["is_route_degradee"]

    # Nombre victimes × gravité comportementale
    df["victimes_risque"] = df["nombre_victimes"] * df["score_risque_comportemental"]

    # Multi-véhicules (>1 véhicule impliqué)
    df["is_multi_vehicule"] = (df["nombre_vehicules"] > 1).astype("int8")

    # Délai d'intervention × gravité
    df["urgence_intervention"] = df["intervention_secours"] * (
        df["gravite"].isin(["Grave", "Mortel"]).astype("int8")
    )

    logger.info("  Features d'interaction créées ✓")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURES GÉOGRAPHIQUES (AGGREGATIONS)
# ─────────────────────────────────────────────────────────────────────────────

def add_geographic_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode le risque historique par zone géographique.
    Utilise le taux de mortalité par région et ville (target encoding robuste).
    """
    df = df.copy()

    # Taux de mortalité par région (% accidents mortels)
    mortalite_region = (
        df.groupby("region")["gravite"]
        .apply(lambda x: (x == "Mortel").mean())
        .rename("taux_mortalite_region")
    )
    df = df.merge(mortalite_region, on="region", how="left")

    # Taux de mortalité par ville
    mortalite_ville = (
        df.groupby("ville")["gravite"]
        .apply(lambda x: (x == "Mortel").mean())
        .rename("taux_mortalite_ville")
    )
    df = df.merge(mortalite_ville, on="ville", how="left")

    # Nombre d'accidents par axe routier (densité de l'axe)
    count_axe = (
        df.groupby("axe_routier")["accident_id"]
        .count()
        .rename("nb_accidents_axe")
    )
    df = df.merge(count_axe, on="axe_routier", how="left")
    df["log_nb_accidents_axe"] = np.log1p(df["nb_accidents_axe"])

    logger.info("  Features géographiques créées ✓")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENCODAGE POUR LE ML
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encoding des variables catégorielles pour le ML.
    (Utilisé dans le notebook de modélisation avec sklearn ColumnTransformer)
    """
    df = df.copy()

    # Conversion binaire Oui/Non → 0/1
    for col in ["port_casque", "port_ceinture", "accident_fete"]:
        df[col] = (df[col] == "Oui").astype("int8")

    return df


def get_feature_sets() -> dict:
    """
    Retourne les ensembles de features pour différents usages.
    """
    return {
        "temporal": [
            "heure", "heure_sin", "heure_cos", "mois_sin", "mois_cos",
            "jour_sin", "jour_cos", "is_weekend", "is_rush_hour",
            "is_nuit_profonde", "is_saison_pluies", "semestre",
        ],
        "behavioral": [
            "exces_vitesse", "ratio_vitesse", "score_risque_comportemental",
            "is_alcool", "is_distracteur", "is_fatigue", "is_exces_vitesse",
            "epi_double", "no_epi", "port_casque", "port_ceinture",
        ],
        "environmental": [
            "is_meteo_risque", "is_nuit", "is_crepuscule",
            "is_route_degradee", "score_conditions_env",
            "type_route_num", "limitation_vitesse",
        ],
        "interaction": [
            "exces_vitesse_nuit", "alcool_nuit", "pluie_route_deg",
            "victimes_risque", "is_multi_vehicule",
        ],
        "geographic": [
            "taux_mortalite_region", "taux_mortalite_ville",
            "log_nb_accidents_axe",
        ],
        "accident": [
            "nombre_vehicules", "nombre_victimes", "intervention_secours",
            "accident_fete",
        ],
        "categorical_to_encode": [
            "region", "type_route", "meteo", "luminosite",
            "type_vehicule", "etat_route", "type_accident",
            "cause", "periode_journaliere", "speed_zone",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. PIPELINE COMPLET
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    clean_path: Path = CLEAN_DATA_FILE,
    save_path: Optional[Path] = FEAT_DATA_FILE,
) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering.
    """
    with Timer("Feature Engineering"):
        logger.info("═" * 60)
        logger.info("DÉMARRAGE DU FEATURE ENGINEERING")
        logger.info("═" * 60)

        df = load_dataframe(clean_path)
        n_orig_cols = df.shape[1]

        logger.info("[1/6] Features temporelles...")
        df = add_temporal_features(df)

        logger.info("[2/6] Features comportementales...")
        df = add_behavioral_risk_features(df)

        logger.info("[3/6] Features environnementales...")
        df = add_environmental_features(df)

        logger.info("[4/6] Features d'interaction...")
        df = add_interaction_features(df)

        logger.info("[5/6] Features géographiques...")
        df = add_geographic_risk_features(df)

        logger.info("[6/6] Encodage binaire...")
        df = encode_categoricals(df)

        n_new_cols = df.shape[1] - n_orig_cols
        logger.info(f"\n  {n_new_cols} nouvelles features créées.")
        logger.info(f"  Shape finale : {df.shape}")

        if save_path:
            save_dataframe(df, save_path, fmt="parquet")

        logger.info("═" * 60)
        logger.info("FEATURE ENGINEERING TERMINÉ")
        logger.info("═" * 60)

    return df


def feature_importance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé statistique des features créées.
    """
    feature_sets = get_feature_sets()
    all_features = [f for group in feature_sets.values() for f in group]
    available = [f for f in all_features if f in df.columns]

    summary = df[available].describe().T
    summary["missing_pct"] = df[available].isnull().mean() * 100
    return summary


if __name__ == "__main__":
    df_feat = run_feature_engineering()
    print(f"\nFeatures finales : {df_feat.shape[1]} colonnes")
    sets = get_feature_sets()
    for name, cols in sets.items():
        available = [c for c in cols if c in df_feat.columns]
        print(f"  {name:<25} : {len(available)} features")
