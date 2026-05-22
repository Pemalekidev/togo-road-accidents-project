"""
data_preprocessing.py
=====================
Pipeline de nettoyage et prétraitement des données d'accidents de la route au Togo.
Transforme les données brutes en données propres prêtes pour l'analyse.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from utils import (
    get_logger, quick_info, memory_optimize, validate_schema,
    save_dataframe, RAW_DATA_FILE, CLEAN_DATA_FILE, GRAVITE_ORDER, Timer
)

logger = get_logger("preprocessing")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Charge les données brutes avec les bons types."""
    logger.info(f"Chargement des données brutes depuis {path}...")
    df = pd.read_csv(
        path,
        dtype={
            "accident_id":         "int32",
            "heure":               "int8",
            "semestre":            "int8",
            "nombre_vehicules":    "int8",
            "nombre_deces":        "int8",
            "limitation_vitesse":  "int16",
            "cout_estime":         "int64",
        },
        parse_dates=["date_accident"],
        encoding="utf-8"
    )
    logger.info(f"  → {len(df):,} lignes, {df.shape[1]} colonnes chargées.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. NETTOYAGE DES COLONNES
# ─────────────────────────────────────────────────────────────────────────────

def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les chaînes : strip, casse uniforme pour les colonnes clés."""
    df = df.copy()
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df


def fix_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Assure que date_accident est bien en datetime."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date_accident"]):
        df["date_accident"] = pd.to_datetime(df["date_accident"], errors="coerce")
    n_invalid = df["date_accident"].isna().sum()
    if n_invalid:
        logger.warning(f"  {n_invalid} dates invalides → supprimées.")
        df = df.dropna(subset=["date_accident"])
    return df


def fix_heure_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les heures aberrantes.
    L'heure doit être dans [0, 23]. 
    La colonne contient des valeurs jusqu'à 30 (erreurs de saisie).
    On applique modulo 24 pour corriger.
    """
    df = df.copy()
    mask_invalid = (df["heure"] < 0) | (df["heure"] > 23)
    n_invalid = mask_invalid.sum()
    if n_invalid:
        logger.info(f"  Heures aberrantes : {n_invalid} → correction par modulo 24.")
        df.loc[mask_invalid, "heure"] = df.loc[mask_invalid, "heure"] % 24
    return df


def fix_vitesse_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plafonne les vitesses estimées aberrantes.
    Seuil : 200 km/h (limite physique réaliste).
    """
    df = df.copy()
    Q1 = df["vitesse_estimee"].quantile(0.01)
    Q99 = df["vitesse_estimee"].quantile(0.99)
    n_clip = ((df["vitesse_estimee"] < Q1) | (df["vitesse_estimee"] > Q99)).sum()
    df["vitesse_estimee"] = df["vitesse_estimee"].clip(lower=0, upper=200)
    logger.info(f"  Vitesses hors plage [0, 200] : {n_clip} valeurs clampées.")
    return df


def fix_cout_negatif(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les coûts estimés négatifs par la médiane."""
    df = df.copy()
    mask = df["cout_estime"] < 0
    n = mask.sum()
    if n:
        median_cout = df.loc[~mask, "cout_estime"].median()
        df.loc[mask, "cout_estime"] = median_cout
        logger.info(f"  Coûts négatifs : {n} → remplacés par médiane ({median_cout:,.0f} FCFA).")
    return df


def standardize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise les colonnes binaires Oui/Non."""
    df = df.copy()
    binary_cols = ["port_casque", "port_ceinture", "accident_fete"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.capitalize()
            unexpected = set(df[col].dropna().unique()) - {"Oui", "Non"}
            if unexpected:
                logger.warning(f"  {col} : valeurs inattendues {unexpected} → NaN")
                df.loc[~df[col].isin(["Oui", "Non"]), col] = np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. GESTION DES VALEURS MANQUANTES
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratégie d'imputation par colonne :
    - meteo       → mode global (valeur la plus fréquente)
    - cause       → "Inconnue"
    - etat_route  → mode par region
    - vitesse_estimee → médiane par type_route
    - nombre_victimes → 1 (minimum logique)
    - port_casque / port_ceinture → mode global
    - intervention_secours → médiane globale
    """
    df = df.copy()

    # meteo
    mode_meteo = df["meteo"].mode()[0]
    n = df["meteo"].isna().sum()
    df["meteo"] = df["meteo"].fillna(mode_meteo)
    logger.info(f"  meteo : {n} NaN → '{mode_meteo}'")

    # cause
    n = df["cause"].isna().sum()
    df["cause"] = df["cause"].fillna("Inconnue")
    logger.info(f"  cause : {n} NaN → 'Inconnue'")

    # etat_route → mode par region
    n = df["etat_route"].isna().sum()
    df["etat_route"] = (
        df.groupby("region")["etat_route"]
        .transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Bon"))
    )
    logger.info(f"  etat_route : {n} NaN → mode par region")

    # vitesse_estimee → médiane par type_route
    n = df["vitesse_estimee"].isna().sum()
    df["vitesse_estimee"] = (
        df.groupby("type_route")["vitesse_estimee"]
        .transform(lambda x: x.fillna(x.median()))
    )
    logger.info(f"  vitesse_estimee : {n} NaN → médiane par type_route")

    # nombre_victimes → 1 par défaut (accident = au moins 1 personne impliquée)
    n = df["nombre_victimes"].isna().sum()
    df["nombre_victimes"] = df["nombre_victimes"].fillna(1.0).clip(lower=0)
    logger.info(f"  nombre_victimes : {n} NaN → 1")

    # port_casque / port_ceinture → mode global
    for col in ["port_casque", "port_ceinture"]:
        n = df[col].isna().sum()
        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Non"
        df[col] = df[col].fillna(mode_val)
        logger.info(f"  {col} : {n} NaN → '{mode_val}'")

    # intervention_secours → médiane
    n = df["intervention_secours"].isna().sum()
    median_inter = df["intervention_secours"].median()
    df["intervention_secours"] = df["intervention_secours"].fillna(median_inter)
    logger.info(f"  intervention_secours : {n} NaN → {median_inter:.0f} min")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUPPRESSION DES DOUBLONS & OUTLIERS EXTRÊMES
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons sur accident_id."""
    df = df.copy()
    n_before = len(df)
    df = df.drop_duplicates(subset=["accident_id"], keep="first")
    n_removed = n_before - len(df)
    if n_removed:
        logger.info(f"  Doublons supprimés : {n_removed}")
    return df.reset_index(drop=True)


def remove_extreme_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes avec des valeurs impossibles :
    - nombre_deces > nombre_victimes → incohérence logique
    - cout_estime > 50M FCFA (outlier extrême)
    """
    df = df.copy()
    n_before = len(df)

    # Incohérence logique : plus de décès que de victimes
    mask_incoherent = df["nombre_deces"] > df["nombre_victimes"]
    df = df[~mask_incoherent]

    # Coût aberrant (> 20M FCFA → garder, mais cap à 99.9 percentile)
    cap = df["cout_estime"].quantile(0.999)
    df["cout_estime"] = df["cout_estime"].clip(upper=cap)

    n_removed = n_before - len(df)
    logger.info(f"  Lignes aberrantes supprimées : {n_removed}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENCODAGE DE LA VARIABLE CIBLE
# ─────────────────────────────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne gravite_num (0=Léger, 1=Grave, 2=Mortel)."""
    df = df.copy()
    df["gravite_num"] = df["gravite"].map(GRAVITE_ORDER)
    n_na = df["gravite_num"].isna().sum()
    if n_na:
        logger.warning(f"  {n_na} valeurs de gravite non mappées → supprimées.")
        df = df.dropna(subset=["gravite_num"])
    df["gravite_num"] = df["gravite_num"].astype("int8")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE COMPLET
# ─────────────────────────────────────────────────────────────────────────────

def run_cleaning_pipeline(
    raw_path: Path = RAW_DATA_FILE,
    save_path: Optional[Path] = CLEAN_DATA_FILE,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Exécute le pipeline de nettoyage complet.
    
    Returns
    -------
    pd.DataFrame : données nettoyées
    """
    with Timer("Cleaning Pipeline"):
        logger.info("═" * 60)
        logger.info("DÉMARRAGE DU PIPELINE DE NETTOYAGE")
        logger.info("═" * 60)

        df = load_raw(raw_path)
        if verbose:
            quick_info(df)

        logger.info("\n[1/8] Nettoyage des chaînes de caractères...")
        df = clean_strings(df)

        logger.info("[2/8] Correction des dates...")
        df = fix_date_column(df)

        logger.info("[3/8] Correction des heures aberrantes...")
        df = fix_heure_anomalies(df)

        logger.info("[4/8] Correction des vitesses aberrantes...")
        df = fix_vitesse_anomalies(df)

        logger.info("[5/8] Correction des coûts négatifs...")
        df = fix_cout_negatif(df)

        logger.info("[6/8] Standardisation colonnes binaires...")
        df = standardize_binary_columns(df)

        logger.info("[7/8] Imputation des valeurs manquantes...")
        df = impute_missing_values(df)

        logger.info("[8/8] Suppression doublons & outliers...")
        df = remove_duplicates(df)
        df = remove_extreme_outliers(df)

        # Encodage de la cible
        df = encode_target(df)

        # Optimisation mémoire
        df = memory_optimize(df)

        # Validation du schéma
        issues = validate_schema(df)
        if issues:
            logger.warning(f"Problèmes de schéma détectés : {issues}")
        else:
            logger.info("Validation du schéma : OK ✓")

        if verbose:
            logger.info("\nDonnées nettoyées :")
            quick_info(df)

        if save_path:
            save_dataframe(df, save_path, fmt="parquet")

        logger.info("═" * 60)
        logger.info("PIPELINE DE NETTOYAGE TERMINÉ")
        logger.info("═" * 60)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATISTIQUES DE QUALITÉ
# ─────────────────────────────────────────────────────────────────────────────

def data_quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un rapport de qualité comparant données brutes et nettoyées.
    """
    report = {
        "Lignes (brut)":            len(df_raw),
        "Lignes (propre)":          len(df_clean),
        "Lignes supprimées":        len(df_raw) - len(df_clean),
        "Taux de rétention (%)":    round(len(df_clean) / len(df_raw) * 100, 2),
        "Colonnes":                 df_clean.shape[1],
        "Mémoire (MB)":             round(df_clean.memory_usage(deep=True).sum() / 1e6, 2),
        "Valeurs manquantes":       df_clean.isnull().sum().sum(),
        "Doublons restants":        df_clean.duplicated().sum(),
    }
    return pd.DataFrame.from_dict(report, orient="index", columns=["Valeur"])


if __name__ == "__main__":
    df_clean = run_cleaning_pipeline(verbose=True)
    print(f"\nDonnées nettoyées : {df_clean.shape}")
    print(df_clean.head(3))
