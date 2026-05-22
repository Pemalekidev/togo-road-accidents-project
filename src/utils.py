"""
utils.py
========
Utilitaires communs pour le projet Togo Road Accidents.
Logging, chemins, configuration, helpers génériques.
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Union
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS DU PROJET
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "raw":       PROJECT_ROOT / "data" / "raw",
    "processed": PROJECT_ROOT / "data" / "processed",
    "external":  PROJECT_ROOT / "data" / "external",
    "notebooks": PROJECT_ROOT / "notebooks",
    "reports":   PROJECT_ROOT / "reports",
    "figures":   PROJECT_ROOT / "reports" / "figures",
    "tables":    PROJECT_ROOT / "reports" / "tables",
    "models":    PROJECT_ROOT / "reports" / "models",
    "docs":      PROJECT_ROOT / "docs",
}

RAW_DATA_FILE  = PATHS["raw"] / "togo_road_accidents_dataset.csv"
CLEAN_DATA_FILE = PATHS["processed"] / "accidents_clean.parquet"
FEAT_DATA_FILE  = PATHS["processed"] / "accidents_features.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Retourne un logger formaté."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger("utils")


# ─────────────────────────────────────────────────────────────────────────────
# SCHÉMA DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes catégorielles avec leurs valeurs attendues
CATEGORICAL_SCHEMA = {
    "gravite":            ["Léger", "Grave", "Mortel"],
    "periode_journaliere":["Matin", "Après-midi", "Soir", "Nuit"],
    "meteo":              ["Soleil", "Nuageux", "Pluie", "Brouillard"],
    "luminosite":         ["Jour", "Nuit", "Crépuscule"],
    "type_route":         ["Urbaine", "Interurbaine", "Rurale"],
    "etat_route":         ["Bon", "Dégradé", "Très dégradé"],
    "port_casque":        ["Oui", "Non"],
    "port_ceinture":      ["Oui", "Non"],
    "accident_fete":      ["Oui", "Non"],
    "type_accident":      [
        "Collision frontale", "Collision arrière", "Collision latérale",
        "Collision piéton", "Sortie de route", "Renversement"
    ],
    "cause":              [
        "Alcool", "Téléphone", "Fatigue", "Excès de vitesse",
        "Dépassement dangereux", "Priorité non respectée", "Freinage brusque"
    ],
    "type_vehicule":      ["Moto", "Voiture", "Camion", "Bus", "Tricycle", "Vélo"],
    "region":             ["Maritime", "Plateaux", "Centrale", "Kara", "Savanes"],
}

# Encodage ordinal pour la gravité
GRAVITE_ORDER = {"Léger": 0, "Grave": 1, "Mortel": 2}
GRAVITE_COLORS = {"Léger": "#2ecc71", "Grave": "#f39c12", "Mortel": "#e74c3c"}

# Couleurs palette projet
PALETTE = {
    "primary":   "#2c3e50",
    "secondary": "#3498db",
    "success":   "#27ae60",
    "warning":   "#f39c12",
    "danger":    "#e74c3c",
    "light":     "#ecf0f1",
    "muted":     "#95a5a6",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def quick_info(df: pd.DataFrame) -> None:
    """Affiche un résumé rapide du DataFrame."""
    print(f"{'='*60}")
    print(f"Shape     : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"Mémoire   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Doublons  : {df.duplicated().sum():,}")
    print(f"\nValeurs manquantes (top 10):")
    nulls = df.isnull().sum().sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    if len(nulls):
        for col, n in nulls.head(10).items():
            pct = n / len(df) * 100
            print(f"  {col:<30} {n:>6,}  ({pct:.1f}%)")
    else:
        print("  Aucune valeur manquante ✓")
    print(f"{'='*60}")


def memory_optimize(df: pd.DataFrame) -> pd.DataFrame:
    """Optimise la mémoire d'un DataFrame en downcasting les types numériques."""
    df = df.copy()
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")
    return df


def validate_schema(df: pd.DataFrame) -> dict:
    """Valide que les valeurs catégorielles respectent le schéma attendu."""
    issues = {}
    for col, expected in CATEGORICAL_SCHEMA.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().unique())
        unexpected = actual - set(expected)
        if unexpected:
            issues[col] = list(unexpected)
    return issues


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division sécurisée (évite ZeroDivisionError)."""
    return numerator / denominator if denominator != 0 else default


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE / CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: Union[str, Path], fmt: str = "parquet") -> None:
    """Sauvegarde un DataFrame au format spécifié."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False, compression="snappy")
    elif fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif fmt == "feather":
        df.to_feather(path)
    logger.info(f"DataFrame sauvegardé → {path} ({len(df):,} lignes)")


def load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Charge un DataFrame selon l'extension du fichier."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    ext = path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif ext == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Format non supporté : {ext}")
    logger.info(f"DataFrame chargé ← {path} ({len(df):,} lignes)")
    return df


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Sauvegarde un objet en JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Charge un fichier JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTIQUES
# ─────────────────────────────────────────────────────────────────────────────

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Calcule une moyenne pondérée."""
    return np.average(values.dropna(), weights=weights.loc[values.dropna().index])


def confidence_interval_proportion(p: float, n: int, alpha: float = 0.05) -> tuple:
    """Intervalle de confiance Wilson pour une proportion."""
    from scipy import stats
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Mesure d'association Cramér's V entre deux variables catégorielles."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))


# ─────────────────────────────────────────────────────────────────────────────
# TIMER
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager pour mesurer le temps d'exécution."""

    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start).total_seconds()
        label = f"[{self.name}] " if self.name else ""
        logger.info(f"{label}Temps d'exécution : {self.elapsed:.2f}s")


if __name__ == "__main__":
    logger.info("Module utils chargé avec succès.")
    for k, v in PATHS.items():
        v.mkdir(parents=True, exist_ok=True)
    logger.info("Répertoires du projet vérifiés / créés.")
