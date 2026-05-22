"""
sql_queries.py
==============
Requêtes SQL analytiques avec DuckDB pour le projet Togo Road Accidents.
DuckDB permet d'exécuter du SQL directement sur des DataFrames Pandas.
"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional, Union
import warnings
warnings.filterwarnings("ignore")

from utils import get_logger, CLEAN_DATA_FILE, load_dataframe

logger = get_logger("sql_queries")


# ─────────────────────────────────────────────────────────────────────────────
# CONNEXION DUCKDB
# ─────────────────────────────────────────────────────────────────────────────

class AccidentsDB:
    """
    Interface SQL sur les données d'accidents via DuckDB.
    Enregistre le DataFrame Pandas comme table virtuelle.
    """

    def __init__(self, df: pd.DataFrame):
        self.con = duckdb.connect(":memory:")
        self.con.register("accidents", df)
        logger.info(f"DuckDB initialisé avec {len(df):,} accidents.")

    def query(self, sql: str) -> pd.DataFrame:
        """Exécute une requête SQL et retourne un DataFrame."""
        return self.con.execute(sql).fetchdf()

    def close(self):
        self.con.close()


def load_db(path: Path = CLEAN_DATA_FILE) -> AccidentsDB:
    """Charge le dataset et retourne une instance AccidentsDB."""
    df = load_dataframe(path)
    return AccidentsDB(df)


# ─────────────────────────────────────────────────────────────────────────────
# REQUÊTES ANALYTIQUES
# ─────────────────────────────────────────────────────────────────────────────

class AccidentsQueries:
    """Catalogue de requêtes analytiques métier."""

    def __init__(self, db: AccidentsDB):
        self.db = db

    # ── 1. VUE D'ENSEMBLE ────────────────────────────────────────────────────

    def kpi_globaux(self) -> pd.DataFrame:
        """KPIs globaux du dataset."""
        return self.db.query("""
            SELECT
                COUNT(*)                                       AS total_accidents,
                SUM(nombre_deces)                              AS total_deces,
                SUM(nombre_victimes)                           AS total_victimes,
                ROUND(AVG(nombre_victimes), 2)                 AS moy_victimes_par_accident,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2)    AS taux_mortalite_pct,
                ROUND(AVG(cout_estime)/1e6, 3)                AS cout_moyen_MFCFA,
                SUM(cout_estime)/1e9                           AS cout_total_GFCFA,
                MIN(date_accident)                             AS date_debut,
                MAX(date_accident)                             AS date_fin,
                COUNT(DISTINCT region)                         AS nb_regions,
                COUNT(DISTINCT ville)                          AS nb_villes,
                COUNT(DISTINCT axe_routier)                    AS nb_axes_routiers
            FROM accidents
        """)

    def repartition_par_gravite(self) -> pd.DataFrame:
        """Distribution des accidents par niveau de gravité."""
        return self.db.query("""
            SELECT
                gravite,
                COUNT(*)                                AS nb_accidents,
                ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2) AS pct,
                SUM(nombre_victimes)                    AS total_victimes,
                SUM(nombre_deces)                       AS total_deces,
                ROUND(AVG(cout_estime), 0)              AS cout_moyen_FCFA
            FROM accidents
            GROUP BY gravite
            ORDER BY CASE gravite
                WHEN 'Mortel' THEN 1
                WHEN 'Grave' THEN 2
                ELSE 3 END
        """)

    # ── 2. ANALYSE TEMPORELLE ─────────────────────────────────────────────────

    def tendance_annuelle(self) -> pd.DataFrame:
        """Évolution annuelle des accidents et indicateurs clés."""
        return self.db.query("""
            SELECT
                YEAR(date_accident)                     AS annee,
                COUNT(*)                                AS nb_accidents,
                SUM(nombre_deces)                       AS deces,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2) AS taux_mortalite,
                ROUND(AVG(vitesse_estimee), 1)          AS vitesse_moyenne,
                ROUND(SUM(cout_estime)/1e6, 1)          AS cout_total_MFCFA
            FROM accidents
            GROUP BY annee
            ORDER BY annee
        """)

    def accidents_par_heure(self) -> pd.DataFrame:
        """Distribution horaire des accidents et taux de mortalité."""
        return self.db.query("""
            SELECT
                heure,
                COUNT(*)                                        AS nb_accidents,
                SUM(nombre_deces)                               AS deces,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2)     AS taux_mortalite,
                ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2)  AS pct_total
            FROM accidents
            GROUP BY heure
            ORDER BY heure
        """)

    def accidents_par_jour_semaine(self) -> pd.DataFrame:
        """Sinistralité par jour de la semaine."""
        return self.db.query("""
            WITH jours AS (
                SELECT jour_semaine,
                       COUNT(*) AS nb_accidents,
                       SUM(nombre_deces) AS deces,
                       ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2) AS taux_mortalite,
                       ROUND(AVG(cout_estime), 0) AS cout_moyen
                FROM accidents
                GROUP BY jour_semaine
            )
            SELECT *,
                CASE jour_semaine
                    WHEN 'Monday'    THEN 1
                    WHEN 'Tuesday'   THEN 2
                    WHEN 'Wednesday' THEN 3
                    WHEN 'Thursday'  THEN 4
                    WHEN 'Friday'    THEN 5
                    WHEN 'Saturday'  THEN 6
                    WHEN 'Sunday'    THEN 7
                END AS ordre
            FROM jours
            ORDER BY ordre
        """)

    def heatmap_heure_jour(self) -> pd.DataFrame:
        """Matrice heure × jour (pour heatmap temporelle)."""
        return self.db.query("""
            SELECT
                heure,
                jour_semaine,
                COUNT(*) AS nb_accidents,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2) AS taux_mortalite
            FROM accidents
            GROUP BY heure, jour_semaine
            ORDER BY heure, jour_semaine
        """)

    def accidents_fetes(self) -> pd.DataFrame:
        """Comparaison accidents jours fériés vs jours normaux."""
        return self.db.query("""
            SELECT
                accident_fete,
                COUNT(*)                                      AS nb_accidents,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2)  AS taux_mortalite,
                ROUND(AVG(vitesse_estimee), 1)               AS vitesse_moyenne,
                ROUND(AVG(cout_estime), 0)                   AS cout_moyen
            FROM accidents
            GROUP BY accident_fete
        """)

    # ── 3. ANALYSE GÉOGRAPHIQUE ───────────────────────────────────────────────

    def top_regions_risque(self, n: int = 5) -> pd.DataFrame:
        """Régions les plus dangereuses (par taux de mortalité)."""
        return self.db.query(f"""
            SELECT
                region,
                COUNT(*)                                        AS nb_accidents,
                SUM(nombre_deces)                               AS deces,
                SUM(nombre_victimes)                            AS victimes,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2)     AS taux_mortalite,
                ROUND(AVG(vitesse_estimee), 1)                  AS vitesse_moy,
                ROUND(SUM(cout_estime)/1e6, 1)                  AS cout_total_MFCFA,
                ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2)  AS pct_accidents
            FROM accidents
            GROUP BY region
            ORDER BY taux_mortalite DESC
            LIMIT {n}
        """)

    def top_villes_accidentogenes(self, n: int = 10) -> pd.DataFrame:
        """Top N villes avec le plus d'accidents."""
        return self.db.query(f"""
            SELECT
                ville,
                region,
                COUNT(*)                                    AS nb_accidents,
                SUM(nombre_deces)                           AS deces,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2) AS taux_mortalite,
                ROUND(AVG(lat||lon), 2)                     AS coord_check
            FROM (
                SELECT *, CAST(latitude AS VARCHAR) AS lat, CAST(longitude AS VARCHAR) AS lon
                FROM accidents
            )
            GROUP BY ville, region
            ORDER BY nb_accidents DESC
            LIMIT {n}
        """)

    def top_axes_dangereux(self, n: int = 10) -> pd.DataFrame:
        """Axes routiers les plus dangereux."""
        return self.db.query(f"""
            SELECT
                axe_routier,
                COUNT(*)                                    AS nb_accidents,
                SUM(nombre_deces)                           AS deces,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2) AS taux_mortalite,
                ROUND(AVG(vitesse_estimee), 1)              AS vitesse_moy,
                COUNT(DISTINCT ville)                       AS nb_villes_touchees
            FROM accidents
            GROUP BY axe_routier
            HAVING COUNT(*) >= 50
            ORDER BY deces DESC
            LIMIT {n}
        """)

    # ── 4. ANALYSE DES CAUSES ─────────────────────────────────────────────────

    def causes_par_gravite(self) -> pd.DataFrame:
        """Distribution des causes par niveau de gravité."""
        return self.db.query("""
            SELECT
                cause,
                gravite,
                COUNT(*)                                        AS nb_accidents,
                ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(PARTITION BY gravite), 2) AS pct_dans_gravite
            FROM accidents
            GROUP BY cause, gravite
            ORDER BY cause, gravite
        """)

    def pivot_causes_gravite(self) -> pd.DataFrame:
        """Pivot : causes × gravité (pour heatmap)."""
        return self.db.query("""
            PIVOT (
                SELECT cause, gravite, COUNT(*) AS n
                FROM accidents
                GROUP BY cause, gravite
            )
            ON gravite
            USING SUM(n)
            ORDER BY "Mortel" DESC NULLS LAST
        """)

    def analyse_vitesse(self) -> pd.DataFrame:
        """Statistiques de vitesse par gravité et type de route."""
        return self.db.query("""
            SELECT
                gravite,
                type_route,
                ROUND(AVG(vitesse_estimee), 1)   AS vitesse_moy,
                ROUND(MEDIAN(vitesse_estimee), 1) AS vitesse_mediane,
                ROUND(STDDEV(vitesse_estimee), 1) AS vitesse_std,
                ROUND(AVG(vitesse_estimee - limitation_vitesse), 1) AS exces_moy,
                ROUND(AVG(limitation_vitesse), 0) AS limitation_moy,
                COUNT(*)                          AS nb_accidents
            FROM accidents
            GROUP BY gravite, type_route
            ORDER BY gravite, type_route
        """)

    # ── 5. ANALYSE COMPORTEMENTALE ────────────────────────────────────────────

    def impact_epi(self) -> pd.DataFrame:
        """Impact du port du casque et de la ceinture sur la gravité."""
        return self.db.query("""
            SELECT
                port_casque,
                port_ceinture,
                COUNT(*)                                        AS nb_accidents,
                ROUND(SUM(CASE WHEN gravite='Mortel' THEN 1 ELSE 0 END)*100.0/COUNT(*), 2)
                    AS taux_mortalite,
                ROUND(SUM(CASE WHEN gravite='Grave' THEN 1 ELSE 0 END)*100.0/COUNT(*), 2)
                    AS taux_grave,
                ROUND(AVG(cout_estime), 0)                     AS cout_moyen
            FROM accidents
            GROUP BY port_casque, port_ceinture
            ORDER BY taux_mortalite DESC
        """)

    def analyse_alcool_vs_heure(self) -> pd.DataFrame:
        """Accidents liés à l'alcool selon l'heure."""
        return self.db.query("""
            SELECT
                heure,
                COUNT(*) AS total_accidents,
                SUM(CASE WHEN cause = 'Alcool' THEN 1 ELSE 0 END) AS accidents_alcool,
                ROUND(SUM(CASE WHEN cause = 'Alcool' THEN 1 ELSE 0 END)*100.0/COUNT(*), 2)
                    AS pct_alcool
            FROM accidents
            GROUP BY heure
            ORDER BY heure
        """)

    # ── 6. ANALYSE ÉCONOMIQUE ─────────────────────────────────────────────────

    def cout_par_region(self) -> pd.DataFrame:
        """Coût économique des accidents par région."""
        return self.db.query("""
            SELECT
                region,
                COUNT(*)                         AS nb_accidents,
                ROUND(SUM(cout_estime)/1e6, 1)  AS cout_total_MFCFA,
                ROUND(AVG(cout_estime), 0)       AS cout_moyen_FCFA,
                ROUND(MAX(cout_estime), 0)       AS cout_max_FCFA,
                ROUND(MEDIAN(cout_estime), 0)    AS cout_median_FCFA
            FROM accidents
            GROUP BY region
            ORDER BY cout_total_MFCFA DESC
        """)

    def cout_par_cause(self) -> pd.DataFrame:
        """Coût économique par cause d'accident."""
        return self.db.query("""
            SELECT
                cause,
                COUNT(*)                                AS nb_accidents,
                ROUND(SUM(cout_estime)/1e6, 2)         AS cout_total_MFCFA,
                ROUND(AVG(cout_estime), 0)              AS cout_moyen,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2) AS taux_mortalite
            FROM accidents
            GROUP BY cause
            ORDER BY cout_total_MFCFA DESC
        """)

    def evolution_cout_annuel(self) -> pd.DataFrame:
        """Évolution du coût économique par année."""
        return self.db.query("""
            SELECT
                YEAR(date_accident)                 AS annee,
                ROUND(SUM(cout_estime)/1e9, 3)     AS cout_total_GFCFA,
                ROUND(AVG(cout_estime)/1e6, 3)     AS cout_moyen_MFCFA,
                COUNT(*)                            AS nb_accidents
            FROM accidents
            GROUP BY annee
            ORDER BY annee
        """)

    # ── 7. ANALYSE VÉHICULES ──────────────────────────────────────────────────

    def risque_par_type_vehicule(self) -> pd.DataFrame:
        """Taux de mortalité et gravité par type de véhicule."""
        return self.db.query("""
            SELECT
                type_vehicule,
                COUNT(*)                                        AS nb_accidents,
                SUM(nombre_victimes)                            AS total_victimes,
                SUM(nombre_deces)                               AS total_deces,
                ROUND(SUM(nombre_deces)*100.0/COUNT(*), 2)     AS taux_mortalite,
                ROUND(AVG(nombre_victimes), 2)                  AS moy_victimes,
                ROUND(AVG(vitesse_estimee), 1)                  AS vitesse_moy,
                ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2)  AS pct_total
            FROM accidents
            GROUP BY type_vehicule
            ORDER BY taux_mortalite DESC
        """)

    # ── 8. REQUÊTES AVANCÉES AVEC WINDOW FUNCTIONS ───────────────────────────

    def running_total_deces_par_annee(self) -> pd.DataFrame:
        """Cumul glissant des décès par année et mois."""
        return self.db.query("""
            WITH monthly AS (
                SELECT
                    YEAR(date_accident) AS annee,
                    MONTH(date_accident) AS mois,
                    SUM(nombre_deces) AS deces_mois,
                    COUNT(*) AS accidents_mois
                FROM accidents
                GROUP BY annee, mois
            )
            SELECT
                annee, mois, deces_mois, accidents_mois,
                SUM(deces_mois) OVER (
                    PARTITION BY annee ORDER BY mois
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS deces_cumul,
                ROUND(AVG(accidents_mois) OVER (
                    PARTITION BY annee ORDER BY mois
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ), 1) AS moy_mobile_3mois
            FROM monthly
            ORDER BY annee, mois
        """)

    def top_zones_nuit(self, n: int = 10) -> pd.DataFrame:
        """Zones les plus dangereuses la nuit."""
        return self.db.query(f"""
            SELECT
                ville,
                region,
                COUNT(*) FILTER (WHERE luminosite = 'Nuit') AS accidents_nuit,
                COUNT(*) AS total_accidents,
                ROUND(
                    COUNT(*) FILTER (WHERE luminosite = 'Nuit') * 100.0 / COUNT(*),
                    2
                ) AS pct_nuit,
                ROUND(
                    SUM(nombre_deces) FILTER (WHERE luminosite = 'Nuit') * 100.0 /
                    NULLIF(COUNT(*) FILTER (WHERE luminosite = 'Nuit'), 0),
                    2
                ) AS taux_mortalite_nuit
            FROM accidents
            GROUP BY ville, region
            HAVING COUNT(*) >= 100
            ORDER BY accidents_nuit DESC
            LIMIT {n}
        """)

    def correlation_vitesse_gravite(self) -> pd.DataFrame:
        """Analyse de la vitesse excessive par niveau de gravité."""
        return self.db.query("""
            SELECT
                gravite,
                ROUND(AVG(vitesse_estimee - limitation_vitesse), 2) AS exces_moy,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY vitesse_estimee - limitation_vitesse), 1) AS q25,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vitesse_estimee - limitation_vitesse), 1) AS mediane,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY vitesse_estimee - limitation_vitesse), 1) AS q75,
                ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY vitesse_estimee - limitation_vitesse), 1) AS p90,
                COUNT(*) AS n
            FROM accidents
            GROUP BY gravite
            ORDER BY CASE gravite WHEN 'Mortel' THEN 1 WHEN 'Grave' THEN 2 ELSE 3 END
        """)

    def rapport_complet(self) -> dict:
        """Retourne toutes les analyses en un dictionnaire."""
        return {
            "kpi_globaux":              self.kpi_globaux(),
            "gravite":                  self.repartition_par_gravite(),
            "tendance_annuelle":        self.tendance_annuelle(),
            "accidents_par_heure":      self.accidents_par_heure(),
            "top_regions":              self.top_regions_risque(),
            "top_axes":                 self.top_axes_dangereux(),
            "causes_gravite":           self.causes_par_gravite(),
            "analyse_vitesse":          self.analyse_vitesse(),
            "impact_epi":               self.impact_epi(),
            "cout_region":              self.cout_par_region(),
            "risque_vehicule":          self.risque_par_type_vehicule(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT RAPPORT SQL
# ─────────────────────────────────────────────────────────────────────────────

def export_sql_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Exporte toutes les tables analytiques en CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    db = AccidentsDB(df)
    queries = AccidentsQueries(db)
    rapport = queries.rapport_complet()

    for name, result_df in rapport.items():
        path = output_dir / f"{name}.csv"
        result_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"  Exporté : {path.name}")

    db.close()
    logger.info(f"Rapport SQL exporté dans {output_dir}")


if __name__ == "__main__":
    from utils import PATHS
    df = load_dataframe(CLEAN_DATA_FILE)
    db = AccidentsDB(df)
    q = AccidentsQueries(db)

    print("═" * 60)
    print("KPIs GLOBAUX")
    print("═" * 60)
    print(q.kpi_globaux().to_string(index=False))

    print("\n" + "═" * 60)
    print("RÉPARTITION PAR GRAVITÉ")
    print("═" * 60)
    print(q.repartition_par_gravite().to_string(index=False))

    print("\n" + "═" * 60)
    print("TOP RÉGIONS À RISQUE")
    print("═" * 60)
    print(q.top_regions_risque().to_string(index=False))

    db.close()
