"""
run_pipeline.py
===============
Script maître pour exécuter l'intégralité du pipeline
Data Science Togo Road Accidents en une seule commande.

Usage:
    python run_pipeline.py [--skip-modeling] [--tune] [--dashboard]

Options:
    --skip-modeling : Sauter l'étape de modélisation ML (rapide)
    --tune          : Activer l'optimisation Optuna (lent, ~10 min)
    --dashboard     : Lancer le dashboard après le pipeline
    --help          : Afficher cette aide
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Ajout du dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def banner(text: str, char: str = "═") -> None:
    width = 65
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def section(text: str) -> None:
    print(f"\n▶  {text}")
    print("─" * 50)


def success(text: str) -> None:
    print(f"  ✓ {text}")


def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{int(s//60)}m {int(s%60)}s" if s >= 60 else f"{s:.1f}s"


def run_step1_cleaning() -> object:
    """Étape 1 : Nettoyage des données."""
    section("ÉTAPE 1/4 — Nettoyage des données")
    from data_preprocessing import run_cleaning_pipeline
    df_clean = run_cleaning_pipeline(verbose=False)
    success(f"Données nettoyées : {df_clean.shape[0]:,} lignes × {df_clean.shape[1]} colonnes")
    success("Sauvegardé → data/processed/accidents_clean.parquet")
    return df_clean


def run_step2_features(df_clean) -> object:
    """Étape 2 : Feature engineering."""
    section("ÉTAPE 2/4 — Feature Engineering")
    from feature_engineering import run_feature_engineering
    df_feat = run_feature_engineering(verbose=False) if hasattr(run_feature_engineering, 'verbose') else run_feature_engineering()
    success(f"Features créées : {df_feat.shape[1]} colonnes")
    success("Sauvegardé → data/processed/accidents_features.parquet")
    return df_feat


def run_step3_sql(df_clean) -> None:
    """Étape 3 : Analyse SQL."""
    section("ÉTAPE 3/4 — Analyse SQL (DuckDB)")
    from sql_queries import AccidentsDB, AccidentsQueries, export_sql_report
    from utils import PATHS

    db = AccidentsDB(df_clean)
    q  = AccidentsQueries(db)

    kpis = q.kpi_globaux()
    row = kpis.iloc[0]

    success(f"Total accidents  : {int(row.get('total_accidents', 0)):,}")
    success(f"Total décès      : {int(row.get('total_deces', 0)):,}")
    success(f"Taux mortalité   : {float(row.get('taux_mortalite_pct', 0)):.2f}%")
    success(f"Coût total       : {float(row.get('cout_total_GFCFA', 0)):.2f} GFCFA")

    export_sql_report(df_clean, PATHS["tables"])
    success(f"Tables analytiques exportées → reports/tables/")

    db.close()


def run_step4_modeling(df_feat, tune: bool = False) -> dict:
    """Étape 4 : Modélisation ML."""
    section("ÉTAPE 4/4 — Modélisation Prédictive")
    from modeling import run_modeling_pipeline

    results = run_modeling_pipeline(
        df=df_feat,
        tune=tune,
        n_trials=30 if tune else 0,
        save_models=True
    )

    best = results["best_result"]
    success(f"Meilleur modèle : {best['model_name']}")
    success(f"F1-macro (test) : {best['f1_macro']:.4f}")
    success(f"AUC-OvR         : {best['auc_ovr']:.4f}")
    success(f"Balanced Acc.   : {best['balanced_acc']:.4f}")

    print("\n  Comparaison des modèles :")
    print(results["comparison"].to_string(index=False))

    return results


def run_dashboard() -> None:
    """Lancement du dashboard Dash."""
    import subprocess
    import os

    dashboard_path = Path(__file__).parent / "dashboard"
    print(f"\n  Lancement du dashboard...")
    print(f"  → http://localhost:8050")
    print(f"  (Ctrl+C pour arrêter)\n")

    try:
        subprocess.run(
            [sys.executable, "app.py"],
            cwd=str(dashboard_path)
        )
    except KeyboardInterrupt:
        print("\n  Dashboard arrêté.")


def generate_final_report(results: dict, start_time: float) -> None:
    """Génère un rapport final JSON."""
    import json
    from utils import PATHS

    report = {
        "timestamp":       datetime.now().isoformat(),
        "duree_pipeline":  elapsed(start_time),
        "meilleur_modele": results["best_result"]["model_name"],
        "f1_macro":        round(results["best_result"]["f1_macro"], 4),
        "auc_ovr":         round(results["best_result"]["auc_ovr"], 4),
        "balanced_acc":    round(results["best_result"]["balanced_acc"], 4),
        "comparaison":     results["comparison"].to_dict(orient="records"),
    }

    report_path = PATHS["tables"] / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    success(f"Rapport pipeline → {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Data Science — Togo Road Accidents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--skip-modeling", action="store_true",
                        help="Sauter l'étape ML (plus rapide)")
    parser.add_argument("--tune", action="store_true",
                        help="Activer Optuna hyperparameter tuning")
    parser.add_argument("--dashboard", action="store_true",
                        help="Lancer le dashboard après le pipeline")
    args = parser.parse_args()

    start_time = time.time()

    banner("🚦 TOGO ROAD ACCIDENTS — PIPELINE DATA SCIENCE")
    print(f"  Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Options   : skip-modeling={args.skip_modeling} | tune={args.tune} | dashboard={args.dashboard}")

    try:
        # Étape 1 : Nettoyage
        t1 = time.time()
        df_clean = run_step1_cleaning()
        success(f"[{elapsed(t1)}]")

        # Étape 2 : Features
        t2 = time.time()
        df_feat = run_step2_features(df_clean)
        success(f"[{elapsed(t2)}]")

        # Étape 3 : SQL
        t3 = time.time()
        run_step3_sql(df_clean)
        success(f"[{elapsed(t3)}]")

        # Étape 4 : Modélisation
        results = None
        if not args.skip_modeling:
            t4 = time.time()
            results = run_step4_modeling(df_feat, tune=args.tune)
            success(f"[{elapsed(t4)}]")

            # Rapport final
            generate_final_report(results, start_time)
        else:
            banner("Étape ML ignorée (--skip-modeling)", char="─")

        # Résumé
        banner("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print(f"  Durée totale : {elapsed(start_time)}")
        print(f"\n  Livrables :")
        print(f"  {'data/processed/accidents_clean.parquet':<45} ✓")
        print(f"  {'data/processed/accidents_features.parquet':<45} ✓")
        print(f"  {'reports/tables/ (CSV analytiques)':<45} ✓")
        if results:
            print(f"  {'reports/models/ (modèles entraînés)':<45} ✓")
        print(f"\n  → Lancer le dashboard : cd dashboard && python app.py")
        print(f"  → Explorer les notebooks : jupyter lab")

        # Dashboard
        if args.dashboard:
            run_dashboard()

    except KeyboardInterrupt:
        print("\n\n  Pipeline interrompu par l'utilisateur.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ✗ Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
