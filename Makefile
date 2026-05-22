# ============================================================
# Makefile — Togo Road Accidents Capstone
# Usage : make <target>
# ============================================================

PYTHON   = python
PIP      = pip
JUPYTER  = jupyter
VENV     = venv
SRC      = src
DASH_DIR = dashboard

.PHONY: all install clean pipeline pipeline-fast dashboard notebooks help

# ── Aide ────────────────────────────────────────────────────
help:
	@echo "Togo Road Accidents — Commandes disponibles :"
	@echo ""
	@echo "  make install        Installer les dépendances"
	@echo "  make pipeline       Exécuter le pipeline complet (avec ML)"
	@echo "  make pipeline-fast  Pipeline sans modélisation ML"
	@echo "  make pipeline-tune  Pipeline avec Optuna tuning"
	@echo "  make dashboard      Lancer le dashboard Dash"
	@echo "  make notebooks      Lancer JupyterLab"
	@echo "  make clean          Nettoyer les fichiers générés"
	@echo "  make test           Vérifier l'intégrité du projet"
	@echo ""

# ── Installation ─────────────────────────────────────────────
install:
	@echo "→ Installation des dépendances..."
	$(PIP) install -r requirements.txt
	@echo "✓ Installation terminée"

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Environnement virtuel créé"
	@echo "  Activer avec : source $(VENV)/bin/activate"

# ── Pipeline ─────────────────────────────────────────────────
pipeline:
	@echo "→ Lancement du pipeline complet..."
	$(PYTHON) run_pipeline.py

pipeline-fast:
	@echo "→ Lancement du pipeline (sans ML)..."
	$(PYTHON) run_pipeline.py --skip-modeling

pipeline-tune:
	@echo "→ Pipeline avec Optuna hyperparameter tuning..."
	$(PYTHON) run_pipeline.py --tune

# ── Étapes individuelles ─────────────────────────────────────
clean-data:
	$(PYTHON) $(SRC)/data_preprocessing.py

features:
	$(PYTHON) $(SRC)/feature_engineering.py

modeling:
	$(PYTHON) $(SRC)/modeling.py

sql:
	$(PYTHON) $(SRC)/sql_queries.py

# ── Dashboard ────────────────────────────────────────────────
dashboard:
	@echo "→ Lancement du dashboard..."
	@echo "   http://localhost:8050"
	cd $(DASH_DIR) && $(PYTHON) app.py

# ── Notebooks ────────────────────────────────────────────────
notebooks:
	$(JUPYTER) lab --notebook-dir=notebooks

# ── Tests & Vérifications ────────────────────────────────────
test:
	@echo "→ Vérification de l'intégrité du projet..."
	@$(PYTHON) -c "import pandas; print('  ✓ pandas', pandas.__version__)"
	@$(PYTHON) -c "import numpy; print('  ✓ numpy', numpy.__version__)"
	@$(PYTHON) -c "import sklearn; print('  ✓ scikit-learn', sklearn.__version__)"
	@$(PYTHON) -c "import xgboost; print('  ✓ xgboost', xgboost.__version__)"
	@$(PYTHON) -c "import lightgbm; print('  ✓ lightgbm', lightgbm.__version__)"
	@$(PYTHON) -c "import plotly; print('  ✓ plotly', plotly.__version__)"
	@$(PYTHON) -c "import dash; print('  ✓ dash', dash.__version__)"
	@$(PYTHON) -c "import folium; print('  ✓ folium', folium.__version__)"
	@$(PYTHON) -c "import duckdb; print('  ✓ duckdb', duckdb.__version__)"
	@$(PYTHON) -c "import shap; print('  ✓ shap', shap.__version__)"
	@$(PYTHON) -c "import optuna; print('  ✓ optuna', optuna.__version__)"
	@echo "  ✓ Toutes les dépendances sont présentes"

# ── Nettoyage ────────────────────────────────────────────────
clean:
	@echo "→ Nettoyage des fichiers générés..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "  ✓ Fichiers temporaires supprimés"

clean-all: clean
	@echo "→ Suppression des données traitées et modèles..."
	rm -f data/processed/*.parquet
	rm -f data/processed/*.pkl
	rm -f reports/models/*.pkl
	rm -f reports/models/*.joblib
	rm -f reports/figures/*.png
	rm -f reports/tables/*.csv
	rm -f reports/tables/*.json
	rm -f reports/*.html
	@echo "  ✓ Données et modèles supprimés"

# ── Rapports ────────────────────────────────────────────────
report:
	@echo "→ Génération du rapport final..."
	$(PYTHON) -c "from src.sql_queries import *; from src.utils import *; \
	    df = load_dataframe('data/processed/accidents_clean.parquet'); \
	    export_sql_report(df, Path('reports/tables'))"
	@echo "✓ Rapport SQL exporté"
