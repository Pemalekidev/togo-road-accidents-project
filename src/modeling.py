"""
modeling.py
===========
Pipeline complet de modélisation prédictive pour les accidents de la route au Togo.

Tâche : Classification multiclasse de la gravité (Léger / Grave / Mortel)
Modèles : RandomForest, XGBoost, LightGBM, LogisticRegression
Pipeline : Preprocessing → Hyperparameter Tuning (Optuna) → Calibration → SHAP
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    learning_curve
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, balanced_accuracy_score, f1_score,
    roc_curve, auc
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Boosting
import xgboost as xgb
import lightgbm as lgb

# Optimisation
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# SHAP
import shap

from utils import (
    get_logger, save_json, FEAT_DATA_FILE, PATHS, Timer, GRAVITE_ORDER
)

logger = get_logger("modeling")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TARGET = "gravite_num"
CLASSES = ["Léger", "Grave", "Mortel"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MODELS_DIR = PATHS["reports"] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRÉPARATION DES FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Retourne les listes de features numériques et catégorielles."""
    # Colonnes à exclure (identifiants, cibles, dates)
    exclude = {
        "accident_id", "date_accident", "gravite", "gravite_num",
        "nombre_deces",  # Fuite : le nombre de décès détermine la gravité
        "cout_estime",   # Fuite potentielle (corrélé à la gravité)
        "lat_str", "lon_str",  # colonnes intermédiaires
    }

    # Features numériques
    num_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    # Features catégorielles (object / category)
    cat_features = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in exclude
        and c not in ["axe_routier", "ville"]  # Trop de modalités → risque d'overfit
    ]

    logger.info(f"  Features numériques    : {len(num_features)}")
    logger.info(f"  Features catégorielles : {len(cat_features)}")
    return num_features, cat_features


def build_preprocessor(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    """
    Construire un préprocesseur sklearn ColumnTransformer :
    - Numériques : StandardScaler
    - Catégorielles : OneHotEncoder (drop first pour éviter multicolinéarité)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple:
    """
    Prépare X, y et sépare train/test avec stratification.
    """
    num_features, cat_features = get_feature_lists(df)
    X = df[num_features + cat_features].copy()
    y = df[TARGET].values

    # Vérifications
    assert y.min() >= 0 and y.max() <= 2, "Cible hors plage [0, 2]"
    assert X.isnull().sum().sum() == 0, f"NaN dans X : {X.isnull().sum()[X.isnull().sum()>0]}"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")
    logger.info(f"  Distribution y_train : {np.bincount(y_train)}")
    logger.info(f"  Distribution y_test  : {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test, num_features, cat_features


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUCTION DES PIPELINES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def build_logistic_regression(preprocessor: ColumnTransformer) -> Pipeline:
    model = LogisticRegression(
        max_iter=1000, multi_class="multinomial",
        class_weight="balanced", random_state=RANDOM_STATE
    )
    return Pipeline([("prep", preprocessor), ("clf", model)])


def build_random_forest(preprocessor: ColumnTransformer, **kwargs) -> Pipeline:
    defaults = dict(n_estimators=300, max_depth=None, min_samples_leaf=5,
                    class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    defaults.update(kwargs)
    model = RandomForestClassifier(**defaults)
    return Pipeline([("prep", preprocessor), ("clf", model)])


def build_xgboost(preprocessor: ColumnTransformer, **kwargs) -> Pipeline:
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]),
                                          y=np.array([0, 1, 2]))
    defaults = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RANDOM_STATE,
        n_jobs=-1, use_label_encoder=False
    )
    defaults.update(kwargs)
    model = xgb.XGBClassifier(**defaults)
    return Pipeline([("prep", preprocessor), ("clf", model)])


def build_lightgbm(preprocessor: ColumnTransformer, **kwargs) -> Pipeline:
    defaults = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    defaults.update(kwargs)
    model = lgb.LGBMClassifier(**defaults)
    return Pipeline([("prep", preprocessor), ("clf", model)])


# ─────────────────────────────────────────────────────────────────────────────
# 3. ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Évaluation complète d'un modèle :
    - Classification report
    - Matrice de confusion
    - AUC-OvR
    - Balanced Accuracy
    - F1-macro
    - Cross-validation
    """
    logger.info(f"\n{'─'*50}")
    logger.info(f"Évaluation : {model_name}")
    logger.info(f"{'─'*50}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Métriques
    f1_macro   = f1_score(y_test, y_pred, average="macro")
    bal_acc    = balanced_accuracy_score(y_test, y_pred)
    auc_ovr    = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    logger.info(f"  F1-macro         : {f1_macro:.4f}")
    logger.info(f"  Balanced Accuracy: {bal_acc:.4f}")
    logger.info(f"  AUC-OvR (macro)  : {auc_ovr:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=CLASSES)}")

    # Cross-validation (sur train uniquement)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring={"f1_macro": "f1_macro", "bal_acc": "balanced_accuracy"},
        n_jobs=-1, return_train_score=False
    )
    cv_f1_mean = cv_scores["test_f1_macro"].mean()
    cv_f1_std  = cv_scores["test_f1_macro"].std()
    logger.info(f"  CV F1-macro      : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    # ROC par classe
    roc_data = {}
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc(fpr, tpr)}

    return {
        "model_name":    model_name,
        "pipeline":      pipeline,
        "y_pred":        y_pred,
        "y_proba":       y_proba,
        "f1_macro":      f1_macro,
        "balanced_acc":  bal_acc,
        "auc_ovr":       auc_ovr,
        "cv_f1_mean":    cv_f1_mean,
        "cv_f1_std":     cv_f1_std,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_data":      roc_data,
        "report":        classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING AVEC OPTUNA
# ─────────────────────────────────────────────────────────────────────────────

def tune_lightgbm(
    X_train: pd.DataFrame, y_train: np.ndarray,
    preprocessor: ColumnTransformer,
    n_trials: int = 50
) -> Dict:
    """
    Optimisation des hyperparamètres LightGBM avec Optuna.
    Retourne les meilleurs hyperparamètres.
    """
    logger.info(f"\nOptimisation LightGBM avec Optuna ({n_trials} trials)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":     trial.suggest_int("num_leaves", 20, 150),
            "max_depth":      trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight":   "balanced",
            "random_state":   RANDOM_STATE,
            "verbose":        -1,
            "n_jobs":         -1,
        }
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", lgb.LGBMClassifier(**params))
        ])
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring="f1_macro", n_jobs=1
        )
        return scores["test_score"].mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"  Meilleurs params : {best}")
    logger.info(f"  Meilleur F1-macro : {study.best_value:.4f}")
    return best


def tune_xgboost(
    X_train: pd.DataFrame, y_train: np.ndarray,
    preprocessor: ColumnTransformer,
    n_trials: int = 50
) -> Dict:
    """Optimisation XGBoost avec Optuna."""
    logger.info(f"\nOptimisation XGBoost avec Optuna ({n_trials} trials)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "eval_metric":       "mlogloss",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
        }
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", xgb.XGBClassifier(**params))
        ])
        scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring="f1_macro", n_jobs=1)
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"  Meilleur F1-macro XGBoost : {study.best_value:.4f}")
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_type: str = "tree"
) -> Tuple[np.ndarray, List[str]]:
    """
    Calcule les valeurs SHAP pour un modèle.
    Retourne (shap_values, feature_names).
    """
    logger.info("Calcul des valeurs SHAP...")

    # Transformer X avec le préprocesseur
    prep = pipeline.named_steps["prep"]
    clf  = pipeline.named_steps["clf"]

    X_train_t = prep.transform(X_train)
    X_test_t  = prep.transform(X_test)

    # Noms des features après transformation
    try:
        feature_names = prep.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    # Explainer selon le type de modèle
    # ✅ Nouveau code — compatible XGBoost 3.x / LightGBM / RF
    if model_type == "tree":
        try:
            # shap.Explainer (API moderne) gère automatiquement XGBoost 3.x
            explainer        = shap.Explainer(clf, X_test_t[:500])
            shap_values_obj  = explainer(X_test_t[:500])
            shap_values      = shap_values_obj.values          # (n_samples, n_features, n_classes)
        except Exception:
            # Fallback KernelExplainer si shap.Explainer échoue aussi
            background   = shap.sample(X_test_t, 100)
            explainer    = shap.KernelExplainer(clf.predict_proba, background)
            shap_values  = explainer.shap_values(X_test_t[:200])
            shap_values  = np.stack(shap_values, axis=-1)      # liste → array 3D
    else:
        explainer = shap.LinearExplainer(clf, X_train_t)
        shap_values = explainer.shap_values(X_test_t[:500])

    logger.info("  Valeurs SHAP calculées ✓")
    return shap_values, feature_names, X_test_t[:500]


def get_shap_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.Series:
    """
    Importance SHAP globale (mean |SHAP|) toutes classes confondues.
    """
    if isinstance(shap_values, list):
        # Multiclasse : liste de matrices
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values)

    importance = pd.Series(
        mean_abs.mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)
    return importance


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARAISON DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(results: List[Dict]) -> pd.DataFrame:
    """
    Tableau comparatif des modèles.
    """
    rows = []
    for r in results:
        rows.append({
            "Modèle":           r["model_name"],
            "F1-macro (test)":  f"{r['f1_macro']:.4f}",
            "Balanced Acc.":    f"{r['balanced_acc']:.4f}",
            "AUC-OvR":          f"{r['auc_ovr']:.4f}",
            "CV F1 (mean±std)": f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}",
        })
    df_cmp = pd.DataFrame(rows).sort_values("F1-macro (test)", ascending=False)
    return df_cmp


# ─────────────────────────────────────────────────────────────────────────────
# 7. PIPELINE COMPLET D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_modeling_pipeline(
    df: pd.DataFrame,
    tune: bool = False,
    n_trials: int = 30,
    save_models: bool = True
) -> Dict:
    """
    Pipeline complet d'entraînement et d'évaluation.
    
    Parameters
    ----------
    df : DataFrame avec features engineerées
    tune : Si True, optimise les hyperparamètres avec Optuna
    n_trials : Nombre de trials Optuna
    save_models : Si True, sauvegarde les modèles entraînés
    
    Returns
    -------
    dict : résultats de tous les modèles + meilleur modèle
    """
    with Timer("Modeling Pipeline"):
        logger.info("═" * 60)
        logger.info("DÉMARRAGE DU PIPELINE DE MODÉLISATION")
        logger.info("═" * 60)

        # Préparation
        logger.info("[1/5] Préparation des données...")
        X_train, X_test, y_train, y_test, num_features, cat_features = prepare_data(df)
        preprocessor = build_preprocessor(num_features, cat_features)

        # Modèles
        logger.info("[2/5] Entraînement des modèles de base...")
        all_results = []

        # Logistic Regression (baseline)
        lr_pipe = build_logistic_regression(build_preprocessor(num_features, cat_features))
        lr_result = evaluate_model(lr_pipe, X_train, X_test, y_train, y_test, "Logistic Regression")
        all_results.append(lr_result)

        # Random Forest
        rf_pipe = build_random_forest(build_preprocessor(num_features, cat_features))
        rf_result = evaluate_model(rf_pipe, X_train, X_test, y_train, y_test, "Random Forest")
        all_results.append(rf_result)

        # XGBoost
        if tune:
            logger.info("[3/5] Optimisation XGBoost (Optuna)...")
            xgb_params = tune_xgboost(X_train, y_train,
                                       build_preprocessor(num_features, cat_features), n_trials)
        else:
            xgb_params = {}
        xgb_pipe = build_xgboost(build_preprocessor(num_features, cat_features), **xgb_params)
        xgb_result = evaluate_model(xgb_pipe, X_train, X_test, y_train, y_test, "XGBoost")
        all_results.append(xgb_result)

        # LightGBM
        if tune:
            logger.info("[4/5] Optimisation LightGBM (Optuna)...")
            lgbm_params = tune_lightgbm(X_train, y_train,
                                         build_preprocessor(num_features, cat_features), n_trials)
        else:
            lgbm_params = {}
        lgbm_pipe = build_lightgbm(build_preprocessor(num_features, cat_features), **lgbm_params)
        lgbm_result = evaluate_model(lgbm_pipe, X_train, X_test, y_train, y_test, "LightGBM")
        all_results.append(lgbm_result)

        # Comparaison
        logger.info("\n" + "═" * 60)
        logger.info("COMPARAISON DES MODÈLES")
        logger.info("═" * 60)
        comparison = compare_models(all_results)
        print(comparison.to_string(index=False))

        # Meilleur modèle
        best_result = max(all_results, key=lambda r: r["f1_macro"])
        logger.info(f"\n✓ Meilleur modèle : {best_result['model_name']} (F1={best_result['f1_macro']:.4f})")

        # SHAP
        logger.info("\n[5/5] Calcul SHAP pour le meilleur modèle...")
        model_type = "tree" if "LightGBM" in best_result["model_name"] or "XGBoost" in best_result["model_name"] or "Forest" in best_result["model_name"] else "linear"
        try:
            shap_values, feat_names, X_test_t = compute_shap_values(
                best_result["pipeline"], X_train, X_test, model_type
            )
            shap_importance = get_shap_importance(shap_values, feat_names)
        except Exception as e:
            logger.warning(f"  SHAP échoué : {e}")
            shap_values, feat_names, shap_importance = None, None, None

        # Sauvegarde
        if save_models:
            for result in all_results:
                path = MODELS_DIR / f"{result['model_name'].replace(' ', '_').lower()}.pkl"
                joblib.dump(result["pipeline"], path)
                logger.info(f"  Modèle sauvegardé : {path.name}")

            # Métriques JSON
            metrics_export = [
                {k: v for k, v in r.items() if k not in ["pipeline", "y_pred", "y_proba", "confusion_matrix", "roc_data"]}
                for r in all_results
            ]
            save_json(metrics_export, MODELS_DIR / "metrics.json")

        logger.info("═" * 60)
        logger.info("PIPELINE DE MODÉLISATION TERMINÉ")
        logger.info("═" * 60)

        return {
            "all_results":    all_results,
            "best_result":    best_result,
            "comparison":     comparison,
            "shap_values":    shap_values,
            "shap_importance": shap_importance,
            "feature_names":  feat_names,
            "X_train":        X_train,
            "X_test":         X_test,
            "y_train":        y_train,
            "y_test":         y_test,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. PRÉDICTION (INFÉRENCE)
# ─────────────────────────────────────────────────────────────────────────────

def predict_accident_severity(
    model_path: Path,
    accident_data: Dict
) -> Dict:
    """
    Prédit la gravité d'un accident à partir de ses caractéristiques.
    
    Parameters
    ----------
    model_path : chemin vers le modèle sauvegardé (.pkl)
    accident_data : dict avec les features de l'accident
    
    Returns
    -------
    dict : {gravite_predite, probabilites, classe_num}
    """
    pipeline = joblib.load(model_path)
    df_input = pd.DataFrame([accident_data])

    # Probabilités et classe prédite
    proba = pipeline.predict_proba(df_input)[0]
    classe = pipeline.predict(df_input)[0]
    gravite = CLASSES[int(classe)]

    return {
        "gravite_predite":   gravite,
        "classe_num":        int(classe),
        "probabilites": {
            "Léger":  round(proba[0], 3),
            "Grave":  round(proba[1], 3),
            "Mortel": round(proba[2], 3),
        }
    }


if __name__ == "__main__":
    from utils import load_dataframe, FEAT_DATA_FILE
    df = load_dataframe(FEAT_DATA_FILE)
    results = run_modeling_pipeline(df, tune=False, save_models=True)
    print("\nTerminé.")
