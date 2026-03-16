"""
model_training.py – Train, compare, tune, and evaluate models for 30-day readmission.

Provides:
  • 4 base models (LogReg, RandomForest, XGBoost, LightGBM)
  • 5-fold stratified cross-validation comparison
  • Optuna hyper-parameter tuning for the best base model
  • Fairness analysis  (disparate impact, equalized odds by race & age)
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, accuracy_score,
)
import xgboost as xgb
import lightgbm as lgb
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# Base model catalogue

def get_base_models(scale_pos_weight: float = 8.0) -> dict:
    """Return a dict of named baseline models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs",
            random_state=42, n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
        ),
    }



# Evaluation helpers

def evaluate_model(model, X, y, name: str) -> dict:
    """Compute full evaluation metrics on a hold-out set."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        "auc": roc_auc_score(y, y_prob),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def _roc_data(y_true, y_prob, name: str) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"model": name, "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": roc_auc_score(y_true, y_prob)}



# Model comparison (5-fold CV + final test evaluation)

def compare_models(X_train, y_train, X_train_res, y_train_res,
                   X_test, y_test,
                   scale_pos_weight: float = 8.0) -> dict:
    """
    Train 4 models with 5-fold CV (SMOTE inside each fold via ImbPipeline),
    then fit the final model on the pre-resampled data and evaluate on the test set.

    Parameters
    ----------
    X_train / y_train     : raw (unbalanced) scaled training data used for CV
    X_train_res / y_train_res : SMOTE-resampled data used for final model fit
    """
    models = get_base_models(scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    roc_curves = []
    confusion_matrices = {}

    for name, model in models.items():
        print(f"    Training {name}...")
        # Cross-validation with SMOTE applied inside each fold only
        cv_pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", model),
        ])
        cv_results = cross_validate(
            cv_pipeline, X_train, y_train, cv=cv,
            scoring=["roc_auc", "f1", "recall", "precision"],
            return_train_score=False, n_jobs=-1,
        )
        cv_auc = cv_results["test_roc_auc"].mean()
        cv_f1 = cv_results["test_f1"].mean()
        cv_recall = cv_results["test_recall"].mean()

        # Fit on full resampled training set
        model.fit(X_train_res, y_train_res)

        # Test evaluation
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics["cv_auc"] = cv_auc
        metrics["cv_f1"] = cv_f1
        metrics["cv_recall"] = cv_recall

        results.append(metrics)
        roc_curves.append(_roc_data(y_test, metrics["y_prob"], name))
        confusion_matrices[name] = metrics["confusion_matrix"]

        print(f"      CV AUC={cv_auc:.4f}  |  Test AUC={metrics['auc']:.4f}  "
              f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

    # Pick best model by test AUC
    best_idx = int(np.argmax([r["auc"] for r in results]))
    best_name = results[best_idx]["model"]
    best_model = models[best_name]
    print(f"\n  ★ Best base model: {best_name}  (AUC={results[best_idx]['auc']:.4f})")

    return {
        "results": results,
        "roc_curves": roc_curves,
        "confusion_matrices": confusion_matrices,
        "best_name": best_name,
        "best_model": best_model,
        "models": models,
    }



# Optuna hyper-parameter tuning

def tune_best_model(X_train, y_train, X_train_res, y_train_res,
                    X_test, y_test,
                    best_name: str, n_trials: int = 50,
                    scale_pos_weight: float = 8.0):
    """
    Tune the selected best model with Optuna and return (tuned_model, study).

    CV uses ImbPipeline so SMOTE only touches training folds.
    Final model is fit on the pre-resampled X_train_res / y_train_res.
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        if best_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "logloss", "random_state": 42,
                "n_jobs": -1, "verbosity": 0,
            }
            model = xgb.XGBClassifier(**params)

        elif best_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "is_unbalance": True, "random_state": 42,
                "n_jobs": -1, "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)

        elif best_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        else:  # Logistic Regression
            params = {
                "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                "max_iter": 2000, "class_weight": "balanced",
                "solver": "lbfgs", "random_state": 42, "n_jobs": -1,
            }
            model = LogisticRegression(**params)

        # CV with SMOTE inside each fold
        cv_pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", model),
        ])
        scores = cross_validate(cv_pipeline, X_train, y_train, cv=cv,
                                scoring="roc_auc", n_jobs=-1)
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Optuna best CV AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Refit with best params
    best_params = study.best_params
    if best_name == "XGBoost":
        tuned_model = xgb.XGBClassifier(
            **best_params, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )
    elif best_name == "LightGBM":
        tuned_model = lgb.LGBMClassifier(
            **best_params, is_unbalance=True, random_state=42,
            n_jobs=-1, verbose=-1,
        )
    elif best_name == "Random Forest":
        tuned_model = RandomForestClassifier(
            **best_params, class_weight="balanced", random_state=42, n_jobs=-1,
        )
    else:
        tuned_model = LogisticRegression(
            **best_params, max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=42, n_jobs=-1,
        )

    tuned_model.fit(X_train_res, y_train_res)
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test, f"{best_name} (Tuned)")
    print(f"  Tuned test AUC={tuned_metrics['auc']:.4f}  "
          f"Recall={tuned_metrics['recall']:.4f}  F1={tuned_metrics['f1']:.4f}")

    return tuned_model, tuned_metrics, study



# Fairness analysis

def compute_fairness_metrics(y_true, y_pred, y_prob,
                             sensitive_feature, group_name: str) -> dict:
    """
    Compute per-group metrics for a sensitive attribute (race, age, etc.).
    Returns a dict keyed by group value with standard metrics.
    """
    groups = np.unique(sensitive_feature)
    group_metrics = {}

    for g in groups:
        mask = sensitive_feature == g
        n = mask.sum()
        if n < 30:
            continue
        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_pr = y_prob[mask]

        try:
            auc = roc_auc_score(y_t, y_pr)
        except ValueError:
            auc = float("nan")

        group_metrics[str(g)] = {
            "n": int(n),
            "prevalence": float(y_t.mean()),
            "predicted_positive_rate": float(y_p.mean()),
            "auc": float(auc),
            "recall": float(recall_score(y_t, y_p, zero_division=0)),
            "precision": float(precision_score(y_t, y_p, zero_division=0)),
            "f1": float(f1_score(y_t, y_p, zero_division=0)),
        }

    # Disparate impact ratio (min PPR / max PPR)
    pprs = [m["predicted_positive_rate"] for m in group_metrics.values() if m["n"] >= 30]
    di_ratio = min(pprs) / max(pprs) if pprs and max(pprs) > 0 else float("nan")

    return {
        "group_name": group_name,
        "group_metrics": group_metrics,
        "disparate_impact_ratio": float(di_ratio),
    }


def run_fairness_analysis(model, X_test, y_test, artifacts: dict) -> dict:
    """Run fairness analysis by race and age group."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {}

    # By race
    race = artifacts.get("race_test")
    if race is not None:
        results["race"] = compute_fairness_metrics(y_test, y_pred, y_prob, race, "Race")
        print(f"  Race fairness: DI ratio = {results['race']['disparate_impact_ratio']:.3f}")

    # By age group (bin age_numeric into groups)
    age = artifacts.get("age_test")
    if age is not None:
        age_bins = pd.cut(age, bins=[0, 40, 60, 75, 100],
                          labels=["<40", "40-60", "60-75", "75+"])
        results["age"] = compute_fairness_metrics(
            y_test, y_pred, y_prob, np.array(age_bins.astype(str)), "Age Group"
        )
        print(f"  Age fairness:  DI ratio = {results['age']['disparate_impact_ratio']:.3f}")

    return results
