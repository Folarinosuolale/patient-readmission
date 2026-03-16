"""
run_pipeline.py – 8-stage orchestrator for the Patient Readmission Prediction pipeline.

Usage:
    cd patient-readmission/
    python -m src.run_pipeline          # or  python src/run_pipeline.py
"""

import os, sys, json, time, joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import run_preprocessing
from src.feature_engineering import (
    group_diagnoses, encode_medications,
    create_derived_features, prepare_features,
)
from src.model_training import compare_models, tune_best_model, run_fairness_analysis, evaluate_model
from src.explainability import (
    compute_shap_values, plot_shap_summary,
    plot_shap_bar, plot_shap_waterfall,
)

DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


def main():
    t0 = time.time()
    _ensure_dirs()

    # ════════════════════════════════════════════════════════════
    # [1/8]  Load and preprocess
    # ════════════════════════════════════════════════════════════
    print("\n[1/8] Loading and preprocessing data...")
    df = run_preprocessing(DATA_DIR)

    # ════════════════════════════════════════════════════════════
    # [2/8]  Feature engineering
    # ════════════════════════════════════════════════════════════
    print("\n[2/8] Engineering features...")
    df = group_diagnoses(df)
    df = encode_medications(df)
    df = create_derived_features(df)

    # ════════════════════════════════════════════════════════════
    # [3/8]  Prepare features (encode, scale, split)
    # ════════════════════════════════════════════════════════════
    print("\n[3/8] Preparing features...")
    X_train, X_test, y_train, y_test, feature_names, artifacts = prepare_features(df)

    # Apply SMOTE here (outside CV) for the final model fit only.
    # CV folds apply their own SMOTE internally via ImbPipeline.
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"  SMOTE (final fit): {len(X_train):,} -> {len(X_train_res):,} training samples")

    # Save dataset summary for the dashboard
    dataset_summary = {
        "n_encounters": int(len(df)),
        "n_features": int(len(feature_names)),
        "positive_rate": float(df["readmitted_binary"].mean()),
        "feature_names": feature_names,
    }

    # ════════════════════════════════════════════════════════════
    # [4/8]  Compare base models
    # ════════════════════════════════════════════════════════════
    print("\n[4/8] Comparing models...")
    # scale_pos_weight derived from raw (unbalanced) y_train
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    comparison = compare_models(
        X_train, y_train, X_train_res, y_train_res,
        X_test, y_test,
        scale_pos_weight=scale_pos_weight,
    )

    # ════════════════════════════════════════════════════════════
    # [5/8]  Tune best model with Optuna
    # ════════════════════════════════════════════════════════════
    print("\n[5/8] Tuning best model...")
    tuned_model, tuned_metrics, study = tune_best_model(
        X_train, y_train, X_train_res, y_train_res,
        X_test, y_test,
        best_name=comparison["best_name"],
        n_trials=50,
        scale_pos_weight=scale_pos_weight,
    )

    # ════════════════════════════════════════════════════════════
    # [6/8]  SHAP explanations
    # ════════════════════════════════════════════════════════════
    print("\n[6/8] Computing SHAP explanations...")
    shap_dict = compute_shap_values(tuned_model, X_test, feature_names)
    plot_shap_summary(shap_dict, os.path.join(ASSETS_DIR, "shap_summary.png"))
    plot_shap_bar(shap_dict, os.path.join(ASSETS_DIR, "shap_bar.png"))
    plot_shap_waterfall(shap_dict, os.path.join(ASSETS_DIR, "shap_waterfall.png"))

    # ════════════════════════════════════════════════════════════
    # [7/8]  Fairness analysis
    # ════════════════════════════════════════════════════════════
    print("\n[7/8] Fairness analysis...")
    fairness_results = run_fairness_analysis(tuned_model, X_test, y_test, artifacts)

    # ════════════════════════════════════════════════════════════
    # [8/8]  Save artefacts
    # ════════════════════════════════════════════════════════════
    print("\n[8/8] Saving artifacts...")

    # Trained model
    joblib.dump(tuned_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    print(f"  Saved best_model.pkl")

    # Artifacts bundle (scaler, encoder, feature_names, etc.)
    save_artifacts = {
        "scaler": artifacts["scaler"],
        "encoder": artifacts["encoder"],
        "feature_names": artifacts["feature_names"],
    }
    joblib.dump(save_artifacts, os.path.join(MODELS_DIR, "artifacts.pkl"))
    print(f"  Saved artifacts.pkl")

    # SHAP dict (for dashboard)
    shap_save = {
        "shap_values": shap_dict["shap_values"],
        "X_sample": shap_dict["X_sample"],
        "feature_names": shap_dict["feature_names"],
        "importance_df": shap_dict["importance_df"],
        "expected_value": float(shap_dict["expected_value"])
                          if not isinstance(shap_dict["expected_value"], (list, np.ndarray))
                          else float(shap_dict["expected_value"]),
        "sample_idx": shap_dict["sample_idx"].tolist()
                      if isinstance(shap_dict["sample_idx"], np.ndarray)
                      else shap_dict["sample_idx"],
    }
    joblib.dump(shap_save, os.path.join(MODELS_DIR, "shap_dict.pkl"))
    print(f"  Saved shap_dict.pkl")

    # Model comparison CSV
    comp_rows = []
    for r in comparison["results"]:
        comp_rows.append({
            "model": r["model"],
            "cv_auc": r.get("cv_auc", ""),
            "cv_f1": r.get("cv_f1", ""),
            "cv_recall": r.get("cv_recall", ""),
            "test_auc": r["auc"],
            "test_f1": r["f1"],
            "test_recall": r["recall"],
            "test_precision": r["precision"],
            "test_specificity": r["specificity"],
            "test_accuracy": r["accuracy"],
        })
    # Append tuned model row
    comp_rows.append({
        "model": tuned_metrics["model"],
        "cv_auc": study.best_value,
        "cv_f1": "",
        "cv_recall": "",
        "test_auc": tuned_metrics["auc"],
        "test_f1": tuned_metrics["f1"],
        "test_recall": tuned_metrics["recall"],
        "test_precision": tuned_metrics["precision"],
        "test_specificity": tuned_metrics["specificity"],
        "test_accuracy": tuned_metrics["accuracy"],
    })
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    print(f"  Saved model_comparison.csv")

    # Feature importance CSV
    shap_dict["importance_df"].to_csv(
        os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
    print(f"  Saved feature_importance.csv")

    # ROC data JSON
    roc_data = comparison["roc_curves"]
    # Add tuned model ROC
    from sklearn.metrics import roc_curve, roc_auc_score
    y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]
    fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_tuned)
    roc_data.append({
        "model": tuned_metrics["model"],
        "fpr": fpr_t.tolist(), "tpr": tpr_t.tolist(),
        "auc": tuned_metrics["auc"],
    })
    with open(os.path.join(MODELS_DIR, "roc_data.json"), "w") as f:
        json.dump(roc_data, f, indent=2)
    print(f"  Saved roc_data.json")

    # Confusion matrices JSON
    cm_data = comparison["confusion_matrices"]
    cm_data[tuned_metrics["model"]] = tuned_metrics["confusion_matrix"]
    with open(os.path.join(MODELS_DIR, "confusion_matrices.json"), "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"  Saved confusion_matrices.json")

    # Pipeline results JSON (master summary)
    pipeline_results = {
        "dataset_summary": dataset_summary,
        "best_base_model": comparison["best_name"],
        "tuned_model_name": tuned_metrics["model"],
        "tuned_metrics": {
            k: v for k, v in tuned_metrics.items()
            if k not in ("y_pred", "y_prob", "confusion_matrix")
        },
        "optuna_best_value": study.best_value,
        "optuna_best_params": study.best_params,
        "fairness": {
            k: {
                "disparate_impact_ratio": v["disparate_impact_ratio"],
                "group_metrics": v["group_metrics"],
            }
            for k, v in fairness_results.items()
        },
    }
    with open(os.path.join(MODELS_DIR, "pipeline_results.json"), "w") as f:
        json.dump(pipeline_results, f, indent=2, default=str)
    print(f"  Saved pipeline_results.json")

    # Fairness results JSON
    with open(os.path.join(MODELS_DIR, "fairness_results.json"), "w") as f:
        json.dump(fairness_results, f, indent=2, default=str)
    print(f"  Saved fairness_results.json")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Best model:  {tuned_metrics['model']}")
    print(f"  Test AUC:    {tuned_metrics['auc']:.4f}")
    print(f"  Test Recall: {tuned_metrics['recall']:.4f}")
    print(f"  Test F1:     {tuned_metrics['f1']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
