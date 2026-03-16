"""
tune_threshold.py: Find optimal decision threshold from saved pipeline artifacts.

Regenerates X_test/y_test by replaying the data + feature-engineering steps
(no SMOTE, no training), then runs predict_proba on the saved model.

Uses Youden's J statistic (max sensitivity + specificity) to pick the threshold.
Updates pipeline_results.json, confusion_matrices.json, and saves threshold.json.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_curve, confusion_matrix,
    f1_score, recall_score, precision_score, roc_auc_score
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"


def regenerate_test_set():
    """Replay data loading + feature engineering to get X_test, y_test."""
    from src.data_loader import run_preprocessing
    from src.feature_engineering import (
        group_diagnoses, encode_medications,
        create_derived_features, prepare_features,
    )

    print("  Re-running data loader...")
    df = run_preprocessing(data_dir=str(DATA_DIR))

    print("  Re-running feature engineering (no SMOTE)...")
    df = group_diagnoses(df)
    df = encode_medications(df)
    df = create_derived_features(df)

    # Returns: X_train, X_test, y_train, y_test, feature_names, artifacts
    _, X_test, _, y_test, _, _ = prepare_features(df, apply_smote=False)

    print(f"  Test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    return X_test, y_test


def find_youden_threshold(y_true, y_prob):
    """Return threshold maximising Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def main():
    print("Loading saved model...")
    model = joblib.load(MODELS_DIR / "best_model.pkl")

    print("Regenerating test set...")
    X_test, y_test = regenerate_test_set()

    print("Computing probabilities...")
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    # ── Default threshold baseline ────────────────────────────────────────────
    y_default = (y_prob >= 0.5).astype(int)
    print(f"\n  Default threshold (0.50):")
    print(f"    AUC    = {auc:.4f}")
    print(f"    Recall = {recall_score(y_test, y_default, zero_division=0):.4f}")
    print(f"    F1     = {f1_score(y_test, y_default, zero_division=0):.4f}")

    # ── Youden's J threshold ──────────────────────────────────────────────────
    chosen_t = find_youden_threshold(y_test, y_prob)
    y_final  = (y_prob >= chosen_t).astype(int)

    recall    = recall_score(y_test, y_final, zero_division=0)
    precision = precision_score(y_test, y_final, zero_division=0)
    f1        = f1_score(y_test, y_final, zero_division=0)

    cm = confusion_matrix(y_test, y_final)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"\n  ★ Youden's J threshold ({chosen_t:.4f}):")
    print(f"    AUC         = {auc:.4f}")
    print(f"    Recall      = {recall:.4f}")
    print(f"    Precision   = {precision:.4f}")
    print(f"    F1          = {f1:.4f}")
    print(f"    Specificity = {specificity:.4f}")
    print(f"\n    Confusion Matrix:")
    print(f"      TN={tn}  FP={fp}")
    print(f"      FN={fn}  TP={tp}")

    # ── Update pipeline_results.json ─────────────────────────────────────────
    results_path = MODELS_DIR / "pipeline_results.json"
    results = json.loads(results_path.read_text())

    results["optimal_threshold"] = round(chosen_t, 4)
    results["tuned_metrics_at_optimal_threshold"] = {
        "threshold":   round(chosen_t, 4),
        "auc":         round(auc, 4),
        "recall":      round(recall, 4),
        "precision":   round(precision, 4),
        "f1":          round(f1, 4),
        "specificity": round(specificity, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Updated pipeline_results.json ✓")

    # ── Update confusion_matrices.json ────────────────────────────────────────
    cm_path = MODELS_DIR / "confusion_matrices.json"
    cm_data = json.loads(cm_path.read_text())
    cm_data["optimal_threshold"] = {
        "threshold": round(chosen_t, 4),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }
    cm_path.write_text(json.dumps(cm_data, indent=2))
    print(f"  Updated confusion_matrices.json ✓")

    # ── Save standalone threshold.json ────────────────────────────────────────
    threshold_path = MODELS_DIR / "threshold.json"
    threshold_path.write_text(json.dumps({"optimal_threshold": round(chosen_t, 4)}, indent=2))
    print(f"  Saved threshold.json ✓")

    print("\n  Done.")


if __name__ == "__main__":
    main()
