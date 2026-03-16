"""
explainability.py – SHAP-based model explanations for the readmission predictor.

Generates:
  • Global feature importance (bar chart)
  • SHAP beeswarm summary plot
  • Single-prediction waterfall explanation
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────
# SHAP computation
# ──────────────────────────────────────────────────────────────

def compute_shap_values(model, X_test, feature_names: list,
                        max_samples: int = 1000) -> dict:
    """
    Compute SHAP values for a trained model.
    Uses TreeExplainer for tree models, LinearExplainer for logistic regression.
    """
    # Subsample for speed
    if len(X_test) > max_samples:
        idx = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx] if isinstance(X_test, np.ndarray) else X_test.iloc[idx]
    else:
        X_sample = X_test
        idx = np.arange(len(X_test))

    model_type = type(model).__name__
    print(f"  Model type: {model_type}, computing SHAP on {len(X_sample):,} samples...")

    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                       "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        # LinearExplainer for logistic regression
        if isinstance(X_sample, np.ndarray):
            bg = X_sample[:100]
        else:
            bg = X_sample.values[:100]
        explainer = shap.LinearExplainer(model, bg)

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, some explainers return a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # Build feature importance ranking
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return {
        "shap_values": shap_values,
        "X_sample": X_sample,
        "feature_names": feature_names,
        "importance_df": importance_df,
        "expected_value": (explainer.expected_value[1]
                           if isinstance(explainer.expected_value, (list, np.ndarray))
                              and len(explainer.expected_value) > 1
                           else explainer.expected_value),
        "sample_idx": idx,
    }


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────

def plot_shap_summary(shap_dict: dict, save_path: str) -> None:
    """Beeswarm summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_dict["shap_values"],
        features=shap_dict["X_sample"],
        feature_names=shap_dict["feature_names"],
        show=False, max_display=15,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary → {save_path}")


def plot_shap_bar(shap_dict: dict, save_path: str) -> None:
    """Bar plot of mean |SHAP| values (top 15 features)."""
    df = shap_dict["importance_df"].head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][::-1], df["mean_abs_shap"][::-1], color="#06B6D4")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 15 Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP bar → {save_path}")


def plot_shap_waterfall(shap_dict: dict, save_path: str, sample_index: int = 0) -> None:
    """Waterfall plot for a single prediction."""
    sv = shap_dict["shap_values"][sample_index]
    ev = shap_dict["expected_value"]

    explanation = shap.Explanation(
        values=sv,
        base_values=ev if isinstance(ev, (int, float, np.floating)) else float(ev),
        feature_names=shap_dict["feature_names"],
    )
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP waterfall → {save_path}")


def get_feature_importance_df(shap_dict: dict) -> pd.DataFrame:
    """Return the feature importance DataFrame."""
    return shap_dict["importance_df"]
