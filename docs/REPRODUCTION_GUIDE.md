# Reproduction Guide: Patient Readmission Prediction

**Author:** Folarin Osuolale
**Date:** March 2026

This guide documents every step required to reproduce all results in this project from a clean environment.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| RAM | 8 GB minimum (16 GB recommended) |
| Disk space | ~500 MB (data + models) |
| OS | Linux / macOS / Windows (WSL2 recommended) |

---

## 1. Clone the Repository

```bash
git clone https://github.com/Folarinosuolale/patient-readmission.git
cd patient-readmission
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Purpose |
|---|---|
| lightgbm | Best model |
| xgboost | Comparison model |
| scikit-learn | Preprocessing, metrics, cross-validation |
| imbalanced-learn | SMOTE oversampling |
| optuna | Hyperparameter tuning (50 trials) |
| shap | TreeExplainer for feature importance |
| pandas, numpy | Data processing |
| matplotlib, seaborn | Static plots |
| streamlit, plotly | Interactive dashboard |
| joblib | Model serialisation |

---

## 4. Download the Dataset

The dataset is **not included** in the repository due to size. Download it manually:

1. Go to: [UCI ML Repository: Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
2. Download the zip file and extract it
3. Place the following two files in `data/`:

```
patient-readmission/
└── data/
    ├── diabetic_data.csv       # 101,766 rows × 50 columns
    └── IDS_mapping.csv         # Admission/discharge/readmission code mappings
```

---

## 5. Run the ML Pipeline

From the project root directory:

```bash
python -m src.run_pipeline
```

This executes all 8 pipeline stages in sequence:

| Stage | Module | Description |
|---|---|---|
| 1 | `data_loader.py` | Load raw CSV, apply IDS mapping, drop excluded rows |
| 2 | `feature_engineering.py` → `group_diagnoses` | Map ICD-9 codes to 20 clinical categories |
| 3 | `feature_engineering.py` → `encode_medications` | Compress 23 medication columns into 4 summary features |
| 4 | `feature_engineering.py` → `create_derived_features` | Engineer 11 clinical composite features |
| 5 | `feature_engineering.py` → `prepare_features` | Encode, scale, train/test split (no SMOTE here) |
| 6 | `model_training.py` → `compare_models` | 5-fold stratified CV on 4 base models |
| 7 | `model_training.py` → `tune_model` | Optuna 50-trial hyperparameter search (LightGBM) |
| 8 | `model_training.py` → `explain_model` | SHAP values + fairness analysis + save all artifacts |

**Expected runtime:** 15–30 minutes (depending on CPU and SMOTE size)

**Expected output files in `models/`:**

```
models/
├── best_model.pkl              # Trained LightGBM classifier (joblib)
├── artifacts.pkl               # Scaler, encoder, feature names (joblib)
├── shap_dict.pkl               # SHAP values for 1,000 test samples
├── model_comparison.csv        # CV results for all 4 models
├── feature_importance.csv      # LightGBM native feature importances
├── roc_data.json               # FPR/TPR/thresholds for all models
├── confusion_matrices.json     # Confusion matrices at default threshold
├── pipeline_results.json       # Full results summary
└── fairness_results.json       # Disparate impact ratios by race and age
```

**Expected output files in `assets/`:**

```
assets/
├── shap_summary.png            # Beeswarm SHAP summary plot
├── shap_bar.png                # Bar chart of mean |SHAP| values
└── shap_waterfall.png          # Waterfall plot for first test sample
```

---

## 6. Run Threshold Optimisation

The default 0.5 classification threshold produces a recall of only 1.2%, which is clinically unacceptable. Run the threshold tuning script to find the Youden's J optimal threshold:

```bash
python -m src.tune_threshold
```

**What this script does:**
1. Loads `models/best_model.pkl`
2. Regenerates the test set by replaying the full preprocessing pipeline
3. Computes the ROC curve and finds the threshold that maximises `TPR − FPR` (Youden's J)
4. Saves the threshold and updated metrics

**Expected output:**

```
Optimal threshold (Youden's J): 0.1517
AUC:         0.6331
Recall:      0.4565
Precision:   0.1796
F1:          0.2578
Specificity: 0.7320
```

**Files updated/created:**

```
models/
├── threshold.json              # {"threshold": 0.1517, "method": "youden_j", ...}
├── pipeline_results.json       # Updated with optimal_threshold + tuned_metrics_at_optimal_threshold
└── confusion_matrices.json     # Updated with optimal_threshold confusion matrix
```

---

## 7. Launch the Dashboard

```bash
cd patient-readmission
streamlit run app/streamlit_app.py --server.port 8502
```

Then open your browser at: [http://localhost:8502](http://localhost:8502)

For the full feature set of the dashboard to work correctly, all of the following files must exist:

- `models/best_model.pkl`
- `models/artifacts.pkl`
- `models/shap_dict.pkl`
- `models/pipeline_results.json`
- `models/confusion_matrices.json`
- `models/fairness_results.json`
- `models/roc_data.json`
- `models/threshold.json`
- `assets/shap_summary.png`

---

## 8. Verifying Results

After running the full pipeline + threshold tuning, verify the following key metrics:

| Metric | Expected Value |
|---|---|
| Best model | LightGBM (Tuned) |
| Test AUC | 0.6331 |
| Recall @ threshold 0.1517 | 0.4565 (45.7%) |
| Precision @ threshold 0.1517 | 0.1796 (18.0%) |
| F1 @ threshold 0.1517 | 0.2578 |
| Specificity @ threshold 0.1517 | 0.7320 (73.2%) |
| True Positives (test set) | 1,033 |
| False Negatives (test set) | 1,230 |
| False Positives (test set) | 4,718 |
| True Negatives (test set) | 12,888 |

Top 5 features by SHAP importance:
1. `time_in_hospital`
2. `age_numeric`
3. `num_procedures`
4. `number_diagnoses`
5. `number_inpatient`

---

## 9. Known Limitations

- **SMOTE leakage in cross-validation (fixed):** SMOTE is now applied inside each CV fold using `imblearn.pipeline.Pipeline`, preventing synthetic samples from leaking into validation folds. A separate SMOTE resampling is applied to the full training set for the final model fit. The inflated CV AUC scores (~0.95) seen in prior runs were caused by the old approach and will not appear in re-runs with the corrected code.
- **Fairness analysis at default threshold:** The fairness report in `fairness_results.json` was computed at threshold=0.5, where positive predictions are near-zero. Re-run fairness analysis at threshold=0.1517 for meaningful disparity metrics.
- **Temporal leakage:** The dataset spans 1999–2008 and was split randomly. A temporally ordered split (training on earlier years, testing on later years) would better simulate production deployment.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root (`patient-readmission/`), not from inside `src/` |
| `FileNotFoundError: diabetic_data.csv` | Place the UCI dataset files in `data/` as described in Step 4 |
| `KeyError: 'tuned_metrics_at_optimal_threshold'` | Run `python -m src.tune_threshold` first |
| Dashboard shows 0.01 recall | Threshold not tuned. Run `python -m src.tune_threshold` |
| SHAP plots missing | Run `python -m src.run_pipeline` first to generate `assets/` |
| Memory error during SMOTE | Reduce `k_neighbors` in `prepare_features()` or use a smaller dataset sample |

---

*For a non-technical summary of the system, see the Stakeholder Brief. For full model performance details, see the Project Report.*
