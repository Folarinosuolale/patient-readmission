# Patient Readmission Prediction

Explainable ML system for predicting **30-day hospital readmission risk** in diabetic patients, with SHAP-based explanations and fairness auditing across race and age groups.

---

## Problem Statement

Hospital readmissions within 30 days are a key quality-of-care indicator and a major financial burden on healthcare systems. The CMS Hospital Readmissions Reduction Program (HRRP) penalises hospitals for excessive readmission rates. Early, accurate identification of high-risk patients enables targeted discharge planning, timely follow-up calls, and medication reconciliation, directly improving outcomes.

---

## Dataset

**UCI Diabetes 130-US Hospitals Dataset (1999–2008)**

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Encounters | 101,766 (before filtering) |
| Period | 1999–2008 |
| Hospitals | 130 US hospitals |
| Features (raw) | 50 |
| Target | `readmitted < 30 days` (binary) |
| Positive rate | ~11% (imbalanced) |

Key feature groups:
- **Demographics:** age (10-year buckets), gender, race
- **Admission:** type, discharge disposition, admission source, medical specialty
- **Clinical:** time in hospital, lab procedures, procedures, medications, diagnoses
- **Diagnoses:** ICD-9 codes (diag_1 / diag_2 / diag_3)
- **Medications:** 23 individual drug features (dosage: No / Steady / Up / Down)
- **Lab results:** A1C result, max glucose serum

---

## Pipeline Overview

```
[1/8] Load & preprocess
      → CSV loading, ICD-9 mapping, deceased/hospice filtering, missing-value handling

[2/8] Feature engineering
      → Diagnosis grouping (21 ICD-9 clinical groups)
      → Medication aggregation (23 cols → 4 summary features)
      → Derived clinical features (utilisation, comorbidities, flags)

[3/8] Prepare features
      → Target encoding of high-cardinality categoricals
      → StandardScaler on numeric features
      → 80/20 stratified train/test split
      → SMOTE oversampling on training set

[4/8] Compare base models (5-fold stratified CV)
      → Logistic Regression, Random Forest, XGBoost, LightGBM
      → Scoring: ROC AUC, F1, Recall, Precision

[5/8] Hyperparameter tuning (Optuna, 50 trials)
      → TPE Bayesian optimisation on best base model
      → Maximise CV ROC AUC

[6/8] SHAP explainability
      → TreeExplainer / LinearExplainer
      → Summary (beeswarm), bar, and waterfall plots

[7/8] Fairness analysis
      → Disparate impact by race and age group
      → 4/5ths rule evaluation
      → Per-group: prevalence, PPR, AUC, recall, precision, F1

[8/8] Save artefacts
      → best_model.pkl, artifacts.pkl, shap_dict.pkl
      → model_comparison.csv, feature_importance.csv
      → roc_data.json, confusion_matrices.json
      → pipeline_results.json, fairness_results.json
```

---

## Feature Engineering

### 1. ICD-9 Diagnosis Grouping
The three diagnosis columns (`diag_1`, `diag_2`, `diag_3`) are mapped from raw ICD-9 codes to 21 clinical groups:

| Group | ICD-9 Range | Description |
|---|---|---|
| Infectious | 1–139 | Infectious and parasitic diseases |
| Neoplasms | 140–239 | Cancers and tumours |
| Endocrine_Other | 240–249 | Non-diabetes endocrine disorders |
| Diabetes | 250–250 | Primary diabetes codes |
| Endocrine_Post | 251–279 | Post-diabetes endocrine and metabolic |
| Blood | 280–289 | Blood and blood-forming organ diseases |
| Mental | 290–319 | Mental disorders |
| Nervous | 320–389 | Nervous system and sense organ diseases |
| Circulatory | 390–459 | Circulatory system diseases |
| Respiratory | 460–519 | Respiratory system diseases |
| Digestive | 520–579 | Digestive system diseases |
| Genitourinary | 580–629 | Genitourinary system diseases |
| Pregnancy | 630–679 | Pregnancy, childbirth and puerperium |
| Skin | 680–709 | Skin and subcutaneous tissue diseases |
| Musculoskeletal | 710–739 | Musculoskeletal and connective tissue |
| Congenital | 740–759 | Congenital anomalies |
| Perinatal | 760–779 | Perinatal conditions |
| Ill_Defined | 780–799 | Symptoms, signs and ill-defined conditions |
| Injury | 800–999 | Injury and poisoning |
| External | E/V prefix | External causes and supplemental codes |
| Other | — | Unrecognised or unparseable codes |

### 2. Medication Aggregation
23 individual drug columns (metformin, insulin, glipizide…) are compressed into:
- `num_active_medications`: count of drugs prescribed (not "No")
- `medication_change_count`: count of dosage changes (Up or Down)
- `has_insulin`: binary flag
- `insulin_changed`: binary flag

### 3. Derived Clinical Features

| Feature | Formula |
|---|---|
| `total_visits_prior` | outpatient + emergency + inpatient |
| `high_utilizer` | total_visits_prior ≥ 3 |
| `diabetes_primary` | diag_1 starts with "250" |
| `num_comorbidities` | distinct diagnosis groups across diag_1/2/3 |
| `long_stay` | time_in_hospital > 7 |
| `emergency_admission` | admission_type_id == 1 |
| `age_numeric` | midpoint of age bucket |
| `med_changed` | change == "Ch" |
| `on_diabetes_med` | diabetesMed == "Yes" |
| `a1c_ordinal` | None=0, Norm=1, >7=2, >8=3 |
| `glu_ordinal` | None=0, Norm=1, >200=2, >300=3 |

---

## Models

| Model | Configuration |
|---|---|
| Logistic Regression | balanced class weight, lbfgs, max_iter=1000 |
| Random Forest | 200 trees, max_depth=12, balanced weights |
| XGBoost | 200 estimators, scale_pos_weight from class ratio |
| LightGBM | 200 estimators, is_unbalance=True |

Best model selected by test ROC AUC, then tuned with Optuna.

---

## Explainability (SHAP)

- **Global:** Mean |SHAP| bar chart ranks features by average impact across test set
- **Summary:** Beeswarm plot shows per-patient SHAP values (direction + magnitude)
- **Local:** Waterfall plot explains a single patient's prediction step-by-step
- **Live:** SHAP computed in real-time for any patient entered through the dashboard

Top clinical risk factors identified by SHAP:
1. `number_inpatient`: prior inpatient hospitalisations are the strongest predictor
2. `total_visits_prior`: overall healthcare utilisation history
3. `num_medications`: polypharmacy correlates with disease severity
4. `time_in_hospital`: longer stays indicate more serious presentations
5. `num_lab_procedures`: intensity of diagnostic workup

---

## Fairness Analysis

Disparate impact is assessed by **race** and **age group** using:
- **Disparate Impact Ratio (DIR):** min group predicted positive rate ÷ max group rate
- **4/5ths rule:** DIR < 0.80 indicates potential disparate impact
- **Per-group metrics:** prevalence, PPR, AUC, recall, precision, F1

Recall disparity is especially critical in healthcare: if the model misses high-risk patients in a specific demographic, those patients receive less follow-up care, compounding existing health equity gaps.

---

## Project Structure

```
patient-readmission/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Loading, ICD-9 mapping, preprocessing
│   ├── feature_engineering.py  # Medication encoding, derived features, scaling
│   ├── model_training.py       # Model comparison, Optuna tuning, fairness
│   ├── explainability.py       # SHAP computation and plotting
│   └── run_pipeline.py         # 8-stage orchestrator
├── app/
│   └── streamlit_app.py        # Interactive Streamlit dashboard
├── data/
│   ├── diabetic_data.csv       # Raw UCI dataset
│   └── IDS_mapping.csv         # ID → description mappings
├── models/                     # Generated by pipeline
│   ├── best_model.pkl
│   ├── artifacts.pkl
│   ├── shap_dict.pkl
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── roc_data.json
│   ├── confusion_matrices.json
│   ├── pipeline_results.json
│   └── fairness_results.json
├── assets/                     # Generated SHAP plots
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_waterfall.png
├── docs/
└── requirements.txt
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full ML pipeline
```bash
cd patient-readmission/
python -m src.run_pipeline
```
This will train models, generate SHAP plots, run fairness analysis, and save all artefacts to `models/` and `assets/`.

### 3. Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Requirements

```
pandas, numpy, scikit-learn, xgboost, lightgbm, shap, optuna,
imbalanced-learn, category_encoders, fairlearn, streamlit, plotly,
matplotlib, seaborn, joblib
```

See `requirements.txt` for pinned versions.

---

## Key Results (Expected)

| Metric | Expected Range |
|---|---|
| ROC AUC | 0.65 – 0.72 |
| Recall (30-day readmitted) | 0.55 – 0.70 |
| F1 Score | 0.25 – 0.40 |
| Fairness (Race DIR) | assessed per run |

> Note: Readmission prediction is a well-studied hard problem. AUC of 0.65–0.72 is consistent with
> published clinical literature (e.g., LACE index achieves ~0.70). The class imbalance (~11% positive)
> means F1 will be modest; Recall is the primary clinical metric.

---

## Clinical Context

This project demonstrates responsible AI deployment considerations relevant to regulated healthcare environments:
- **Explainability:** Every prediction can be explained at the individual patient level (SHAP waterfall)
- **Fairness auditing:** Systematic checks for racial and age-based disparity before deployment
- **Threshold flexibility:** The classification threshold can be adjusted to prioritise Recall (catching more at-risk patients) at the cost of Precision
- **Documentation:** Full reproducibility guide and artefact versioning

---

*Dataset: Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International.*
