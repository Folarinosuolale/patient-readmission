"""
feature_engineering.py – Derive clinical features, encode categoricals, and prepare
train / test splits for the patient-readmission model.

Feature groups:
  • ICD-9 diagnosis grouping  (diag_1 / 2 / 3 → clinical categories)
  • Medication aggregation    (23 drug cols → counts + flags)
  • Clinical derived features (utilization, polypharmacy, severity)
  • Encoding + scaling pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

from src.data_loader import map_icd9_to_group, MEDICATION_COLS

# Diagnosis grouping

def group_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """Map diag_1 / diag_2 / diag_3 to clinical group names."""
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            new_col = f"{col}_group"
            df[new_col] = df[col].apply(map_icd9_to_group)
            print(f"  {col} → {new_col}  ({df[new_col].nunique()} groups)")
    return df


# Medication aggregation

def encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 23 medication columns into summary features."""
    med_cols = [c for c in MEDICATION_COLS if c in df.columns]

    # Number of active (prescribed) medications
    df["num_active_medications"] = (
        df[med_cols].apply(lambda row: (row != "No").sum(), axis=1)
    )

    # Number of medication changes (dosage up or down)
    df["medication_change_count"] = (
        df[med_cols].apply(lambda row: row.isin(["Up", "Down"]).sum(), axis=1)
    )

    # Insulin-specific flags
    if "insulin" in df.columns:
        df["has_insulin"] = (df["insulin"] != "No").astype(int)
        df["insulin_changed"] = df["insulin"].isin(["Up", "Down"]).astype(int)
    else:
        df["has_insulin"] = 0
        df["insulin_changed"] = 0

    # Drop raw medication columns (replaced by aggregates)
    df = df.drop(columns=med_cols, errors="ignore")
    print(f"  Encoded {len(med_cols)} medication columns → 4 summary features")
    return df


# Derived clinical features

_AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer clinically-motivated features."""

    # --- Prior utilisation -----
    df["total_visits_prior"] = (
        df.get("number_outpatient", 0)
        + df.get("number_emergency", 0)
        + df.get("number_inpatient", 0)
    )
    df["high_utilizer"] = (df["total_visits_prior"] >= 3).astype(int)

    # --- Diagnosis-level flags ----
    if "diag_1" in df.columns:
        df["diabetes_primary"] = (
            df["diag_1"].astype(str).str.startswith("250")
        ).astype(int)
    else:
        df["diabetes_primary"] = 0

    # Number of distinct comorbidity groups across diag_1/2/3
    diag_group_cols = [c for c in ["diag_1_group", "diag_2_group", "diag_3_group"]
                       if c in df.columns]
    if diag_group_cols:
        df["num_comorbidities"] = df[diag_group_cols].nunique(axis=1)
    else:
        df["num_comorbidities"] = 0

    # --- Hospitalisation flags ----
    if "time_in_hospital" in df.columns:
        df["long_stay"] = (df["time_in_hospital"] > 7).astype(int)

    if "admission_type_id" in df.columns:
        df["emergency_admission"] = (df["admission_type_id"] == 1).astype(int)

    # --- Age numeric ----
    if "age" in df.columns:
        df["age_numeric"] = df["age"].map(_AGE_MAP).fillna(55)

    # --- change & diabetesMed as binary ----
    if "change" in df.columns:
        df["med_changed"] = (df["change"] == "Ch").astype(int)
    if "diabetesMed" in df.columns:
        df["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)

    # --- A1Cresult & max_glu_serum as ordinal ----
    if "A1Cresult" in df.columns:
        a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
        df["a1c_ordinal"] = df["A1Cresult"].map(a1c_map).fillna(0).astype(int)
    if "max_glu_serum" in df.columns:
        glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
        df["glu_ordinal"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(int)

    n_new = sum(1 for c in ["total_visits_prior", "high_utilizer", "diabetes_primary",
                             "num_comorbidities", "long_stay", "emergency_admission",
                             "age_numeric", "med_changed", "on_diabetes_med",
                             "a1c_ordinal", "glu_ordinal"] if c in df.columns)
    print(f"  Created {n_new} derived features")
    return df


# Encoding + scaling + split

# Columns to drop before modelling (raw / ID columns superseded by features)
_DROP_BEFORE_MODEL = [
    "readmitted",           # original multi-class target
    "diag_1", "diag_2", "diag_3",   # raw ICD codes (replaced by groups)
    "age",                  # replaced by age_numeric
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "change", "diabetesMed", "A1Cresult", "max_glu_serum",
    "gender",               # will be encoded below
    "race",                 # kept for fairness, encoded below
    "admission_type", "discharge_disposition", "admission_source",
    "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group",
]


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "readmitted_binary",
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = False,   # deprecated: SMOTE is now applied in run_pipeline via imblearn.Pipeline
):
    """
    Encode categoricals, scale numerics, and split into train/test.
    SMOTE is no longer applied here; it is applied inside CV folds in model_training
    using imblearn.Pipeline to prevent synthetic-sample leakage into validation folds.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    feature_names : list[str]
    artifacts : dict   (scaler, encoder, columns, race/age for fairness)
    """
    df = df.copy()

    # ---- Preserve sensitive attributes for fairness (before encoding) ----
    race_series = df["race"].copy() if "race" in df.columns else pd.Series("Unknown", index=df.index)
    age_series = df["age_numeric"].copy() if "age_numeric" in df.columns else pd.Series(55, index=df.index)
    gender_series = df["gender"].copy() if "gender" in df.columns else pd.Series("Unknown", index=df.index)

    # ---- Separate target ------
    y = df[target_col].values
    df = df.drop(columns=[target_col], errors="ignore")

    # ---- Encode gender ------
    if "gender" in df.columns:
        df["gender_male"] = (df["gender"] == "Male").astype(int)

    # ---- Target-encode high-cardinality categoricals ------
    cat_cols_to_encode = [
        c for c in ["admission_type", "discharge_disposition", "admission_source",
                     "medical_specialty", "diag_1_group", "diag_2_group", "diag_3_group"]
        if c in df.columns
    ]

    # ---- Drop raw / superseded columns ------
    drop_existing = [c for c in _DROP_BEFORE_MODEL if c in df.columns]
    df = df.drop(columns=drop_existing, errors="ignore")

    # ---- Ensure only numeric columns remain (safety) -----
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  ⚠  Dropping unexpected non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # ---- Train / test split (stratified) ------
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    train_idx, test_idx = X_train.index, X_test.index
    print(f"  Split: {len(X_train):,} train / {len(X_test):,} test  "
          f"(positive rate: train {y_train.mean():.3f}, test {y_test.mean():.3f})")

    # ---- Target encoding (fit on train only) ----
    enc_cols = [c for c in cat_cols_to_encode if c in X_train.columns]
    encoder = None
    if enc_cols:
        encoder = ce.TargetEncoder(cols=enc_cols, smoothing=1.0)
        X_train = encoder.fit_transform(X_train, y_train)
        X_test = encoder.transform(X_test)
        print(f"  Target-encoded {len(enc_cols)} categorical columns")

    # ---- Standard scaling -----
    scaler = StandardScaler()
    feature_names = X_train.columns.tolist()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Build artifacts dict -----
    artifacts = {
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": feature_names,
        "race_train": race_series.loc[train_idx].values,
        "race_test":  race_series.loc[test_idx].values,
        "age_train":  age_series.loc[train_idx].values,
        "age_test":   age_series.loc[test_idx].values,
        "gender_train": gender_series.loc[train_idx].values,
        "gender_test":  gender_series.loc[test_idx].values,
        "X_test_unscaled": X_test.copy(),
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, artifacts
