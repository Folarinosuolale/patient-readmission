"""
data_loader.py – Load, clean, and preprocess the UCI Diabetes 130-US Hospitals dataset.

Handles:
  • CSV loading with proper NA markers
  • ID-to-description mapping (admission type, discharge disposition, admission source)
  • ICD-9 diagnosis code → clinical group mapping
  • Deceased / hospice patient filtering
  • Missing-value strategy (drop useless cols, impute / flag remainder)
  • Binary target creation  (<30-day readmission vs not)
"""

import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# ICD-9 clinical groupings  (range-based, plus E/V prefix codes)
# ──────────────────────────────────────────────────────────────
ICD9_GROUPS = {
    "Infectious":       (1, 139),
    "Neoplasms":        (140, 239),
    "Endocrine_Other":  (240, 249),     # non-diabetes endocrine
    "Diabetes":         (250, 250),     # primary diabetes codes
    "Endocrine_Post":   (251, 279),     # post-diabetes endocrine
    "Blood":            (280, 289),
    "Mental":           (290, 319),
    "Nervous":          (320, 389),
    "Circulatory":      (390, 459),
    "Respiratory":      (460, 519),
    "Digestive":        (520, 579),
    "Genitourinary":    (580, 629),
    "Pregnancy":        (630, 679),
    "Skin":             (680, 709),
    "Musculoskeletal":  (710, 739),
    "Congenital":       (740, 759),
    "Perinatal":        (760, 779),
    "Ill_Defined":      (780, 799),
    "Injury":           (800, 999),
}

# Discharge disposition IDs that indicate the patient expired or went to hospice
DECEASED_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]

# Medication columns in the dataset (23 drug features)
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


# ──────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────

def map_icd9_to_group(code: str) -> str:
    """Map a single ICD-9 code string to a clinical group name."""
    if pd.isna(code) or str(code).strip() == "?" or str(code).strip() == "":
        return "Unknown"
    code = str(code).strip()
    # E / V codes
    if code.startswith("E"):
        return "External"
    if code.startswith("V"):
        return "External"
    # Numeric codes – take integer part
    try:
        num = float(code)
        num_int = int(num)
    except ValueError:
        return "Other"
    for group, (lo, hi) in ICD9_GROUPS.items():
        if lo <= num_int <= hi:
            return group
    return "Other"


def _parse_ids_mapping(mapping_path: str) -> dict:
    """Parse the multi-section IDS_mapping.csv into three dicts."""
    maps = {}
    current_key = None
    current_rows = []

    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_key and current_rows:
                    maps[current_key] = dict(current_rows)
                current_key = None
                current_rows = []
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            col, val = parts[0].strip(), parts[1].strip().strip('"')
            # Header row
            if col.endswith("_id") and val == "description":
                current_key = col
                continue
            try:
                col_int = int(col)
                current_rows.append((col_int, val))
            except ValueError:
                continue
    # Flush last section
    if current_key and current_rows:
        maps[current_key] = dict(current_rows)
    return maps


# ──────────────────────────────────────────────────────────────
# Pipeline functions
# ──────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str = "data/") -> pd.DataFrame:
    """Load diabetic_data.csv and return a raw DataFrame."""
    path = os.path.join(data_dir, "diabetic_data.csv")
    df = pd.read_csv(path, na_values=["?", "Unknown/Invalid", "None"])
    print(f"  Loaded {len(df):,} encounters, {df.shape[1]} columns")
    return df


def decode_admission_ids(df: pd.DataFrame, data_dir: str = "data/") -> pd.DataFrame:
    """Replace numeric IDs with human-readable descriptions using IDS_mapping.csv."""
    mapping_path = os.path.join(data_dir, "IDS_mapping.csv")
    if not os.path.exists(mapping_path):
        print("  ⚠  IDS_mapping.csv not found – skipping ID decode")
        return df

    maps = _parse_ids_mapping(mapping_path)
    id_cols = {
        "admission_type_id":        "admission_type",
        "discharge_disposition_id":  "discharge_disposition",
        "admission_source_id":       "admission_source",
    }
    for id_col, new_col in id_cols.items():
        if id_col in df.columns and id_col in maps:
            df[new_col] = df[id_col].map(maps[id_col]).fillna("Unknown")
            print(f"  Decoded {id_col} → {new_col}  ({df[new_col].nunique()} categories)")
    return df


def filter_deceased_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Remove encounters where the patient expired or went to hospice."""
    mask = df["discharge_disposition_id"].isin(DECEASED_DISPOSITION_IDS)
    n_removed = mask.sum()
    df = df[~mask].copy()
    print(f"  Removed {n_removed:,} deceased / hospice encounters → {len(df):,} remaining")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `readmitted` to binary: 1 = <30 days, 0 = otherwise."""
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
    pos = df["readmitted_binary"].sum()
    total = len(df)
    print(f"  Binary target: {pos:,} positive ({pos/total*100:.2f}%) / "
          f"{total-pos:,} negative ({(total-pos)/total*100:.2f}%)")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop useless high-missing columns, clean markers, and impute remaining.
    """
    # Columns to drop entirely
    drop_cols = ["weight", "payer_code", "encounter_id", "patient_nbr"]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"  Dropped columns: {existing_drops}")

    # medical_specialty: high missing → fill NaN with 'Unknown', then group
    if "medical_specialty" in df.columns:
        df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
        top_specialties = df["medical_specialty"].value_counts().head(10).index.tolist()
        df["medical_specialty"] = df["medical_specialty"].apply(
            lambda x: x if x in top_specialties else "Other"
        )
        print(f"  medical_specialty: grouped into {df['medical_specialty'].nunique()} categories")

    # race: fill NaN with 'Unknown'
    if "race" in df.columns:
        df["race"] = df["race"].fillna("Unknown")

    # diag_1/2/3: fill NaN with 'Unknown' (will be grouped later)
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Drop any remaining rows with NaN in critical numeric columns
    num_before = len(df)
    critical_numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                        "num_medications", "number_outpatient", "number_emergency",
                        "number_inpatient", "number_diagnoses"]
    existing_numeric = [c for c in critical_numeric if c in df.columns]
    df = df.dropna(subset=existing_numeric)
    if len(df) < num_before:
        print(f"  Dropped {num_before - len(df):,} rows with missing numeric values")

    print(f"  Final shape after cleaning: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────
# Convenience: full preprocessing chain
# ──────────────────────────────────────────────────────────────

def run_preprocessing(data_dir: str = "data/") -> pd.DataFrame:
    """Execute the full data-loading pipeline and return a clean DataFrame."""
    df = load_raw_data(data_dir)
    df = decode_admission_ids(df, data_dir)
    df = filter_deceased_patients(df)
    df = create_binary_target(df)
    df = handle_missing_values(df)
    return df
