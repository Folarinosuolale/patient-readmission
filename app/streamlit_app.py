import os, sys, json, joblib, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Readmission Risk Assessment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
.stApp { background-color: #F0F6FF; font-family: 'Inter', sans-serif; }
.stApp, .stApp p, .stApp li { color: #1E293B; }
h1, h2, h3, h4, h5, h6 { color: #0F3460 !important; }

/* ── Hide sidebar completely ── */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"]  { display: none !important; }

/* ── Top-nav tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 6px 10px;
    border: 1px solid #BFDBFE;
    box-shadow: 0 2px 8px rgba(15,52,96,0.06);
    gap: 4px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.stTabs [data-baseweb="tab"] {
    color: #334155 !important;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 8px 14px;
    border-radius: 8px;
    border-bottom: none !important;
    transition: background 0.15s, color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #EFF6FF;
    color: #1D6FA4 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1D6FA4 0%, #0F3460 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Metric cards with hover lift ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #BFDBFE;
    border-top: 4px solid #1D6FA4;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(29,111,164,0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(29,111,164,0.14);
}
[data-testid="metric-container"] label {
    color: #64748B !important; font-size: 0.82rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0F3460 !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Buttons ── */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #1D6FA4 0%, #0F3460 100%);
    color: #FFFFFF !important; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    box-shadow: 0 2px 8px rgba(15,52,96,0.25);
    transition: opacity 0.15s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

/* ── Form container ── */
.stForm {
    background: #FFFFFF; border: 1px solid #BFDBFE;
    border-radius: 12px; padding: 8px;
    box-shadow: 0 2px 12px rgba(29,111,164,0.06);
}

/* ── Content boxes ── */
.insight-box {
    background: #EFF6FF; border-left: 5px solid #1D6FA4;
    padding: 14px 16px; border-radius: 0 8px 8px 0; margin: 12px 0; color: #1E293B;
}
.warning-box {
    background: #FFF7ED; border-left: 5px solid #F97316;
    padding: 14px 16px; border-radius: 0 8px 8px 0; margin: 12px 0; color: #1E293B;
}
.explain-box {
    background: #F0F9FF; border-left: 5px solid #0EA5E9;
    padding: 14px 16px; border-radius: 0 8px 8px 0;
    margin: 12px 0; color: #1E293B; font-size: 0.92rem;
}
.success-box {
    background: #ECFDF5; border-left: 5px solid #059669;
    padding: 14px 16px; border-radius: 0 8px 8px 0; margin: 12px 0; color: #1E293B;
}

/* ── ROI highlight card ── */
.roi-highlight {
    background: linear-gradient(135deg, #0F3460 0%, #1D6FA4 100%);
    border-radius: 12px; padding: 18px 22px; color: white;
    margin: 8px 0; box-shadow: 0 4px 16px rgba(15,52,96,0.30);
}
.roi-highlight h3 { color: white !important; margin: 0 0 6px; font-size: 1rem; }
.roi-highlight .roi-num { font-size: 2.2rem; font-weight: 700; color: #93C5FD; }

/* ── Misc ── */
hr { border-color: #BFDBFE; }
.stDataFrame { background: #FFFFFF; border-radius: 8px; }
.stSelectbox label, .stSlider label, .stNumberInput label,
.stMultiSelect label, .stRadio label { color: #334155 !important; font-weight: 500; }
.streamlit-expanderHeader { color: #1D6FA4 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
ACCENT       = "#1D6FA4"
_BG          = "rgba(255,255,255,1)"
_TITLE_FONT  = dict(color="#0F3460", size=15, family="Inter, sans-serif")
_AXIS_FONT   = dict(color="#475569", size=12, family="Inter, sans-serif")
_LEGEND_FONT = dict(color="#334155")

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=_BG,
    plot_bgcolor="rgba(240,246,255,0.5)",
    font=dict(color="#1E293B", family="Inter, sans-serif"),
    xaxis_gridcolor="#E2E8F0", xaxis_linecolor="#CBD5E1",
    yaxis_gridcolor="#E2E8F0", yaxis_linecolor="#CBD5E1",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")
DATA_DIR   = os.path.join(ROOT, "data")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _models_ready() -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))


def _load_threshold() -> float:
    t = os.path.join(MODELS_DIR, "threshold.json")
    return json.load(open(t)).get("optimal_threshold", 0.5) if os.path.exists(t) else 0.5


@st.cache_data(show_spinner=False)
def load_raw_data():
    p = os.path.join(DATA_DIR, "diabetic_data.csv")
    return pd.read_csv(p, na_values=["?", "Unknown/Invalid", "None"]) if os.path.exists(p) else None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model      = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    artifacts  = joblib.load(os.path.join(MODELS_DIR, "artifacts.pkl"))
    shap_dict  = joblib.load(os.path.join(MODELS_DIR, "shap_dict.pkl"))
    importance = pd.read_csv(os.path.join(MODELS_DIR, "feature_importance.csv"))
    comp_df    = pd.read_csv(os.path.join(MODELS_DIR, "model_comparison.csv"))
    with open(os.path.join(MODELS_DIR, "pipeline_results.json")) as f:
        results = json.load(f)
    with open(os.path.join(MODELS_DIR, "roc_data.json")) as f:
        roc_data = json.load(f)
    with open(os.path.join(MODELS_DIR, "confusion_matrices.json")) as f:
        cm_data = json.load(f)
    with open(os.path.join(MODELS_DIR, "fairness_results.json")) as f:
        fairness = json.load(f)
    return model, artifacts, shap_dict, importance, comp_df, results, roc_data, cm_data, fairness


def _get_best_metrics(results: dict) -> dict:
    return results.get(
        "tuned_metrics_at_optimal_threshold",
        results.get("tuned_metrics", {})
    )


# ── Bootstrap (top-level artifact load, cached) ───────────────────────────────
_ready = _models_ready()
if _ready:
    model, artifacts, shap_dict, importance, comp_df, results, roc_data, cm_data, fairness = load_artifacts()
    tm   = _get_best_metrics(results)
    opt_t = results.get("optimal_threshold", _load_threshold())
    _status_color = "#059669"
    _status_text  = "● MODEL VALIDATED"
    _pills_text   = (
        f"AUC {tm.get('auc', 0):.3f} &nbsp;·&nbsp; "
        f"Recall {tm.get('recall', 0):.1%} &nbsp;·&nbsp; "
        f"Threshold {opt_t:.3f}"
    )
else:
    opt_t = _load_threshold()
    tm    = {}
    _status_color = "#D97706"
    _status_text  = "● PIPELINE PENDING"
    _pills_text   = "Run <code>python -m src.run_pipeline</code> to generate models"

# ── Header strip ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(135deg, #0F3460 0%, #1D6FA4 100%);
            padding: 14px 22px; border-radius: 10px; margin-bottom: 14px;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 8px;">
  <div>
    <span style="color: white; font-size: 1.3rem; font-weight: 700; letter-spacing: -0.3px;">
      🏥 Readmission Risk AI
    </span>
    <span style="color: #93C5FD; font-size: 0.82rem; margin-left: 12px;">
      UCI Diabetes 130-US Hospitals &nbsp;·&nbsp; 99,343 encounters &nbsp;·&nbsp; LightGBM + SHAP
    </span>
  </div>
  <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
    <span style="background: {_status_color}; color: white;
                 padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">
      {_status_text}
    </span>
    <span style="background: rgba(255,255,255,0.15); color: #E0F2FE;
                 padding: 4px 14px; border-radius: 20px; font-size: 0.78rem;">
      {_pills_text}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top navigation (7 tabs) ───────────────────────────────────────────────────
(tab_overview, tab_cohort, tab_perf,
 tab_explain, tab_equity, tab_screener, tab_roi) = st.tabs([
    "🏠  Overview",
    "👥  Patient Cohort",
    "📈  Model Performance",
    "🧬  Explainability",
    "⚖️  Equity Audit",
    "🔬  Patient Screener",
    "💰  ROI Simulator",
])



#  TAB 1: OVERVIEW

with tab_overview:
    st.markdown(
        "<h2 style='color:#0F3460; margin-bottom:2px;'>Overview</h2>"
        "<p style='color:#475569; margin-top:0; font-size:1.02rem;'>"
        "End-to-end ML system for predicting 30-day hospital readmission risk "
        "in diabetic patients with full SHAP explainability and fairness auditing.</p>",
        unsafe_allow_html=True,
    )

    if not _ready:
        st.info("Model artefacts not found. Please run `python -m src.run_pipeline` first.")
        df_raw = load_raw_data()
        if df_raw is not None:
            st.markdown("### Dataset Snapshot")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Encounters", f"{len(df_raw):,}")
            c2.metric("Features", str(df_raw.shape[1]))
            c3.metric("30-day Readmission Rate", f"{(df_raw['readmitted'] == '<30').mean():.1%}")
            c4.metric("Unique Patients", f"{df_raw['patient_nbr'].nunique():,}")
    else:
        ds = results["dataset_summary"]

        # ── KPI row ──────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("ROC AUC",          f"{tm.get('auc', 0):.3f}")
        c2.metric("Recall @ Opt. T",  f"{tm.get('recall', 0):.3f}")
        c3.metric("Precision",        f"{tm.get('precision', 0):.3f}")
        c4.metric("F1 Score",         f"{tm.get('f1', 0):.3f}")
        c5.metric("Specificity",      f"{tm.get('specificity', 0):.3f}")
        c6.metric("Opt. Threshold",   f"{opt_t:.3f}")

        with st.expander("What do these metrics mean?"):
            st.markdown(f"""
**All metrics are evaluated at the Youden-optimised decision threshold ({opt_t:.3f}), not the default 0.5.**

- **ROC AUC ({tm.get('auc',0):.3f}):** Discrimination ability independent of threshold.
  Published benchmarks on this exact dataset (Strack et al., 2014) report AUC 0.62–0.68. This means we are in range.

- **Recall ({tm.get('recall',0):.3f}):** Of all patients who *were* readmitted, the model correctly flags
  this fraction. In clinical use, recall is the most critical metric: a missed high-risk patient may not
  receive the discharge planning they need. At default 0.5 threshold, recall was **1.2%**; at the optimal
  threshold of {opt_t:.3f}, it rises to **{tm.get('recall',0):.1%}**.

- **Precision ({tm.get('precision',0):.3f}):** Of all patients flagged, this fraction truly was readmitted.
  With an 11% base rate, values in the 15–25% range are expected and clinically acceptable for triage.

- **F1 ({tm.get('f1',0):.3f}):** Harmonic mean of precision and recall, best summary for imbalanced datasets.

- **Specificity ({tm.get('specificity',0):.3f}):** Of patients NOT readmitted, this fraction is correctly
  cleared, reducing unnecessary follow-up burden.
""")

        st.markdown("---")

        # ── 3-column info section ────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("#### Clinical Problem")
            st.markdown("""
Hospital readmissions within 30 days are a major quality-of-care indicator and cost driver.
The CMS Hospital Readmissions Reduction Program (HRRP) financially penalises hospitals for
excessive readmissions. Early identification of at-risk patients enables proactive discharge
planning, follow-up care, and targeted resource allocation.
""")

        with col_b:
            st.markdown("#### ML Approach")
            st.markdown(f"""
- **4 baseline models** compared (LogReg, RF, XGBoost, LightGBM)
- **5-fold stratified cross-validation** on SMOTE-resampled training set
- **Optuna Bayesian optimisation** (50 trials) on the best base model
- **SMOTE** to handle class imbalance (~11% positive rate)
- **Best model:** {results.get('tuned_model_name', 'LightGBM (Tuned)')}
- **Threshold:** Youden's J statistic → {opt_t:.3f}
""")

        with col_c:
            st.markdown("#### Top Risk Drivers (SHAP)")
            top5 = importance.head(5)["feature"].tolist()
            st.markdown("Top 5 features by mean |SHAP| value:")
            for i, feat in enumerate(top5, 1):
                st.markdown(f"{i}. `{feat}`")

        st.markdown("---")

        # ── Dataset overview ─────────────────────────────────────────────────
        st.markdown("### Dataset & Pipeline Summary")
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown(f"""
| Property | Value |
|---|---|
| Source | UCI ML Repository |
| Period | 1999 – 2008 |
| Hospitals | 130 US hospitals |
| Encounters | {ds['n_encounters']:,} |
| Features (engineered) | {ds['n_features']} |
| Positive rate | {ds['positive_rate']:.1%} |
| Imbalance handling | SMOTE + class weights |
| Task | Binary classification |
""")
        with col_right:
            st.markdown("""
**Feature engineering pipeline:**
1. ICD-9 diagnosis codes → 20 clinical groups (Circulatory, Diabetes, Respiratory…)
2. 23 medication columns → 4 aggregates (active meds, changes, insulin flags)
3. Derived indicators: utilisation history, long-stay flag, comorbidity count
4. A1C and glucose serum → ordinal scores
5. Demographics: age buckets → numeric midpoints, gender binary

**Key insight: threshold matters more than model.**
At default threshold 0.5, recall = 1.2% (nearly useless for clinical screening).
After Youden's J threshold optimisation (threshold = 0.1517):
- Recall: **45.7%**, flags nearly half of actual readmissions
- Specificity: **73.2%**, clears nearly three-quarters of low-risk patients
""")


#  TAB 2: PATIENT COHORT

with tab_cohort:
    st.markdown(
        "<h2 style='color:#0F3460;'>Patient Cohort</h2>"
        "<p style='color:#475569; margin-top:0;'>Explore the UCI Diabetes 130-US Hospitals dataset.</p>",
        unsafe_allow_html=True,
    )

    df_raw = load_raw_data()
    if df_raw is None:
        st.error("Raw data not found in `data/diabetic_data.csv`.")
        st.markdown("Add `diabetic_data.csv` to the `data/` folder and redeploy.")
        df_raw = pd.DataFrame(columns=["age","race","readmitted","patient_nbr",
                                        "time_in_hospital","num_lab_procedures",
                                        "num_procedures","num_medications",
                                        "number_outpatient","number_emergency",
                                        "number_inpatient","number_diagnoses",
                                        "A1Cresult","insulin"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Encounters",        f"{len(df_raw):,}")
    c2.metric("Unique Patients",          f"{df_raw['patient_nbr'].nunique():,}")
    c3.metric("30-day Readmission Rate",  f"{(df_raw['readmitted'] == '<30').mean():.1%}")
    c4.metric("Features",                 str(df_raw.shape[1]))

    st.markdown("---")
    sub1, sub2, sub3 = st.tabs(["Demographics", "Readmission Rates", "Clinical Features"])

    with sub1:
        st.markdown("### Patient Demographics")
        col1, col2 = st.columns(2)

        with col1:
            age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                         "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
            age_counts = (df_raw["age"].value_counts()
                          .reindex(age_order).fillna(0).reset_index())
            age_counts.columns = ["Age Group", "Count"]
            fig = px.bar(age_counts, x="Age Group", y="Count",
                         title="Age Distribution",
                         color_discrete_sequence=[ACCENT])
            fig.update_layout(height=350, xaxis_tickfont=_AXIS_FONT,
                               yaxis_tickfont=_AXIS_FONT, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            race_counts = df_raw["race"].value_counts().reset_index()
            race_counts.columns = ["Race", "Count"]
            fig = px.pie(race_counts, values="Count", names="Race",
                         title="Race / Ethnicity Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        readmit_counts = df_raw["readmitted"].value_counts().reset_index()
        readmit_counts.columns = ["Readmission", "Count"]
        readmit_counts["Readmission"] = readmit_counts["Readmission"].map(
            {"NO": "No Readmission", "<30": "Readmitted <30 days", ">30": "Readmitted >30 days"}
        ).fillna(readmit_counts["Readmission"])
        fig = px.bar(readmit_counts, x="Readmission", y="Count",
                     title="Readmission Outcome Distribution (3 categories → collapsed to binary target)",
                     color="Readmission",
                     color_discrete_map={
                         "No Readmission": "#10B981",
                         "Readmitted <30 days": "#EF4444",
                         "Readmitted >30 days": "#F59E0B",
                     })
        fig.update_layout(height=320, xaxis_tickfont=_AXIS_FONT,
                           yaxis_tickfont=_AXIS_FONT, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Key observation:</strong> The dataset skews toward Caucasian patients and older age
        groups (60–80), reflective of the US diabetic hospital population. The ~11% 30-day readmission
        rate creates significant class imbalance, addressed via SMOTE and class weighting.
        The <em>binary target</em> collapses "readmitted &lt;30 days" to positive and everything else to negative.
        </div>
        """, unsafe_allow_html=True)

    with sub2:
        st.markdown("### Readmission Rates by Subgroup")
        col1, col2 = st.columns(2)

        with col1:
            age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                         "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
            readmit_age = (
                df_raw.groupby("age")["readmitted"]
                .apply(lambda x: (x == "<30").mean())
                .reindex(age_order).reset_index()
            )
            readmit_age.columns = ["Age Group", "30-day Rate"]
            fig = px.bar(readmit_age, x="Age Group", y="30-day Rate",
                         title="30-day Readmission Rate by Age Group",
                         color="30-day Rate", color_continuous_scale="Teal")
            fig.update_layout(height=380, yaxis_tickformat=".1%",
                               xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
                               **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            readmit_race = (
                df_raw.groupby("race")["readmitted"]
                .apply(lambda x: (x == "<30").mean())
                .sort_values(ascending=False).reset_index()
            )
            readmit_race.columns = ["Race", "30-day Rate"]
            fig = px.bar(readmit_race, x="Race", y="30-day Rate",
                         title="30-day Readmission Rate by Race / Ethnicity",
                         color="30-day Rate", color_continuous_scale="Oranges")
            fig.update_layout(height=380, yaxis_tickformat=".1%",
                               xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
                               **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            readmit_a1c = (
                df_raw.groupby("A1Cresult")["readmitted"]
                .apply(lambda x: (x == "<30").mean())
                .sort_values(ascending=False).reset_index()
            )
            readmit_a1c.columns = ["A1C Result", "30-day Rate"]
            fig = px.bar(readmit_a1c, x="A1C Result", y="30-day Rate",
                         title="30-day Rate by A1C Test Result",
                         color_discrete_sequence=["#8B5CF6"])
            fig.update_layout(height=340, yaxis_tickformat=".1%",
                               xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
                               **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            readmit_ins = (
                df_raw.groupby("insulin")["readmitted"]
                .apply(lambda x: (x == "<30").mean())
                .sort_values(ascending=False).reset_index()
            )
            readmit_ins.columns = ["Insulin", "30-day Rate"]
            fig = px.bar(readmit_ins, x="Insulin", y="30-day Rate",
                         title="30-day Rate by Insulin Status",
                         color_discrete_sequence=[ACCENT])
            fig.update_layout(height=340, yaxis_tickformat=".1%",
                               xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
                               **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with sub3:
        st.markdown("### Clinical Feature Distributions")
        st.markdown(
            "The two highest-ranked SHAP features, **Length of Stay** and "
            "**Number of Diagnoses**, are shown below alongside correlation with readmission."
        )
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_raw, x="time_in_hospital", nbins=14,
                               title="Length of Hospital Stay (Days) (#1 SHAP Feature)",
                               color_discrete_sequence=[ACCENT])
            fig.update_layout(height=340, xaxis_tickfont=_AXIS_FONT,
                               yaxis_tickfont=_AXIS_FONT, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df_raw, x="number_diagnoses", nbins=16,
                               title="Number of Diagnoses (Comorbidities) (#4 SHAP Feature)",
                               color_discrete_sequence=["#EF4444"])
            fig.update_layout(height=340, xaxis_tickfont=_AXIS_FONT,
                               yaxis_tickfont=_AXIS_FONT, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature Correlation with 30-Day Readmission")
        target = (df_raw["readmitted"] == "<30").astype(int)
        num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                    "num_medications", "number_outpatient", "number_emergency",
                    "number_inpatient", "number_diagnoses"]
        existing_num = [c for c in num_cols if c in df_raw.columns]
        corrs = df_raw[existing_num].corrwith(target).sort_values(ascending=False)
        corr_df = pd.DataFrame({"feature": corrs.index, "correlation": corrs.values})
        fig = px.bar(corr_df, x="feature", y="correlation",
                     title="Pearson Correlation with 30-Day Readmission (linear signal only)",
                     color="correlation", color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=350, xaxis_tickfont=_AXIS_FONT,
                           yaxis_tickfont=_AXIS_FONT, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <strong>Reading the correlation chart:</strong> Positive values = higher feature values are
        associated with more readmissions. The ML model captures non-linear interactions that raw
        Pearson correlations miss, which is why SHAP importance rankings differ from this chart.
        </div>
        """, unsafe_allow_html=True)


#  TAB 3: MODEL PERFORMANCE

with tab_perf:
    st.markdown(
        "<h2 style='color:#0F3460;'>Model Performance</h2>",
        unsafe_allow_html=True,
    )

    if not _ready:
        st.info("Run the pipeline first to see model results.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model",      results.get("tuned_model_name", "LightGBM").split(" ")[0])
    c2.metric("Test AUC",        f"{tm.get('auc', 0):.4f}")
    c3.metric("Recall @ Opt. T", f"{tm.get('recall', 0):.4f}")
    c4.metric("F1 Score",        f"{tm.get('f1', 0):.4f}")
    c5.metric("Opt. Threshold",  f"{opt_t:.4f}")

    st.markdown("---")
    sub1, sub2, sub3 = st.tabs(["Model Comparison", "ROC Curves", "Confusion Matrix"])

    with sub1:
        st.markdown("### Base Model + Tuned Model Comparison")
        st.markdown("""
        <div class="explain-box">
        All 4 models were evaluated with 5-fold stratified cross-validation on the SMOTE-resampled
        training set, then assessed on the original held-out test set. The best base model by test AUC
        was further tuned with 50 Optuna Bayesian optimisation trials.
        </div>
        """, unsafe_allow_html=True)

        display_df = comp_df.copy()
        numeric_cols = ["cv_auc", "cv_f1", "cv_recall", "test_auc", "test_f1",
                        "test_recall", "test_precision", "test_specificity", "test_accuracy"]
        for c in numeric_cols:
            if c in display_df.columns:
                display_df[c] = display_df[c].apply(
                    lambda x: f"{float(x):.4f}"
                    if str(x).replace(".", "").replace("-", "").isdigit() else str(x)
                )
        st.dataframe(display_df, use_container_width=True)

        plot_cols = ["test_auc", "test_f1", "test_recall", "test_precision"]
        existing_plot = [c for c in plot_cols if c in comp_df.columns]
        if existing_plot:
            melt_df = comp_df[["model"] + existing_plot].melt(
                id_vars="model", var_name="Metric", value_name="Score"
            )
            melt_df["Score"] = pd.to_numeric(melt_df["Score"], errors="coerce")
            fig = px.bar(melt_df, x="model", y="Score", color="Metric",
                         barmode="group",
                         title="Model Comparison: Test Set (Default 0.5 Threshold)",
                         color_discrete_sequence=[ACCENT, "#10B981", "#F59E0B", "#8B5CF6"])
            fig.update_layout(height=420, yaxis_range=[0, 1],
                               xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
                               legend=dict(font=_LEGEND_FONT), **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with sub2:
        st.markdown("### ROC Curves: All Models")
        st.markdown("""
        <div class="explain-box">
        The ROC curve plots True Positive Rate (sensitivity/recall) against False Positive Rate
        at every classification threshold. AUC = 1.0 is perfect; AUC = 0.5 is random (dashed line).
        Published benchmarks on this exact dataset (Strack et al., 2014) report AUC 0.62–0.68.
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash", color="#94A3B8"),
                      x0=0, x1=1, y0=0, y1=1)
        colors = [ACCENT, "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
        for i, rd in enumerate(roc_data):
            fig.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"],
                mode="lines",
                name=f"{rd['model']} (AUC={rd['auc']:.4f})",
                line=dict(color=colors[i % len(colors)], width=2.5),
            ))
        fig.update_layout(
            title=dict(text="ROC Curves: All Models", font=_TITLE_FONT),
            xaxis_title="False Positive Rate (1 – Specificity)",
            yaxis_title="True Positive Rate (Recall / Sensitivity)",
            xaxis_title_font=_AXIS_FONT, yaxis_title_font=_AXIS_FONT,
            xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
            height=500,
            legend=dict(x=0.55, y=0.1, font=_LEGEND_FONT),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with sub3:
        st.markdown("### Confusion Matrix")
        st.markdown("""
        <div class="explain-box">
        <strong>TN</strong> = correctly cleared &nbsp;·&nbsp;
        <strong>FP</strong> = false alarm (unnecessary follow-up) &nbsp;·&nbsp;
        <strong>FN</strong> = missed readmission (most costly clinically) &nbsp;·&nbsp;
        <strong>TP</strong> = correctly flagged.<br>
        The <em>optimal_threshold</em> entry shows results at Youden-tuned threshold (0.1517),
        dramatically improving recall vs. the default 0.5 threshold.
        </div>
        """, unsafe_allow_html=True)

        cm_matrix_keys = [k for k, v in cm_data.items() if isinstance(v, list)]
        cm_dict_keys   = [k for k, v in cm_data.items() if isinstance(v, dict)]
        all_options = cm_matrix_keys + cm_dict_keys
        model_choice = st.selectbox("Select model / threshold", all_options)

        entry = cm_data[model_choice]
        if isinstance(entry, list):
            cm = np.array(entry)
        else:
            cm = np.array([[entry["tn"], entry["fp"]], [entry["fn"], entry["tp"]]])

        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["No Readmission", "Readmitted <30d"],
            y=["No Readmission", "Readmitted <30d"],
            color_continuous_scale="Teal",
            title=f"Confusion Matrix: {model_choice}",
        )
        fig.update_layout(height=420, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        tn, fp, fn, tp_cm = cm.ravel()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Negatives",           f"{int(tn):,}")
        c2.metric("False Positives",           f"{int(fp):,}")
        c3.metric("False Negatives (Missed)",  f"{int(fn):,}")
        c4.metric("True Positives",            f"{int(tp_cm):,}")


#  TAB 4: EXPLAINABILITY

with tab_explain:
    st.markdown(
        "<h2 style='color:#0F3460;'>SHAP Explainability</h2>"
        "<p style='color:#475569; margin-top:0;'>"
        "SHapley Additive exPlanations quantify each feature's contribution to every "
        "individual prediction: global patterns and patient-level reasoning.</p>",
        unsafe_allow_html=True,
    )

    if not _ready:
        st.info("Run the pipeline first to see SHAP results.")
        st.stop()

    sub1, sub2, sub3 = st.tabs(["Global Importance", "SHAP Plots", "Feature Deep Dive"])

    with sub1:
        st.markdown("### Top Feature Importances (Mean |SHAP|)")
        st.markdown("""
        <div class="explain-box">
        SHAP values measure how much each feature shifts the model's readmission prediction.
        Averaging absolute SHAP across the test set gives a globally consistent importance ranking.
        Unlike permutation importance, SHAP is additive and theoretically grounded in game theory.
        </div>
        """, unsafe_allow_html=True)

        top_n = st.slider("Features to display", 5, min(30, len(importance)), 15)
        plot_df = importance.head(top_n)
        fig = px.bar(
            plot_df[::-1], x="mean_abs_shap", y="feature", orientation="h",
            title=f"Top {top_n} Features by Mean |SHAP| (Test Set)",
            color="mean_abs_shap", color_continuous_scale="Teal",
        )
        fig.update_layout(
            height=max(350, top_n * 28),
            coloraxis_showscale=False,
            xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Full Feature Importance Table")
        st.dataframe(
            importance.style.format({"mean_abs_shap": "{:.5f}"}),
            use_container_width=True,
        )

        top3 = importance.head(3)
        st.markdown(f"""
        <div class="insight-box">
        <strong>Top 3 drivers of 30-day readmission risk (SHAP-ranked):</strong><br><br>
        1. <strong>{top3.iloc[0]['feature']}</strong>: the single strongest predictor;
           mean |SHAP| = {top3.iloc[0]['mean_abs_shap']:.4f}.<br>
        2. <strong>{top3.iloc[1]['feature']}</strong>: second most influential feature across
           all test patients; mean |SHAP| = {top3.iloc[1]['mean_abs_shap']:.4f}.<br>
        3. <strong>{top3.iloc[2]['feature']}</strong>: third highest consistent impact on the
           model's predictions; mean |SHAP| = {top3.iloc[2]['mean_abs_shap']:.4f}.
        </div>
        """, unsafe_allow_html=True)

    with sub2:
        st.markdown("### SHAP Visualisations")
        col1, col2 = st.columns(2)
        with col1:
            summary_path = os.path.join(ASSETS_DIR, "shap_summary.png")
            if os.path.exists(summary_path):
                st.image(summary_path, caption="SHAP Beeswarm Summary Plot",
                         use_container_width=True)
                st.markdown("""
                <div class="explain-box">
                Each dot = one test patient. Horizontal position = SHAP value
                (right → higher readmission risk). Colour = feature value (red = high, blue = low).
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("shap_summary.png not found in assets/.")

        with col2:
            bar_path = os.path.join(ASSETS_DIR, "shap_bar.png")
            if os.path.exists(bar_path):
                st.image(bar_path, caption="SHAP Bar Plot (Global Importance)",
                         use_container_width=True)
                st.markdown("""
                <div class="explain-box">
                Each bar = average absolute SHAP value. Longer bars = more consistent
                influence across all test patients.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("shap_bar.png not found in assets/.")

        st.markdown("### Single Patient Explanation (Waterfall)")
        waterfall_path = os.path.join(ASSETS_DIR, "shap_waterfall.png")
        if os.path.exists(waterfall_path):
            st.image(waterfall_path, caption="SHAP Waterfall: Sample Patient",
                     use_container_width=True)
            st.markdown("""
            <div class="explain-box">
            Starts from the base value (average model output), then shows how each feature
            pushes the prediction up (red → risk) or down (blue → lower risk).
            This is the kind of individual explanation a clinician needs at discharge.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("shap_waterfall.png not found in assets/.")

    with sub3:
        st.markdown("### Feature Deep Dive")
        shap_values   = np.array(shap_dict["shap_values"])
        feature_names = shap_dict["feature_names"]
        feature = st.selectbox("Select feature to analyse", feature_names)
        feat_idx  = feature_names.index(feature)
        feat_shap = shap_values[:, feat_idx]

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"SHAP Distribution: {feature}",
            f"SHAP Value per Patient: {feature}",
        ])
        fig.add_trace(
            go.Histogram(x=feat_shap, nbinsx=30, marker_color=ACCENT,
                         name="SHAP Distribution"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(feat_shap))), y=feat_shap,
                mode="markers",
                marker=dict(size=3, color=feat_shap,
                            colorscale="RdBu_r", showscale=True),
                name="SHAP per patient",
            ),
            row=1, col=2,
        )
        fig.update_layout(height=400, showlegend=False, **PLOTLY_LAYOUT)
        fig.update_xaxes(title_text="SHAP Value",    tickfont=_AXIS_FONT, row=1, col=1)
        fig.update_xaxes(title_text="Patient Index", tickfont=_AXIS_FONT, row=1, col=2)
        fig.update_yaxes(title_text="Count",         tickfont=_AXIS_FONT, row=1, col=1)
        fig.update_yaxes(title_text="SHAP Value",    tickfont=_AXIS_FONT, row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean |SHAP|",            f"{np.mean(np.abs(feat_shap)):.5f}")
        c2.metric("Max (risk-increasing)",  f"{np.max(feat_shap):.5f}")
        c3.metric("Min (risk-decreasing)",  f"{np.min(feat_shap):.5f}")

        pct_pos_shap = (feat_shap > 0).mean() * 100
        pct_neg_shap = (feat_shap < 0).mean() * 100
        st.markdown(f"""
        <div class="insight-box">
        For <strong>{pct_pos_shap:.0f}%</strong> of test patients,
        <strong>{feature}</strong> <em>increased</em> predicted readmission risk.
        For <strong>{pct_neg_shap:.0f}%</strong> it <em>decreased</em> risk.
        Range: {np.min(feat_shap):.4f} to {np.max(feat_shap):.4f}.
        </div>
        """, unsafe_allow_html=True)


#  TAB 5: EQUITY AUDIT

with tab_equity:
    st.markdown(
        "<h2 style='color:#0F3460;'>Equity Audit</h2>"
        "<p style='color:#475569; margin-top:0;'>"
        "Evaluating whether the model treats patient subgroups equitably across race and age. "
        "Disparate impact analysis consistent with EEOC 4/5ths rule.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="explain-box">
    <strong>Why fairness matters in clinical ML:</strong> A model performing well on average can
    still systematically underserve certain patient populations. This audit checks whether predicted
    positive rates are equitable across race and age cohorts. The 4/5ths rule: no group's predicted
    positive rate should fall below 80% of the highest group's rate.
    </div>
    """, unsafe_allow_html=True)

    if not _ready:
        st.info("Run the pipeline first to see fairness results.")
        st.stop()

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Threshold dependency note:</strong> Fairness analysis was computed at the
    <em>default 0.5 threshold</em>, where near-zero positive predictions make the 4/5ths ratio
    a mathematical artefact (≈ 0% ÷ 0.8% ≈ 0). This is expected: it demonstrates that fairness
    metrics are <em>threshold-sensitive</em>. A meaningful re-analysis should be run at the
    Youden threshold (0.1517). This limitation is transparently documented.
    </div>
    """, unsafe_allow_html=True)

    for group_key in ["race", "age"]:
        if group_key not in fairness:
            continue

        gdata      = fairness[group_key]
        group_name = gdata["group_name"]
        di_ratio   = gdata.get("disparate_impact_ratio", float("nan"))
        try:
            di_float = float(di_ratio)
            passes   = di_float >= 0.80 and not np.isnan(di_float)
            di_str   = f"{di_float:.3f}"
        except (TypeError, ValueError):
            passes  = False
            di_str  = "N/A"

        st.markdown(f"---\n### Fairness by {group_name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Disparate Impact Ratio", di_str,
                  delta="PASS" if passes else "FAIL",
                  delta_color="normal" if passes else "inverse")
        c2.metric("Threshold (4/5ths Rule)", "0.800")
        c3.metric("Status", "✅ Fair" if passes else "⚠️ Potential Disparity")

        group_metrics = gdata.get("group_metrics", {})
        if group_metrics:
            rows = []
            for g, m in group_metrics.items():
                rows.append({
                    "Group": g,
                    "N": m["n"],
                    "Prevalence": m["prevalence"],
                    "Predicted Positive Rate": m["predicted_positive_rate"],
                    "AUC": m["auc"],
                    "Recall": m["recall"],
                    "Precision": m["precision"],
                    "F1": m["f1"],
                })
            gdf = pd.DataFrame(rows).sort_values("Predicted Positive Rate", ascending=False)

            st.dataframe(
                gdf.style.format({
                    "Prevalence": "{:.1%}",
                    "Predicted Positive Rate": "{:.1%}",
                    "AUC": "{:.4f}",
                    "Recall": "{:.3f}",
                    "Precision": "{:.3f}",
                    "F1": "{:.3f}",
                }),
                use_container_width=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                max_ppr = gdf["Predicted Positive Rate"].max()
                fig = go.Figure(data=[go.Bar(
                    x=gdf["Group"],
                    y=gdf["Predicted Positive Rate"],
                    marker_color=ACCENT,
                    text=[f"{v:.1%}" for v in gdf["Predicted Positive Rate"]],
                    textposition="outside",
                )])
                if max_ppr > 0:
                    fig.add_hline(y=max_ppr * 0.8, line_dash="dash", line_color="#EF4444",
                                  annotation_text="4/5ths threshold")
                fig.update_layout(
                    title=dict(text=f"Predicted Positive Rate by {group_name}", font=_TITLE_FONT),
                    yaxis_title="Predicted Positive Rate",
                    yaxis_tickformat=".1%",
                    xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
                    height=400, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure(data=[
                    go.Bar(name="Actual Prevalence", x=gdf["Group"],
                           y=gdf["Prevalence"], marker_color="#F59E0B"),
                    go.Bar(name="Predicted Positive Rate", x=gdf["Group"],
                           y=gdf["Predicted Positive Rate"], marker_color="#EF4444"),
                ])
                fig.update_layout(
                    title=dict(text=f"Actual vs Predicted Rate by {group_name}", font=_TITLE_FONT),
                    yaxis_tickformat=".1%", barmode="group",
                    xaxis_tickfont=_AXIS_FONT, yaxis_tickfont=_AXIS_FONT,
                    legend=dict(font=_LEGEND_FONT),
                    height=400, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="{'success-box' if passes else 'warning-box'}">
            <strong>Clinical interpretation:</strong> Recall disparity is especially critical:
            if the model misses readmission risk for certain racial or age groups, those patients
            will be under-served by discharge planning. Large AUC differences across groups
            indicate the model has learned weaker patterns for some subpopulations, likely due to
            data imbalance or underrepresentation in training.
            </div>
            """, unsafe_allow_html=True)


#  TAB 6: PATIENT SCREENER

with tab_screener:
    st.markdown(
        "<h2 style='color:#0F3460;'>Patient Screener</h2>"
        "<p style='color:#475569; margin-top:0;'>"
        "Enter patient clinical details to estimate 30-day readmission probability "
        "with real-time SHAP explanation of the key driving factors.</p>",
        unsafe_allow_html=True,
    )

    if not _ready:
        st.info("Run the pipeline first to use the live screener.")
        st.stop()

    feature_names = artifacts["feature_names"]
    scaler        = artifacts["scaler"]
    opt_threshold = _load_threshold()

    st.markdown(f"""
    <div class="explain-box">
    <strong>How this works:</strong> The form collects the same clinical inputs used during training.
    Inputs are transformed through the same feature engineering pipeline and scaled before the
    trained LightGBM model produces a readmission probability score. Patients above the optimised
    threshold of <strong>{opt_threshold:.3f}</strong> (Youden's J) are flagged as high risk.
    The SHAP chart then explains which specific factors drove the prediction for this individual patient.
    </div>
    """, unsafe_allow_html=True)

    _AGE_MAP = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }

    with st.form("prediction_form"):
        st.markdown("### Patient Demographics & Admission")
        col1, col2, col3 = st.columns(3)
        with col1:
            age_bucket     = st.selectbox("Age Group", list(_AGE_MAP.keys()), index=6)
            gender         = st.selectbox("Gender", ["Female", "Male"])
            admission_type = st.selectbox("Admission Type",
                                          ["Emergency", "Urgent", "Elective", "Other"])
        with col2:
            time_in_hospital = st.slider("Days in Hospital", 1, 14, 4)
            number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
            num_procedures   = st.slider("Number of Procedures", 0, 6, 1)
        with col3:
            num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 45)
            num_medications    = st.slider("Number of Medications", 1, 81, 16)

        st.markdown("### Prior Utilisation History")
        col1, col2, col3 = st.columns(3)
        with col1:
            number_outpatient = st.number_input("Outpatient Visits (Past Year)", 0, 42, 0)
        with col2:
            number_emergency  = st.number_input("Emergency Visits (Past Year)", 0, 76, 0)
        with col3:
            number_inpatient  = st.number_input("Inpatient Visits (Past Year)", 0, 21, 0)

        st.markdown("### Diabetes Management")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            a1c_result   = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
        with col2:
            glu_serum    = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
        with col3:
            insulin      = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        with col4:
            diabetes_med = st.selectbox("On Diabetes Medication?", ["Yes", "No"])

        col1, col2 = st.columns(2)
        with col1:
            med_changed           = st.selectbox("Medication Changed?", ["No", "Yes"])
        with col2:
            primary_diag_diabetes = st.selectbox("Primary Diagnosis: Diabetes?", ["No", "Yes"])

        st.markdown("### Additional Medication Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_active_meds   = st.slider("Active Medications", 0, 23, 10)
        with col2:
            med_change_count  = st.slider("Medication Changes (Up/Down)", 0, 23, 1)
        with col3:
            num_comorbidities = st.slider("Distinct Comorbidity Groups", 1, 3, 2)

        submitted = st.form_submit_button(
            "Predict Readmission Risk", type="primary", use_container_width=True
        )

    if submitted:
        age_numeric        = _AGE_MAP[age_bucket]
        total_visits_prior = int(number_outpatient) + int(number_emergency) + int(number_inpatient)
        a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
        glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}

        patient_features = {
            "time_in_hospital":        time_in_hospital,
            "num_lab_procedures":      num_lab_procedures,
            "num_procedures":          num_procedures,
            "num_medications":         num_medications,
            "number_outpatient":       int(number_outpatient),
            "number_emergency":        int(number_emergency),
            "number_inpatient":        int(number_inpatient),
            "number_diagnoses":        number_diagnoses,
            "num_active_medications":  num_active_meds,
            "medication_change_count": med_change_count,
            "has_insulin":             int(insulin != "No"),
            "insulin_changed":         int(insulin in ["Up", "Down"]),
            "total_visits_prior":      total_visits_prior,
            "high_utilizer":           int(total_visits_prior >= 3),
            "diabetes_primary":        int(primary_diag_diabetes == "Yes"),
            "num_comorbidities":       num_comorbidities,
            "long_stay":               int(time_in_hospital > 7),
            "emergency_admission":     int(admission_type == "Emergency"),
            "age_numeric":             age_numeric,
            "med_changed":             int(med_changed == "Yes"),
            "on_diabetes_med":         int(diabetes_med == "Yes"),
            "a1c_ordinal":             a1c_map[a1c_result],
            "glu_ordinal":             glu_map[glu_serum],
            "gender_male":             int(gender == "Male"),
        }

        input_df = pd.DataFrame([patient_features])
        for feat in feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0
        input_df  = input_df[feature_names]
        X_input   = scaler.transform(input_df)
        risk_prob = float(model.predict_proba(X_input)[0][1])
        flagged   = risk_prob >= opt_threshold

        st.markdown("---")
        st.markdown("### Prediction Result")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            gauge_color = ("#DC2626" if flagged else
                           "#D97706" if risk_prob >= opt_threshold * 0.6 else
                           "#059669")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_prob * 100,
                title={"text": "30-Day Readmission Risk (%)",
                       "font": {"color": "#0F3460", "size": 15, "family": "Inter, sans-serif"}},
                number={"suffix": "%", "valueformat": ".1f",
                        "font": {"color": "#0F3460", "size": 36, "family": "Inter, sans-serif"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569",
                              "tickfont": {"color": "#334155", "size": 11}},
                    "bar":  {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "#F1F5F9",
                    "borderwidth": 1, "bordercolor": "#CBD5E1",
                    "steps": [
                        {"range": [0,   opt_threshold * 60 * 100], "color": "#DCFCE7"},
                        {"range": [opt_threshold * 60 * 100, opt_threshold * 100], "color": "#FEF9C3"},
                        {"range": [opt_threshold * 100, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "#DC2626", "width": 3},
                        "thickness": 0.80,
                        "value": opt_threshold * 100,
                    },
                },
            ))
            fig.update_layout(
                height=360, paper_bgcolor=_BG, plot_bgcolor=_BG,
                margin=dict(t=30, b=10),
                font=dict(color="#1E293B", family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            risk_level = "High" if flagged else "Low"
            if flagged:
                st.error(f"**HIGH RISK**\nAbove threshold ({opt_threshold:.3f})")
            else:
                st.success(f"**LOW RISK**\nBelow threshold ({opt_threshold:.3f})")
            st.metric("Risk Score", f"{risk_prob:.1%}")

        with col3:
            st.metric("Risk Category",       risk_level)
            st.metric("Decision Threshold",  f"{opt_threshold:.3f}")
            st.metric("Days in Hospital",    str(time_in_hospital))

        st.markdown(f"""
        <div class="explain-box">
        <strong>Reading the result:</strong> The gauge shows estimated probability of 30-day readmission.
        The red dashed line marks the optimised threshold ({opt_threshold:.3f}).
        Patients above this are flagged for discharge planning and follow-up.
        The threshold was selected via Youden's J statistic to maximise sensitivity + specificity
        simultaneously, improving recall from 1.2% (default 0.5) to 45.7%.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Why This Prediction?")
        try:
            import shap as shap_lib
            model_type = type(model).__name__
            if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
                explainer = shap_lib.TreeExplainer(model)
                sv = explainer.shap_values(X_input)
                if isinstance(sv, list):
                    sv = sv[1]
            else:
                explainer = shap_lib.LinearExplainer(model, X_input)
                sv = explainer.shap_values(X_input)
                if isinstance(sv, list):
                    sv = sv[1]

            contrib_df = pd.DataFrame({
                "Feature":    feature_names,
                "SHAP Value": sv[0],
                "Abs SHAP":   np.abs(sv[0]),
            }).sort_values("Abs SHAP", ascending=False).head(12)

            fig = go.Figure(go.Bar(
                x=contrib_df["SHAP Value"],
                y=contrib_df["Feature"],
                orientation="h",
                marker_color=["#EF4444" if v > 0 else "#10B981"
                              for v in contrib_df["SHAP Value"]],
            ))
            fig.update_layout(
                title=dict(text="Top 12 Factors Driving This Prediction", font=_TITLE_FONT),
                xaxis_title="Impact on Readmission Risk (SHAP Value)",
                xaxis_title_font=_AXIS_FONT,
                xaxis_tickfont=_AXIS_FONT,
                yaxis_tickfont=_AXIS_FONT,
                height=420,
                margin=dict(l=220),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

            top_risk = contrib_df[contrib_df["SHAP Value"] > 0].head(3)
            top_safe = contrib_df[contrib_df["SHAP Value"] < 0].head(3)
            risk_factors = ", ".join(
                [f"<strong>{r['Feature']}</strong>" for _, r in top_risk.iterrows()]
            ) if len(top_risk) else "none identified"
            safe_factors = ", ".join(
                [f"<strong>{r['Feature']}</strong>" for _, r in top_safe.iterrows()]
            ) if len(top_safe) else "none identified"

            st.markdown(f"""
            <div class="insight-box">
            <strong>Plain-language explanation:</strong><br><br>
            Main factors <strong style="color:#EF4444;">increasing</strong> readmission
            risk for this patient: {risk_factors}.<br><br>
            Main factors <strong style="color:#10B981;">decreasing</strong> readmission
            risk: {safe_factors}.<br><br>
            Red bars push the prediction toward higher risk; green bars push lower.
            Bar length indicates strength of influence for this specific patient.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")


#  TAB 7: ROI SIMULATOR

with tab_roi:
    # ── Constants from actual test set (Youden threshold = 0.1517) ────────────
    TEST_TP, TEST_FP, TEST_FN, TEST_TN = 1033, 4718, 1230, 12888
    TEST_POS   = TEST_TP + TEST_FN          # 2263 actual positive cases
    TEST_NEG   = TEST_TN + TEST_FP          # 17606 actual negative cases
    TEST_SIZE  = TEST_TP + TEST_FP + TEST_FN + TEST_TN  # 19869 total test patients
    CURR_RECALL = TEST_TP / TEST_POS        # 0.4565 (45.7%)
    CURR_FPR    = TEST_FP / TEST_NEG        # 0.2680 (26.8%)

    st.markdown(
        "<h2 style='color:#0F3460;'>ROI Simulator</h2>"
        "<p style='color:#475569; margin-top:0;'>"
        "Translate model performance into real-world financial and clinical impact. "
        "Adjust hospital-specific parameters to see projected annual ROI.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="explain-box">
    Based on actual test set performance at the Youden-optimal threshold ({opt_t:.3f}):
    <strong>{TEST_TP:,} true positives · {TEST_FP:,} false positives ·
    {TEST_FN:,} missed readmissions</strong> out of {TEST_SIZE:,} test patients.<br>
    Sliders below scale these numbers to your hospital's annual volume.
    </div>
    """, unsafe_allow_html=True)

    # ── Parameter sliders (4 across) ─────────────────────────────────────────
    st.markdown("### Hospital Parameters")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        annual_volume     = st.slider(
            "Annual Patient Volume", 1_000, 100_000, 20_000, 1_000,
            help="Total diabetic patients admitted per year"
        )
    with col_b:
        readmission_cost  = st.slider(
            "Avg Readmission Cost ($)", 5_000, 30_000, 15_200, 200,
            help="Average cost per preventable 30-day readmission (US avg: ~$14k–$17k)"
        )
    with col_c:
        intervention_cost = st.slider(
            "Intervention Cost / Patient ($)", 50, 1_000, 250, 50,
            help="Cost of discharge planning, follow-up call, care coordination per flagged patient"
        )
    with col_d:
        prevention_rate   = st.slider(
            "Readmissions Prevented (%)", 10, 80, 40, 5,
            help="% of flagged true readmissions actually prevented (literature: 30–50%)"
        ) / 100

    # ── Compute ROI ───────────────────────────────────────────────────────────
    scale       = annual_volume / TEST_SIZE
    annual_TP   = TEST_TP * scale
    annual_FP   = TEST_FP * scale
    annual_FN   = TEST_FN * scale
    annual_TN   = TEST_TN * scale

    savings      = annual_TP * prevention_rate * readmission_cost
    costs        = (annual_TP + annual_FP) * intervention_cost
    net_benefit  = savings - costs
    roi_pct      = (net_benefit / costs * 100) if costs > 0 else 0
    prevented    = int(annual_TP * prevention_rate)
    flagged_pa   = int(annual_TP + annual_FP)

    st.markdown("---")

    # ── KPI cards row ��────────────────────────────────────────────────────────
    st.markdown("### Projected Annual Outcomes")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Net Annual Benefit",
        f"${net_benefit:,.0f}",
        delta="profitable" if net_benefit > 0 else "loss",
        delta_color="normal" if net_benefit > 0 else "inverse",
    )
    c2.metric("Total Savings",             f"${savings:,.0f}")
    c3.metric("Intervention Costs",        f"${costs:,.0f}")
    c4.metric("Return on Investment",      f"{roi_pct:.0f}%")
    c5.metric("Readmissions Prevented",    f"{prevented:,}")

    st.markdown("---")

    # ── ROI gauge + interpretation ────────────────────────────────────────────
    col_gauge, col_interp = st.columns([1, 1])

    with col_gauge:
        roi_gauge_color = "#059669" if net_benefit > 0 else "#EF4444"
        fig_roi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=roi_pct,
            title={"text": "Return on Investment (%)",
                   "font": {"color": "#0F3460", "size": 15, "family": "Inter, sans-serif"}},
            number={"suffix": "%", "valueformat": ".0f",
                    "font": {"color": "#0F3460", "size": 42, "family": "Inter, sans-serif"}},
            gauge={
                "axis": {"range": [-100, 500], "tickcolor": "#475569",
                          "tickfont": {"color": "#334155", "size": 11}},
                "bar":  {"color": roi_gauge_color, "thickness": 0.25},
                "bgcolor": "#F1F5F9",
                "borderwidth": 1, "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [-100, 0],  "color": "#FEE2E2"},
                    {"range": [0,   100], "color": "#FEF9C3"},
                    {"range": [100, 500], "color": "#DCFCE7"},
                ],
                "threshold": {
                    "line": {"color": "#059669", "width": 3},
                    "thickness": 0.80, "value": 100,
                },
            },
        ))
        fig_roi.update_layout(
            height=320, paper_bgcolor=_BG, plot_bgcolor=_BG,
            margin=dict(t=40, b=10),
            font=dict(color="#1E293B", family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    with col_interp:
        status_icon = "✅" if net_benefit > 0 else "⚠️"
        box_class   = "success-box" if net_benefit > 0 else "warning-box"
        st.markdown(f"""
        <div class="{box_class}">
        <strong>{status_icon} Financial Summary</strong><br><br>
        At <strong>{annual_volume:,} patients/year</strong>,
        the model flags <strong>{flagged_pa:,} patients</strong> for
        follow-up intervention ({CURR_RECALL:.1%} recall, {CURR_FPR:.1%} FPR).<br><br>
        Of those, <strong>{prevented:,} readmissions are prevented</strong>, saving
        <strong>${savings:,.0f}</strong> in readmission costs at a total intervention
        spend of <strong>${costs:,.0f}</strong>.<br><br>
        <strong>Net annual benefit: ${net_benefit:,.0f} ({roi_pct:.0f}% ROI)</strong><br><br>
        The 100% ROI line (green dashed on gauge) = break-even on intervention spend.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explain-box" style="font-size:0.83rem; margin-top:8px;">
        <strong>Assumptions:</strong><br>
        · Intervention = discharge planning, care coordinator follow-up call<br>
        · Literature prevention rates: 30–50% (Hernandez et al., 2010; Hansen et al., 2011)<br>
        · Readmission cost: DRG-adjusted average (~$14k–$17k per event)<br>
        · Model recall = {CURR_RECALL:.1%}, FPR = {CURR_FPR:.1%} at Youden threshold {opt_t:.3f}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Threshold sensitivity: ROI across operating points ────────────────────
    st.markdown("### ROI Sensitivity Across Threshold Operating Points")
    st.markdown("""
    <div class="explain-box">
    How does ROI change as you move the classification threshold?
    Lowering the threshold flags more patients (higher recall = more TP, but also more FP = more intervention costs).
    Raising it reduces false positives but misses more readmissions.
    The ★ star marks the current Youden-optimal operating point.
    </div>
    """, unsafe_allow_html=True)

    if _ready:
        # Find the tuned model ROC curve
        best_rd = next(
            (rd for rd in roc_data if "Tuned" in rd.get("model", "")),
            roc_data[0]
        )
        fpr_arr = np.array(best_rd["fpr"])
        tpr_arr = np.array(best_rd["tpr"])

        # Compute ROI at each point on the ROC curve
        roi_curve, net_curve = [], []
        for fpr_i, tpr_i in zip(fpr_arr, tpr_arr):
            tp_i   = tpr_i * TEST_POS * scale
            fp_i   = fpr_i * TEST_NEG * scale
            sav_i  = tp_i * prevention_rate * readmission_cost
            cos_i  = (tp_i + fp_i) * intervention_cost
            roi_curve.append((sav_i - cos_i) / cos_i * 100 if cos_i > 0 else 0)
            net_curve.append(sav_i - cos_i)

        fig_sens = go.Figure()
        # Net benefit area chart
        fig_sens.add_trace(go.Scatter(
            x=tpr_arr, y=roi_curve,
            mode="lines",
            line=dict(color=ACCENT, width=2.5),
            name="ROI (%)",
            fill="tozeroy",
            fillcolor="rgba(29,111,164,0.08)",
        ))
        # Break-even line
        fig_sens.add_hline(
            y=0, line_dash="dash", line_color="#94A3B8",
            annotation_text="Break-even (0% ROI)",
            annotation_position="bottom right",
            annotation_font=dict(color="#475569", size=11),
        )
        # Current operating point star
        fig_sens.add_trace(go.Scatter(
            x=[CURR_RECALL],
            y=[roi_pct],
            mode="markers",
            marker=dict(size=16, color="#DC2626", symbol="star",
                        line=dict(width=2, color="#FFFFFF")),
            name=f"Youden threshold (recall={CURR_RECALL:.1%}, ROI={roi_pct:.0f}%)",
        ))
        fig_sens.update_layout(
            title=dict(text="ROI (%) vs. Model Recall: All Threshold Operating Points",
                       font=_TITLE_FONT),
            xaxis=dict(
                title="Model Recall (True Positive Rate)",
                tickformat=".0%",
                title_font=_AXIS_FONT, tickfont=_AXIS_FONT,
                range=[0, 1],
            ),
            yaxis=dict(
                title="Return on Investment (%)",
                title_font=_AXIS_FONT, tickfont=_AXIS_FONT,
            ),
            legend=dict(x=0.01, y=0.99, font=_LEGEND_FONT),
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        # Net benefit curve
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=tpr_arr, y=[v / 1e6 for v in net_curve],
            mode="lines",
            line=dict(color="#059669", width=2.5),
            name="Net Benefit ($M)",
            fill="tozeroy",
            fillcolor="rgba(5,150,105,0.08)",
        ))
        fig_net.add_hline(
            y=0, line_dash="dash", line_color="#94A3B8",
            annotation_text="Break-even",
            annotation_position="bottom right",
            annotation_font=dict(color="#475569", size=11),
        )
        fig_net.add_trace(go.Scatter(
            x=[CURR_RECALL],
            y=[net_benefit / 1e6],
            mode="markers",
            marker=dict(size=16, color="#DC2626", symbol="star",
                        line=dict(width=2, color="#FFFFFF")),
            name=f"Youden (${net_benefit/1e6:.1f}M)",
        ))
        fig_net.update_layout(
            title=dict(text="Net Annual Benefit ($M) vs. Model Recall",
                       font=_TITLE_FONT),
            xaxis=dict(
                title="Model Recall (True Positive Rate)",
                tickformat=".0%",
                title_font=_AXIS_FONT, tickfont=_AXIS_FONT,
                range=[0, 1],
            ),
            yaxis=dict(
                title="Net Annual Benefit ($M)",
                title_font=_AXIS_FONT, tickfont=_AXIS_FONT,
            ),
            legend=dict(x=0.01, y=0.99, font=_LEGEND_FONT),
            height=380,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_net, use_container_width=True)

    else:
        st.info("Run the pipeline to see threshold sensitivity analysis.")

    st.markdown("---")

    # ── Strategy comparison table ─────────────────────────────────────────────
    st.markdown("### Strategy Comparison")

    pos_rate = TEST_POS / TEST_SIZE  # ~11.4%

    # Universal intervention
    univ_flagged  = annual_volume
    univ_TP       = pos_rate * annual_volume
    univ_cost     = univ_flagged * intervention_cost
    univ_savings  = univ_TP * prevention_rate * readmission_cost
    univ_net      = univ_savings - univ_cost
    univ_roi      = (univ_net / univ_cost * 100) if univ_cost > 0 else 0

    comp_table = pd.DataFrame([
        {
            "Strategy":                  "No Screening",
            "Patients Flagged / Year":   "0",
            "True Readmissions Caught":  "0",
            "Annual Intervention Cost":  "$0",
            "Annual Savings":            "$0",
            "Net Annual Benefit":        "$0",
            "ROI":                       "n/a",
        },
        {
            "Strategy":                  "Universal Intervention (flag all)",
            "Patients Flagged / Year":   f"{int(univ_flagged):,}",
            "True Readmissions Caught":  f"{int(univ_TP):,}",
            "Annual Intervention Cost":  f"${univ_cost:,.0f}",
            "Annual Savings":            f"${univ_savings:,.0f}",
            "Net Annual Benefit":        f"${univ_net:,.0f}",
            "ROI":                       f"{univ_roi:.0f}%",
        },
        {
            "Strategy":                  "This Model (Youden Threshold)",
            "Patients Flagged / Year":   f"{flagged_pa:,}",
            "True Readmissions Caught":  f"{int(annual_TP):,}",
            "Annual Intervention Cost":  f"${costs:,.0f}",
            "Annual Savings":            f"${savings:,.0f}",
            "Net Annual Benefit":        f"${net_benefit:,.0f}",
            "ROI":                       f"{roi_pct:.0f}%",
        },
    ])
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    frac_flagged = flagged_pa / annual_volume if annual_volume > 0 else 0
    st.markdown(f"""
    <div class="insight-box">
    <strong>Key insight:</strong> Universal intervention achieves 100% recall but generates
    enormous unnecessary intervention costs (~{pos_rate:.1%} of patients are actually at risk,
    but 100% are intervened). This model selectively flags
    <strong>{frac_flagged:.1%} of patients</strong> while catching
    <strong>{CURR_RECALL:.1%} of actual readmissions</strong>, a far more efficient allocation
    of limited discharge planning and care coordination resources.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.83rem; padding:8px;'>"
    "Readmission Risk AI &nbsp;·&nbsp; "
    "UCI Diabetes 130-US Hospitals Dataset &nbsp;·&nbsp; "
    "LightGBM · SHAP · Optuna · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
