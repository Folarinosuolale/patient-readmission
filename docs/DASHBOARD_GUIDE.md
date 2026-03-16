# Dashboard Guide: Patient Readmission Prediction System

**Author:** Folarin Osuolale
**Date:** March 2026

This guide explains how to use the interactive Streamlit dashboard for the Patient Readmission Prediction System.

---

## Launching the Dashboard

```bash
cd patient-readmission
streamlit run app/streamlit_app.py --server.port 8502
```

Open your browser at: **http://localhost:8502**

Or visit the deployed app at: **https://patient-readmission.streamlit.app** *(Streamlit Cloud)*

---

## Dashboard Structure

The dashboard has **7 tabs**, displayed as a horizontal navigation bar at the top of the page. There is no sidebar. Click any tab to switch between views instantly.

---

## Tab 1: Overview

**Purpose:** High-level summary of the system for new users and stakeholders.

**What you will see:**
- **Key Performance Metrics** -- four headline cards showing AUC, Recall, Precision, and F1 score at the optimal threshold (0.1517)
- **How the System Works** -- step-by-step explanation of the prediction pipeline (Input, Model, Output, Decision)
- **What the Model Is NOT** -- important disclaimers about clinical use

**How to use it:**
- Share this tab with clinical leadership or administrators for a quick system overview
- The metrics shown here reflect the **Youden's J optimal threshold (0.1517)**, not the default 0.5

---

## Tab 2: Patient Cohort

**Purpose:** Explore the training dataset distributions and understand the patient population.

**What you will see:**
- **Dataset summary** -- total encounters, positive rate, feature count
- **Age Distribution** -- histogram of patient ages across the 10-year dataset
- **Readmission by Race** -- bar chart comparing readmission rates across racial groups
- **Length of Stay vs Readmission** -- box plot comparing hospital stay duration between readmitted and non-readmitted patients
- **Number of Diagnoses vs Readmission** -- comparison of comorbidity burden between groups

**How to use it:**
- Use the charts to understand which patient subgroups are at elevated baseline risk
- Note that the dataset spans **1999--2008 across 130 US hospitals** (diabetic patients only)
- Readmission rate in this dataset: **11.4%**

---

## Tab 3: Model Performance

**Purpose:** Detailed technical evaluation of the LightGBM model.

**What you will see:**
- **Model Comparison Table** -- AUC, Recall, and F1 for all 4 base models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- **ROC Curves** -- all models plotted together; AUC shown in legend
- **Threshold Analysis** -- side-by-side comparison of default (0.5) vs optimal (0.1517) threshold performance
- **Confusion Matrix** -- interactive heatmap at the optimal threshold
  - TN = 12,888 | FP = 4,718
  - FN = 1,230 | TP = 1,033

**Reading the Confusion Matrix:**

| Cell | Meaning | Clinical Impact |
|---|---|---|
| True Positive (TP) | Readmitted, correctly flagged | Patient receives proactive follow-up |
| False Positive (FP) | Not readmitted, but flagged | Unnecessary follow-up call (low cost) |
| False Negative (FN) | Readmitted, but missed | Patient receives no extra support (high cost) |
| True Negative (TN) | Not readmitted, correctly cleared | No action needed |

**Key insight:** The model is deliberately tuned to minimise false negatives (missed readmissions), accepting a higher false positive rate. The cost of a follow-up call is far lower than the cost of a preventable readmission (~$15,000).

---

## Tab 4: Explainability

**Purpose:** Understand *why* the model makes predictions -- both globally and for individual patients.

**What you will see:**
- **SHAP Summary Plot** (beeswarm) -- Each dot is one patient sample. Horizontal position = SHAP value (impact on prediction). Colour = feature value (red = high, blue = low). Features are ranked by mean |SHAP|.
- **SHAP Bar Plot** -- Mean absolute SHAP value for each feature (global feature importance)
- **Feature Importance Table** -- Sortable table of all features with their SHAP importance scores

**How to interpret SHAP values:**
- A **positive SHAP value** pushes the prediction toward *readmitted* (higher risk)
- A **negative SHAP value** pushes the prediction toward *not readmitted* (lower risk)
- The magnitude indicates how strongly that feature influenced the prediction

**Top 5 most influential features:**

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `time_in_hospital` | Longer stays indicate higher readmission risk |
| 2 | `age_numeric` | Older patients carry higher baseline risk |
| 3 | `num_procedures` | More procedures reflect greater clinical complexity |
| 4 | `number_diagnoses` | More diagnoses indicate higher comorbidity burden |
| 5 | `number_inpatient` | Prior inpatient visits are the strongest utilisation signal |

---

## Tab 5: Equity Audit

**Purpose:** Audit the model for potential bias across demographic subgroups.

**What you will see:**
- **Disparate Impact Ratios** -- for race and age groups, using the EEOC 4/5ths rule
- **Predicted Positive Rate by Group** -- bar chart comparing how often each group is flagged as high risk
- **Fairness Interpretation** -- explanation of what near-zero DI ratios mean in context

**Understanding the Fairness Results:**

The fairness analysis was run at the default 0.5 threshold, where the overall positive prediction rate is near zero (~1%). When almost no one is flagged, the ratio of lowest-to-highest group prediction rate also collapses to near zero -- this reflects the conservative threshold, not selective bias.

**Important caveat:** Fairness analysis at the Youden threshold (0.1517) is recommended before any clinical rollout. This is flagged as a recommended next step in the Project Report.

| DI Ratio | Interpretation |
|---|---|
| >= 0.80 | No disparate impact detected |
| < 0.80 | Potential disparate impact (EEOC 4/5ths rule) |
| ~0.00 | Near-zero positive rate (threshold artifact -- see note above) |

---

## Tab 6: Patient Screener

**Purpose:** Enter a patient's clinical features and receive a real-time readmission risk score.

**How to use it:**

1. Fill in the **Patient Information** form in the left column:
   - Age, time in hospital, number of procedures, number of diagnoses
   - Prior visit counts (outpatient, emergency, inpatient)
   - Lab results (A1C level, max glucose serum)
   - Medication flags (insulin, medication changes)
   - Admission type (Emergency vs. other)

2. Click **"Predict Readmission Risk"**

3. Review the results in the right column:
   - **Risk Gauge** -- visual dial showing the predicted probability (0--100%)
   - The gauge turns **red** above the optimal threshold (15.2%) and **green** below it
   - **Risk Classification** -- HIGH RISK or LOW RISK label with clinical guidance
   - **Top Contributing Factors** -- SHAP waterfall chart showing which features pushed the score up or down for this specific patient

**Clinical guidance:**

| Risk Score | Classification | Suggested Action |
|---|---|---|
| >= 15.2% | HIGH RISK | Flag for enhanced discharge planning and follow-up |
| < 15.2% | LOW RISK | Standard discharge pathway |

**Important:** This score is one input among many. Clinical staff should use their judgment alongside the model's output. The system is **not** a diagnostic tool and does **not** replace clinical assessment.

---

## Tab 7: ROI Simulator

**Purpose:** Model the financial return of deploying the risk prediction system at your hospital, based on your specific patient volume and cost assumptions.

**How to use it:**

1. Use the sliders in the left column to set your parameters:
   - **Annual patient volume** -- total annual diabetic patient encounters at your facility
   - **Readmission cost** -- estimated cost of one preventable readmission (default: $15,000)
   - **Intervention cost** -- cost of one follow-up call or post-discharge intervention (default: $150)
   - **Prevention rate** -- share of flagged readmissions actually prevented by intervention (default: 30%)

2. The right column updates in real time with:
   - **Annual Net Benefit** -- projected cost savings from prevented readmissions minus intervention costs
   - **Return on Investment** -- net benefit divided by total intervention cost, expressed as a percentage
   - **ROI vs Threshold chart** -- how net benefit changes as the operating threshold shifts; the current Youden point is marked with a star

**Key assumptions:**
- Model performance is extrapolated from the test set (TP = 1,033 of 19,869 encounters)
- The prevention rate represents additional prevented readmissions attributable to the intervention, above baseline
- Intervention costs apply to all flagged patients (true positives + false positives)

**Interpreting the ROI chart:**
- The curve shows how ROI changes as you trade off recall vs precision along the ROC curve
- Moving left (lower threshold) catches more readmissions but flags more false positives, increasing intervention cost
- The optimal clinical operating point balances these trade-offs

---

## Frequently Asked Questions

**Q: Why is the risk threshold 15.2% and not 50%?**
A: The default 50% threshold flags almost no patients (recall: 1.2%). The 15.2% threshold was selected using Youden's J statistic to maximise the balance of sensitivity and specificity, raising recall to 45.7% while maintaining 73.2% specificity.

**Q: What does it mean when the model flags a patient who does not get readmitted?**
A: This is a false positive. It means the patient was flagged for follow-up but did not actually return within 30 days. The cost of this is a follow-up call or check-in, a low-cost intervention. The model is deliberately tuned to accept these in order to catch more true readmissions.

**Q: Is the model validated on current patients?**
A: No. The model was trained on historical data from 1999--2008. Prospective validation on a current patient cohort is **required** before clinical deployment.

**Q: What patient population does this apply to?**
A: The model was trained exclusively on **diabetic patients** from US hospitals. Its performance on other patient populations (non-diabetic, paediatric, non-US healthcare systems) has not been validated.

---

*For technical pipeline details, see the Project Report. For a non-technical summary, see the Stakeholder Brief.*
