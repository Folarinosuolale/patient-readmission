# Stakeholder Brief: Patient Readmission Prediction System

**For:** Hospital Administration, Clinical Leadership, Quality Improvement Teams
**Prepared by:** Folarin Osuolale, Data Scientist
**Date:** March 2026

---

## What This System Does

This tool uses machine learning to predict which diabetic patients are likely to be readmitted to hospital within 30 days of discharge. It analyses 24 clinical and demographic factors from the patient's admission record and produces a risk score between 0% and 100%.

Patients above the risk threshold (12.6%) are flagged for proactive follow-up planning before or immediately after discharge.

---

## Why This Matters

| Metric | Value |
|---|---|
| Average cost of a preventable readmission | ~$15,000 (CMS estimate) |
| CMS penalty for excessive readmissions | Up to 3% Medicare payment reduction |
| 30-day readmission rate in this dataset | 11.4% |
| Patients correctly flagged by this model | 45.7% of actual readmissions |

Catching nearly half of future readmissions before they happen gives clinical teams the opportunity to intervene through enhanced discharge planning, medication reconciliation, or scheduled follow-up calls.

---

## How It Works (Non-Technical)

1. **Input:** Patient's clinical record at time of discharge (age, diagnoses, medications, lab results, length of stay, prior visit history)
2. **Processing:** An AI model trained on 99,343 historical patient encounters analyses the record
3. **Output:** A risk score (%) and a plain-language explanation of which factors most influenced the score
4. **Decision:** Clinical staff use the score to prioritise follow-up resources

---

## Model Performance in Plain Language

Out of every 100 patients who will actually be readmitted within 30 days:
- **46 will be correctly flagged** as high risk
- **54 will be missed** (false negatives)

Out of every 100 patients the model flags as high risk:
- **18 will actually be readmitted** (true positives)
- **82 will not be readmitted** (false alarms)

**Clinical context:** A false alarm means a follow-up call or check-in that was not strictly necessary, a low-cost intervention. A missed readmission means a patient who needed support did not receive it, a high-cost outcome. The model is tuned to minimise missed readmissions.

---

## Key Risk Factors Identified

The model has identified the following as the strongest predictors of 30-day readmission:

1. **Length of hospital stay:** Longer admissions signal greater severity
2. **Patient age:** Older patients have higher baseline readmission risk
3. **Number of procedures:** More procedures indicate clinical complexity
4. **Number of diagnoses:** Multiple comorbidities increase risk
5. **Prior inpatient visits:** History of hospitalisation is the strongest utilisation predictor

---

## Fairness Considerations

The model was audited for potential bias across racial and age groups. Results indicate near-zero predicted positive rates across all groups at the default threshold, suggesting the model is conservative rather than selectively biased. However, this also highlights the need for threshold calibration per subgroup in a production deployment, a recommended next step for any clinical rollout.

---

## What This System Is NOT

- **Not a diagnostic tool:** It does not diagnose conditions or recommend treatments
- **Not a replacement for clinical judgment:** Risk scores are one input among many
- **Not a final decision maker:** A high score should prompt review, not automatic intervention
- **Not validated for all patient populations:** Trained on diabetic patients in US hospitals (1999--2008); performance on other populations requires validation

---

## Recommended Next Steps for Clinical Deployment

1. **Prospective validation:** Test on current patient cohort before full rollout
2. **EHR integration:** Connect to live patient data via FHIR API
3. **Workflow design:** Define escalation protocols for flagged patients
4. **Staff training:** Educate clinical staff on interpreting SHAP explanations
5. **Outcome monitoring:** Track actual vs. predicted readmissions monthly to detect model drift

---

*For technical details, see the full Project Report. For dashboard instructions, see the Dashboard Guide.*
