# LinkedIn Post: Patient Readmission Prediction

*Copy and paste the post below. Add the GitHub repo link and Streamlit app link before posting.*

---

## Post (Long-Form)

🏥 **New Portfolio Project: Predicting 30-Day Hospital Readmissions with Machine Learning**

Every year, preventable hospital readmissions cost the US healthcare system billions of dollars and patients their health. The Centers for Medicare & Medicaid Services (CMS) financially penalises hospitals with excessive readmission rates. The question is: can we identify high-risk patients *before* they leave the hospital?

I built a full end-to-end ML system to do exactly that.

---

**The Dataset**
UCI Diabetes 130-US Hospitals (1999-2008), 99,343 patient encounters across 130 US hospitals. Target: 30-day readmission (~11.4% positive rate, ~8:1 class imbalance).

---

**The Pipeline**
- ICD-9 diagnosis code grouping into 20 clinical categories
- Medication aggregation (23 drug columns to 4 summary features)
- Clinical feature engineering: length of stay, comorbidity count, prior visit history
- SMOTE applied inside each CV fold via imblearn Pipeline (prevents synthetic-sample leakage into validation folds)
- 4-model comparison (Logistic Regression, Random Forest, XGBoost, LightGBM) with 5-fold stratified CV
- Optuna hyperparameter tuning (50 trials)
- SHAP TreeExplainer for global + individual-level explainability
- Fairness audit across race and age groups

---

**The Insight That Changed Everything**

At the default 0.5 threshold, recall was 1.2%, nearly useless for clinical screening.

After applying **Youden's J threshold optimisation** (threshold = 0.1517):
- Recall jumped from **1.2% to 45.7%**
- AUC: **0.6323** (consistent with Strack et al. 2014 benchmarks on this dataset)
- Specificity: **73.2%**

The model now correctly flags **nearly half of actual readmissions** before discharge, giving clinical teams the window to intervene.

---

**Top Risk Factors (SHAP)**
1. Length of hospital stay
2. Patient age
3. Number of procedures
4. Number of diagnoses (comorbidities)
5. Prior inpatient visit history

---

**The Financial Case**
Hospital administrators do not speak AUC. They speak dollars. So I built a financial layer on top of the model.

The CMS Hospital Readmissions Reduction Program (HRRP) penalises hospitals with excessive readmission rates by up to 3% of total Medicare payments. For a mid-size hospital, that is millions per year.

The ROI Simulator lets you plug in your hospital's patient volume, cost assumptions, and prevention rate to get a projected net annual benefit. At default assumptions ($15,000 per preventable readmission, $150 per intervention call, 30% prevention rate), the model pays for itself many times over, because the cost of a false positive is one phone call, while the cost of a missed readmission is a $15,000 admission plus potential CMS penalties.

One insight from the simulator: the Youden threshold that maximises clinical recall (0.1517) is not necessarily the threshold that maximises net financial return. The ROI curve lets teams find their own operating point depending on intervention capacity and cost structure.

---

**The Dashboard**
Built a 7-tab Streamlit app covering:
- Model performance with interactive ROC curves and confusion matrix
- SHAP explainability (global beeswarm + individual waterfall)
- Fairness analysis across demographic subgroups
- Live prediction interface: enter patient data, get a risk score + plain-language explanation
- ROI Simulator: model the financial return of deploying the system at your hospital

---

**What I would do differently in production:**
- Use temporally ordered train/test splits across the 10-year dataset
- Re-run fairness analysis at the optimal threshold
- Prospective validation on current patient cohort before deployment
- FHIR API integration for live EHR data

I've documented everything transparently because honest ML is the only kind that should be deployed in healthcare.

🔗 **GitHub:** https://github.com/Folarinosuolale/patient-readmission
🚀 **Live Demo:** [link]

---

#MachineLearning #HealthcareAI #DataScience #Python #LightGBM #SHAP #Streamlit #MLEngineering #ClinicalAI #ReadmissionPrediction #ResponsibleAI

---

## Short Version (for lower engagement / repost)

🏥 Just shipped a machine learning system for predicting 30-day hospital readmissions.

The catch: at default settings, recall was 1.2%. After threshold optimisation (Youden's J), recall jumped to **45.7%**, flagging nearly half of actual readmissions before discharge.

Built on 99,343 real patient encounters with full SHAP explainability, fairness auditing across demographic groups, and a 7-tab Streamlit dashboard for clinical teams.

Key insight: picking the right threshold matters more than picking the right model.

🔗 https://github.com/Folarinosuolale/patient-readmission | 🚀 [Live demo link]

#MachineLearning #HealthcareAI #DataScience #SHAP #Streamlit
