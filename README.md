#  Loan Default Risk Prediction

This project uses a **Logistic Regression** model to predict whether a customer is likely to default on a loan, based on financial and behavioral features. The goal is to assist financial institutions in early identification of high-risk clients.

---

##  Problem Statement
Financial distress and loan default are major concerns for lenders. This project predicts the probability of default using real-world anonymized credit data from the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) competition.

---

## Dataset
- Source: Kaggle – Give Me Some Credit
- File Used: `cs-training.csv`
- Target variable: `SeriousDlqin2yrs` (1 = default, 0 = non-default)
- Total records: ~150,000
- Features include:
  - Credit utilization
  - Late payments
  - Income
  - Debt ratio
  - Age and dependents

---

##  Tools & Libraries
- Python (Jupyter Notebook)
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` for ML modeling
- `shap` for explainability

---

##  ML Pipeline Steps
1. Data Cleaning (handling missing income & dependents)
2. Feature Selection and Scaling
3. Train-test split
4. Model Training (Logistic Regression)
5. Model Evaluation (Accuracy, Precision, Recall, Confusion Matrix)
6. Explainability with SHAP to understand feature impact

---

##  Evaluation Results

| Metric     | Value     |
|------------|-----------|
| Accuracy   | ~93%      |
| Precision  | ~55%      |
| Recall     | ~3.6%     |

* Recall is low due to class imbalance (only ~6.6% of samples are defaulters). This can be improved with advanced techniques like SMOTE or XGBoost.

---

##  Top Features Influencing Default
- Revolving credit utilization
- Number of times 90+ days late
- Debt ratio
- Monthly income
- Number of open credit lines

---

##  Files Included
- `loan_default_risk_analysis.ipynb` → Final notebook with full code
- `cs-training.csv` → Dataset (optional if large)
- `README.md` → This documentation


##  Project Highlights

- Real-world ML project on financial data  
- Clear step-by-step logic with professional metrics  
- Explainable AI using SHAP for model transparency  
