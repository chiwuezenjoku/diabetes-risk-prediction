# Diabetes Risk Prediction from Clinical Measurements

A first-pass machine learning model predicting diabetes risk from routine clinical measurements (glucose, BMI, insulin, blood pressure, and related features), using the Pima Indians Diabetes dataset.

## Problem

Can basic clinical measurements — the kind collected in a routine checkup — predict whether a patient has diabetes? And can this specific dataset be trusted enough to build something meaningful on it?

## Data

- Source: Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases)
- 768 patients, 8 clinical features (Glucose, BMI, Blood Pressure, Insulin, Skin Thickness, Pregnancies, Age, Diabetes Pedigree Function), 1 target (Outcome: diabetic / not diabetic)

## Data Cleaning

Several columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) contained biologically impossible zero values (e.g., BMI = 0), which were identified as disguised missing data rather than true measurements. Missingness ranged from under 1% (Glucose, BMI) to nearly 49% (Insulin). Missing values were replaced using each column's median, chosen deliberately over the mean for resistance to outlier skew, and over row-deletion to avoid losing large portions of the dataset — a documented tradeoff, particularly for Insulin, where heavy imputation likely understates its true relationship with the outcome.

## Exploratory Analysis

- Diabetic patients showed a clearly higher average Glucose (142 mg/dL) than non-diabetic patients (111 mg/dL), consistent with known clinical thresholds.
- Correlation analysis identified Glucose (0.49) and BMI (0.31) as the strongest linear predictors of diabetes outcome, with other features showing weaker but still clinically relevant relationships. All features were retained for modeling rather than dropped on correlation strength alone, since correlation only captures linear relationships and domain knowledge indicates some weaker-correlation features (e.g., Insulin) remain biologically important.

## Model v1 — Baseline Logistic Regression

- Logistic Regression, trained on an 80/20 train-test split
- **Accuracy: 75.3%**
- **Recall (diabetic class): 61.8%**

### Key Finding & Limitation

While overall accuracy looks reasonable, recall on the diabetic class is notably lower — the model misses roughly 38% of actual diabetic patients in the test set. Because the test set is imbalanced (99 non-diabetic vs. 55 diabetic patients), overall accuracy is inflated by strong performance on the larger, easier-to-classify healthy group, masking weaker performance on the smaller diabetic group — the group where a missed classification (false negative) carries the most real-world risk.

## Model v2 — Class-Weighted Logistic Regression

To directly address the class imbalance identified in v1, the model was retrained with `class_weight='balanced'`, which increases the training penalty for misclassifying the minority (diabetic) class.

| Metric | v1 (Baseline) | v2 (Balanced) | Change |
|---|---|---|---|
| Accuracy | 75.3% | 70.1% | -5.2 pts |
| Recall (diabetic class) | 61.8% | 70.9% | **+9.1 pts** |

### Decision: v2 is the preferred model

Although v2 has lower overall accuracy, it is the more clinically appropriate model for this problem. In a screening context, a false negative (telling a diabetic patient they are healthy) carries substantially more real-world harm than a false positive (an unnecessary follow-up test for a healthy patient). Prioritizing recall — even at the cost of some overall accuracy — better reflects the actual clinical stakes rather than optimizing for a single blended metric that treats both error types as equally costly.

## Model v3 — Random Forest

To test whether a more flexible, non-linear algorithm could recover relationships the linear logistic regression models might miss — particularly around Insulin, whose correlation (0.20) was suspected to be artificially suppressed by heavy median-imputation (~49% missing) — a Random Forest classifier was trained on the same data.

| Metric | v1: Baseline LogReg | v2: Balanced LogReg | v3: Random Forest |
|---|---|---|---|
| Accuracy | 75.3% | 70.1% | 74.7% |
| Recall (diabetic class) | 61.8% | 70.9% | 67.3% |

Random Forest produced a modest improvement over the baseline in recall, but did not outperform the class-weighted model. This is itself an informative result: three meaningfully different modeling approaches converging on a similar performance range suggests the ceiling here is set by the information content of the 8 available features, not by model choice — no algorithm can extract predictive signal that isn't present in the underlying data.

### Feature Importance & the Insulin Hypothesis

```
Glucose                     0.263
BMI                         0.164
Age                         0.135
DiabetesPedigreeFunction    0.122
Insulin                     0.089
BloodPressure               0.084
SkinThickness               0.072
Pregnancies                 0.070
```

Insulin ranked 5th of 8 by Random Forest feature importance, an improvement over its near-bottom rank by simple correlation — mild support for the theory that heavy imputation suppressed (but did not eliminate) its true relationship with the outcome. The evidence is not strong enough to call Insulin a top predictor here, but it does support treating correlation-based feature screening with caution on heavily-imputed columns, rather than as a final word on a feature's real-world importance. Glucose remains the dominant predictor across all three methods used in this project (mean comparison, correlation, and feature importance) — a consistent, triangulated finding.

## Next Steps

- Try a lower missingness-tolerance threshold (e.g., drop Insulin as a column, or use a smarter imputation method like KNN or regression-based imputation) to test whether Insulin's true importance is higher than this analysis suggests
- Explore adjusting the classification threshold directly, for finer control over the precision/recall tradeoff than `class_weight` alone provides
- Investigate resampling techniques (e.g., SMOTE) as an alternative approach to the imbalance problem

## Tools

Python, pandas, scikit-learn, Jupyter Notebook
