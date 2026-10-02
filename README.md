# Telco Customer Churn Prediction — Professional Data Science Project

End-to-end customer churn prediction using the IBM Telco Customer Churn dataset.

## Workflow

Business Problem → Data Loading → Data Audit → Cleaning → EDA → Statistical Testing → Feature Engineering → Stratified Split → Preprocessing → Baseline Models → Cross-Validation → Hyperparameter Tuning → SMOTENC → Balanced Models → Threshold Optimization → Final Evaluation → SHAP → New Customer Prediction → Model Export

## Repository structure

```text
telco-customer-churn/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── Telco_Customer_Churn_Professional_Report_v2.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── prediction.py
├── models/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── predictions/
└── reports/
    └── Telco_Customer_Churn_Report.md
```

## Dataset
- **Source:** [Kaggle — Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- **Records:** 7,043 customers
- **Columns:** 21 in the raw dataset
- **Target:** `Churn` (`Yes` / `No`)
- **Feature domains:** customer profile, tenure, subscribed services, contract, billing, and payment method.
## Processed Data
- [`Datasets/Processed/Processed_Train.csv`](Datasets/Processed/Processed_Train.csv)
- [`Datasets/Processed/Processed_Test.csv`](Datasets/Processed/Processed_Test.csv)

## Analytical and modeling findings

### Modeling workflow
The notebook follows data understanding and cleaning, EDA/statistical analysis, feature engineering, train/test preparation, preprocessing, baseline/model comparison, evaluation, hyperparameter tuning, SHAP interpretation, threshold-based classification, and deployment preparation.

### Feature preparation
- Numerical and categorical inputs are processed separately.
- Numerical fields are scaled with the fitted scaler.
- Categorical fields are one-hot encoded.
- Feature engineering includes `AvgChargesPerMonth`, `TenureGroup`, and `NumServices`.
- The final transformed feature space contains **51 features**.
- The test set contains **1,409 rows**.

### Final model selection
The notebook selects a tuned **XGBoost (`XGBClassifier`)** estimator using `RandomizedSearchCV`.

Best parameters recorded in the notebook:
| Parameter | Selected value |
|---|---:|
| `subsample` | 0.9 |
| `reg_lambda` | 2 |
| `reg_alpha` | 0.1 |
| `n_estimators` | 200 |
| `min_child_weight` | 1 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `gamma` | 0 |
| `colsample_bytree` | 0.8 |

### Threshold decision
- **Decision threshold:** `0.45`
- A customer is flagged as predicted churn when the model's churn probability is at least 0.45; otherwise, the prediction is `No Churn`.
- This threshold is the notebook's selected operating point. The business should validate the trade-off between missed churners and unnecessary retention outreach before using it operationally.

### Example predictions verified in the notebook
| Example | Predicted label | Churn probability | Threshold |
|---|---|---:|---:|
| First held-out test customer | No Churn | 0.31% | 0.45 |
| Constructed new-customer example | Churn | 76.52% | 0.45 |

These are individual examples, not overall model-performance metrics.

## Explainability
The notebook creates a `shap.TreeExplainer` for the tuned XGBoost model and calculates SHAP values for the test set:
- SHAP array: **1,409 × 51**
- Feature dimensions were checked against the model input dimensions.
- Global SHAP summary and bar plots are generated.
- A local waterfall explanation is generated for an example customer and the constructed new-customer example.

SHAP values explain how features contribute to a particular model output; they do not establish that a feature causes churn. Review the notebook's plots when communicating specific top drivers—the README does not assign feature rankings that are not available as readable output here.

## Decisions and practical implications
1. **Use the tuned XGBoost estimator as the notebook's final model** for the demonstrated prediction workflow.
2. **Keep the 0.45 threshold explicit and configurable.** Revisit it using business costs and validation data rather than treating it as universally optimal.
3. **Use probabilities to prioritize human review**, not as a guarantee that a customer will churn.
4. **Use SHAP explanations alongside customer context** to help analysts understand individual predictions.
5. **Validate before deployment:** monitor precision, recall, F1, ROC-AUC, confusion matrix, calibration, and performance across relevant customer groups on fresh data.
6. **Avoid automatic customer actions** based only on a prediction. Retention outreach and offers should be reviewed against customer context, cost, and fairness considerations.

## Saved deployment artifact
The notebook saves a deployment bundle at:

```text
models/churn_pipeline.pkl
```

The bundle contains:
- `model`
- `scaler`
- `encoder`
- `feature_names`
- `threshold`

## Final candidate reported by the notebook

Balanced Tuned XGBoost, evaluated on an untouched test set with an OOF-selected threshold of 0.45.

- Accuracy ≈ 0.769
- Precision ≈ 0.553
- Recall ≈ 0.655
- F1 ≈ 0.600
- ROC-AUC ≈ 0.819
- PR-AUC ≈ 0.614

These values are descriptive of this notebook's evaluation and should not be treated as guarantees on future data.


The notebook reloads the bundle and verifies the constructed customer's prediction again (76.52% churn probability, threshold 0.45).

## Limitations
- The example predictions do not describe overall test-set performance.
- A selected threshold should be justified with a documented validation trade-off.
- Historical customer patterns may not reflect future behavior or a different telecom provider.
- SHAP explains model behavior, not causal effects.
- The model and preprocessing artifacts must be kept version-compatible; retraining or changing feature engineering requires regenerating the bundle.

---
**Project:** Customer Churn Prediction System using the IBM Telco Customer Churn dataset  
**Model:** Tuned XGBoost  
**Decision threshold:** 0.45
