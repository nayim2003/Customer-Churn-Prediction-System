
# 📚 Detailed Final Analytical Interpretation

## 1. Data Understanding and Quality

The dataset contains **7,043 customer records and 21 raw columns**. `customerID` is treated as an identifier and excluded from predictive modeling. `TotalCharges` requires explicit numeric conversion because the raw column is stored as text. After conversion, **11 missing values** are identified; all belong to customers with `tenure = 0`, supporting the notebook's domain-based decision to replace them with zero.

The cleaned dataset contains **7,043 observations and 20 analytical columns with zero remaining missing values**.

## 2. EDA — Main Findings

### Target imbalance
Churn = Yes represents **26.54%** of customers. This establishes the need for minority-class-aware evaluation.

### Contract
Month-to-month customers have a **42.71% observed churn rate**, compared with **11.27%** for one-year contracts and **2.83%** for two-year contracts. This is the strongest clearly demonstrated categorical relationship in the notebook's EDA.

### Tenure
The retained group has a median tenure of **38 months**, while the churned group has a median of only **10 months**. The distributions are therefore strongly separated toward shorter tenure among churners.

### Monthly charges
The churned population has a higher central tendency for MonthlyCharges, with significant distributional separation confirmed by the Mann–Whitney test.

### Service and payment configuration
Fiber-optic service and electronic-check payment show visually large churn components. These patterns are useful for segmentation but are not causal conclusions.

### Correlation structure
`tenure` and `TotalCharges` are strongly correlated (**≈0.83**), while `MonthlyCharges` and `TotalCharges` have a moderate-to-strong positive correlation (**≈0.65**). The model therefore receives partially redundant information, which tree-based models can still exploit.

## 3. Statistical Testing

The notebook applies tests that match the variable types:

- **Chi-square test** for categorical Contract × Churn.
- **Mann–Whitney U tests** for numerical variables where the churn groups are compared without assuming normality.

The resulting p-values are extremely small, so the null hypotheses of independence/equal distributional location are rejected. The findings support retaining these variables for predictive modeling.

## 4. Feature Engineering

Three derived variables are created:
- `AvgChargesPerMonth`
- `TenureGroup`
- `NumServices`

These features translate raw customer attributes into more directly interpretable lifecycle, service-breadth and charge-related representations.

## 5. Imbalance Handling

The training partition contains **4,139 non-churn and 1,495 churn customers** (73.46% / 26.54%). SMOTENC then creates a balanced training set of **4,139 / 4,139**.

This is an important methodological distinction: the model learns from synthetic minority examples, but the final test set remains at the original **26.5% churn prevalence**.

## 6. Model Development

The baseline comparison demonstrates why model selection requires multiple metrics. Logistic Regression has the highest baseline ROC-AUC (**0.8458**), while Decision Tree has the highest baseline F1 (**0.5989**) among those initial models.

The workflow then evaluates balanced and tuned tree-based models. The tuned balanced XGBoost achieves **CV F1 = 0.8369** during its 5-fold randomized search.

## 7. Final Model

The final candidate is the **Balanced Tuned XGBoost** with:

`n_estimators=200`  
`max_depth=8`  
`learning_rate=0.1`  
`subsample=0.9`  
`colsample_bytree=0.8`  
`min_child_weight=1`  
`gamma=0`  
`reg_alpha=0.1`  
`reg_lambda=2`

On the untouched test set at threshold 0.50, it achieves:
- F1 = **0.6000**
- ROC-AUC = **0.8189**
- PR-AUC = **0.6141**

## 8. Threshold Optimization

The selected threshold is **0.45**, determined from 5-fold out-of-fold predictions on the balanced training data.

At threshold 0.45 on the untouched test set:
- Accuracy ≈ **0.769**
- Precision ≈ **0.553**
- Recall ≈ **0.655**
- F1 ≈ **0.600**
- ROC-AUC = **0.8189**
- PR-AUC = **0.6141**

The confusion matrix is:
- TN = **837**
- FP = **198**
- FN = **129**
- TP = **245**

Thus, among the 374 actual churners in the test set, the selected threshold correctly identifies **245** and misses **129**.

## 9. Explainability

SHAP analysis shows that the model's predictions are driven substantially by contract, service configuration, payment method, charges and tenure-related variables.

`Contract_Month-to-month` appears as the strongest global contributor in the SHAP importance plot. The individual waterfall explanation demonstrates how multiple features combine for one customer's risk score.

## 10. Overall Analytical Conclusion

The notebook establishes a coherent predictive pipeline in which:

**customer characteristics → data quality → EDA → statistical evidence → feature engineering → stratified splitting → preprocessing → imbalance handling → model tuning → threshold optimization → final evaluation → SHAP explanation**

The strongest recurring signal across the analysis is the relationship between **customer lifecycle/tenure, contract structure, service/payment configuration and churn**.

The final model should therefore be interpreted as a **risk-ranking and classification tool**, not as a causal model of why customers leave.

## ***Md. Nayim Howlader***
### ***BSc (Honours), Department of Statistics,***
### ***Dhaka College, Dhaka.***
