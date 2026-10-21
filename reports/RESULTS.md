# Evaluation results

Selected using training-only CV ROC-AUC: **Logistic Regression**.

| Model | CV ROC-AUC | Test accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.9958 | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 |
| SVM | 0.9949 | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| XGBoost | 0.9921 | 0.9561 | 1.0000 | 0.8810 | 0.9367 | 0.9967 |
| RandomForest | 0.9889 | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9947 |

Test set: 114 observations (72 benign, 42 malignant). Selected model: 71 true negatives, 39 true positives, 1 false positive and 3 false negatives. Malignant is the positive class.

SVM has higher test accuracy in this split; selection remains Logistic Regression because the rule was fixed to training-only CV ROC-AUC. The test set was not used to revise selection.

Ten inference/data checks passed. All four notebook code cells executed successfully in a shared Python namespace; separate Jupyter kernel startup was restricted by the execution environment. No notebook cell errors.

This is a small retrospective educational evaluation, not evidence of clinical readiness.