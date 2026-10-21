# CodeAlpha_DiseasePrediction

CodeAlpha Machine Learning Task 4: Disease Prediction from Medical Data.

Classify breast tumors as benign or malignant using SVM, Logistic Regression, Random Forest and XGBoost. Educational demonstration only, not a clinical diagnosis tool.

## Run
Use Python 3.11 or 3.12. Open a terminal inside this folder.

```bash
python -m venv .venv
```
Windows: `.venv\Scripts\activate`

macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python src/train.py
python src/predict.py --input data/example_input.csv --output reports/predictions.csv
```

Notebook: install Jupyter with `python -m pip install notebook`, then run `python -m notebook DiseasePrediction.ipynb`. Run all cells from the project root.

## Files
- src/train.py: data export, EDA, preprocessing, model tuning, selection and evaluation.
- src/predict.py: batch CSV predictions and input validation.
- DiseasePrediction.ipynb: explained, executed workflow.
- data/: full dataset, split indices, sample input.
- models/best_model.joblib: trained pipeline and feature schema.
- reports/: actual metrics, charts, predictions, validation and environment versions.
- SUBMISSION.md: internship instructions and video outline.

## Dataset and method
UCI Breast Cancer Wisconsin (Diagnostic), loaded offline via sklearn: 569 observations and 30 numeric tumor measurements derived from digitized fine needle aspirate images. It does not contain age, symptoms or blood test fields; none are fabricated. Class mapping is explicitly remapped to 1=malignant, 0=benign.

A fixed stratified 80/20 split uses seed 42. Median imputation and standard scaling are inside pipelines and learn only from training folds. Five-fold training-only CV ROC-AUC selects hyperparameters and the winning model. The held-out test partition is excluded from selection. All four models receive final test evaluation. Threshold=0.5 is fixed in advance. Precision, recall and F1 use malignant as positive. The saved artifact remains fitted only on training data so its test results are honest.

See reports/RESULTS.md for actual results. Model selection across candidates can make CV estimates optimistic; the untouched test partition supplies the final check. No external validation has been performed.

## Prediction inputs
Use exactly the 30 feature columns shown in data/example_input.csv. Column order is restored automatically. Extra/missing columns, nonnumeric values, infinity, negative values and entirely blank rows are rejected. Numeric blanks are imputed using fitted training medians. Measurements must match dataset definitions and units. Example inputs come from the held-out split, not an independent population. Model probability scores are not calibrated clinical risk estimates.

## Reproducibility and limitations
See reports/environment.txt for exact versions used. Reinstall matching versions or retrain if artifact loading fails. Only load trusted joblib files. If XGBoost is missing, the script explicitly reports its omission and runs three models; install requirements for the complete comparison.

Small historical dataset, single test split, no prospective, external or subgroup validation. Does not predict future disease onset or diagnose from general symptoms. False negatives matter: malignant classified as benign. Clinical use would need independent validation, calibration and appropriate oversight.

## Sources
- UCI: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn loader: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. DOI: 10.24432/C5DW2B.
