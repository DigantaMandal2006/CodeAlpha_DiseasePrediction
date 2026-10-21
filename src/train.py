"""Reproducible CodeAlpha Task 4: train-only model selection, held-out testing."""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay)

ROOT = Path(__file__).resolve().parents[1]

def main():
    for name in ['data', 'reports', 'models']:
        (ROOT / name).mkdir(exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    # sklearn uses 0=malignant. Remap so the disease class is positive.
    y = (dataset.target == 0).astype(int).rename('malignant')
    pd.concat([X, y], axis=1).to_csv(ROOT/'data/breast_cancer.csv', index=False)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    Xtest.head(5).to_csv(ROOT/'data/example_input.csv', index=False)
    pd.DataFrame({'row_index': X.index, 'split': np.where(
        X.index.isin(Xtest.index), 'test', 'train')}).to_csv(ROOT/'data/split.csv', index=False)
    summary = {'samples': len(X), 'features': X.shape[1],
        'missing_values': int(X.isna().sum().sum()), 'duplicate_feature_rows': int(X.duplicated().sum()),
        'benign': int((y == 0).sum()), 'malignant': int(y.sum()),
        'train_rows': len(Xtrain), 'test_rows': len(Xtest)}
    (ROOT/'reports/data_summary.json').write_text(json.dumps(summary, indent=2))
    X.describe().to_csv(ROOT/'reports/feature_statistics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y.value_counts().reindex([0, 1]).plot.bar(ax=axes[0], color=['#237b84', '#d46658'])
    axes[0].set(xticklabels=['Benign', 'Malignant'], title='Class distribution', ylabel='Samples')
    for cls, label in [(0, 'Benign'), (1, 'Malignant')]:
        axes[1].hist(X.loc[y == cls, 'mean radius'], bins=22, alpha=.6, label=label)
    axes[1].set(title='Mean radius by class', xlabel='Mean radius', ylabel='Samples')
    axes[1].legend(); fig.tight_layout(); fig.savefig(ROOT/'reports/eda.png', dpi=160); plt.close(fig)
    candidates = {
        'LogisticRegression': (LogisticRegression(max_iter=3000, random_state=42), {'model__C': [.1, 1, 10]}),
        'SVM': (SVC(probability=True, random_state=42), {'model__C': [.1, 1, 10]}),
        'RandomForest': (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
                         {'model__max_depth': [None, 5], 'model__min_samples_leaf': [1, 3]})}
    try:
        from xgboost import XGBClassifier
        candidates['XGBoost'] = (XGBClassifier(n_estimators=100, learning_rate=.05,
            eval_metric='logloss', random_state=42, n_jobs=1), {'model__max_depth': [2, 3]})
    except ImportError:
        print('XGBoost is unavailable; install requirements.txt to include it.')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    searches = {}
    for name, (model, grid) in candidates.items():
        pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('scale', StandardScaler()), ('model', model)])
        search = GridSearchCV(pipe, grid, scoring='roc_auc', cv=cv, n_jobs=1)
        search.fit(Xtrain, ytrain)
        searches[name] = search
    # Freeze selection before looking at the test set.
    winner = max(searches, key=lambda name: searches[name].best_score_)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, search in searches.items():
        pred = search.predict(Xtest); proba = search.predict_proba(Xtest)[:, 1]
        rows.append({'model': name, 'cv_roc_auc': search.best_score_,
            'cv_std': search.cv_results_['std_test_score'][search.best_index_],
            'accuracy': accuracy_score(ytest, pred), 'precision': precision_score(ytest, pred, zero_division=0),
            'recall': recall_score(ytest, pred), 'f1': f1_score(ytest, pred),
            'roc_auc': roc_auc_score(ytest, proba), 'best_parameters': json.dumps(search.best_params_)})
        RocCurveDisplay.from_predictions(ytest, proba, name=name, ax=ax)
    ax.set_title('Held-out test ROC curves'); fig.tight_layout()
    fig.savefig(ROOT/'reports/roc_curves.png', dpi=160); plt.close(fig)
    best = searches[winner].best_estimator_
    pred = best.predict(Xtest); proba = best.predict_proba(Xtest)[:, 1]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(ytest, pred, display_labels=['Benign', 'Malignant'], ax=ax, colorbar=False)
    ax.set_title(f'{winner}: held-out test'); fig.tight_layout()
    fig.savefig(ROOT/'reports/confusion_matrix.png', dpi=160); plt.close(fig)
    metrics = pd.DataFrame(rows).sort_values('cv_roc_auc', ascending=False)
    metrics.to_csv(ROOT/'reports/model_comparison.csv', index=False)
    (ROOT/'reports/classification_report.txt').write_text(classification_report(
        ytest, pred, target_names=['Benign', 'Malignant'], zero_division=0))
    pd.DataFrame({'row_index': Xtest.index, 'actual': ytest.to_numpy(), 'predicted': pred,
        'malignant_probability': proba}).to_csv(ROOT/'reports/test_predictions.csv', index=False)
    joblib.dump({'pipeline': best, 'features': list(X.columns), 'model_name': winner,
        'positive_class': 'malignant', 'threshold': .5}, ROOT/'models/best_model.joblib')
    (ROOT/'reports/selection.json').write_text(json.dumps({'selected_model': winner,
        'selection_rule': 'Highest mean training-only 5-fold CV ROC-AUC',
        'seed': 42, 'threshold': .5, 'test_used_for_selection': False}, indent=2))
    print(metrics.to_string(index=False)); print('\nSelected:', winner)
    return metrics

if __name__ == '__main__':
    main()
