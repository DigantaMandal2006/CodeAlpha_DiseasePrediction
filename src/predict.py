"""Batch inference using the exact training feature schema."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def predict_frame(frame, bundle):
    required = bundle['features']
    if frame.empty:
        raise ValueError('Input CSV contains no rows.')
    missing = set(required) - set(frame.columns)
    extra = set(frame.columns) - set(required)
    if missing or extra:
        raise ValueError(f'Feature mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}')
    values = frame[required].apply(pd.to_numeric, errors='raise')
    if np.isinf(values.to_numpy()).any() or (values < 0).any().any():
        raise ValueError('Features must be nonnegative, finite measurements (blank values may be imputed).')
    if values.isna().all(axis=1).any():
        raise ValueError('A row cannot have all features missing.')
    probabilities = bundle['pipeline'].predict_proba(values)[:, 1]
    result = frame.copy()
    result['malignant_probability'] = probabilities
    result['prediction'] = np.where(probabilities >= bundle['threshold'], 'Malignant', 'Benign')
    return result

def main():
    parser = argparse.ArgumentParser(description='Educational breast tumor classifier; not for clinical use.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/predictions.csv')
    args = parser.parse_args()
    # Load only this project's trusted model artifact.
    bundle = joblib.load(ROOT/'models/best_model.joblib')
    result = predict_frame(pd.read_csv(args.input), bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[['prediction', 'malignant_probability']].to_string(index=False))

if __name__ == '__main__':
    main()
