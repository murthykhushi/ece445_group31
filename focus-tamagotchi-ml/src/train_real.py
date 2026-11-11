"""Train a v2 model with held-out sessions; never split adjacent rows at random."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from vision.features import FEATURES, SCHEMA_ID

ROOT = Path(__file__).resolve().parents[1]


def train(data_path, output_path):
    data = pd.read_csv(data_path)
    required = ['session_id', 'schema_id', 'landmark_schema', *FEATURES, 'focus_score']
    if any(column not in data for column in required) or not (data['schema_id'] == SCHEMA_ID).all():
        raise ValueError('Expected schema-v2 data from data_collection.py; legacy data cannot be relabeled as v2')
    if len(data) < 10 or data.session_id.nunique() < 2 or data.session_id.isna().any():
        raise ValueError('Collect at least 10 labeled rows across at least 2 independent sessions')
    numeric = data[FEATURES + ['focus_score']].apply(pd.to_numeric, errors='raise')
    if np.isinf(numeric.to_numpy()).any() or numeric.focus_score.isna().any() or not numeric.focus_score.between(0, 100).all():
        raise ValueError('Invalid non-finite or out-of-range labels/features')
    if numeric[['ear_left', 'ear_right', 'closure_fraction_60s', 'blink_count_60s']].isna().any().any():
        raise ValueError('Required eye metrics are missing; invalid windows must not enter training')
    x, y = numeric[FEATURES], numeric.focus_score
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(x, y, groups=data.session_id))
    model = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                      ('regressor', RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=1))])
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(x.iloc[test_idx])
    report = dict(train_rows=len(train_idx), test_rows=len(test_idx),
                  train_sessions=sorted(data.session_id.iloc[train_idx].unique().tolist()),
                  test_sessions=sorted(data.session_id.iloc[test_idx].unique().tolist()),
                  held_out_mae=float(mean_absolute_error(y.iloc[test_idx], predictions)),
                  constant_baseline_mae=float(mean_absolute_error(y.iloc[test_idx], np.full(len(test_idx), y.iloc[train_idx].mean()))),
                  evaluation='single held-out session split; not a population accuracy claim')
    bundle = dict(schema_id=SCHEMA_ID, features=FEATURES, model=model, report=report,
                  landmark_schemas=sorted(data.landmark_schema.unique().tolist()))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    output_path.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Saved {output_path}; fitted on training sessions only.')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/focus_v2.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'focus_v2.joblib')
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == '__main__':
    main()
