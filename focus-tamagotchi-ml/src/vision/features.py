"""One explicit feature contract for collection, training and inference."""
import hashlib
import json
import math

FEATURES = ['ear_left', 'ear_right', 'closure_fraction_60s', 'blink_count_60s', 'median_blink_ms',
            'yaw_deg', 'pitch_deg', 'roll_deg', 'active_app', 'keystrokes', 'mouse_clicks']
SCHEMA_ID = hashlib.sha256(json.dumps({'version': 2, 'features': FEATURES}, sort_keys=True).encode()).hexdigest()


def feature_row(metric, desktop):
    if (metric.eye_state == 'UNKNOWN' or metric.closure_fraction_60s is None
            or metric.valid_coverage_60s < 0.8 or 'stale_frame' in metric.quality_flags):
        return None
    values = metric.to_dict()
    values.update(active_app=desktop['active_app_encoded'], keystrokes=desktop['keystrokes'], mouse_clicks=desktop['mouse_clicks'])
    return {key: values[key] if values[key] is not None else float('nan') for key in FEATURES}


def load_focus_model(path):
    import joblib
    bundle = joblib.load(path)  # Load only your own trusted training artifacts.
    if not isinstance(bundle, dict) or bundle.get('schema_id') != SCHEMA_ID or bundle.get('features') != FEATURES:
        raise ValueError('Incompatible focus model. Collect v2 sessions and run train_real.py; legacy six-feature pickles are not accepted.')
    return bundle


def predict_focus(bundle, row):
    import pandas as pd
    score = float(bundle['model'].predict(pd.DataFrame([row], columns=FEATURES))[0])
    if not math.isfinite(score):
        raise ValueError('Focus model returned a non-finite score')
    return min(100.0, max(0.0, score))
