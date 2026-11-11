"""Compare timestamped replay metrics to independent labels; no training in this tool."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_events(predicted, reference, tolerance=0.1):
    if not predicted or not reference:
        return []
    distances = np.abs(np.asarray(predicted)[:, None] - np.asarray(reference)[None, :])
    # Penalty exceeds sum of all feasible distances: maximize matches before minimizing error.
    penalty = (min(len(predicted), len(reference)) + 1) * max(1, tolerance)
    rows, columns = linear_sum_assignment(np.where(distances <= tolerance, distances, penalty))
    return [(int(i), int(j)) for i, j in zip(rows, columns) if distances[i, j] <= tolerance]


def match_samples(predicted, reference, tolerance):
    # Linear memory/time alignment; frame streams can contain tens of thousands of samples.
    pairs = []
    i = j = 0
    while i < len(predicted) and j < len(reference):
        delta = predicted[i] - reference[j]
        if abs(delta) <= tolerance:
            pairs.append((i, j))
            i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return pairs


def evaluate(predictions, labels, tolerance=0.05):
    if predictions.empty or labels.empty:
        raise ValueError('Predictions and labels must be nonempty')
    for frame in (predictions, labels):
        if not frame.timestamp.is_monotonic_increasing or frame.timestamp.duplicated().any():
            raise ValueError('Timestamps must increase strictly')
    pairs = match_samples(predictions.timestamp.tolist(), labels.timestamp.tolist(), tolerance)
    report = dict(predicted_frames=len(predictions), labeled_frames=len(labels), matched_frames=len(pairs),
                  aligned_label_fraction=len(pairs)/len(labels))
    for column in ['closure_fraction_60s', 'yaw_deg', 'pitch_deg', 'roll_deg']:
        if column not in labels or column not in predictions:
            continue
        differences = [abs(predictions.iloc[i][column] - labels.iloc[j][column]) for i, j in pairs
                       if pd.notna(predictions.iloc[i][column]) and pd.notna(labels.iloc[j][column])]
        report[column + '_evaluated_pairs'] = len(differences)
        report[column + '_mae'] = float(np.mean(differences)) if differences else None
    if 'eye_state' in labels:
        valid_pairs = [(i,j) for i,j in pairs if predictions.iloc[i].eye_state != 'UNKNOWN' and labels.iloc[j].eye_state != 'UNKNOWN']
        report['eye_state_evaluated_pairs'] = len(valid_pairs)
        report['eye_state_accuracy'] = float(np.mean([predictions.iloc[i].eye_state == labels.iloc[j].eye_state for i,j in valid_pairs])) if valid_pairs else None
    if 'blink_event_duration_ms' in labels and 'blink_event_duration_ms' in predictions:
        pe = predictions[predictions.blink_event_duration_ms.notna()]
        re = labels[labels.blink_event_duration_ms.notna()]
        event_pairs = match_events(pe.timestamp.tolist(), re.timestamp.tolist(), 0.1)
        tp, fp, fn = len(event_pairs), len(pe)-len(event_pairs), len(re)-len(event_pairs)
        report.update(blink_tp=tp, blink_fp=fp, blink_fn=fn,
                      blink_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                      blink_duration_mae_ms=float(np.mean([abs(pe.iloc[i].blink_event_duration_ms-re.iloc[j].blink_event_duration_ms) for i,j in event_pairs])) if event_pairs else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(pd.read_json(args.predictions, lines=True), pd.read_csv(args.labels), args.tolerance)
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.write_text(text+'\n')


if __name__ == '__main__':
    main()
