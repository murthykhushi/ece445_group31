"""Fit eye thresholds from labeled samples or intrinsics from checkerboard images."""
import argparse
from dataclasses import asdict
import glob
import json
from pathlib import Path
import numpy as np
from vision.config import Calibration


def fit_eyes(samples):
    result = {}
    for side in ('left', 'right'):
        opened = samples.loc[samples.eye_state == 'OPEN', 'ear_' + side].dropna().to_numpy()
        closed = samples.loc[samples.eye_state == 'CLOSED', 'ear_' + side].dropna().to_numpy()
        if min(len(opened), len(closed)) < 20:
            raise ValueError('Need at least 20 independently labeled OPEN and CLOSED EAR samples per eye')
        if not np.isfinite(opened).all() or not np.isfinite(closed).all():
            raise ValueError('Calibration EAR samples must be finite')
        low, high = np.quantile(closed, .9), np.quantile(opened, .1)
        if not 0 < low < high < 1 or high-low < .02:
            raise ValueError(f'{side} eye distributions overlap; collect better samples or inspect landmarks')
        result['close_' + side] = float(low + .35 * (high-low))
        result['open_' + side] = float(low + .65 * (high-low))
    return dict(result, calibrated_eyes=True)


def fit_camera(paths, columns, rows, square_size):
    import cv2
    if min(columns, rows) < 3 or square_size <= 0:
        raise ValueError('Use at least 3x3 interior corners and a positive square size')
    object_points = np.zeros((rows * columns, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    objects, images, size = [], [], None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        current_size = image.shape[::-1]
        if size is not None and current_size != size:
            raise ValueError('All calibration images must use the same resolution')
        size = current_size
        found, corners = cv2.findChessboardCornersSB(image, (columns, rows))
        if found:
            objects.append(object_points.copy())
            images.append(corners)
    if len(images) < 10:
        raise ValueError('Need at least 10 detectable checkerboard views across angles and image regions')
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(objects, images, size, None, None)
    if not np.isfinite(rms) or rms > 2:
        raise ValueError(f'Camera calibration RMS {rms:.2f}px is too large; recapture sharper, varied views')
    print(f'Calibrated {len(images)} views; reprojection RMS={rms:.3f}px')
    return dict(camera_matrix=matrix.tolist(), distortion=distortion.ravel().tolist(), image_size=list(size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing', type=Path, help='Preserve the other calibration fields from this JSON')
    parser.add_argument('--output', type=Path, required=True)
    sub = parser.add_subparsers(dest='mode', required=True)
    eye = sub.add_parser('eyes')
    eye.add_argument('--samples', type=Path, required=True, help='CSV: eye_state (manual OPEN/CLOSED), ear_left, ear_right')
    camera = sub.add_parser('camera')
    camera.add_argument('--images', required=True, help='Quoted glob of checkerboard images')
    camera.add_argument('--columns', type=int, required=True, help='Interior corner count')
    camera.add_argument('--rows', type=int, required=True)
    camera.add_argument('--square-size', type=float, default=1)
    args = parser.parse_args()
    data = asdict(Calibration.load(args.existing))
    data.pop('calibration_id')
    if args.mode == 'eyes':
        import pandas as pd
        data.update(fit_eyes(pd.read_csv(args.samples)))
    else:
        data.update(fit_camera(sorted(glob.glob(args.images)), args.columns, args.rows, args.square_size))
    Calibration(**data)  # validate before publishing the config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(f'Saved {args.output}. Eye calibration is a closure proxy, not validated anatomical PERCLOS80.')


if __name__ == '__main__':
    main()
