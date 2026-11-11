import math
import numpy as np
import cv2

# Face coordinates: +x image-right, +y down, +z away from camera.
POSE_MODEL = np.array([(0, 0, 0), (0, 330, 65), (-225, -170, 135),
                       (225, -170, 135), (-150, 150, 125), (150, 150, 125)], dtype=np.float64)
SCHEMAS = {
    'ibug68': {'left': list(range(36, 42)), 'right': list(range(42, 48)), 'pose': [30, 8, 36, 45, 48, 54]},
    'mediapipe478': {'left': [33, 160, 158, 133, 153, 144], 'right': [362, 385, 387, 263, 373, 380],
                     'pose': [1, 152, 33, 263, 61, 291]},
}


def eye_aspect_ratio(points):
    pts = np.asarray(points, dtype=float)
    if pts.shape != (6, 2) or not np.isfinite(pts).all():
        return None
    width = np.linalg.norm(pts[0] - pts[3])
    if width < 3:
        return None
    ear = (np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])) / (2 * width)
    return float(ear) if 0 <= ear <= 1 else None


class Geometry:
    def __init__(self, calibration, smoothing_tau=0.12):
        self.calibration = calibration
        self.tau = smoothing_tau
        self.last_time = None
        self.last_angles = None

    def measure(self, packet, width, height, timestamp):
        result = dict(ear_left=None, ear_right=None, yaw_deg=None, pitch_deg=None, roll_deg=None,
                      pose_reprojection_px=None)
        flags = []
        if packet.points is None:
            self.last_angles = None
            return result, ['no_landmarks']
        points = np.asarray(packet.points, dtype=float)
        schema = SCHEMAS.get(packet.schema)
        if (schema is None or points.ndim != 2 or points.shape[1] != 2
                or len(points) <= max(sum(schema.values(), [])) or not np.isfinite(points).all()):
            self.last_angles = None
            return result, ['invalid_landmarks']
        # Landmarks outside the image imply a clipped face, not measurable eyelids.
        for side in ('left', 'right'):
            eye = points[schema[side]]
            if ((eye >= 0).all() and (eye[:, 0] < width).all() and (eye[:, 1] < height).all()):
                result['ear_' + side] = eye_aspect_ratio(eye)
        c = self.calibration
        if c.camera_matrix is None:
            self.last_angles = None
            return result, ['pose_uncalibrated']
        k = np.array(c.camera_matrix, dtype=float)
        sx, sy = width / c.image_size[0], height / c.image_size[1]
        if abs(sx - sy) > 0.01:
            return result, ['camera_aspect_mismatch']
        k[0, :] *= sx
        k[1, :] *= sy
        distortion = np.array(c.distortion or [0.0] * 5)
        image_points = np.ascontiguousarray(points[schema['pose']], dtype=np.float64)
        try:
            ok, rotation, translation = cv2.solvePnP(POSE_MODEL, image_points, k, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or translation[2, 0] <= 0:
                raise ValueError('pose fit failed')
            projected, _ = cv2.projectPoints(POSE_MODEL, rotation, translation, k, distortion)
            error = float(np.sqrt(np.mean(np.sum((projected[:, 0] - image_points) ** 2, axis=1))))
            result['pose_reprojection_px'] = error
            if not math.isfinite(error) or error > max(3.0, width * 0.015):
                raise ValueError('pose reprojection error')
            matrix, _ = cv2.Rodrigues(rotation)
            angles = np.asarray(cv2.RQDecomp3x3(matrix)[0]) - np.asarray(c.neutral_angles)
            angles = (angles + 180) % 360 - 180  # pitch, yaw, roll
            if not np.isfinite(angles).all() or np.any(np.abs(angles) > [70, 75, 80]):
                raise ValueError('pose outside validated operating range')
            if self.last_angles is not None and self.last_time is not None and timestamp - self.last_time <= 0.2:
                alpha = 1 - math.exp(-(timestamp - self.last_time) / self.tau)
                angles = self.last_angles + alpha * ((angles - self.last_angles + 180) % 360 - 180)
            self.last_angles, self.last_time = angles, timestamp
            result.update(pitch_deg=float(angles[0]), yaw_deg=float(angles[1]), roll_deg=float(angles[2]))
        except (cv2.error, ValueError):
            self.last_angles = None
            flags.append('pose_rejected')
        return result, flags
