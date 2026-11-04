import unittest
import numpy as np
import cv2
from vision.config import Calibration
from vision.contracts import LandmarkPacket
from vision.geometry import Geometry, POSE_MODEL, SCHEMAS, eye_aspect_ratio


class GeometryTests(unittest.TestCase):
    def test_ear_known_shape_and_degenerate(self):
        points = np.array([[0, 0], [2, -1], [6, -1], [8, 0], [6, 1], [2, 1]])
        self.assertAlmostEqual(eye_aspect_ratio(points), .25)
        self.assertIsNone(eye_aspect_ratio(np.zeros((6, 2))))
        self.assertIsNone(eye_aspect_ratio(np.full((6, 2), np.nan)))

    def test_missing_face(self):
        values, flags = Geometry(Calibration()).measure(LandmarkPacket(), 640, 480, 0)
        self.assertIsNone(values['yaw_deg'])
        self.assertIn('no_landmarks', flags)

    def test_calibrated_pose_recovers_known_projection(self):
        k = np.array([[700., 0, 320], [0, 700, 240], [0, 0, 1]])
        rotation = np.array([.1, -.15, .05])
        projected, _ = cv2.projectPoints(POSE_MODEL, rotation, np.array([0., 0., 1800.]), k, np.zeros(5))
        points = np.zeros((68, 2))
        points[SCHEMAS['ibug68']['pose']] = projected[:, 0]
        values, flags = Geometry(Calibration(camera_matrix=k.tolist(), image_size=[640, 480])).measure(LandmarkPacket(points), 640, 480, 0)
        expected = cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0]
        np.testing.assert_allclose([values['pitch_deg'], values['yaw_deg'], values['roll_deg']], expected, atol=.01)
        self.assertLess(values['pose_reprojection_px'], .01)

    def test_intrinsics_required(self):
        values, flags = Geometry(Calibration()).measure(LandmarkPacket(np.ones((68, 2))*100), 640, 480, 0)
        self.assertIsNone(values['yaw_deg'])
        self.assertIn('pose_uncalibrated', flags)

    def test_calibration_requires_image_size(self):
        with self.assertRaises(ValueError):
            Calibration(camera_matrix=[[700, 0, 320], [0, 700, 240], [0, 0, 1]])
