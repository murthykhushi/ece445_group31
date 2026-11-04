from pathlib import Path
import hashlib
import numpy as np
from vision.contracts import LandmarkPacket


class MediaPipeBackend:
    def __init__(self, model_path, gpu=False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f'MediaPipe model missing: {path}. Run src/download_models.py --mediapipe.')
        self.mp = mp
        self.actual_backend = 'mediapipe-gpu' if gpu else 'mediapipe-cpu'
        self.detector = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.GPU if gpu else BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO, num_faces=2,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5))
        self.last_ms = -1
        self.identity = {'backend': self.actual_backend, 'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'mediapipe_version': mp.__version__, 'schema': 'mediapipe478'}

    def infer(self, frame):
        timestamp_ms = int(frame.timestamp * 1000)
        if timestamp_ms <= self.last_ms:
            raise ValueError('MediaPipe requires strictly increasing millisecond timestamps')
        self.last_ms = timestamp_ms
        rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
        result = self.detector.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if len(result.face_landmarks) != 1:
            return LandmarkPacket(schema='mediapipe478', flags=['multiple_faces' if result.face_landmarks else 'no_face'])
        h, w = frame.image.shape[:2]
        points = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
        return LandmarkPacket(points=points, schema='mediapipe478')

    def close(self):
        self.detector.close()
