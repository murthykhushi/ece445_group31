from dataclasses import asdict
from pathlib import Path
import time
from vision.config import Calibration
from vision.contracts import FramePacket, MetricPacket
from vision.geometry import Geometry
from vision.temporal import TemporalMetrics

ROOT = Path(__file__).resolve().parents[2]


class VisionPipeline:
    def __init__(self, backend='torch-cuda', calibration=None, model_path=None,
                 profile='optimized', detector_hz=5, fp16=False, threads=1,
                 allow_cpu_fallback=False, backend_instance=None, window_sec=60):
        self.requested_backend = backend
        self.fallback_reason = None
        self.calibration = calibration or Calibration()
        self.geometry = Geometry(self.calibration)
        self.temporal = TemporalMetrics(self.calibration, window_sec=window_sec)
        self.seq = 0
        if backend_instance is not None:
            self.backend = backend_instance
        else:
            try:
                self.backend = self._create_backend(backend, model_path, profile, detector_hz, fp16, threads)
            except Exception as exc:
                if not allow_cpu_fallback or backend not in ('torch-cuda', 'mediapipe-gpu'):
                    raise
                self.fallback_reason = f'{type(exc).__name__}: {exc}'
                fallback = 'torch-cpu' if backend == 'torch-cuda' else 'mediapipe-cpu'
                self.backend = self._create_backend(fallback, model_path, profile, detector_hz, False, threads)
        self.actual_backend = self.backend.actual_backend
        self.identity = dict(self.backend.identity, requested_backend=backend,
                             fallback_reason=self.fallback_reason, accelerated_run_valid=self.fallback_reason is None and self.actual_backend in ('torch-cuda', 'mediapipe-gpu'),
                             calibration=asdict(self.calibration), window_sec=window_sec)

    @staticmethod
    def _create_backend(backend, model_path, profile, detector_hz, fp16, threads):
        if backend in ('torch-cuda', 'torch-cpu'):
            from vision.backends.torch_backend import TorchBackend
            return TorchBackend(model_path or ROOT / 'models/2DFAN4-cd938726ad.zip',
                                device=backend.split('-')[1], profile=profile, detector_hz=detector_hz,
                                fp16=fp16, threads=threads)
        if backend in ('mediapipe-cpu', 'mediapipe-gpu'):
            from vision.backends.mediapipe_backend import MediaPipeBackend
            candidates = [ROOT / 'models/face_landmarker.task', ROOT / 'face_landmarker.task', ROOT / 'src/face_landmarker.task']
            path = model_path or next((p for p in candidates if p.exists()), candidates[0])
            return MediaPipeBackend(path, gpu=backend.endswith('gpu'))
        raise ValueError(f'Unknown backend: {backend}')

    def process(self, frame):
        start = time.monotonic()
        landmarks = self.backend.infer(frame)
        h, w = frame.image.shape[:2]
        geometry, flags = self.geometry.measure(landmarks, w, h, frame.timestamp)
        temporal = self.temporal.update(frame.timestamp, geometry['ear_left'], geometry['ear_right'])
        flags.extend(landmarks.flags)
        flags.extend(temporal.pop('quality_flags'))
        if not self.calibration.calibrated_eyes:
            flags.append('eyes_uncalibrated')
        if self.fallback_reason:
            flags.append('cpu_fallback')
        now = time.monotonic()
        age = (now - frame.read_completed) * 1000
        if age > 100:
            flags.append('stale_frame')
        return MetricPacket(seq=frame.seq, timestamp=frame.timestamp, backend_actual=self.actual_backend,
                            **geometry, **temporal, frame_age_ms=age, processing_ms=(now-start)*1000,
                            detection_ran=landmarks.detection_ran, gpu_ms=landmarks.gpu_ms,
                            quality_flags=sorted(set(flags)), calibration_id=self.calibration.calibration_id)

    def process_frame(self, image):
        """Compatibility entry point returning a v2 dictionary (no false PERCLOS alias)."""
        self.seq += 1
        now = time.monotonic()
        return self.process(FramePacket(self.seq, now, image, now)).to_dict()

    def close(self):
        self.backend.close()
