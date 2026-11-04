import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from vision.capture import LatestQueue
from vision.contracts import FramePacket, LandmarkPacket, MetricPacket
from vision.pipeline import VisionPipeline
from vision.features import FEATURES, feature_row, load_focus_model


class FakeBackend:
    actual_backend = 'fake-cpu'
    identity = {'backend': 'fake-cpu', 'schema': 'ibug68'}
    def infer(self, frame):
        return LandmarkPacket(flags=['no_face'])
    def close(self):
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def test_latest_queue_drops_pending_only(self):
        queue = LatestQueue()
        queue.put('one'); queue.put('two')
        self.assertEqual(queue.dropped, 1)
        current = queue.get()
        queue.put('three')
        self.assertEqual(current, 'two')
        self.assertEqual(queue.get(), 'three')

    def test_close_wakes_blocked_consumer(self):
        queue = LatestQueue()
        results = []
        worker = threading.Thread(target=lambda: results.append(queue.get()))
        worker.start()
        queue.close()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [None])

    def test_pipeline_no_face_and_actual_backend(self):
        backend = FakeBackend()
        pipeline = VisionPipeline(backend_instance=backend)
        now = time.monotonic()
        packet = pipeline.process(FramePacket(1, now, np.zeros((480,640,3), dtype=np.uint8), now))
        self.assertEqual(packet.eye_state, 'UNKNOWN')
        self.assertEqual(packet.backend_actual, 'fake-cpu')
        self.assertIsNone(packet.closure_fraction_60s)
        pipeline.close()
        self.assertTrue(backend.closed)

    def test_fallback_requires_opt_in_and_records_failure(self):
        with patch.object(VisionPipeline, '_create_backend', side_effect=RuntimeError('CUDA absent')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA absent'):
                VisionPipeline()
        with patch.object(VisionPipeline, '_create_backend', side_effect=[RuntimeError('CUDA absent'), FakeBackend()]):
            pipeline = VisionPipeline(allow_cpu_fallback=True)
            self.assertIn('CUDA absent', pipeline.identity['fallback_reason'])
            self.assertFalse(pipeline.identity['accelerated_run_valid'])
            self.assertEqual(pipeline.actual_backend, 'fake-cpu')

    def test_features_block_missing_and_stale(self):
        metric = MetricPacket(1, 1, 'test')
        desktop = dict(active_app_encoded=0, keystrokes=2, mouse_clicks=1)
        self.assertIsNone(feature_row(metric, desktop))
        metric.eye_state='OPEN'; metric.closure_fraction_60s=.1; metric.valid_coverage_60s=.9
        row = feature_row(metric, desktop)
        self.assertEqual(list(row), FEATURES)
        self.assertTrue(np.isnan(row['yaw_deg']))
        metric.quality_flags=['stale_frame']
        self.assertIsNone(feature_row(metric, desktop))

    def test_legacy_model_is_rejected(self):
        import joblib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.joblib'
            joblib.dump({'features':['ear','perclos']},path)
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                load_focus_model(path)
