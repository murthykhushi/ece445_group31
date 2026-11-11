import unittest
from vision.config import Calibration
from vision.temporal import TemporalMetrics


class TemporalTests(unittest.TestCase):
    def engine(self, **kwargs):
        return TemporalMetrics(Calibration(), **kwargs)

    def test_missing_face_is_unknown_and_not_open(self):
        engine = self.engine(window_sec=1, max_gap_sec=1)
        engine.update(0, .3, .3)
        engine.update(.5, None, None)
        result = engine.update(1, .3, .3)
        self.assertEqual(result['valid_coverage_60s'], 0)
        self.assertIsNone(result['closure_fraction_60s'])

    def test_variable_rate_uses_duration_not_frame_count(self):
        engine = self.engine(window_sec=1, max_gap_sec=1)
        engine.update(0, .1, .1)
        engine.update(.7, .3, .3)
        engine.update(.8, .3, .3)
        result = engine.update(1, .3, .3)
        self.assertAlmostEqual(result['closure_fraction_60s'], .7)
        self.assertAlmostEqual(result['valid_coverage_60s'], 1)

    def test_window_boundary_clips_old_interval(self):
        engine = self.engine(window_sec=1, max_gap_sec=1)
        engine.update(0, .1, .1)
        engine.update(.7, .3, .3)
        engine.update(1, .3, .3)
        result = engine.update(1.2, .3, .3)
        self.assertAlmostEqual(result['closure_fraction_60s'], .5)

    def test_gap_is_entirely_unknown_and_cancels_blink(self):
        engine = self.engine(window_sec=1)
        engine.update(0, .3, .3)
        engine.update(.05, .1, .1)
        result = engine.update(.4, .3, .3)
        self.assertIn('sampling_gap', result['quality_flags'])
        self.assertEqual(len(engine.blinks), 0)
        self.assertAlmostEqual(result['valid_coverage_60s'], .05)

    def test_blink_hysteresis_and_event_duration(self):
        engine = self.engine(window_sec=.3)
        engine.update(0, .3, .3)
        engine.update(.05, .1, .1)
        engine.update(.1, .21, .21)
        event = engine.update(.15, .3, .3)
        self.assertAlmostEqual(event['blink_event_duration_ms'], 100)
        engine.update(.2, .3, .3)
        result = engine.update(.3, .3, .3)
        self.assertEqual(result['blink_count_60s'], 1)
        self.assertAlmostEqual(result['median_blink_ms'], 100)

    def test_initial_closed_is_not_a_blink(self):
        engine = self.engine()
        engine.update(0, .1, .1)
        result = engine.update(.1, .3, .3)
        self.assertIsNone(result['blink_event_duration_ms'])

    def test_long_closure_is_not_blink(self):
        engine = self.engine(max_gap_sec=2)
        engine.update(0, .3, .3)
        engine.update(.05, .1, .1)
        result = engine.update(1.1, .3, .3)
        self.assertIsNone(result['blink_event_duration_ms'])

    def test_unknown_eye_or_disagreement_is_unknown(self):
        for left, right in [(None, .3), (.1, .3), (float('nan'), .3)]:
            result = self.engine().update(0, left, right)
            self.assertEqual(result['eye_state'], 'UNKNOWN')

    def test_warmup_and_bad_timestamps(self):
        engine = self.engine()
        self.assertIsNone(engine.update(1, .3, .3)['closure_fraction_60s'])
        for timestamp in (1, .9, float('nan')):
            with self.assertRaises(ValueError):
                engine.update(timestamp, .3, .3)

    def test_closed_fraction_not_claimed_as_perclos(self):
        from vision.contracts import MetricPacket
        packet = MetricPacket(1, 1, 'test', closure_fraction_60s=.2)
        self.assertIsNone(packet.perclos80_60s)

    def test_calibration_validation(self):
        with self.assertRaises(ValueError):
            Calibration(close_left=.3, open_left=.2)
