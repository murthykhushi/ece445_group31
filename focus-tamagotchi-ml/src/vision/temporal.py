"""Duration-weighted eye closure with explicit missing intervals."""
from collections import deque
from statistics import median
import math


class TemporalMetrics:
    def __init__(self, calibration, window_sec=60.0, max_gap_sec=0.1,
                 min_coverage=0.8, blink_min_sec=0.06, blink_max_sec=0.8):
        if window_sec <= 0 or max_gap_sec <= 0 or not 0 <= min_coverage <= 1:
            raise ValueError("Invalid temporal configuration")
        self.calibration = calibration
        self.window = window_sec
        self.max_gap = max_gap_sec
        self.min_coverage = min_coverage
        self.blink_min = blink_min_sec
        self.blink_max = blink_max_sec
        self.intervals = deque()
        self.blinks = deque()
        self.first = self.previous = None
        self.previous_state = "UNKNOWN"
        self.left_state = self.right_state = "UNKNOWN"
        self.closure_start = None

    @staticmethod
    def _state(ear, previous, close, opened):
        if ear is None or not math.isfinite(ear) or not 0 <= ear <= 1:
            return "UNKNOWN"
        if ear < close:
            return "CLOSED"
        if ear > opened:
            return "OPEN"
        return previous

    def update(self, timestamp, left, right):
        if not math.isfinite(timestamp) or (self.previous is not None and timestamp <= self.previous):
            raise ValueError("Metric timestamps must be finite and strictly increasing")
        if self.first is None:
            self.first = timestamp
        gap = self.previous is not None and timestamp - self.previous > self.max_gap + 1e-9
        if gap:
            self.left_state = self.right_state = "UNKNOWN"
            self.closure_start = None
        c = self.calibration
        self.left_state = self._state(left, self.left_state, c.close_left, c.open_left)
        self.right_state = self._state(right, self.right_state, c.close_right, c.open_right)
        state = self.left_state if self.left_state == self.right_state else "UNKNOWN"
        if self.previous is not None:
            valid = not gap and state != "UNKNOWN" and self.previous_state != "UNKNOWN"
            self.intervals.append((self.previous, timestamp, self.previous_state == "CLOSED", valid))
        blink_event = None
        if state == "UNKNOWN":
            self.closure_start = None
        elif not gap and self.previous_state == "OPEN" and state == "CLOSED":
            self.closure_start = timestamp
        elif state == "OPEN" and self.previous_state == "CLOSED" and self.closure_start is not None:
            duration = timestamp - self.closure_start
            if self.blink_min <= duration <= self.blink_max:
                self.blinks.append((timestamp, duration))
                blink_event = duration * 1000
            self.closure_start = None
        cutoff = timestamp - self.window
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        while self.blinks and self.blinks[0][0] <= cutoff:
            self.blinks.popleft()
        valid_time = closed_time = 0.0
        for start, end, closed, valid in self.intervals:
            dt = end - max(start, cutoff)
            if valid:
                valid_time += dt
                closed_time += dt * closed
        coverage = min(1.0, valid_time / self.window)
        ready = timestamp - self.first >= self.window - 1e-9 and coverage >= self.min_coverage - 1e-9
        flags = []
        if timestamp - self.first < self.window - 1e-9:
            flags.append("warming_up")
        if coverage < self.min_coverage - 1e-9:
            flags.append("low_eye_coverage")
        if gap:
            flags.append("sampling_gap")
        if state == "UNKNOWN":
            flags.append("eyes_unknown")
        self.previous, self.previous_state = timestamp, state
        return dict(eye_state=state, blink_event_duration_ms=blink_event, closure_fraction_60s=closed_time / valid_time if ready and valid_time else None,
                    valid_coverage_60s=coverage, blink_count_60s=len(self.blinks) if ready else None,
                    median_blink_ms=median(d * 1000 for _, d in self.blinks) if ready and self.blinks else None,
                    quality_flags=flags)
