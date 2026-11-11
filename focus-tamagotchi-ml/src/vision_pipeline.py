from pathlib import Path
from collections import deque
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request

# width frames are shrunk to before hitting the model (optimized mode only)
PROC_WIDTH = 320


class VisionPipeline:
    # mode toggles the A/B test:
    #   baseline  = CPU delegate, IMAGE mode (full detection every frame), full-res frames
    #   optimized = GPU delegate (falls back to CPU), VIDEO mode (detect once then track), downscaled frames
    def __init__(self, ear_threshold=0.21, perclos_window_sec=60, mode="optimized"):
        self.ear_threshold = ear_threshold
        self.perclos_window_sec = perclos_window_sec
        self.optimized = (mode == "optimized")

        # deque for O(1) popping and a running total to avoid O(n) summation
        self.closed_frames_history = deque()
        self.closed_count = 0

        # standard mediapipe landmark indices for eye contours
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # VIDEO mode needs strictly increasing timestamps in ms
        self._t0 = time.monotonic()
        self._last_ts = -1

        # download the face landmarker task file if we dont have it already
        self.model_path = str(Path(__file__).resolve().parent / 'face_landmarker.task')
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe FaceLandmarker task file...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        target_delegate = (python.BaseOptions.Delegate.GPU if self.optimized
                           else python.BaseOptions.Delegate.CPU)

        try:
            self.detector = self._build_detector(target_delegate)
            print(f"Vision initialized successfully in {mode.upper()} mode.")
        except Exception as e:
            # if gpu fails (common on desktop python builds), fall back to cpu
            print(f"Warning: Failed to initialize {target_delegate}. Falling back to CPU. ({e})")
            self.detector = self._build_detector(python.BaseOptions.Delegate.CPU)

    def _build_detector(self, delegate):
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=delegate
        )
        running_mode = (vision.RunningMode.VIDEO if self.optimized
                        else vision.RunningMode.IMAGE)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def calculate_ear(self, landmarks, eye_indices, frame_width, frame_height):
        def dist(p1_idx, p2_idx):
            dx = (landmarks[p1_idx].x - landmarks[p2_idx].x) * frame_width
            dy = (landmarks[p1_idx].y - landmarks[p2_idx].y) * frame_height
            return (dx * dx + dy * dy) ** 0.5

        v1 = dist(eye_indices[1], eye_indices[5])
        v2 = dist(eye_indices[2], eye_indices[4])
        h = dist(eye_indices[0], eye_indices[3])

        # normal EAR formula, 1e-6 so we dont divide by zero
        return (v1 + v2) / (2.0 * h + 1e-6)

    def process_frame(self, frame):
        if self.optimized:
            # downscale BEFORE color conversion so cvtColor + the model touch ~4x fewer pixels
            h0, w0 = frame.shape[:2]
            w_frame = PROC_WIDTH
            h_frame = int(h0 * PROC_WIDTH / w0)
            frame = cv2.resize(frame, (w_frame, h_frame), interpolation=cv2.INTER_AREA)
        else:
            h_frame, w_frame = frame.shape[:2]

        # Metal/GPU needs 4 channels (RGBA) for memory alignment
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)

        if self.optimized:
            ts = int((time.monotonic() - self._t0) * 1000)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            results = self.detector.detect_for_video(mp_image, ts)
        else:
            results = self.detector.detect(mp_image)

        ear_avg = 0.0
        is_eyes_closed = 0

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE, w_frame, h_frame)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, w_frame, h_frame)

            # average both eyes to get one score
            ear_avg = (left_ear + right_ear) / 2.0

            if ear_avg < self.ear_threshold:
                is_eyes_closed = 1

        # keep track of closed frames for PERCLOS
        current_time = time.time()
        self.closed_frames_history.append((current_time, is_eyes_closed))
        self.closed_count += is_eyes_closed

        # pop old frames and decrement running total without iterating the window
        while self.closed_frames_history and current_time - self.closed_frames_history[0][0] > self.perclos_window_sec:
            _, old_val = self.closed_frames_history.popleft()
            self.closed_count -= old_val

        perclos = self.closed_count / len(self.closed_frames_history) if self.closed_frames_history else 0.0

        return {
            "ear": ear_avg,
            "perclos": perclos,
            "is_closed": is_eyes_closed
        }