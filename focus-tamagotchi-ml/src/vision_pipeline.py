from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request

class VisionPipeline:
    # added the mode parameter here so we can toggle gpu/cpu from terminal for the A/B test
    def __init__(self, ear_threshold=0.21, perclos_window_sec=60, mode="optimized"):
        self.ear_threshold = ear_threshold
        self.perclos_window_sec = perclos_window_sec
        
        # Tuple format: (timestamp_float, is_closed_int)
        self.closed_frames_history = [] 
        
        # standard mediapipe landmark indices for eye contours
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # download the face landmarker task file if we dont have it already
        self.model_path = str(Path(__file__).resolve().parent / 'face_landmarker.task')
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe FaceLandmarker task file...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        # figure out if we are using GPU (optimized) or CPU (baseline) 
        target_delegate = python.BaseOptions.Delegate.GPU if mode == "optimized" else python.BaseOptions.Delegate.CPU

        try:
            # try to load with the target delegate
            base_options = python.BaseOptions(
                model_asset_path=self.model_path,
                delegate=target_delegate
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print(f"Vision initialized successfully in {mode.upper()} mode.")
        except Exception as e:
            # if gpu fails for some reason on mac, fallback to cpu so it doesnt crash
            print(f"Warning: Failed to initialize {target_delegate}. Falling back to CPU. ({e})")
            base_options = python.BaseOptions(
                model_asset_path=self.model_path,
                delegate=python.BaseOptions.Delegate.CPU
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
    

    def calculate_ear(self, landmarks, eye_indices, frame_width, frame_height):
        # extract (x, y) pixel coordinates for the eye landmarks
        pts = np.array([
            [landmarks[idx].x * frame_width, landmarks[idx].y * frame_height] 
            for idx in eye_indices
        ])
        
        # compute vertical and horizontal distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        
        # normal EAR formula, add 1e-6 to denominator so we dont divide by zero
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear

    def process_frame(self, frame):
        # format frame for mediapipe
        h_frame, w_frame, _ = frame.shape
        # MAC GPU FIX: Metal requires 4 channels (RGBA) for memory alignment, not 3 (RGB)
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)

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
        
        # filter out frames older than our perclos window
        self.closed_frames_history = [
            (t, val) for t, val in self.closed_frames_history 
            if current_time - t <= self.perclos_window_sec
        ]
        
        if self.closed_frames_history:
            perclos = sum(val for _, val in self.closed_frames_history) / len(self.closed_frames_history)
        else:
            perclos = 0.0
            
        return {
            "ear": ear_avg,
            "perclos": perclos,
            "is_closed": is_eyes_closed
        }