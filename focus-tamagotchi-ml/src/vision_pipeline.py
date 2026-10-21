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
    def __init__(self, ear_threshold=0.21, perclos_window_sec=60):
        self.ear_threshold = ear_threshold
        self.perclos_window_sec = perclos_window_sec
        
        # Tuple format: (timestamp_float, is_closed_int)
        self.closed_frames_history = [] 
        
        # Standard MediaPipe landmark indices for eye contours
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Download model bundle if needed
        self.model_path = str(Path(__file__).resolve().parents[1] / 'face_landmarker.task')
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe FaceLandmarker task file...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    

    def calculate_ear(self, landmarks, eye_indices, frame_width, frame_height):
        """
        Calculates the Eye Aspect Ratio (EAR) using Euclidean distances 
        between the vertical and horizontal eye landmarks.
        """
        # Extract (x, y) pixel coordinates for the specified eye landmarks
        pts = np.array([
            [landmarks[idx].x * frame_width, landmarks[idx].y * frame_height] 
            for idx in eye_indices
        ])
        
        # Compute vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        
        # Compute horizontal distance
        h = np.linalg.norm(pts[0] - pts[3])
        
        # EAR formula (epsilon added to denominator to prevent division by zero)
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear

    def process_frame(self, frame):
        """
        Processes a single BGR frame, extracting face landmarks, calculating EAR, 
        and updating the rolling PERCLOS history.
        """
        h_frame, w_frame, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.detect(mp_image)
        
        ear_avg = 0.0
        is_eyes_closed = 0
        
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE, w_frame, h_frame)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, w_frame, h_frame)
            ear_avg = (left_ear + right_ear) / 2.0
            
            if ear_avg < self.ear_threshold:
                is_eyes_closed = 1
                
        # Update PERCLOS rolling window
        current_time = time.time()
        self.closed_frames_history.append((current_time, is_eyes_closed))
        
        # Filter out data points older than the defined window
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

# Standalone testing mode
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    pipeline = VisionPipeline()
    
    print("Starting Vision Pipeline (Press 'q' to quit)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        metrics = pipeline.process_frame(frame)
        
        cv2.putText(frame, f"EAR: {metrics['ear']:.3f} | PERCLOS: {metrics['perclos']:.3f}", 
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Vision Pipeline Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()