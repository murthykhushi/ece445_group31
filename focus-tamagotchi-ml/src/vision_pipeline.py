# vision_pipeline.py
import cv2
import mediapipe as mp
import numpy as np
import time

class VisionPipeline:
    def __init__(self, ear_threshold=0.21, perclos_window_sec=60):
        self.ear_threshold = ear_threshold
        self.perclos_window_sec = perclos_window_sec
        self.closed_frames_history = [] # stores timestamps and 0/1 for closed eyes
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for Left and Right eyes in MediaPipe Face Mesh
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def calculate_ear(self, landmarks, eye_indices, frame_width, frame_height):
        pts = np.array([[landmarks[idx].x * frame_width, landmarks[idx].y * frame_height] for idx in eye_indices])
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear

    def process_frame(self, frame):
        h_frame, w_frame, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        ear_avg = 0.0
        is_eyes_closed = 0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE, w_frame, h_frame)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, w_frame, h_frame)
            ear_avg = (left_ear + right_ear) / 2.0
            
            if ear_avg < self.ear_threshold:
                is_eyes_closed = 1
                
        current_time = time.time()
        self.closed_frames_history.append((current_time, is_eyes_closed))
        
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
        cv2.imshow("Vision Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
